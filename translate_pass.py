"""
Monggu Translate - Pass 2: Dịch theo ngữ cảnh kịch bản
Load ocr_data.json → chuẩn hóa chữ OCR → dịch theo kịch bản (Phase 1-4) → xuất translation_map.json
Mọi lời gọi LLM qua llm.ask_json (Gemini, dự phòng DeepSeek; có cache → chạy lại không tốn lại).

Usage:
    python translate_pass.py
"""

import sys

# Đảm bảo console in được tiếng Việt Unicode (kể cả khi stdout là pipe/subprocess)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import json
import os
import re
import time
import argparse
from difflib import get_close_matches

from config import (
    GLOSSARY_PATH,
    TRANSLATION_GENRE, TRANSLATION_SYSTEM_PROMPT,
    NORMALIZATION_SYSTEM_PROMPT,
    OCR_DATA_PATH, TRANSLATION_MAP_PATH,
)
from llm import ask_json, LLMError
from utils import write_json_atomic


def load_glossary() -> dict:
    """Load bảng glossary tên riêng."""
    if os.path.exists(GLOSSARY_PATH):
        with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
            glossary = json.load(f)
        print(f"  Loaded glossary: {len(glossary)} entries")
        return glossary
    print("  [WARN] Không tìm thấy glossary.json, bỏ qua")
    return {}


def load_ocr_data() -> dict:
    """Load kết quả OCR từ Pass 1."""
    if not os.path.exists(OCR_DATA_PATH):
        print(f"  [ERROR] Không tìm thấy {OCR_DATA_PATH}")
        print(f"  Hãy chạy ocr_pass.py trước!")
        return None

    with open(OCR_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded OCR data: {len(data.get('unique_texts', []))} unique texts")
    return data


def format_glossary_for_prompt(glossary: dict) -> str:
    """Format glossary thành chuỗi đẹp cho prompt."""
    if not glossary:
        return "(Không có)"
    lines = [f"  {zh} → {vi}" for zh, vi in glossary.items()]
    return "\n".join(lines)


def normalize_texts(unique_texts: list, batch_size: int = 150) -> dict:
    """Làm sạch chữ OCR bằng LLM: {raw: cleaned, hoặc "" nếu là rác}.
    Câu model bỏ sót hoặc trả giá trị không phải chuỗi → giữ nguyên chữ gốc."""
    mapping = {}
    total = (len(unique_texts) + batch_size - 1) // batch_size
    for b, start in enumerate(range(0, len(unique_texts), batch_size), 1):
        batch = unique_texts[start:start + batch_size]
        print(f"    Batch {b}/{total} ({len(batch)} câu)...")
        got = ask_json(NORMALIZATION_SYSTEM_PROMPT, json.dumps(batch, ensure_ascii=False), temperature=0.0)
        got = got if isinstance(got, dict) else {}
        for zh in batch:
            v = got.get(zh, zh)
            mapping[zh] = v if isinstance(v, str) else zh
    return mapping


# ──────────────────────────────────────────────
# CONTEXTUAL TRANSLATION (Phase 1-4): dịch theo ngữ cảnh kịch bản
#   Phase 1: dịch cả transcript theo thứ tự thời gian, có ngữ cảnh
#   Phase 2: gán người nói (speaker attribution) từ nhãn tên nhân vật
#   Phase 3: hồ sơ nhân vật (giới tính/vai vế/xưng hô) tự suy luận
#   Phase 4: pass QA rà soát nhất quán chủ ngữ/xưng hô
# ──────────────────────────────────────────────

CHAR_PROFILES_PATH = os.path.join(os.path.dirname(GLOSSARY_PATH), "character_profiles.json")

_PROFILE_SYS = (
    "Bạn là trợ lý phân tích nhân vật phim Trung Quốc. "
    "Tôi cung cấp các câu thoại theo thứ tự thời gian (kèm vài tên đã biết để tham khảo). "
    "Hãy TỰ NHẬN DIỆN các nhân vật chính và suy luận hồ sơ từng người từ ngữ cảnh hội thoại. "
    "Trả về DUY NHẤT một JSON object: key = tên (tiếng Việt nếu suy ra được, không thì để tiếng Trung), "
    'value = {"gender":"nam|nữ|?","role":"vai trò/quan hệ ngắn gọn","xung_ho":"cách tự xưng và gọi người khác"}. '
    "Chỉ liệt kê nhân vật thực sự (bỏ qua rác OCR). Nếu không chắc dùng '?'. KHÔNG giải thích, CHỈ JSON."
)

_QA_SYS = (
    "Bạn là biên tập viên phụ đề tiếng Việt cho phim Trung Quốc. "
    "Tôi đưa kịch bản đã dịch (mỗi dòng: số | người nói | tiếng Trung | tiếng Việt) kèm hồ sơ nhân vật. "
    "Hãy SOÁT các dòng bị SAI chủ ngữ / xưng hô / nhầm người nói so với ngữ cảnh hội thoại và hồ sơ nhân vật "
    "(tiếng Trung hay lược bỏ chủ ngữ nên dễ dịch nhầm 'tôi/anh/nó'). "
    "Trả về DUY NHẤT một JSON object CHỈ gồm các dòng CẦN SỬA: key = số dòng, value = câu tiếng Việt đã sửa. "
    "Nếu không dòng nào sai, trả về {}. KHÔNG giải thích."
)


def _is_name_label(cleaned: str, glossary: dict) -> bool:
    """NHÃN TÊN nhân vật — chỉ tin glossary (chính xác cao, tránh nhận nhầm verb/rác OCR)."""
    c = (cleaned or "").strip()
    if not c:
        return False
    return c in glossary or c in glossary.values()


def build_ordered_transcript(text_segments: list, normalized_mapping: dict, glossary: dict) -> list:
    """
    Phase 1+2: dựng transcript theo thứ tự thời gian, DEDUP theo lần xuất hiện đầu
    (mỗi câu unique 1 dòng) để giữ ngữ cảnh mà không phí API cho text lặp lại.
    Mỗi item: {idx, raw, cleaned, start, end, is_label, onscreen, speaker}.
    """
    raw_items = []
    for seg in text_segments:
        raw = seg.get("text", "")
        cleaned = normalized_mapping.get(raw, raw)
        if not cleaned:  # bị lọc thành rác
            continue
        raw_items.append({
            "raw": raw,
            "cleaned": cleaned,
            "start": int(seg.get("start_frame", 0)),
            "end": int(seg.get("end_frame", seg.get("start_frame", 0))),
            "is_label": _is_name_label(cleaned, glossary),
        })
    raw_items.sort(key=lambda it: (it["start"], it["end"]))

    # Cửa sổ thời gian của các nhãn tên (dùng cho speaker attribution, lấy từ TOÀN BỘ segment)
    labels = [it for it in raw_items if it["is_label"]]

    # Dedup theo cleaned, giữ lần xuất hiện ĐẦU (thứ tự thời gian)
    seen, transcript = set(), []
    for it in raw_items:
        if it["cleaned"] in seen:
            continue
        seen.add(it["cleaned"])
        transcript.append(it)

    for i, it in enumerate(transcript):
        it["idx"] = i + 1
        if it["is_label"]:
            it["onscreen"], it["speaker"] = [], None
            continue
        onscreen = []
        for lb in labels:
            if lb["start"] <= it["end"] and it["start"] <= lb["end"]:  # giao thời gian
                name = glossary.get(lb["cleaned"], lb["cleaned"])
                if name not in onscreen:
                    onscreen.append(name)
        it["onscreen"] = onscreen
        it["speaker"] = onscreen[0] if len(onscreen) == 1 else None  # chỉ chắc khi đúng 1
    return transcript


def infer_character_profiles(transcript: list, glossary: dict) -> dict:
    """Phase 3: suy luận hồ sơ nhân vật từ THOẠI (không phụ thuộc nhãn). Lưu file tái dùng."""
    sample = [f"[{it['idx']}] {it['cleaned']}" for it in transcript[:90]]
    known = sorted(set(list(glossary.values())))
    user = ("TÊN ĐÃ BIẾT (tham khảo, có thể không đủ):\n" + json.dumps(known, ensure_ascii=False)
            + "\n\nCÂU THOẠI THEO THỨ TỰ (hãy tự nhận diện nhân vật chính từ đây):\n"
            + "\n".join(sample))
    try:
        profiles = ask_json(_PROFILE_SYS, user, temperature=0.0)
    except LLMError as e:
        print(f"  [Phase 3][WARN] Suy luận hồ sơ nhân vật lỗi: {e}")
        return {}
    if not isinstance(profiles, dict):
        return {}

    # Trộn với hồ sơ thủ công nếu đã có (ưu tiên file cũ do người dùng chỉnh)
    existing = {}
    if os.path.exists(CHAR_PROFILES_PATH):
        try:
            existing = json.load(open(CHAR_PROFILES_PATH, encoding="utf-8"))
        except Exception:
            existing = {}
    merged = {**profiles, **existing}
    try:
        write_json_atomic(CHAR_PROFILES_PATH, merged)
        print(f"  [Phase 3] Hồ sơ nhân vật: {len(merged)} người -> {os.path.basename(CHAR_PROFILES_PATH)}")
    except Exception as e:
        print(f"  [Phase 3][WARN] Không lưu được hồ sơ: {e}")
    return merged


def _format_profiles(profiles: dict) -> str:
    if not profiles:
        return "(Không có)"
    lines = []
    for name, info in profiles.items():
        if isinstance(info, dict):
            g = info.get("gender", "?"); r = info.get("role", "?"); x = info.get("xung_ho", "")
            lines.append(f"  - {name}: giới tính={g}, vai={r}" + (f", xưng hô: {x}" if x else ""))
        else:
            lines.append(f"  - {name}: {info}")
    return "\n".join(lines)


def _line_tag(it: dict) -> str:
    if it["is_label"]:
        return "(Nhãn tên) "
    if it.get("speaker"):
        return f"(Người nói: {it['speaker']}) "
    if it.get("onscreen"):
        return f"(Trên màn hình: {', '.join(it['onscreen'])}) "
    return ""


def translate_script(transcript: list, profiles: dict, glossary: dict,
                     window: int = 120, overlap: int = 15) -> dict:
    """
    Phase 1: dịch cả kịch bản theo NGỮ CẢNH. Chia cửa sổ có gối đầu để giữ mạch
    hội thoại cho video dài. Trả về {idx: vi}.
    """
    glossary_str = format_glossary_for_prompt(glossary)
    base_sys = TRANSLATION_SYSTEM_PROMPT.format(genre=TRANSLATION_GENRE, glossary=glossary_str)
    sys_prompt = (
        base_sys
        + "\n\nHỒ SƠ NHÂN VẬT (dùng để xưng hô đúng & nhất quán):\n" + _format_profiles(profiles)
        + "\n\nQUAN TRỌNG:\n"
        "- Mỗi dòng được đánh số [n] và có thể kèm gợi ý '(Người nói: X)' hoặc '(Trên màn hình: ...)'.\n"
        "- Tiếng Trung hay LƯỢC BỎ chủ ngữ. Hãy dựa vào MẠCH hội thoại liên tiếp + gợi ý người nói "
        "để xác định ĐÚNG chủ ngữ và xưng hô tiếng Việt.\n"
        "- Giữ xưng hô NHẤT QUÁN xuyên suốt theo hồ sơ nhân vật.\n"
        "- Trả về DUY NHẤT một JSON object: key = số dòng (trong []), value = câu tiếng Việt. KHÔNG giải thích."
    )

    vi_by_idx = {}
    n = len(transcript)
    step = max(1, window - overlap)
    for start in range(0, n, step):
        chunk = transcript[start:start + window]
        if not chunk:
            break
        user = "\n".join(f"[{it['idx']}] {_line_tag(it)}{it['cleaned']}" for it in chunk)
        _take_translations(ask_json(sys_prompt, user, temperature=0.1), chunk, vi_by_idx)
        missing = [it for it in chunk if it["idx"] not in vi_by_idx]
        if missing:
            # Hỏi lại riêng các dòng thiếu (prompt khác → khóa cache khác), vẫn kèm cả cửa sổ làm ngữ cảnh
            ids = ", ".join(str(it["idx"]) for it in missing)
            _take_translations(ask_json(sys_prompt, user + f"\n\nCHỈ dịch các dòng: {ids}", temperature=0.1),
                               missing, vi_by_idx)
        if start + window >= n:
            break
    missing = [it["idx"] for it in transcript if it["idx"] not in vi_by_idx]
    if missing:
        raise LLMError(f"Model bỏ sót {len(missing)} dòng kể cả khi hỏi lại: {missing[:20]}")
    return vi_by_idx


def _take_translations(data, items: list, vi_by_idx: dict) -> None:
    """Lấy bản dịch {"idx": vi} cho đúng các dòng của cửa sổ.
    Chuỗi rỗng = model cố ý để trống (rác) → vẫn tính là đã trả lời; thiếu key mới là bỏ sót."""
    if not isinstance(data, dict):
        return
    for it in items:
        vi = data.get(str(it["idx"]))
        if isinstance(vi, str):
            vi_by_idx[it["idx"]] = vi.strip()


def qa_consistency_pass(transcript: list, vi_by_idx: dict, profiles: dict,
                        window: int = 150) -> dict:
    """Phase 4: rà soát & sửa các dòng sai chủ ngữ/xưng hô. Cập nhật vi_by_idx tại chỗ."""
    prof_str = _format_profiles(profiles)
    n = len(transcript)
    fixed_total = 0
    for start in range(0, n, window):
        chunk = transcript[start:start + window]
        window_ids = {it["idx"] for it in chunk}
        rows = []
        for it in chunk:
            vi = vi_by_idx.get(it["idx"], "")
            if not vi:
                continue
            sp = it.get("speaker") or (", ".join(it.get("onscreen", [])) or "?")
            rows.append(f"[{it['idx']}] | {sp} | {it['cleaned']} | {vi}")
        if not rows:
            continue
        user = "HỒ SƠ NHÂN VẬT:\n" + prof_str + "\n\nKỊCH BẢN:\n" + "\n".join(rows)
        try:
            fixes = ask_json(_QA_SYS, user, temperature=0.0)
        except LLMError as e:
            print(f"  [Phase 4][WARN] QA cửa sổ {start} lỗi: {e}")
            fixes = None
        if isinstance(fixes, dict):
            for k, v in fixes.items():
                try:
                    idx = int(k)
                except (ValueError, TypeError):
                    continue
                if idx in window_ids and v and isinstance(v, str):
                    vi_by_idx[idx] = v.strip()
                    fixed_total += 1
    if fixed_total:
        print(f"  [Phase 4] QA đã sửa {fixed_total} dòng cho nhất quán chủ ngữ/xưng hô")
    return vi_by_idx


def translate_contextual(text_segments: list, normalized_mapping: dict,
                         texts_to_translate: list, glossary: dict) -> dict:
    """
    Orchestrator Phase 1-4. Trả về {cleaned_zh: vi} phủ tối đa texts_to_translate.
    Ném LLMError nếu LLM hỏng / vẫn bỏ sót dòng → caller dừng (exit 1); phần đã xong nằm trong cache.
    """
    print("  [Phase 1] Dựng transcript theo thứ tự thời gian...")
    transcript = build_ordered_transcript(text_segments, normalized_mapping, glossary)
    if not transcript:
        raise RuntimeError("Transcript rỗng (không có segment hợp lệ)")
    n_label = sum(1 for it in transcript if it["is_label"])
    print(f"    {len(transcript)} dòng ({n_label} nhãn tên, {len(transcript)-n_label} thoại)")

    print("  [Phase 3] Suy luận hồ sơ nhân vật...")
    profiles = infer_character_profiles(transcript, glossary)

    print("  [Phase 1+2] Dịch kịch bản theo ngữ cảnh + người nói...")
    vi_by_idx = translate_script(transcript, profiles, glossary)

    print("  [Phase 4] Rà soát nhất quán...")
    qa_consistency_pass(transcript, vi_by_idx, profiles)

    # Dựng cleaned -> vi
    cleaned_to_vi = {}
    for it in transcript:
        vi = vi_by_idx.get(it["idx"])
        if vi:
            cleaned_to_vi[it["cleaned"]] = vi

    # Câu sạch không có trong transcript = biến thể OCR của một dòng đã dịch → mượn bản dịch dòng
    # giống nhất (giữ đúng xưng hô theo ngữ cảnh), không dịch rời rạc.
    leftover = [t for t in texts_to_translate if t not in cleaned_to_vi]
    if leftover:
        known = list(cleaned_to_vi)
        borrowed = 0
        for t in leftover:
            match = get_close_matches(t, known, n=1, cutoff=0.7)
            if match:
                cleaned_to_vi[t] = cleaned_to_vi[match[0]]
                borrowed += 1
        print(f"  [Bổ sung] {len(leftover)} biến thể ngoài transcript: {borrowed} mượn bản dịch dòng "
              f"giống nhất, {len(leftover) - borrowed} để trống")

    return cleaned_to_vi


def run_translation():
    """Chạy dịch thuật, xuất translation_map.json. Trả về True nếu thành công."""
    print("=" * 55)
    print("  Monggu Translate - Pass 2: Translation")
    print("=" * 55)

    # Load data
    print("\n[1/3] Đọc dữ liệu...")
    ocr_data = load_ocr_data()
    if not ocr_data:
        return False

    glossary = load_glossary()
    unique_texts = ocr_data.get("unique_texts", [])

    if not unique_texts:
        print("  [ERROR] Không có text nào để dịch!")
        return False

    text_segments = ocr_data.get("text_segments", [])
    if not text_segments:
        print("  [ERROR] ocr_data.json thiếu text_segments — hãy chạy lại OCR.")
        return False

    print(f"  Tổng cộng {len(unique_texts)} câu ban đầu")

    try:
        print("\n[1.5/3] Normalize Texts...")
        t_start_norm = time.time()
        normalized_mapping = normalize_texts(unique_texts)

        # Câu đã làm sạch, có nghĩa, bỏ trùng (giữ thứ tự)
        texts_to_translate = list(dict.fromkeys(c for c in normalized_mapping.values() if c))
        print(f"  => Sau khi clean, còn lại {len(texts_to_translate)} câu cần dịch "
              f"(thời gian: {time.time() - t_start_norm:.1f}s)")
        if not texts_to_translate:
            print("  [ERROR] Tất cả text đều bị lọc thành rác. Không có gì để dịch!")
            return False

        print(f"\n[2/3] Dịch theo NGỮ CẢNH kịch bản (Phase 1-4) — {len(text_segments)} segment...")
        t0 = time.time()
        cleaned_translation_map = translate_contextual(
            text_segments, normalized_mapping, texts_to_translate, glossary)
    except (LLMError, RuntimeError) as e:
        print(f"  [ERROR] Dịch thất bại: {e}")
        print("  Phần đã dịch xong nằm trong cache/llm.sqlite — chạy lại sẽ làm tiếp, không tốn lại.")
        return False

    # Dựng translation_map theo chữ OCR THÔ; rỗng = Pass 3 xóa chữ mà không vẽ (rác)
    translation_map = {raw: (cleaned_translation_map.get(cleaned, "") if cleaned else "")
                       for raw, cleaned in normalized_mapping.items()}

    elapsed = time.time() - t0

    # Không dịch được câu nào (mạng/quota hỏng hết) → đừng ghi map toàn chữ Trung/rỗng:
    # render sẽ xóa sạch phụ đề gốc mà không vẽ gì, mà vẫn "thành công".
    if not any(vi and not re.search(r'[\u4e00-\u9fff]', vi) for vi in translation_map.values()):
        print("  [ERROR] Không dịch được câu nào — không ghi translation_map.json")
        return False

    # Xuất JSON
    print(f"\n[3/3] Xuất translation_map.json...")
    write_json_atomic(TRANSLATION_MAP_PATH, translation_map)

    print(f"\n{'=' * 55}")
    print(f"  Dịch thuật hoàn tất!")
    print(f"  Thời gian  : {elapsed:.1f}s")
    print(f"  Số câu dịch: {len(translation_map)}")
    print(f"  Output     : {TRANSLATION_MAP_PATH}")
    print(f"{'=' * 55}")

    # Preview 5 câu đầu
    print("\n  Preview bản dịch:")
    for i, (zh, vi) in enumerate(translation_map.items()):
        if i >= 5:
            print(f"  ... và {len(translation_map) - 5} câu nữa")
            break
        print(f"  [{zh}]")
        print(f"  → {vi}\n")

    return True


if __name__ == "__main__":
    argparse.ArgumentParser(description="Monggu Translate - Translation Pass").parse_args()
    sys.exit(0 if run_translation() else 1)

