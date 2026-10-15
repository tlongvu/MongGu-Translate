# B3 — Dịch theo cốt truyện, đúng nhân vật — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pass dịch hiểu cốt truyện và biết ai nói với ai.
- Hồ sơ truyện có bảng xưng hô theo cặp.
- Gán người nói/người nghe cho từng câu, gợi ý bằng nhãn tên trên màn hình.
- Dịch **từng lần xuất hiện**, xưng hô tra từ bảng.
- Rà soát.
- Render tra bản dịch theo segment.

**Architecture:** Viết lại lõi `translate_pass.py`: bỏ phase 1–4 cũ, thay bằng `build_lines` → `build_story` → `mark_labels_and_onscreen` → `attribute_speakers` → `translate_lines` → `qa_lines`. Dữ liệu nằm trong job: `lines.json` và `story.json` (của video lẻ hoặc của series). `script.py` đọc `lines.json`.

**Tech Stack:** thư viện chuẩn và `llm.ask_json` (B1).

**Spec:** `docs/superpowers/specs/2026-09-26-dich-theo-cot-truyen-design.md`, mục 4, 5 và 9.

## Global Constraints

- Nhánh `nang-cap`. Backup vào `_backup_nang-cap-B3_<timestamp>/`.
- **Không gọi API thật** cho tới khi user cho phép (Task 4). Test dùng `ask_json` giả.
- Giữ sửa tay:
  - `noi`/`nghe`/`chac`/`vi_sua` trong `lines.json`;
  - mọi khóa đã có trong `story.json`.
- Windows + PowerShell, `$env:PYTHONIOENCODING="utf-8"`.

---

### Task 1: `jobs` — đường dẫn `story` và `lines`

**Files:** Modify `jobs.py`, `tests/test_jobs.py`

- [ ] **Step 1: Test.** Trong `tests/test_jobs.py`:
  - thay `assert os.path.dirname(j.profiles) == j.dir, "Video lẻ: hồ sơ nằm trong job"` bằng:

```python
        assert j.story == os.path.join(j.dir, "story.json"), "Video lẻ: hồ sơ truyện nằm trong job"
        assert j.lines == os.path.join(j.dir, "lines.json")
```

  - thay dòng `assert j1.profiles == j2.profiles == ...` bằng:

```python
        assert j1.story == j2.story == os.path.join(jobs.SERIES_DIR, "daosi", "story.json")
```

Run: `python tests/test_jobs.py` → FAIL (`no attribute 'story'`)

- [ ] **Step 2: `jobs.py`.** Khối `return SimpleNamespace(...)` thành:

```python
    return SimpleNamespace(
        name=name, dir=d, series=meta.get("series"),
        ocr_data=os.path.join(d, "ocr_data.json"),
        lines=os.path.join(d, "lines.json"),
        output=os.path.join(d, f"{name}_vi.mp4"),
        story=os.path.join(shared, "story.json"),
    )
```

Sửa docstring module, dòng 2, thành `Video thuộc series thì hồ sơ truyện (story.json) nằm ở series/<bộ>/, dùng chung giữa các tập.`

- [ ] **Step 3:** `python tests/test_jobs.py` → `3/3`. Commit sau Task 3, vì giữa chừng pipeline chưa nhất quán.

---

### Task 2: Lõi dịch mới trong `translate_pass.py`

**Files:**
- Rewrite: `translate_pass.py`
- Modify: `config.py` (xóa `TRANSLATION_GENRE`, `TRANSLATION_SYSTEM_PROMPT`)
- Test: `tests/test_core.py`

**Interfaces — Produces:**
- `build_lines(text_segments: list, normalized: dict, old_lines: list) -> list[dict]`
- `long_lived_texts(lines: list, fps: float) -> list[str]`
- `merge_story(old: dict, new: dict) -> dict`
- `build_story(lines: list, candidates: list, existing: dict) -> dict`
- `mark_labels_and_onscreen(lines: list, story: dict) -> None`
- `attribute_speakers(lines: list, story: dict) -> None`
- `translate_lines(lines: list, story: dict) -> None` (raise `LLMError`)
- `qa_lines(lines: list, story: dict) -> int`
- `_context(story: dict, rows: list, owned: list) -> str`
- `run_translation(video_path: str) -> bool`
- Mỗi dòng trong `lines`: `{"id", "start", "end", "zh", "loai": "thoai"|"nhan"|"rac"}`, thêm tùy dòng: `"noi", "nghe", "chac", "tren_man_hinh", "vi", "vi_sua"`.

- [ ] **Step 1: Test (fail).** Trong `tests/test_core.py`:
  - xóa `_fake_llm`, `_run_contextual`, `test_translate_contextual_retries_missing_lines_and_borrows_variants`, `test_translate_contextual_raises_when_lines_still_missing`;
  - thêm (sau `test_run_translation_fails_without_input`):

```python
# ── Dịch theo cốt truyện (translate_pass, B3) ──

def _line(i, zh, start, end, **kw):
    return dict({"id": i, "start": start, "end": end, "zh": zh, "loai": "thoai" if zh else "rac"}, **kw)


def _with_fake_llm(fake, fn, *args):
    orig = tp.ask_json
    try:
        tp.ask_json = fake
        return fn(*args)
    finally:
        tp.ask_json = orig


def test_build_lines_keeps_manual_edits_only_for_same_text():
    segs = [{'text': 'A1', 'start_frame': 0, 'end_frame': 5}, {'text': 'B1', 'start_frame': 10, 'end_frame': 15},
            {'text': 'rác', 'start_frame': 20, 'end_frame': 25}]
    norm = {'A1': '我知道了', 'B1': '走吧', 'rác': ''}
    old = [{"id": 0, "zh": "我知道了", "noi": "yiyi", "nghe": "tieu_thu", "chac": 1.0, "vi_sua": "Nô tỳ biết rồi"},
           {"id": 1, "zh": "câu cũ khác", "noi": "x"}]
    lines = tp.build_lines(segs, norm, old)
    assert lines[0]["noi"] == "yiyi" and lines[0]["vi_sua"] == "Nô tỳ biết rồi", "Sửa tay phải được giữ"
    assert "noi" not in lines[1], "Chữ đổi (OCR chạy lại) → bỏ dữ liệu cũ của id đó"
    assert lines[2]["loai"] == "rac"


def test_long_lived_texts_are_label_candidates():
    lines = [_line(i, '孙晴', i * 30, i * 30 + 29) for i in range(4)]          # nhãn: 4 × 1.2s = 4.8s
    lines += [_line(10, '我要去修仙！', 200, 240), _line(11, '这是一句很长很长的对白啊', 300, 500)]
    assert tp.long_lived_texts(lines, 25.0) == ['孙晴'], "Thoại ngắn hạn / câu dài không phải nhãn"


def test_labels_mark_onscreen_characters():
    lines = [_line(0, '孙晴', 0, 100), _line(1, '你来了', 10, 20), _line(2, '走', 200, 210)]
    tp.mark_labels_and_onscreen(lines, {"nhan": {"孙晴": "sun_qing"}})
    assert lines[0]["loai"] == "nhan"
    assert lines[1]["tren_man_hinh"] == ["sun_qing"] and lines[2]["tren_man_hinh"] == []


def test_merge_story_only_adds():
    old = {"tom_tat": "cũ", "nhan_vat": {"yiyi": {"ten": "Y Y (sửa tay)"}},
           "xung_ho": {"yiyi>tieu_thu": {"xung": "nô tỳ", "goi": "tiểu thư"}}}
    new = {"tom_tat": "mới", "nhan_vat": {"yiyi": {"ten": "Y Y"}, "bach_lang": {"ten": "Bạch Lang"}},
           "xung_ho": {"yiyi>tieu_thu": {"xung": "tôi", "goi": "cô"}, "bach_lang>*": {"xung": "ta", "goi": "các ngươi"}}}
    m = tp.merge_story(old, new)
    assert m["tom_tat"] == "cũ" and m["nhan_vat"]["yiyi"]["ten"] == "Y Y (sửa tay)"
    assert m["xung_ho"]["yiyi>tieu_thu"]["xung"] == "nô tỳ", "Cặp xưng hô đã có không bị ghi đè"
    assert "bach_lang" in m["nhan_vat"] and "bach_lang>*" in m["xung_ho"], "Cái mới được bổ sung"


def test_context_includes_only_what_is_in_window():
    story = {"nhan_vat": {"a": {"ten": "A"}, "b": {"ten": "B"}, "c": {"ten": "C"}},
             "xung_ho": {"a>b": {"xung": "ta", "goi": "ngươi"}, "c>a": {"xung": "đệ", "goi": "huynh"},
                         "b>*": {"xung": "tôi", "goi": "mọi người"}},
             "ten_rieng": {"天元宗": "Thiên Nguyên Tông", "青云": "Thanh Vân"}}
    rows = [_line(1, '去天元宗', 0, 5, noi='a', nghe='b'), _line(2, '好', 6, 9, noi='b', nghe='a')]
    ctx = tp._context(story, rows, rows)
    assert 'a→b' in ctx and 'b→*' in ctx, "Cặp trực tiếp, hoặc mặc định của người nói"
    assert 'c→a' not in ctx and '- c:' not in ctx, "Nhân vật / cặp không có trong cửa sổ thì không đưa vào"
    assert 'Thiên Nguyên Tông' in ctx and 'Thanh Vân' not in ctx


def test_attribute_speakers_fills_only_missing():
    story = {"nhan_vat": {"a": {"ten": "A"}, "b": {"ten": "B"}}}
    lines = [_line(1, '你好', 0, 5, noi='b', nghe='a', chac=1.0),       # đã có (sửa tay) → giữ
             _line(2, '走吧', 6, 9), _line(3, '什么', 10, 12), _line(4, '好', 13, 15)]

    def fake(system, user, temperature=0.1, **kw):
        if 'CHỈ xác định' in user:
            return {}                                                    # hỏi lại vẫn thiếu dòng 4
        return {"1": {"noi": "a", "nghe": "b", "chac": 1}, "2": {"noi": "a", "nghe": "b", "chac": 0.9},
                "3": {"noi": "zz", "nghe": "a", "chac": 0.8}}            # id lạ → "?"
    _with_fake_llm(fake, tp.attribute_speakers, lines, story)
    assert (lines[0]["noi"], lines[0]["nghe"]) == ("b", "a"), "Dòng đã có không bị ghi đè"
    assert (lines[1]["noi"], lines[1]["nghe"], lines[1]["chac"]) == ("a", "b", 0.9)
    assert lines[2]["noi"] == "?" and lines[2]["chac"] == 0.0, "id nhân vật lạ → ?"
    assert lines[3]["noi"] == "?" and lines[3]["chac"] == 0.0, "Bỏ sót sau khi hỏi lại → ? (không dừng)"


def test_translate_lines_each_line_once():
    lines = [_line(i, f'句{i}', i * 10, i * 10 + 5, noi='a', nghe='b') for i in range(250)]
    owned = []

    def fake(system, user, temperature=0.1, **kw):
        todo = user.split("CẦN DỊCH:\n", 1)[1]
        ids = [int(x[1:x.index("]")]) for x in todo.splitlines() if x.startswith("[")]
        owned.extend(ids)
        return {str(i): f"VI{i}" for i in ids}
    _with_fake_llm(fake, tp.translate_lines, lines, {})
    assert sorted(owned) == list(range(250)), "Mỗi dòng dịch đúng 1 lần (dòng ngữ cảnh không dịch lại)"
    assert all(l["vi"] == f"VI{l['id']}" for l in lines)


def test_translate_lines_retries_missing_then_raises():
    lines = [_line(1, '你好', 0, 5, noi='a', nghe='b'), _line(2, '走吧', 6, 9, noi='b', nghe='a')]
    calls = []

    def fake(system, user, temperature=0.1, **kw):
        calls.append(user)
        return {"1": "Xin chào"} if 'CHỈ dịch' not in user else {"2": "Đi thôi"}
    _with_fake_llm(fake, tp.translate_lines, lines, {})
    assert [l["vi"] for l in lines] == ["Xin chào", "Đi thôi"] and len(calls) == 2
    try:
        _with_fake_llm(lambda s, u, temperature=0.1, **kw: {"1": "Xin chào"}, tp.translate_lines,
                       [_line(1, '你好', 0, 5), _line(2, '走吧', 6, 9)], {})
        raise AssertionError("Vẫn thiếu dòng phải ném LLMError")
    except tp.LLMError:
        pass
```

Run: `python -m pytest tests/test_core.py -q` → các test mới FAIL (`AttributeError`)

- [ ] **Step 2: Viết lại `translate_pass.py`** — toàn bộ file:

```python
"""
Monggu Translate - Pass 2: Dịch theo cốt truyện, đúng nhân vật
jobs/<tên>/ocr_data.json → chuẩn hóa chữ OCR → hồ sơ truyện (story.json) → người nói / người nghe từng câu
→ dịch TỪNG LẦN XUẤT HIỆN, xưng hô tra bảng theo cặp → rà soát → jobs/<tên>/lines.json
Mọi lời gọi LLM qua llm.ask_json (Gemini, dự phòng DeepSeek; có cache → chạy lại không tốn lại).

Usage:
    python translate_pass.py --input video.mp4

Sửa tay rồi chạy lại (cache → chỉ phần liên quan bị gọi lại LLM):
  - story.json: nhân vật, bảng xưng hô "A>B" / "A>*", tên riêng
  - lines.json: "noi"/"nghe" của một dòng (người nói sai), hoặc "vi_sua" (bản dịch sửa tay, render ưu tiên)
"""

import sys

# Đảm bảo console in được tiếng Việt Unicode (kể cả khi stdout là pipe/subprocess)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import argparse
import json
import os
import time

from config import NORMALIZATION_SYSTEM_PROMPT, VIDEO_PATH
from jobs import open_job, JobError
from llm import ask_json, LLMError
from utils import write_json_atomic

LABEL_MIN_SEC, LABEL_MAX_LEN = 3.0, 8        # chữ hiện ≥3s và ≤8 ký tự → ứng viên nhãn tên / tiêu đề
MAX_LABEL_CANDIDATES = 60
SPEAKER_WINDOW, SPEAKER_CONTEXT = 200, 20
TRANSLATE_WINDOW, TRANSLATE_CONTEXT = 120, 15
QA_WINDOW = 150

STORY_SYS = """Bạn là biên tập viên phim Trung Quốc, lập HỒ SƠ TRUYỆN để dịch phụ đề sang tiếng Việt cho đúng nhân vật.
Tôi đưa: (1) HỒ SƠ ĐÃ CÓ của bộ phim (có thể rỗng) — GIỮ NGUYÊN mọi id và nội dung, chỉ bổ sung cái mới;
(2) CHỮ HIỆN LÂU trên màn hình — thường là NHÃN TÊN gắn cạnh nhân vật, hoặc tiêu đề / địa danh / watermark;
(3) LỜI THOẠI của video theo thứ tự thời gian.

Trả về DUY NHẤT một JSON object:
{
  "tom_tat": "tóm tắt cốt truyện 3-6 câu",
  "the_loai": "vd: tu tiên hài, ngôn tình hiện đại, cổ trang cung đấu",
  "nhan_vat": {"<id>": {"ten": "tên tiếng Việt (Hán Việt)", "ten_goc": ["các cách gọi tiếng Trung"], "gioi_tinh": "nam|nữ|?", "vai": "vai trò, quan hệ với nhân vật khác"}},
  "xung_ho": {"<idA>><idB>": {"xung": "A tự xưng", "goi": "A gọi B"}, "<idA>>*": {"xung": "...", "goi": "..."}},
  "ten_rieng": {"<tiếng Trung>": "<tiếng Việt>"},
  "nhan": {"<chữ hiện lâu>": "<id nhân vật>"}
}

Quy tắc:
- id: chữ thường không dấu nối bằng "_" (vd "tieu_thu", "bach_lang"). Người kể chuyện xưng "我" cũng là một nhân vật.
- xung_ho: cho MỌI cặp thực sự nói chuyện với nhau, theo lối xưng hô tiếng Việt hợp thể loại và quan hệ (cổ trang / tu tiên: ta–ngươi, huynh–muội, sư phụ–đồ nhi, bản tiểu thư, nô tỳ…; hiện đại: tôi–cậu, anh–em, bố/mẹ–con, chị–em…). Thêm "<id>>*" = cách xưng mặc định khi nói với nhiều người hoặc tự nói.
- nhan: chỉ ghi những chữ hiện lâu là NHÃN TÊN của một nhân vật (không ghi tiêu đề, địa danh, watermark).
- ten_rieng: tên người, môn phái, địa danh, pháp bảo… (dùng Hán Việt).
KHÔNG giải thích."""

SPEAKER_SYS = """Bạn xác định AI NÓI và NÓI VỚI AI cho từng dòng phụ đề phim Trung Quốc.
Căn cứ: hồ sơ truyện, mạch hội thoại (ai hỏi – ai đáp), và gợi ý "(Trên màn hình: ...)" = các nhân vật có nhãn tên đang hiện khi dòng đó xuất hiện.
Dòng đã có "(A→B)" là đã xác định — chỉ để tham khảo.
Trả về DUY NHẤT một JSON object cho các dòng CHƯA có (A→B):
{"<id dòng>": {"noi": "<id nhân vật hoặc ?>", "nghe": "<id nhân vật, * nếu nói với nhiều người/tự nói/kể chuyện, ? nếu không rõ>", "chac": <độ chắc chắn 0..1>}}
Chỉ dùng id có trong danh sách nhân vật. Lời dẫn của người kể chuyện: noi = id người kể, nghe = "*". Chữ không phải lời thoại (tiêu đề, chú thích, bảng hiệu): noi = "?".
KHÔNG giải thích."""

TRANSLATE_SYS = """Bạn là dịch giả phụ đề phim Trung Quốc sang tiếng Việt.
Mỗi dòng cần dịch có dạng [id] (A→B) câu tiếng Trung: A là người nói, B là người nghe (* = nhiều người / tự nói, ? = không rõ). (Nhãn) = nhãn tên hiện trên màn hình: chỉ dịch tên.
Quy tắc:
1. Dùng ĐÚNG xưng hô trong bảng XƯNG HÔ cho cặp A→B; cặp không có trong bảng thì dùng cách xưng mặc định của A (A→*); vẫn không có thì suy từ hồ sơ và mạch truyện.
2. Giữ đúng tên riêng theo bảng TÊN RIÊNG.
3. Dịch theo MẠCH TRUYỆN (tóm tắt + NGỮ CẢNH TRƯỚC ĐÓ): hiểu đúng ẩn ý, chơi chữ, câu nói cụt; dịch tự nhiên, ngắn gọn như phụ đề, không thêm thắt.
4. Các dòng NGỮ CẢNH TRƯỚC ĐÓ chỉ để giữ mạch — KHÔNG dịch.
Trả về DUY NHẤT một JSON object {"<id>": "<câu tiếng Việt>"} cho MỌI dòng CẦN DỊCH. KHÔNG giải thích."""

QA_SYS = """Bạn là biên tập viên phụ đề tiếng Việt cho phim Trung Quốc.
Mỗi dòng: [id] A→B | tiếng Trung | tiếng Việt (A nói với B).
Soát các lỗi: (1) xưng hô sai so với bảng XƯNG HÔ của cặp A→B; (2) nhầm chủ ngữ (tôi/anh/nó…); (3) sai ý so với mạch truyện và tóm tắt.
Trả về DUY NHẤT một JSON object CHỈ gồm các dòng CẦN SỬA: {"<id>": "<câu đã sửa>"}; không dòng nào sai thì {}. KHÔNG giải thích."""


def load_ocr_data(path: str) -> dict:
    """Load kết quả OCR từ Pass 1."""
    if not os.path.exists(path):
        print(f"  [ERROR] Không tìm thấy {path}")
        print(f"  Hãy chạy ocr_pass.py trước!")
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded OCR data: {len(data.get('unique_texts', []))} unique texts")
    return data


def _load_json(path: str, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


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


# ── Dòng theo từng lần xuất hiện ──

def build_lines(text_segments: list, normalized: dict, old_lines: list) -> list:
    """Mỗi segment một dòng (id = chỉ số segment), sắp theo thời gian.
    Giữ noi/nghe/chac/vi_sua của lines.json cũ nếu cùng id VÀ cùng chữ (OCR chạy lại thì id đổi → bỏ)."""
    old = {l.get("id"): l for l in old_lines or []}
    lines = []
    for i, seg in enumerate(text_segments):
        raw = seg.get("text", "")
        zh = normalized.get(raw, raw)
        line = {"id": i, "start": int(seg.get("start_frame", 0)),
                "end": int(seg.get("end_frame", seg.get("start_frame", 0))),
                "zh": zh, "loai": "thoai" if zh else "rac"}
        prev = old.get(i)
        if prev and prev.get("zh") == zh:
            for k in ("noi", "nghe", "chac", "vi_sua"):
                if k in prev:
                    line[k] = prev[k]
        lines.append(line)
    lines.sort(key=lambda l: (l["start"], l["end"], l["id"]))
    return lines


def long_lived_texts(lines: list, fps: float) -> list:
    """Chữ hiện lâu (tổng ≥ LABEL_MIN_SEC giây, ≤ LABEL_MAX_LEN ký tự): ứng viên nhãn tên / tiêu đề, lâu nhất trước."""
    total = {}
    for l in lines:
        if l["zh"] and len(l["zh"]) <= LABEL_MAX_LEN:
            total[l["zh"]] = total.get(l["zh"], 0.0) + (l["end"] - l["start"] + 1) / fps
    ranked = sorted(total.items(), key=lambda kv: -kv[1])
    return [t for t, sec in ranked if sec >= LABEL_MIN_SEC][:MAX_LABEL_CANDIDATES]


# ── Bước 1: hồ sơ truyện ──

def merge_story(old: dict, new: dict) -> dict:
    """Trộn hồ sơ: GIỮ NGUYÊN mọi thứ đã có (người dùng có thể đã sửa), chỉ thêm khóa mới."""
    # ponytail: tom_tat của series giữ bản đầu tiên; cần tóm tắt từng tập thì thêm "tom_tat_tap" trong job
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in old.items()}
    for key in ("tom_tat", "the_loai"):
        if not out.get(key) and isinstance(new.get(key), str):
            out[key] = new[key]
    for key in ("nhan_vat", "xung_ho", "ten_rieng", "nhan"):
        cur = out.setdefault(key, {})
        new_part = new.get(key)
        if isinstance(new_part, dict):
            for k, v in new_part.items():
                cur.setdefault(k, v)
    return out


def build_story(lines: list, candidates: list, existing: dict) -> dict:
    """Bước 1: hồ sơ truyện từ TOÀN BỘ lời thoại (1 lời gọi). existing: hồ sơ đã có → chỉ bổ sung."""
    skip = set(candidates)                     # chữ hiện lâu đưa riêng, không lặp trong lời thoại
    dialogue, prev = [], None
    for l in lines:
        if l["loai"] == "thoai" and l["zh"] not in skip and l["zh"] != prev:
            dialogue.append(l["zh"])           # bỏ dòng trùng liên tiếp (1 câu bị OCR tách nhiều segment)
            prev = l["zh"]
    user = ("HỒ SƠ ĐÃ CÓ (giữ nguyên, chỉ bổ sung):\n" + json.dumps(existing or {}, ensure_ascii=False)
            + "\n\nCHỮ HIỆN LÂU TRÊN MÀN HÌNH:\n" + json.dumps(candidates, ensure_ascii=False)
            + "\n\nLỜI THOẠI THEO THỨ TỰ:\n" + "\n".join(dialogue))
    # Hồ sơ không có nhân vật nào = vô dụng cho xưng hô → không cache, hỏi lại / đổi provider (check),
    # hỏng hết thì LLMError. Series đã có nhân vật thì tập mới có thể không thêm ai.
    need_chars = not (existing or {}).get("nhan_vat")
    got = ask_json(STORY_SYS, user, temperature=0.0,
                   check=lambda o: isinstance(o, dict) and (not need_chars or bool(o.get("nhan_vat"))))
    return merge_story(existing or {}, got)


def mark_labels_and_onscreen(lines: list, story: dict) -> None:
    """Dòng là nhãn tên → loai='nhan'. Dòng thoại: 'tren_man_hinh' = nhân vật có nhãn đang hiện cùng lúc."""
    nhan = story.get("nhan") or {}
    labels = []
    for l in lines:
        if l["zh"] and l["zh"] in nhan:
            l["loai"] = "nhan"
            labels.append(l)
        elif l["zh"]:
            l["loai"] = "thoai"
    for l in lines:
        if l["loai"] == "thoai":
            l["tren_man_hinh"] = sorted({nhan[x["zh"]] for x in labels
                                         if x["start"] <= l["end"] and l["start"] <= x["end"]})


# ── Bước 2: người nói / người nghe ──

def _story_brief(story: dict) -> str:
    nv = story.get("nhan_vat") or {}
    rows = [f"  - {i}: {v.get('ten', i)} ({v.get('gioi_tinh', '?')}; {v.get('vai', '')})"
            for i, v in nv.items() if isinstance(v, dict)]
    return ("TÓM TẮT: " + (story.get("tom_tat") or "(chưa có)")
            + "\nNHÂN VẬT (id: tên):\n" + ("\n".join(rows) or "  (chưa có)"))


def _speaker_row(l: dict) -> str:
    tag = f"({l['noi']}→{l.get('nghe') or '?'}) " if l.get("noi") else ""
    on = f"(Trên màn hình: {', '.join(l['tren_man_hinh'])}) " if l.get("tren_man_hinh") else ""
    return f"[{l['id']}] {tag}{on}{l['zh']}"


def _apply_speakers(got, items: list, ids: set) -> list:
    """Gán kết quả bước 2 cho đúng các dòng; trả về các dòng model bỏ sót."""
    got = got if isinstance(got, dict) else {}
    missing = []
    for l in items:
        r = got.get(str(l["id"]))
        if not isinstance(r, dict) or not r.get("noi"):
            missing.append(l)
            continue
        noi, nghe = str(r["noi"]), str(r.get("nghe") or "?")
        l["noi"] = noi if noi in ids else "?"
        l["nghe"] = nghe if (nghe in ids or nghe == "*") else "?"
        try:
            chac = float(r.get("chac", 0.5))
        except (TypeError, ValueError):
            chac = 0.5
        l["chac"] = 0.0 if l["noi"] == "?" else max(0.0, min(1.0, chac))
    return missing


def attribute_speakers(lines: list, story: dict) -> None:
    """Bước 2: điền noi/nghe/chac cho dòng thoại CÒN THIẾU (dòng đã có — kể cả sửa tay — giữ nguyên).
    Model vẫn bỏ sót sau khi hỏi lại → noi='?' chac=0: không dừng, bước dịch vẫn tự suy được."""
    ids = {i for i, v in (story.get("nhan_vat") or {}).items() if isinstance(v, dict)}
    sys_prompt = SPEAKER_SYS + "\n\n" + _story_brief(story)
    dialog = [l for l in lines if l["loai"] == "thoai"]
    for start in range(0, len(dialog), SPEAKER_WINDOW):
        window = dialog[start:start + SPEAKER_WINDOW]
        todo = [l for l in window if not l.get("noi")]
        if not todo:
            continue
        ctx = dialog[max(0, start - SPEAKER_CONTEXT):start]
        user = ("NGỮ CẢNH TRƯỚC ĐÓ:\n" + ("\n".join(_speaker_row(l) for l in ctx) or "(không có)")
                + "\n\nĐOẠN CẦN XÁC ĐỊNH:\n" + "\n".join(_speaker_row(l) for l in window))
        missing = _apply_speakers(ask_json(sys_prompt, user, temperature=0.0), todo, ids)
        if missing:
            need = {str(l["id"]) for l in missing}
            retry = user + "\n\nCHỈ xác định các dòng: " + ", ".join(sorted(need, key=int))
            try:  # check → câu hỏi lại mà vẫn thiếu thì KHÔNG bị cache, lần chạy sau hỏi lại được
                got = ask_json(sys_prompt, retry, temperature=0.0,
                               check=lambda o: isinstance(o, dict) and need <= set(o))
                missing = _apply_speakers(got, missing, ids)
            except LLMError as e:
                print(f"  [Người nói][WARN] {len(missing)} dòng chưa xác định được: {e}")
        for l in missing:
            l.update(noi="?", nghe="?", chac=0.0)


# ── Bước 3 + 4: dịch theo cặp xưng hô, rà soát ──

def _context(story: dict, rows: list, owned: list) -> str:
    """Khối ngữ cảnh cho 1 cửa sổ: CHỈ nhân vật / cặp xưng hô / tên riêng xuất hiện trong cửa sổ
    (prompt gọn hơn; sửa 1 cặp xưng hô chỉ làm lỡ cache của những cửa sổ có cặp đó)."""
    nv = story.get("nhan_vat") or {}
    xh = story.get("xung_ho") or {}
    who = sorted({x for l in rows for x in (l.get("noi"), l.get("nghe")) if x in nv})
    pairs = {}
    for l in owned:
        a, b = l.get("noi"), l.get("nghe")
        for key in (f"{a}>{b}", f"{a}>*"):
            if isinstance(xh.get(key), dict):
                pairs[key] = xh[key]
                break
    text = "".join(l["zh"] for l in rows)
    names = {zh: vi for zh, vi in (story.get("ten_rieng") or {}).items() if zh and zh in text}

    def info(i):
        v = nv[i] if isinstance(nv[i], dict) else {}
        return f"  - {i}: {v.get('ten', i)} ({v.get('gioi_tinh', '?')}; {v.get('vai', '')})"
    return ("THỂ LOẠI: " + (story.get("the_loai") or "phim ngắn Trung Quốc")
            + "\nTÓM TẮT: " + (story.get("tom_tat") or "(chưa có)")
            + "\nNHÂN VẬT:\n" + ("\n".join(info(i) for i in who) or "  (không rõ)")
            + "\nXƯNG HÔ (A→B: A tự xưng «…», gọi B «…»):\n"
            + ("\n".join(f"  - {k.replace('>', '→')}: xưng «{v.get('xung', '')}», gọi «{v.get('goi', '')}»"
                         for k, v in sorted(pairs.items()))
               or "  (không có — tự suy từ hồ sơ và mạch truyện)")
            + "\nTÊN RIÊNG:\n" + ("\n".join(f"  - {zh} → {vi}" for zh, vi in sorted(names.items()))
                                   or "  (không có)"))


def _tr_row(l: dict) -> str:
    tag = "(Nhãn)" if l["loai"] == "nhan" else f"({l.get('noi') or '?'}→{l.get('nghe') or '?'})"
    return f"[{l['id']}] {tag} {l['zh']}"


def _apply_translations(got, items: list, allow_empty: bool) -> list:
    """Gán bản dịch cho đúng các dòng; trả về dòng bỏ sót. Chuỗi rỗng chỉ nhận ở lần HỎI LẠI
    (model khẳng định là rác); lần đầu coi như bỏ sót để hỏi lại — tránh phụ đề trống âm thầm."""
    got = got if isinstance(got, dict) else {}
    missing = []
    for l in items:
        vi = got.get(str(l["id"]))
        if isinstance(vi, str) and (vi.strip() or allow_empty):
            l["vi"] = vi.strip()
        else:
            missing.append(l)
    return missing


def translate_lines(lines: list, story: dict) -> None:
    """Bước 3: dịch TỪNG LẦN XUẤT HIỆN (cùng câu tiếng Trung, khác người nói → khác xưng hô).
    Mỗi cửa sổ chỉ dịch dòng của nó. Ngữ cảnh chỉ gồm tiếng Trung + người nói (KHÔNG kèm bản dịch),
    nên sửa 1 cửa sổ không làm lỡ cache các cửa sổ sau. Bỏ sót sau khi hỏi lại → LLMError."""
    items = [l for l in lines if l["loai"] in ("thoai", "nhan")]
    for start in range(0, len(items), TRANSLATE_WINDOW):
        window = items[start:start + TRANSLATE_WINDOW]
        ctx = items[max(0, start - TRANSLATE_CONTEXT):start]
        sys_prompt = TRANSLATE_SYS + "\n\n" + _context(story, ctx + window, window)
        user = ("NGỮ CẢNH TRƯỚC ĐÓ (không dịch):\n" + ("\n".join(_tr_row(l) for l in ctx) or "(không có)")
                + "\n\nCẦN DỊCH:\n" + "\n".join(_tr_row(l) for l in window))
        missing = _apply_translations(ask_json(sys_prompt, user), window, allow_empty=False)
        if missing:
            need = {str(l["id"]) for l in missing}
            retry = user + "\n\nCHỈ dịch các dòng: " + ", ".join(sorted(need, key=int))
            try:  # check → câu hỏi lại mà vẫn thiếu thì KHÔNG bị cache, lần chạy sau hỏi lại được
                got = ask_json(sys_prompt, retry,
                               check=lambda o: isinstance(o, dict) and all(isinstance(o.get(i), str) for i in need))
                missing = _apply_translations(got, missing, allow_empty=True)
            except LLMError:
                pass
        if missing:
            raise LLMError(f"Model bỏ sót {len(missing)} dòng kể cả khi hỏi lại: "
                           + "; ".join(f"[{l['id']}] {l['zh']}" for l in missing[:10]))
        for l in window:
            if not l["vi"]:
                print(f"  [Dịch][WARN] Model để trống (coi là rác): [{l['id']}] {l['zh']}")


def qa_lines(lines: list, story: dict) -> int:
    """Bước 4: soát xưng hô / chủ ngữ / ý theo cốt truyện. Chỉ nhận id của cửa sổ.
    QA lỗi → giữ bản dịch hiện có, báo WARN. Trả về số dòng đã sửa."""
    items = [l for l in lines if l["loai"] == "thoai" and l.get("vi")]
    fixed = 0
    for start in range(0, len(items), QA_WINDOW):
        window = items[start:start + QA_WINDOW]
        own = {str(l["id"]): l for l in window}
        sys_prompt = QA_SYS + "\n\n" + _context(story, window, window)
        user = "\n".join(f"[{l['id']}] {l.get('noi') or '?'}→{l.get('nghe') or '?'} | {l['zh']} | {l['vi']}"
                         for l in window)
        try:
            got = ask_json(sys_prompt, user, temperature=0.0)
        except LLMError as e:
            print(f"  [Rà soát][WARN] Bỏ qua cửa sổ {start}: {e}")
            continue
        for k, v in (got.items() if isinstance(got, dict) else []):
            if k in own and isinstance(v, str) and v.strip():
                own[k]["vi"] = v.strip()
                fixed += 1
    return fixed


def run_translation(video_path: str):
    """Dịch video (đã OCR) theo cốt truyện → jobs/<tên>/lines.json (+ story.json). Trả về True nếu thành công."""
    print("=" * 55)
    print("  Monggu Translate - Pass 2: Dịch theo cốt truyện")
    print("=" * 55)
    try:
        job = open_job(video_path)
    except JobError as e:
        print(f"  [ERROR] {e}")
        return False
    ocr_data = load_ocr_data(job.ocr_data)
    if not ocr_data:
        return False
    if ocr_data.get("complete") is False:
        print("  [ERROR] OCR của video này chưa chạy xong (bị ngắt?) — chạy tiếp ocr_pass.py trước.")
        return False
    unique_texts = ocr_data.get("unique_texts", [])
    text_segments = ocr_data.get("text_segments", [])
    if not unique_texts:
        print("  [ERROR] Không có text nào để dịch!")
        return False
    if not text_segments:
        print("  [ERROR] ocr_data.json thiếu text_segments — hãy chạy lại OCR.")
        return False

    t0 = time.time()
    try:
        print(f"\n[0/4] Chuẩn hóa {len(unique_texts)} chữ OCR...")
        lines = build_lines(text_segments, normalize_texts(unique_texts), _load_json(job.lines, []))
        if not any(l["loai"] == "thoai" for l in lines):
            print("  [ERROR] Tất cả text đều bị lọc thành rác. Không có gì để dịch!")
            return False

        story = _load_json(job.story, {})
        if job.series or not story:
            print(f"\n[1/4] Hồ sơ truyện ({'series ' + job.series if job.series else 'video lẻ'})...")
            story = build_story(lines, long_lived_texts(lines, ocr_data.get("fps") or 25.0), story)
            write_json_atomic(job.story, story)
        else:
            print(f"\n[1/4] Dùng hồ sơ truyện có sẵn: {job.story}")
        print(f"  {len(story.get('nhan_vat') or {})} nhân vật, {len(story.get('xung_ho') or {})} cặp xưng hô, "
              f"{len(story.get('nhan') or {})} nhãn tên")

        mark_labels_and_onscreen(lines, story)
        print("\n[2/4] Xác định người nói / người nghe...")
        attribute_speakers(lines, story)
        print("\n[3/4] Dịch theo từng lần xuất hiện...")
        translate_lines(lines, story)
        print("\n[4/4] Rà soát...")
        print(f"  Đã sửa {qa_lines(lines, story)} dòng")
    except LLMError as e:
        print(f"  [ERROR] Dịch thất bại: {e}")
        print("  Phần đã xong nằm trong cache/llm.sqlite — chạy lại sẽ làm tiếp, không tốn lại.")
        return False

    if not any(l.get("vi") for l in lines):
        print("  [ERROR] Không dịch được câu nào — không ghi lines.json")
        return False
    write_json_atomic(job.lines, lines)

    dialog = [l for l in lines if l["loai"] == "thoai"]
    sure = sum(1 for l in dialog if (l.get("chac") or 0) >= 0.7)
    print(f"\n{'=' * 55}")
    print(f"  Dịch xong trong {time.time() - t0:.1f}s")
    print(f"  Dòng thoại: {len(dialog)} (chắc người nói: {sure}) | nhãn: "
          f"{sum(l['loai'] == 'nhan' for l in lines)} | rác: {sum(l['loai'] == 'rac' for l in lines)}")
    print(f"  Output    : {job.lines}")
    print(f"{'=' * 55}")
    for l in dialog[:5]:
        print(f"  [{l.get('noi')}→{l.get('nghe')}] {l['zh']}  →  {l.get('vi')}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monggu Translate - Translation Pass")
    parser.add_argument("--input", "-i", type=str, default=VIDEO_PATH, help="Đường dẫn video (đã chạy OCR)")
    args = parser.parse_args()
    sys.exit(0 if run_translation(args.input) else 1)
```

- [ ] **Step 3: `config.py`.** Xóa khối comment `# Prompt dịch ...`, `TRANSLATION_GENRE = ...` và toàn bộ chuỗi `TRANSLATION_SYSTEM_PROMPT = """..."""`. Prompt dịch giờ nằm trong `translate_pass.py`. Kiểm tra không còn chỗ nào dùng: `Select-String -Path *.py,tests\*.py -Pattern 'TRANSLATION_GENRE|TRANSLATION_SYSTEM_PROMPT'`.

- [ ] **Step 4:** `python -m pytest tests -q` → 29 test (test_core 17, test_llm 5, test_jobs 3, cộng các test render hiện có…). Tất cả phải qua, trừ phần render ở Task 3.

---

### Task 3: Render đọc `lines.json`, tra bản dịch theo segment; pipeline

**Files:** Modify `script.py`, `pipeline.py`, `tests/test_core.py`

- [ ] **Step 1: Test (fail).**

```python
def test_segment_index_uses_per_occurrence_translation():
    # Cùng một câu tiếng Trung, 2 lần xuất hiện, 2 người nói khác nhau → 2 bản dịch khác nhau
    box = [100, 900, 340, 940]
    segs = [{'id': 0, 'text': '我知道了', 'start_frame': 0, 'end_frame': 10, 'bbox': box, 'keyframe_bboxes': []},
            {'id': 1, 'text': '我知道了', 'start_frame': 50, 'end_frame': 60, 'bbox': box, 'keyframe_bboxes': []}]
    idx = s._build_segment_index(segs, 0, 70, {'我知道了': 'Tôi biết rồi'}, ocr_interval_frames=1,
                                 vi_for_seg={0: 'Bản tiểu thư biết rồi', 1: 'Nô tỳ biết rồi'})
    assert idx[5][0]['vi_text'] == 'Bản tiểu thư biết rồi'
    assert idx[55][0]['vi_text'] == 'Nô tỳ biết rồi'
```

- [ ] **Step 2: `_build_segment_index`.**
  - Chữ ký thêm tham số cuối `vi_for_seg: dict = None`.
  - Dòng `vi_text = translation_map.get(zh_text, "")` **trong hàm này** thành:

```python
        # Bản dịch theo TỪNG LẦN XUẤT HIỆN (id segment); không có id → tra theo chữ (tương thích)
        vi_text = (vi_for_seg or {}).get(seg.get("id"), translation_map.get(zh_text, ""))
```

- [ ] **Step 3: `process_chunk`.**
  - Tuple `args` thêm phần tử cuối `vi_for_seg`, sau `text_segments_for_chunk`.
  - Lời gọi `_build_segment_index(...)` thêm `vi_for_seg=vi_for_seg`.

- [ ] **Step 4: `__main__`.**
  - Chỗ kiểm tra `job.translation_map` đổi thành `job.lines`, câu báo lỗi giữ nguyên ý.
  - Thay khối nạp `translation_map`:

```python
    with open(job.translation_map, "r", encoding="utf-8") as f:
        translation_map = json.load(f)
```

bằng:

```python
    with open(job.lines, "r", encoding="utf-8") as f:
        lines = json.load(f)
```

  - Ngay sau dòng `text_segments = ocr_data.get("text_segments", [])`, thêm:

```python
    # Bản dịch theo TỪNG LẦN XUẤT HIỆN: id segment → vi_sua (sửa tay) hoặc vi.
    # translation_map (chữ OCR thô → bản dịch lần xuất hiện đầu) chỉ dùng cho chữ thô không thuộc segment nào.
    for i, seg in enumerate(text_segments):
        seg["id"] = i
    vi_for_seg, translation_map = {}, {}
    for l in sorted(lines, key=lambda l: l["start"]):
        vi = l.get("vi_sua") or l.get("vi") or ""
        vi_for_seg[l["id"]] = vi
        if l["id"] < len(text_segments):
            translation_map.setdefault(text_segments[l["id"]]["text"], vi)
```

  - Dòng in `Translations   : {len(translation_map)} entries` thành `Translations   : {len(lines)} dòng (lines.json)`.
  - Trong `worker_args`, thêm `vi_for_seg` làm phần tử cuối.

- [ ] **Step 5: `pipeline.py`.** Khối `--skip-existing`: `job.translation_map` → `job.lines`.

- [ ] **Step 6: Kiểm chứng.**
  - `Select-String -Path *.py -Pattern 'translation_map\.json|job\.translation_map|job\.profiles|job\.glossary'` → không còn.
  - `python -m pytest tests -q` → tất cả qua.
  - Render thử: sửa `$SCRATCH\smoke_render.py` để ghi `job.lines` thay cho `translation_map`:

```python
json.dump([{"id": i, "start": sg["start_frame"], "end": sg["end_frame"], "zh": sg["text"], "loai": "thoai",
            "vi": "Xin chào thế giới"} for i, sg in enumerate(ocr["text_segments"])],
          open(job.lines, "w", encoding="utf-8"), ensure_ascii=False)
```

  Kết quả mong đợi: `EXIT 0`, có `test_vi.mp4` kèm audio, và frame ở giây 1.4 có chữ "Xin chào thế giới".
- [ ] **Step 7: Commit** gộp cả Task 1–3, vì phải đi cùng nhau thì pipeline mới nhất quán: `feat(dịch): dịch theo cốt truyện — hồ sơ truyện, người nói/người nghe, dịch từng lần xuất hiện theo bảng xưng hô` (+ Co-Authored-By)

---

### Task 4: Kiểm chứng tổng + chạy thật (cần user cho phép)

- [ ] **Step 1: Chạy thử trên dữ liệu thật với LLM giả.**
  - Script `$SCRATCH\b3_dryrun.py`: job tạm, `ocr_data` thật (hihi), segment dựng lại bằng code mới.
  - `ask_json` giả trả JSON hợp lệ cho từng bước, phân biệt bước bằng `system`: `tp.STORY_SYS` / `SPEAKER_SYS` / `TRANSLATE_SYS` / `QA_SYS` / `NORMALIZATION_SYSTEM_PROMPT`.
  - Đếm số lời gọi mỗi bước, đo thời gian. Mong đợi: OK, dưới 60 giây, `lines.json` phủ đủ các segment.
- [ ] **Step 2: Review độc lập** cho range B3, rồi sửa các điểm Critical / Important.
- [ ] **Step 3: HỎI USER** cho phép chạy thật:
  - OCR đầy đủ `newvideo.mp4` (khoảng 2 phút video, OCR bằng CPU mất vài phút);
  - dịch bằng Gemini (khoảng 25–30 lời gọi, free tier);
  - render.
- [ ] **Step 4 (sau khi được phép):** chạy pipeline thật.
  - Đọc `lines.json` và `story.json`: đối chiếu tiếng Trung, người nói và bản dịch; soát xưng hô theo cặp và ý theo mạch truyện.
  - Báo cáo cho user bằng ví dụ cụ thể, kèm vài frame.
