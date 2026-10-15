"""
Unit tests cho các hàm thuần (không cần video/API/GPU).
Chạy:  python -m pytest tests/         (nếu có pytest)
   hoặc: python tests/test_core.py       (standalone, không cần pytest)
"""

import os
import sys
import json
import shutil
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import script as s          # noqa: E402
import translate_pass as tp  # noqa: E402
import utils                 # noqa: E402
import ocr_pass as op        # noqa: E402
import jobs                  # noqa: E402

FP = "nonexistent.ttf"  # → ImageFont.load_default()


# ── Thuật toán chống đè phụ đề ──

def test_layout_no_overlap_and_pushdown():
    jobs = [
        {'text': 'Cau thoai chinh rat dai se bi ngat dong', 'cx': 500, 'max_w': 300, 'size': 32, 'y': 100, 'orig_y': 100},
        {'text': 'Nhan ten nhan vat', 'cx': 500, 'max_w': 300, 'size': 32, 'y': 108, 'orig_y': 108},
        {'text': 'Dong thoai thu hai cung kha dai', 'cx': 500, 'max_w': 300, 'size': 32, 'y': 116, 'orig_y': 116},
    ]
    out = s._layout_subtitle_blocks([dict(j) for j in jobs], img_h=1920, font_path=FP,
                                    margin=8, line_spacing=12, min_size=22)
    prev_bottom = None
    for j in out:
        if prev_bottom is not None:
            assert j['y'] >= prev_bottom - 1, "Cụm chữ bị đè lên cụm phía trên!"
        prev_bottom = j['y'] + j['height']


def test_layout_overflow_pushback_fits():
    jobs = [{'text': c, 'cx': 500, 'max_w': 300, 'size': 32, 'y': 100 + i * 10, 'orig_y': 100 + i * 10}
            for i, c in enumerate(['Alpha', 'Beta', 'Gamma'])]
    img_h = 150
    out = s._layout_subtitle_blocks([dict(j) for j in jobs], img_h=img_h, font_path=FP,
                                    margin=8, line_spacing=12, min_size=22)
    assert out[0]['y'] >= 0, "Cụm trên cùng bị âm"
    last_bottom = out[-1]['y'] + out[-1]['height']
    assert last_bottom <= img_h, f"Tràn đáy: {last_bottom} > {img_h}"


# ── Hình học ──

def test_calculate_iou():
    assert s.calculate_iou([0, 0, 10, 10], [0, 0, 10, 10]) > 0.99   # trùng hoàn toàn
    assert s.calculate_iou([0, 0, 10, 10], [100, 100, 110, 110]) == 0.0  # rời nhau
    partial = s.calculate_iou([0, 0, 10, 10], [5, 5, 15, 15])
    assert 0.0 < partial < 1.0


def test_interpolate_bbox():
    kfs = [{'frame': 0, 'bbox': [0, 0, 10, 10]}, {'frame': 10, 'bbox': [100, 0, 110, 10]}]
    assert s.interpolate_bbox(kfs, 0)[0] == 0
    assert s.interpolate_bbox(kfs, 10)[0] == 100
    mid = s.interpolate_bbox(kfs, 5)
    assert abs(mid[0] - 50) < 1e-6, "Nội suy giữa phải = 50"
    # Ngoài khoảng → kẹp về đầu/cuối
    assert s.interpolate_bbox(kfs, -5)[0] == 0
    assert s.interpolate_bbox(kfs, 999)[0] == 100


def test_video_mismatch():
    data = {"width": 1440, "height": 1080, "fps": 25.0, "total_frames": 1000}
    assert s.video_mismatch(data, 1440, 1080, 25.0, 1000) is None
    assert s.video_mismatch(data, 1440, 1080, 25.0, 5000) is None, "Video dài hơn dữ liệu (OCR --test) vẫn cho qua"
    assert s.video_mismatch(data, 1440, 1080, 25.0, 0) is None, "Không đọc được số frame -> bỏ qua kiểm tra này"
    assert "kích thước" in s.video_mismatch(data, 720, 1280, 25.0, 1000)
    assert "fps" in s.video_mismatch(data, 1440, 1080, 30.0, 1000)
    assert "frame" in s.video_mismatch(data, 1440, 1080, 25.0, 900)


# ── Segment OCR (ocr_pass) ──

def _det(text, x1, y1=900, h=40, w_per_char=40):
    x2 = x1 + w_per_char * len(text)
    return {'text': text, 'box': [[x1, y1], [x2, y1], [x2, y1 + h], [x1, y1 + h]], 'confidence': 0.99}


def test_segments_split_different_lines_same_position():
    # 2 câu khác nhau, cùng độ dài, cùng chỗ, nối tiếp nhau — lỗi cũ: câu sau "nuốt" câu trước
    fd = {str(f): [_det('我要去修仙！', 100)] for f in range(0, 3)}
    fd.update({str(f): [_det('让姓白的后悔', 100)] for f in range(3, 6)})
    segs = op.build_text_segments(fd, fps=25.0, ocr_interval_frames=1)
    assert [s['text'] for s in segs] == ['我要去修仙！', '让姓白的后悔']
    assert segs[0]['end_frame'] == 2 and segs[1]['start_frame'] == 3


def test_segments_keep_ocr_misread_in_same_segment():
    # Bảo vệ: OCR đọc sai 1 chữ ở 1 frame vẫn thuộc cùng segment (luật 2, giống >= 0.7)
    fd = {str(f): [_det('我要去修仙！', 100)] for f in range(0, 6)}
    fd['3'] = [_det('我要去修伯！', 100)]
    segs = op.build_text_segments(fd, fps=25.0, ocr_interval_frames=1)
    assert len(segs) == 1 and segs[0]['text'] == '我要去修仙！'


def test_watermark_filter_keeps_dialogue_with_latin_slang():
    # Đo trên 15 video thật: luật "Latin > 30%" bỏ cả thoại chen tiếng lóng → để nguyên chữ Trung, không dịch.
    # Nhãn "meme" giờ đi tiếp: bước chuẩn hóa / dịch của B3 coi là rác → render xóa, không vẽ.
    assert not op.is_watermark('就是这个feel')
    assert not op.is_watermark('多惬意，skr!')
    assert op.is_watermark('@douyin123') and op.is_watermark('抖音号：123')
    assert not op.is_watermark('原创@哈加密'), "mục SUPPRESS_TEXTS phải được giữ lại để render xóa"


def test_segments_typewriter_reveal_stays_one_segment():
    # Chữ hiện dần (typewriter), phụ đề căn giữa: phần đã hiện phải nhập vào câu đầy đủ
    full = '我要成为名动天下的女修'
    fd = {}
    for f in range(30):
        n = min(len(full), 2 + f // 3)                # mỗi 3 frame hiện thêm 1 chữ
        fd[str(f)] = [_det(full[:n], 500 - 20 * n)]   # căn giữa quanh x=500
    segs = op.build_text_segments(fd, fps=25.0, ocr_interval_frames=1)
    assert len(segs) == 1 and segs[0]['text'] == full


def test_segment_index_trim_never_widens():
    # Cặp sau không được nới lại giới hạn mà cặp trước đã cắt (tránh vẽ 2 câu cùng 1 frame)
    box = [100, 900, 340, 940]
    mk = lambda t, a, b: {'text': t, 'start_frame': a, 'end_frame': b, 'bbox': box, 'keyframe_bboxes': []}
    idx = s._build_segment_index([mk('甲', 0, 11), mk('乙', 5, 9), mk('丙', 12, 30)], 0, 40, {},
                                 ocr_interval_frames=1)
    assert '丙' not in [e['text'] for e in idx.get(11, [])]


# ── Wrap & chiều cao ──

def test_wrap_and_height():
    from PIL import ImageFont
    font = ImageFont.load_default()
    lines = s.wrap_text("mot hai ba bon nam sau bay tam", font, max_width=40)
    assert len(lines) >= 2, "Text dài phải wrap thành nhiều dòng"

    assert s.get_text_render_height("", FP, 32, 300) == 0
    short = s.get_text_render_height("A", FP, 32, 300)
    long = s.get_text_render_height("A B C D E F G H I J K L M N", FP, 32, 60)
    assert long >= short > 0, "Text nhiều dòng phải cao hơn text 1 dòng"


def test_layout_cache_hits():
    s._resolve_layout_cached.cache_clear()
    for _ in range(5):
        s._resolve_layout_cached("cau giong nhau", FP, 32, 300, 22)
    info = s._resolve_layout_cached.cache_info()
    assert info.hits >= 4, f"Cache không hoạt động: {info}"


# ── Pass dịch (translate_pass) ──

def test_run_translation_fails_without_input():
    d = tempfile.mkdtemp()
    orig_jobs, orig_load, orig_ask = jobs.JOBS_DIR, tp.load_ocr_data, tp.ask_json

    def no_llm(*a, **kw):
        raise AssertionError("Test này không được gọi LLM (sẽ tốn API thật)")
    try:
        tp.ask_json = no_llm
        jobs.JOBS_DIR = os.path.join(d, "jobs")
        v = os.path.join(d, "tap1.mp4")
        with open(v, "wb") as f:
            f.write(b"video")
        assert tp.run_translation(v) is False                   # chưa có job (chưa OCR)
        jobs.open_job(v, create=True)
        tp.load_ocr_data = lambda path: None                    # có job nhưng thiếu ocr_data.json
        assert tp.run_translation(v) is False
        tp.load_ocr_data = lambda path: {"unique_texts": []}    # OCR không ra chữ nào
        assert tp.run_translation(v) is False
        tp.load_ocr_data = lambda path: {"unique_texts": ["你好"], "text_segments": [{}], "complete": False}
        assert tp.run_translation(v) is False                   # OCR bị ngắt giữa chừng
    finally:
        jobs.JOBS_DIR, tp.load_ocr_data, tp.ask_json = orig_jobs, orig_load, orig_ask
        shutil.rmtree(d, ignore_errors=True)


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
    tp.mark_labels_and_onscreen(lines, {"nhan_vat": {"sun_qing": {"ten": "Tôn Tình"}}, "nhan": {"孙晴": "sun_qing"}})
    assert lines[0]["loai"] == "nhan"
    assert lines[1]["tren_man_hinh"] == ["sun_qing"] and lines[2]["tren_man_hinh"] == []


def test_merge_story_only_adds():
    old = {"tom_tat": "cũ", "nhan_vat": {"yiyi": {"ten": "Y Y (sửa tay)"}},
           "xung_ho": {"yiyi>tieu_thu": {"xung": "nô tỳ", "goi": "tiểu thư"}}}
    new = {"tom_tat": "mới", "nhan_vat": {"yiyi": {"ten": "Y Y"}, "bach_lang": {"ten": "Bạch Lang"}},
           "xung_ho": {"yiyi>tieu_thu": {"xung": "tôi", "goi": "cô"}, "bach_lang>*": {"xung": "ta", "goi": "các ngươi"}}}
    m, dropped = tp.merge_story(old, new)
    assert dropped == 0
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
    assert "noi" not in lines[3], "Bỏ sót sau khi hỏi lại → để trống (không dừng, lần chạy sau hỏi lại)"


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


def test_translate_labels_once_per_text_not_mixed_into_dialogue():
    lines = [_line(0, '孙晴', 0, 50, loai='nhan'), _line(1, '你来了', 10, 20, noi='a', nghe='b'),
             _line(2, '孙晴', 60, 90, loai='nhan'), _line(3, '宛雪（女主）', 60, 90, loai='nhan')]
    seen = []

    def fake(system, user, temperature=0.1, **kw):
        seen.append(user)
        return {"0": "Tôn Tình", "1": "Uyển Tuyết (nữ chính)"}
    _with_fake_llm(fake, tp.translate_labels, lines, {"nhan": {"孙晴": "sun_qing"}})
    assert len(seen) == 1 and seen[0].count('孙晴') == 1, "Mỗi chữ nhãn dịch đúng 1 lần"
    assert '你来了' not in seen[0], "Lời thoại không lẫn vào lời gọi dịch nhãn"
    assert [l.get("vi") for l in lines] == ["Tôn Tình", None, "Tôn Tình", "Uyển Tuyết (nữ chính)"]


def test_translate_labels_drops_echoed_owner_hint():
    # Video Douyin "末世开中华餐厅": model chép luôn gợi ý đầu vào → nhãn hiện "Lợi Lâm (nhãn của: Lilian)"
    lines = [_line(0, '莉琳', 0, 50, loai='nhan'), _line(1, '宛雪（女主）', 60, 90, loai='nhan')]
    _with_fake_llm(lambda s, u, temperature=0.1, **kw: {"0": "Lợi Lâm (nhãn của: Lilian)", "1": "Uyển Tuyết (nữ chính)"},
                   tp.translate_labels, lines, {})
    assert [l["vi"] for l in lines] == ["Lợi Lâm", "Uyển Tuyết (nữ chính)"]


def test_translate_lines_retries_missing_and_empty_then_raises():
    lines = [_line(1, '你好', 0, 5, noi='a', nghe='b'), _line(2, '走吧', 6, 9, noi='b', nghe='a')]
    calls = []

    def fake(system, user, temperature=0.1, **kw):
        calls.append(user)
        return {"1": "Xin chào", "2": ""} if 'CHỈ dịch' not in user else {"2": "Đi thôi"}  # lần đầu để trống dòng 2
    _with_fake_llm(fake, tp.translate_lines, lines, {})
    assert [l["vi"] for l in lines] == ["Xin chào", "Đi thôi"] and len(calls) == 2, "Rỗng lần đầu → hỏi lại"
    try:
        _with_fake_llm(lambda s, u, temperature=0.1, **kw: {"1": "Xin chào"}, tp.translate_lines,
                       [_line(1, '你好', 0, 5), _line(2, '走吧', 6, 9)], {})
        raise AssertionError("Vẫn thiếu dòng phải ném LLMError")
    except tp.LLMError as e:
        assert "走吧" in str(e), "Báo lỗi phải kèm câu tiếng Trung bị thiếu"


def test_segment_index_uses_per_occurrence_translation():
    # Cùng một câu tiếng Trung, 2 lần xuất hiện, 2 người nói khác nhau → 2 bản dịch khác nhau
    box = [100, 900, 340, 940]
    segs = [{'id': 0, 'text': '我知道了', 'start_frame': 0, 'end_frame': 10, 'bbox': box, 'keyframe_bboxes': []},
            {'id': 1, 'text': '我知道了', 'start_frame': 50, 'end_frame': 60, 'bbox': box, 'keyframe_bboxes': []}]
    idx = s._build_segment_index(segs, 0, 70, {'我知道了': 'Tôi biết rồi'}, ocr_interval_frames=1,
                                 vi_for_seg={0: 'Bản tiểu thư biết rồi', 1: 'Nô tỳ biết rồi'})
    assert idx[5][0]['vi_text'] == 'Bản tiểu thư biết rồi'
    assert idx[55][0]['vi_text'] == 'Nô tỳ biết rồi'


def test_lines_mismatch_detects_ocr_rerun():
    segs = [{'text': 'A', 'start_frame': 0, 'end_frame': 5}, {'text': 'B', 'start_frame': 10, 'end_frame': 15}]
    lines = [{"id": 0, "start": 0, "end": 5}, {"id": 1, "start": 10, "end": 15}]
    assert not s.lines_mismatch(lines, segs)
    assert s.lines_mismatch(lines[:1], segs), "Thiếu dòng"
    assert s.lines_mismatch([lines[0], {"id": 1, "start": 11, "end": 15}], segs), "Segment đổi (OCR chạy lại)"


def test_build_lines_matches_edits_on_raw_ocr_not_normalized_text():
    segs = [{'text': '我知道了。', 'start_frame': 0, 'end_frame': 5}]
    old = [{"id": 0, "ocr": "我知道了。", "zh": "我知道了。", "vi_sua": "Nô tỳ biết rồi"}]
    lines = tp.build_lines(segs, {'我知道了。': '我知道了'}, old)       # chuẩn hóa ra khác lần trước
    assert lines[0]["vi_sua"] == "Nô tỳ biết rồi", "Sửa tay không được mất chỉ vì chuẩn hóa đổi"


def test_merge_story_drops_malformed_model_output():
    new = {"nhan_vat": {"a": {"ten": "A"}, "b": ["sai kiểu"]},
           "xung_ho": {"a -> b": {"xung": "ta", "goi": "ngươi"}, "a>*": "sai kiểu", "khong_co_mui_ten": {}},
           "ten_rieng": {"天元宗": None}, "nhan": {"孙晴": None, "萧衡": "khong_ton_tai", "我": "a"}}
    m, dropped = tp.merge_story({}, new)
    assert list(m["nhan_vat"]) == ["a"] and m["xung_ho"] == {"a>b": {"xung": "ta", "goi": "ngươi"}}
    assert m["nhan"] == {"我": "a"} and m["ten_rieng"] == {}
    assert dropped == 6, dropped   # b (nhân vật) + a>*, khong_co_mui_ten (xưng hô) + 天元宗 + 孙晴, 萧衡 (nhãn)


def test_repeated_line_translated_once_and_copied():
    lines = [_line(0, '我要去修仙！', 0, 20), _line(1, '我要去修仙！', 30, 50), _line(2, '走吧', 60, 70)]
    tp.mark_labels_and_onscreen(lines, {})
    assert lines[1].get("lap_lai") == 0 and "lap_lai" not in lines[2]

    def fake(system, user, temperature=0.1, **kw):
        todo = user.split("CẦN DỊCH:\n", 1)[1]
        return {x[1:x.index("]")]: "VI" + x[1:x.index("]")] for x in todo.splitlines() if x.startswith("[")}
    _with_fake_llm(fake, tp.translate_lines, lines, {})
    tp.copy_repeats(lines)
    assert [l["vi"] for l in lines] == ["VI0", "VI0", "VI2"], "Câu bị OCR tách đôi phải cùng một bản dịch"


def test_strip_speaker_prefix_only_the_speakers_own_name():
    # Video Douyin 36 phút: model tự chèn tên người nói vào đầu ~2% câu dịch ("Hệ thống: Chúc mừng ngài...")
    story = {"nhan_vat": {"ht": {"ten": "Hệ thống"}, "lam": {"ten": "Hiệu trưởng Lâm"}}}
    lines = [_line(1, '恭喜获得', 0, 5, noi='ht', vi='Hệ thống: Chúc mừng ngài.'),
             _line(2, '所以成绩', 6, 9, noi='lam', vi='Tôi: Vậy nên thành tích'),
             _line(3, '条件一.当周所有科目', 10, 15, noi='?', vi='Điều kiện 1: Tất cả các môn'),
             _line(4, '公告：明天放假', 16, 20, noi='ht', vi='Hệ thống: Thông báo'),   # câu gốc có ':' → giữ
             _line(5, '你好', 21, 25, noi='lam', vi='Hệ thống: Chào')]                # không phải tên người nói → giữ
    assert tp.strip_speaker_prefix(lines, story) == 2
    assert [l["vi"] for l in lines] == ["Chúc mừng ngài.", "Vậy nên thành tích", "Điều kiện 1: Tất cả các môn",
                                        "Hệ thống: Thông báo", "Hệ thống: Chào"]


def test_attribute_speakers_reasks_lines_with_removed_characters():
    story = {"nhan_vat": {"tt": {"ten": "Tôn Tình"}}}                      # sun_qing đã đổi tên thành tt
    lines = [_line(1, '你好', 0, 5, noi='sun_qing', nghe='*', chac=1.0)]
    _with_fake_llm(lambda s, u, temperature=0.1, **kw: {"1": {"noi": "tt", "nghe": "*", "chac": 0.9}},
                   tp.attribute_speakers, lines, story)
    assert lines[0]["noi"] == "tt", "Người nói trỏ tới id không còn trong hồ sơ phải được xác định lại"


def test_qa_lines_only_accepts_window_ids_and_keeps_on_error():
    lines = [_line(1, '你好', 0, 5, noi='a', nghe='b', vi='Chào'), _line(2, '走吧', 6, 9, noi='b', nghe='a', vi='Đi')]
    fixed = _with_fake_llm(lambda s, u, temperature=0.1, **kw: {"1": "Xin chào", "2": "Đi", "999": "ngoài cửa sổ"},
                           tp.qa_lines, lines, {})
    assert fixed == 1 and [l["vi"] for l in lines] == ["Xin chào", "Đi"], "trả lại y nguyên câu cũ không tính là sửa"

    def broken(*a, **kw):
        raise tp.LLMError("hỏng")
    assert _with_fake_llm(broken, tp.qa_lines, lines, {}) == 0 and lines[0]["vi"] == "Xin chào"


def test_run_translation_rerun_is_free_and_keeps_edits():
    # Chạy thật run_translation (llm.ask_json thật + cache thật trong thư mục tạm, provider GIẢ):
    # chạy lại không đổi gì → 0 lời gọi LLM, cả video lẻ lẫn tập của series; vi_sua được giữ.
    import re
    import llm
    d = tempfile.mkdtemp()
    saved = (jobs.JOBS_DIR, jobs.SERIES_DIR, llm.LLM_CACHE_PATH, llm._db, llm._default_providers, tp.load_ocr_data)
    calls = []

    def ids(block):
        return re.findall(r'^\[(\d+)\]', block, re.M)

    def provider(system, user, temperature):
        calls.append(system[:15])
        if system == tp.NORMALIZATION_SYSTEM_PROMPT:
            return json.dumps({t: t for t in json.loads(user)})
        if system == tp.STORY_SYS:
            return json.dumps({"tom_tat": "t", "tom_tat_tap": "tập", "nhan_vat": {"a": {"ten": "A"}, "b": {"ten": "B"}},
                               "xung_ho": {"a>b": {"xung": "ta", "goi": "ngươi"}}})
        if system.startswith(tp.SPEAKER_SYS):
            return json.dumps({i: {"noi": "a", "nghe": "b", "chac": 0.9}
                               for i in ids(user.split("ĐOẠN CẦN XÁC ĐỊNH:\n", 1)[1])})
        if system.startswith(tp.TRANSLATE_SYS):
            return json.dumps({i: f"VI{i}" for i in ids(user.split("CẦN DỊCH:\n", 1)[1])})
        return "{}"                                                        # QA: không sửa gì
    try:
        jobs.JOBS_DIR, jobs.SERIES_DIR = os.path.join(d, "jobs"), os.path.join(d, "series")
        if llm._db is not None:
            llm._db.close()
        llm._db, llm.LLM_CACHE_PATH = None, os.path.join(d, "cache", "llm.sqlite")
        llm._default_providers = lambda: [("fake", provider)]
        segs = [{"text": t, "start_frame": i * 30, "end_frame": i * 30 + 20} for i, t in enumerate(["你好", "走吧", "什么"])]
        tp.load_ocr_data = lambda path: {"unique_texts": ["你好", "走吧", "什么"], "text_segments": segs,
                                         "fps": 25.0, "complete": True}
        for series in (None, "bo"):
            v = os.path.join(d, f"tap_{series}.mp4")
            with open(v, "wb") as f:
                f.write(os.urandom(16))
            job = jobs.open_job(v, series=series, create=True)
            calls.clear()
            assert tp.run_translation(v) is True and calls, "Lần đầu phải gọi LLM"
            lines = json.load(open(job.lines, encoding="utf-8"))
            lines[0]["vi_sua"] = "Sửa tay"
            json.dump(lines, open(job.lines, "w", encoding="utf-8"), ensure_ascii=False)
            calls.clear()
            assert tp.run_translation(v) is True
            assert calls == [], f"Chạy lại không đổi gì phải tốn 0 lời gọi (series={series}): {calls}"
            assert json.load(open(job.lines, encoding="utf-8"))[0]["vi_sua"] == "Sửa tay"
    finally:
        if llm._db is not None:
            llm._db.close()
        (jobs.JOBS_DIR, jobs.SERIES_DIR, llm.LLM_CACHE_PATH, llm._db, llm._default_providers, tp.load_ocr_data) = saved
        shutil.rmtree(d, ignore_errors=True)


# ── Ghi JSON an toàn (utils) ──

def _run_ocr_with_fakes(fail_ocr_at=None, fail_read_at=None, d=None, max_scans=None, no_text=False):
    """Chạy op.run_ocr_pass trên video tổng hợp (tests/wm_synth) với PaddleOCR giả (không cần paddle):
    OCR giả trả 1 dòng chữ mỗi frame; fail_ocr_at: frame OCR ném lỗi (như hết RAM); fail_read_at: từ frame này
    cap.read() thất bại giữa video. → (kết quả run_ocr_pass, ocr_data đã ghi, thư mục tạm để dọn)."""
    import types
    import cv2
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import wm_synth
    fresh = d is None
    d = d or tempfile.mkdtemp()
    video = os.path.join(d, "clip.mp4")
    if fresh:
        wm_synth.make_video(video)
    fake_paddle = types.ModuleType("paddle")
    fake_paddle.device = types.SimpleNamespace(is_compiled_with_cuda=lambda: False)
    fake_paddleocr = types.ModuleType("paddleocr")
    fake_paddleocr.PaddleOCR = lambda *a, **k: object()
    real_cap, real_ocr_frame = cv2.VideoCapture, op.ocr_frame
    saved = {k: sys.modules.get(k) for k in ("paddle", "paddleocr")}

    class _Cap:
        def __init__(self, path):
            self._c, self._n = real_cap(path), 0

        def read(self):
            if fail_read_at is not None and self._n >= fail_read_at:
                return False, None
            self._n += 1
            return self._c.read()

        def __getattr__(self, name):
            return getattr(self._c, name)

    def fake_ocr_frame(engine, frame, height, width, crop, resize):
        n = fake_ocr_frame.calls
        fake_ocr_frame.calls += 1
        if fail_ocr_at is not None and n == fail_ocr_at:
            raise RuntimeError("could not create a primitive")
        if no_text:
            return []
        return [{"text": "你好世界", "box": [[10, 10], [90, 10], [90, 40], [10, 40]], "confidence": 0.99}]
    fake_ocr_frame.calls = 0
    orig_dirs = (jobs.JOBS_DIR, jobs.SERIES_DIR)
    try:
        sys.modules["paddle"], sys.modules["paddleocr"] = fake_paddle, fake_paddleocr
        cv2.VideoCapture, op.ocr_frame = _Cap, fake_ocr_frame
        jobs.JOBS_DIR, jobs.SERIES_DIR = os.path.join(d, "jobs"), os.path.join(d, "series")
        result = op.run_ocr_pass(video, test_mode=False, test_frames=0, max_scans=max_scans)
        with open(jobs.open_job(video).ocr_data, encoding="utf-8") as f:
            data = json.load(f)
    finally:
        cv2.VideoCapture, op.ocr_frame = real_cap, real_ocr_frame
        jobs.JOBS_DIR, jobs.SERIES_DIR = orig_dirs
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return result, data, d


def test_ocr_error_mid_video_is_not_recorded_as_no_text():
    # Video Douyin 36 phút: OCR lỗi vì hết RAM ở vài frame bị ghi là "không có chữ" rồi vẫn báo xong → mất chữ im lặng
    result, data, d = _run_ocr_with_fakes(fail_ocr_at=60)
    try:
        assert result == op.NEXT, "OCR lỗi giữa chừng (vd hết RAM): lưu tiến trình, báo chạy tiếp bằng tiến trình mới"
        assert data["complete"] is False, "dữ liệu dở dang phải complete=False để chạy tiếp / dịch từ chối"
        # lần gọi OCR thứ 60 lỗi (khoảng frame 180 vì frame trùng được bỏ qua): chỉ giữ 60 lần OCR trước đó,
        # bỏ cả kết quả các frame sau đã nằm sẵn trong hàng đợi — chạy lại phải quét lại từ frame lỗi
        assert 0 < len(data["frame_detections"]) <= 60, len(data["frame_detections"])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_ocr_read_failure_mid_video_is_not_end_of_video():
    # Hết RAM làm cap.read() thất bại ở frame ~29.600/54.606 → vòng đọc tưởng hết video → complete=True, mất 46% cuối
    result, data, d = _run_ocr_with_fakes(fail_read_at=100)
    try:
        assert result == op.NEXT and data["complete"] is False, "đọc hỏng giữa video không phải hết video"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_ocr_fake_run_completes_normally():
    # Đối chứng: không lỗi thì OCR giả chạy hết video, complete=True
    result, data, d = _run_ocr_with_fakes()
    try:
        assert result is not None and data["complete"] is True
        assert max(int(k) for k in data["frame_detections"]) >= 200
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_ocr_in_chunks_equals_one_run():
    # PaddleOCR rò ~4 MB/frame trong tiến trình → OCR theo đợt (mỗi đợt một tiến trình); ghép lại phải y như chạy một mạch
    one, data_one, d1 = _run_ocr_with_fakes()
    d2 = None
    try:
        for _ in range(40):
            result, data, d2 = _run_ocr_with_fakes(d=d2, max_scans=15)
            if result != op.NEXT:
                break
        assert result is not None and result != op.NEXT and data["complete"] is True
        # keyframe có thể lệch chút (trạng thái "bỏ frame trùng" bắt đầu lại mỗi đợt) — thứ dùng về sau là segment
        a, b = data["text_segments"], data_one["text_segments"]
        tol = int(op.OCR_MAX_SKIP_SEC * 25) + 1          # mép segment lệch tối đa một bước "bỏ frame trùng"
        assert len(a) == len(b) and all(x["text"] == y["text"] and abs(x["start_frame"] - y["start_frame"]) <= tol
                                        and abs(x["end_frame"] - y["end_frame"]) <= tol for x, y in zip(a, b)),             ("ghép các đợt phải cho cùng segment như chạy một mạch", a, b)
        assert data["scanned_until"] == data["total_frames"]
    finally:
        shutil.rmtree(d1, ignore_errors=True)
        if d2:
            shutil.rmtree(d2, ignore_errors=True)


def test_ocr_chunks_advance_through_stretch_without_text():
    # Chạy tiếp từng dựa vào "frame có chữ cuối cùng" → đoạn dài không chữ làm đợt sau bắt đầu lại chỗ cũ, lặp mãi
    d = None
    try:
        for n in range(40):
            result, data, d = _run_ocr_with_fakes(d=d, max_scans=15, no_text=True)
            if result != op.NEXT:
                break
        assert result is not None and result != op.NEXT and data["complete"] is True, f"kẹt sau {n + 1} đợt"
    finally:
        if d:
            shutil.rmtree(d, ignore_errors=True)


def test_run_in_chunks_restarts_until_done_and_stops_when_stalled():
    calls = []
    codes = iter([op.EXIT_NEXT_CHUNK, op.EXIT_NEXT_CHUNK, 0])
    progress = iter([0, 100, 200, 300])
    assert op.run_in_chunks(["x"], lambda: next(progress), run=lambda argv: calls.append(argv) or next(codes)) == 0
    assert len(calls) == 3, "chạy con mới tới khi xong"
    assert op.run_in_chunks(["x"], lambda: 0, run=lambda argv: 1) == 1, "lỗi hẳn (mã 1) → dừng ngay"
    stalled = []
    rc = op.run_in_chunks(["x"], lambda: 50, run=lambda argv: stalled.append(1) or op.EXIT_NEXT_CHUNK)
    assert rc == 1 and len(stalled) == 2, "không tiến thêm frame nào 2 lần liền → dừng, không lặp mãi"


def test_write_json_atomic_keeps_old_file_on_error():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "data.json")
    try:
        utils.write_json_atomic(path, {"你好": "Xin chào"})
        with open(path, encoding="utf-8") as f:
            assert json.load(f) == {"你好": "Xin chào"}
        try:
            utils.write_json_atomic(path, {"hỏng": object()})  # object() không serialize được
            raise AssertionError("Phải ném TypeError")
        except TypeError:
            pass
        with open(path, encoding="utf-8") as f:
            assert json.load(f) == {"你好": "Xin chào"}, "File cũ phải còn nguyên"
        assert os.listdir(tmpdir) == ["data.json"], "Không được để lại file tạm"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_imwrite_any_roundtrip_unicode_path():
    d = tempfile.mkdtemp(prefix="ảnh 修仙 ")
    try:
        img = np.zeros((6, 9), np.uint8)
        img[2:4, 3:7] = 255
        p = os.path.join(d, "mặt nạ.png")
        utils.imwrite_any(p, img)
        back = utils.imread_gray(p)
        assert back is not None and (back == img).all()
        assert utils.imread_gray(os.path.join(d, "không có.png")) is None
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── Standalone runner (không cần pytest) ──

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
