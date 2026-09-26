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
    orig_jobs, orig_load = jobs.JOBS_DIR, tp.load_ocr_data
    try:
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
        jobs.JOBS_DIR, tp.load_ocr_data = orig_jobs, orig_load
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


# ── Ghi JSON an toàn (utils) ──

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
