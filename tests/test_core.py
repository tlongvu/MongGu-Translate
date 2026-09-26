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


# ── Cache dịch (translate_pass) ──

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


def _fake_llm(first_window_reply):
    """ask_json giả cho Phase 1-4: hồ sơ {}, QA {}, hỏi lại dòng thiếu → dịch dòng 3."""
    calls = []

    def fake(system, user, temperature=0.1):
        calls.append(user)
        if 'CÂU THOẠI THEO THỨ TỰ' in user or 'KỊCH BẢN:' in user:  # Phase 3 hồ sơ / Phase 4 QA
            return {}
        if 'CHỈ dịch các dòng' in user:
            return {"3": "Đi thôi"}
        return first_window_reply
    return fake, calls


def _run_contextual(fake, segs_texts, norm, texts):
    segs = [{'text': t, 'start_frame': i * 10, 'end_frame': i * 10 + 5} for i, t in enumerate(segs_texts)]
    orig_ask = tp.ask_json
    tmp = tempfile.mkdtemp()
    try:
        tp.ask_json = fake
        return tp.translate_contextual(segs, norm, texts, {}, os.path.join(tmp, "profiles.json"))
    finally:
        tp.ask_json = orig_ask
        shutil.rmtree(tmp, ignore_errors=True)


def test_translate_contextual_retries_missing_lines_and_borrows_variants():
    fake, calls = _fake_llm({"1": "Xin chào", "2": "Ta muốn tu tiên!"})  # lần đầu bỏ sót dòng 3
    norm = {'你好': '你好', '我要去修仙！': '我要去修仙！', '走吧': '走吧', '我要去修仙': '我要去修仙'}
    out = _run_contextual(fake, ['你好', '我要去修仙！', '走吧'], norm, ['你好', '我要去修仙！', '走吧', '我要去修仙'])
    assert out['走吧'] == 'Đi thôi', "Dòng bị bỏ sót phải được hỏi lại"
    assert sum('CHỈ dịch các dòng: 3' in u for u in calls) == 1
    assert out['我要去修仙'] == 'Ta muốn tu tiên!', "Biến thể OCR mượn bản dịch dòng giống nhất"


def test_translate_contextual_raises_when_lines_still_missing():
    fake, _ = _fake_llm({"1": "Xin chào"})  # hỏi lại cũng chỉ ra dòng 3, dòng 2 vẫn thiếu
    try:
        _run_contextual(fake, ['你好', '走吧', '好的'], {'你好': '你好', '走吧': '走吧', '好的': '好的'},
                        ['你好', '走吧', '好的'])
        raise AssertionError("Phải ném LLMError khi vẫn thiếu dòng")
    except tp.LLMError:
        pass


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
