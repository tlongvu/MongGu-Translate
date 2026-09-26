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

def test_translation_cache_roundtrip():
    orig = tp.TRANSLATION_CACHE_PATH
    tmp = os.path.join(tempfile.gettempdir(), "_monggu_test_cache.json")
    try:
        tp.TRANSLATION_CACHE_PATH = tmp
        if os.path.exists(tmp):
            os.remove(tmp)
        assert tp.load_translation_cache() == {}, "Cache rỗng khi chưa có file"
        tp.save_translation_cache({"你好": "Xin chào"})
        assert tp.load_translation_cache().get("你好") == "Xin chào"
    finally:
        tp.TRANSLATION_CACHE_PATH = orig
        if os.path.exists(tmp):
            os.remove(tmp)


def test_run_translation_fails_without_input():
    orig = tp.load_ocr_data
    try:
        tp.load_ocr_data = lambda: None                   # thiếu ocr_data.json
        assert tp.run_translation("google") is False
        tp.load_ocr_data = lambda: {"unique_texts": []}   # OCR không ra chữ nào
        assert tp.run_translation("google") is False
    finally:
        tp.load_ocr_data = orig


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
