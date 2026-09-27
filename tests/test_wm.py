"""
Test xóa watermark chạy (wm_track.py + phần xóa trong render) — video tổng hợp nhỏ, không gọi API.
Chạy:  python -m pytest tests/   hoặc   python tests/test_wm.py
"""

import os
import shutil
import sys
import tempfile
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wm_track  # noqa: E402
import wm_synth  # noqa: E402


def _det(text, cx, cy, w, h, conf=0.9):
    return {"text": text, "confidence": conf,
            "box": [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                    [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]]}


def _ocr(dets_by_frame, n=3000, fps=25.0, w=1440, h=1080):
    return {"fps": fps, "total_frames": n, "width": w, "height": h,
            "frame_detections": {str(f): d for f, d in dets_by_frame.items()}}


def test_find_moving_wm_picks_only_the_moving_watermark():
    fd = defaultdict(list)
    for f in range(0, 3000, 20):                        # watermark chạy: suốt 2 phút, trôi đều
        fd[f].append(_det("原创@哈加密", 200 + f * 0.3, 150 + f * 0.2, 190, 46))
    for f in range(0, 3000, 10):                        # watermark tĩnh
        fd[f].append(_det("原文《桃桃师妹》", 1200, 120, 300, 40))
    for k in range(30):                                 # thoại: mỗi câu 2 s ở giữa dưới
        for f in range(k * 100, k * 100 + 50, 5):
            fd[f].append(_det(f"第{k}句台词内容", 720, 950, 400, 50))
    for f in list(range(100, 400, 5)) + list(range(2000, 2300, 5)):   # nhãn nhân vật: 2 cảnh, có di chuyển
        fd[f].append(_det("执事弟子", 300 + (f % 300) * 2, 600, 120, 40))
    for f in range(40, 3000, 60):                       # OCR đọc thiếu một mảnh watermark
        fd[f].append(_det("原创@哈加", 210 + f * 0.3, 150 + f * 0.2, 160, 46))
    wm = wm_track.find_moving_wm(_ocr(fd))
    assert wm and wm["text"] == "原创@哈加密", wm and wm["text"]
    assert len(wm["reads"]) == 150
    no_wm = {f: [d for d in ds if not d["text"].startswith("原创@")] for f, ds in fd.items()}
    assert wm_track.find_moving_wm(_ocr(no_wm)) is None, "nhãn / watermark tĩnh / thoại không được nhận là watermark chạy"


def test_find_moving_wm_each_threshold_excludes_on_its_own():
    """Mỗi ngưỡng tự loại được ứng viên chỉ trượt đúng ngưỡng đó (mọi điều kiện khác đều đạt)."""
    def moving(text, n_reads, width=lambda f: 190):
        fd = defaultdict(list)
        for f in np.linspace(0, 2999, n_reads).astype(int):
            fd[int(f)].append(_det(text, 200 + f * 0.3, 150 + f * 0.2, width(f), 46))
        return _ocr(fd)
    assert wm_track.find_moving_wm(moving("原创@哈加密", 10))["text"] == "原创@哈加密", "vừa đủ 10 lần đọc"
    assert wm_track.find_moving_wm(moving("原创@哈加密", 9)) is None, "< 10 lần đọc"
    assert wm_track.find_moving_wm(moving("哈加密", 150)) is None, "chữ < 4 ký tự"
    assert wm_track.find_moving_wm(moving("原创@哈加密", 150, width=lambda f: 100 + (f % 7) * 40)) is None, \
        "cỡ box dao động > 15%"
    early = defaultdict(list)
    for f in range(0, 500, 3):
        early[f].append(_det("原创@哈加密", 200 + f * 2, 150 + f, 190, 46))
    assert wm_track.find_moving_wm(_ocr(early)) is None, "chỉ có mặt < 40% số đoạn 10 s"
    static = defaultdict(list)
    for f in range(0, 3000, 20):
        static[f].append(_det("原创@哈加密", 700, 500, 190, 46))
    assert wm_track.find_moving_wm(_ocr(static)) is None, "đứng yên (tản < 5% đường chéo)"


def test_find_moving_wm_reads_are_frame_ordered_tuples():
    fd = {}
    for f in reversed(range(0, 3000, 20)):      # chèn ngược thứ tự thời gian
        fd[f] = [_det("原创@哈加密", 200 + f // 3, 150 + f // 6, 190, 46)]
    reads = wm_track.find_moving_wm(_ocr(fd))["reads"]
    assert [r[0] for r in reads] == list(range(0, 3000, 20)), "reads phải sắp theo frame"
    assert reads[1] == (20, 206.0, 153.0, 190.0, 46.0, 0.9), "tuple (frame, cx, cy, w, h, conf)"


def test_track_follows_faint_turning_watermark_and_ignores_flat_region():
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "synth.mp4")
        centers, size = wm_synth.make_video(path)
        res, mask = wm_track.track_watermark(path, wm_synth.fake_ocr(centers, size, "synth.mp4"))
        assert res["found"] and res["text"] == "WM@12", res
        xy = np.array(res["xy"], float)
        ev = np.array([c == "1" for c in res["evidence"]])
        frames = np.arange(res["start"], res["start"] + len(xy))
        err = np.hypot(*(xy - centers[frames]).T)
        assert len(xy) >= 0.9 * wm_synth.N, "phải phủ gần hết video"
        assert (err[ev] <= 3).mean() >= 0.95, f"frame có bằng chứng lệch ≤ 3px: {(err[ev] <= 3).mean():.0%}"
        assert err.max() <= 8, f"lệch lớn nhất {err.max():.1f}px (kể cả đoạn gần như vô hình)"
        assert mask.dtype == np.uint8 and 0.2 < (mask > 0).mean() < 1.0, "mask hình nét chữ, không phải cả hộp"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_track_reports_reason_when_no_watermark():
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "synth.mp4")
        centers, size = wm_synth.make_video(path)
        data = wm_synth.fake_ocr(centers, size, "synth.mp4")
        for dets in data["frame_detections"].values():
            dets[:] = [x for x in dets if x["text"] != "WM@12"]
        res, mask = wm_track.track_watermark(path, data)
        assert res["found"] is False and "không thấy" in res["reason"] and mask is None
    finally:
        shutil.rmtree(d, ignore_errors=True)


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
