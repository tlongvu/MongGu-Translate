"""
Test xóa watermark chạy (wm_track.py + phần xóa trong render) — video tổng hợp nhỏ, không gọi API.
Chạy:  python -m pytest tests/   hoặc   python tests/test_wm.py
"""

import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wm_track  # noqa: E402
import wm_synth  # noqa: E402

import jobs  # noqa: E402
import utils  # noqa: E402
import script as s  # noqa: E402


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


def test_run_writes_track_mask_preview_and_can_be_disabled():
    d = tempfile.mkdtemp(prefix="wm 修仙 ")
    orig = (jobs.JOBS_DIR, wm_track.WM_TRACK)
    try:
        jobs.JOBS_DIR = os.path.join(d, "jobs")
        video = os.path.join(d, "clip wm.mp4")
        centers, size = wm_synth.make_video(video)
        job = jobs.open_job(video, create=True)
        utils.write_json_atomic(job.ocr_data, wm_synth.fake_ocr(centers, size, "clip wm.mp4"))
        assert wm_track.run(video) == 0
        t = json.load(open(job.wm_track, encoding="utf-8"))
        assert t["found"] and t["total_frames"] == wm_synth.N and len(t["xy"]) == len(t["evidence"])
        assert utils.imread_gray(job.wm_mask) is not None, "mask phải ghi được (đường dẫn Unicode)"
        assert utils.imread_gray(job.wm_preview) is not None
        wm_track.WM_TRACK = False
        assert wm_track.run(video) == 0
        assert json.load(open(job.wm_track, encoding="utf-8"))["found"] is False
        os.remove(job.ocr_data)
        assert wm_track.run(video) == 1, "chưa có OCR → mã 1"
    finally:
        jobs.JOBS_DIR, wm_track.WM_TRACK = orig
        shutil.rmtree(d, ignore_errors=True)


def test_remove_wm_touches_only_mask_and_clips_at_edges():
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, (100, 200, 3), np.uint8)
    mask = np.zeros((20, 40), np.uint8)
    mask[5:15, 10:30] = 255
    for center in [(100, 50), (195, 98), (-5, 3)]:          # giữa, lố mép phải-dưới, lố mép trái-trên
        out = s.remove_wm(frame.copy(), center, mask)
        m = np.zeros(frame.shape[:2], bool)
        x0, y0 = int(round(center[0] - 20)), int(round(center[1] - 10))
        for yy, xx in zip(*np.nonzero(mask)):
            if 0 <= y0 + yy < 100 and 0 <= x0 + xx < 200:
                m[y0 + yy, x0 + xx] = True
        assert (out[~m] == frame[~m]).all(), f"{center}: pixel ngoài mask bị đổi"
        assert m.any() and (out[m] != frame[m]).any(), f"{center}: vùng mask phải được xóa"
    assert (s.remove_wm(frame.copy(), (1000, 1000), mask) == frame).all(), "ngoài khung → không đổi gì"


def test_load_wm_track_ignores_missing_or_stale():
    d = tempfile.mkdtemp()
    try:
        job = SimpleNamespace(wm_track=os.path.join(d, "wm_track.json"), wm_mask=os.path.join(d, "wm_mask.png"))
        assert s.load_wm_track(job, 100) == ({}, None), "chưa chạy wm_track → không xóa gì"
        utils.write_json_atomic(job.wm_track, {"found": True, "total_frames": 100, "start": 10,
                                               "xy": [[5, 6], [7, 8]], "evidence": "10"})
        utils.imwrite_any(job.wm_mask, np.full((4, 6), 255, np.uint8))
        pos, mask = s.load_wm_track(job, 100)
        assert pos == {10: (5.0, 6.0), 11: (7.0, 8.0)} and mask.shape == (4, 6)
        assert s.load_wm_track(job, 999) == ({}, None), "OCR chạy lại (số frame khác) → bỏ qua track cũ"
        utils.write_json_atomic(job.wm_track, {"found": False, "reason": "x"})
        assert s.load_wm_track(job, 100) == ({}, None)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_optional_step_failure_does_not_stop_pipeline():
    import pipeline
    d = tempfile.mkdtemp()
    try:
        bad = os.path.join(d, "fail.py")
        with open(bad, "w", encoding="utf-8") as f:
            f.write("import sys; sys.exit(3)")
        pipeline.run_step("thử", bad, [], required=False)     # không thoát
        try:
            pipeline.run_step("thử", bad, [])
            raise AssertionError("bước bắt buộc lỗi phải dừng pipeline")
        except SystemExit as e:
            assert e.code == 3
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
