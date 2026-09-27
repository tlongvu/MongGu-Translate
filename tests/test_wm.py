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
    assert wm_track.find_moving_wm(moving("粥咪咪", 150))["text"] == "粥咪咪", "watermark 3 ký tự (kênh 粥咪咪) phải nhận được"
    assert wm_track.find_moving_wm(moving("粥咪", 150)) is None, "chữ < 3 ký tự"
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


def test_track_ignores_static_lookalike_while_watermark_is_faint():
    # Lỗi thật trên newvideo: một vật đứng yên khớp mẫu (chữ tĩnh) được nhận làm mốc vì "tự nhất quán"
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "synth.mp4")
        centers, size = wm_synth.make_video(path, distractor=(wm_synth.DISTRACT, wm_synth.DISTRACT_AT))
        res, _ = wm_track.track_watermark(path, wm_synth.fake_ocr(centers, size, "synth.mp4"))
        xy = np.array(res["xy"], float)
        frames = np.arange(res["start"], res["start"] + len(xy))
        err = np.hypot(*(xy - centers[frames]).T)
        assert err.max() <= 8, f"bám nhảy sang vật đứng yên: lệch {err.max():.0f}px ở {int((err > 8).sum())} frame"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_hermite_follows_curve_and_is_linear_on_chord():
    pa, pb = np.array([0.0, 0.0]), np.array([100.0, 100.0])
    # đường y = x²/100 (x = frame): vận tốc đầu (1, 0), cuối (1, 2) → Hermite bậc 3 tái tạo đúng parabol
    assert np.allclose(wm_track._hermite(pa, np.array([1.0, 0.0]), pb, np.array([1.0, 2.0]), 0, 100, 50), [50, 25])
    chord = (pb - pa) / 100
    assert np.allclose(wm_track._hermite(pa, chord, pb, chord, 0, 100, 30), [30, 30]), "vận tốc = dây cung → thẳng"


def test_track_vetoes_reachable_lookalike_contradicted_by_ocr():
    # Vật đứng yên xuất hiện giữa một đoạn mờ dài: với tới được từ quỹ đạo, nối vào thì chuỗi thêm đỉnh.
    # Chỉ cần một lần OCR đọc đúng chữ watermark ở chỗ khác trong lúc đó là loại được nó.
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "synth.mp4")
        faint, dis_frames, dis_at = range(40, 200), range(100, 131), (300, 200)
        centers, size = wm_synth.make_video(path, distractor=(dis_frames, dis_at), faint=faint)
        ocr = wm_synth.fake_ocr(centers, size, "synth.mp4", faint=faint, extra=[115])
        res, _ = wm_track.track_watermark(path, ocr)
        xy = np.array(res["xy"], float)
        frames = np.arange(res["start"], res["start"] + len(xy))
        dist = np.hypot(*(xy[np.isin(frames, list(dis_frames))] - np.array(dis_at, float)).T)
        assert dist.min() > 30, f"bám dính vào vật đứng yên (cách {dist.min():.0f}px)"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_track_watermark_hugging_frame_edge():
    # Kênh 粥咪咪: watermark chạy sát mép khung (hộp OCR chạm mép) → vẫn phải dựng được mẫu và bám đúng
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "synth.mp4")
        gh = wm_synth.glyph()[0].shape[0]
        centers, size = wm_synth.make_video(path, track=lambda t: np.array([60 + 1.5 * t, gh / 2]))
        res, _ = wm_track.track_watermark(path, wm_synth.fake_ocr(centers, size, "synth.mp4"))
        assert res["found"], res.get("reason")
        xy = np.array(res["xy"], float)
        frames = np.arange(res["start"], res["start"] + len(xy))
        err = np.hypot(*(xy - centers[frames]).T)
        assert err.max() <= 3, f"lệch {err.max():.1f}px khi chạy sát mép"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_track_watermark_partly_outside_frame():
    # Kênh 粥咪咪 ở mép dưới: chữ lòi ra ngoài khung vài px (hộp OCR bị cắt) → mẫu phải được lòi ra mép khi dò
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "synth.mp4")
        gh = wm_synth.glyph()[0].shape[0]
        y_out = wm_synth.H - gh / 2 + 5                          # đáy chữ lòi ra ngoài khung 5 px

        def track(t):                                            # chéo xuống tới mép dưới rồi chạy dọc mép sang trái
            if t < 150:
                return np.array([60 + 2.0 * t, 60 + (y_out - 60) * t / 150])
            return np.array([360 - 1.5 * (t - 150), y_out])
        centers, size = wm_synth.make_video(path, track=track)
        res, _ = wm_track.track_watermark(path, wm_synth.fake_ocr(centers, size, "synth.mp4"))
        assert res["found"], res.get("reason")
        xy = np.array(res["xy"], float)
        frames = np.arange(res["start"], res["start"] + len(xy))
        edge = frames >= 160
        assert edge.sum() >= 70, f"phải bám tiếp dọc mép dưới (chỉ có {edge.sum()} frame)"
        err = np.hypot(*(xy[edge] - centers[frames[edge]]).T)
        assert err.max() <= 4, f"lệch {err.max():.1f}px dọc mép dưới"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_fill_gaps_redetects_and_follows_turn_between_distant_anchors():
    # Nhánh dò lại + đổi hướng của fill_gaps: bỏ các mốc quanh chỗ rẽ (frame 150, chữ còn rõ) → phải tự dò lại để bám
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "synth.mp4")
        faint = range(112, 140)                        # một phần khoảng hụt mờ → phải nội suy lại sau chỗ đổi hướng
        centers, size = wm_synth.make_video(path, faint=faint)
        tpl, tpl_small, scale, _ = wm_track.build_template(
            path, wm_track.find_moving_wm(wm_synth.fake_ocr(centers, size, "synth.mp4", faint=faint)))
        pos, score = wm_track.detect_all(path, tpl_small, scale, wm_synth.N)
        anchors = [a for a in wm_track.pick_anchors(pos, score, wm_synth.FPS) if not 110 <= a <= 200]
        xy, ev = wm_track.fill_gaps(path, anchors, pos, tpl, wm_synth.FPS)
        frames = np.arange(anchors[0], anchors[0] + len(xy))
        gap = (frames > 110) & (frames < 200)
        err = np.hypot(*(xy - centers[frames]).T)
        assert ev[gap].mean() > 0.5, "phải dò lại được watermark (còn rõ) trong khoảng không có mốc"
        assert not ev[gap].all(), "đoạn mờ trong khoảng phải được nội suy (không có bằng chứng)"
        assert err[gap].max() <= 3, f"lệch {err[gap].max():.1f}px quanh chỗ rẽ (đi thẳng sẽ lệch ~69px)"
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
        utils.write_json_atomic(job.wm_track, {"found": True, "total_frames": 100})   # sửa tay hỏng: thiếu start/xy
        assert s.load_wm_track(job, 100) == ({}, None), "JSON sai dạng → cảnh báo, không crash render"
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
