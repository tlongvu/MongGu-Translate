"""
Test tách lớp watermark + xóa bằng đảo pha trộn (wm_matte.py) — dữ liệu tổng hợp, không gọi API.
Chạy:  python -m pytest tests/test_wm_matte.py -q
"""

import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils  # noqa: E402
import wm_matte  # noqa: E402
import wm_track  # noqa: E402


def test_invert_recovers_background_when_alpha_and_color_known():
    rng = np.random.default_rng(0)
    inside = np.zeros((16, 24), np.uint8); inside[3:13, 4:20] = 1
    alpha = (inside * 0.3).astype(np.float32)
    color = np.array([240, 230, 220], np.float32)
    aW = alpha[..., None] * color
    bg = rng.uniform(0, 255, (16, 24, 3)).astype(np.float32)
    img = np.clip(np.round((1 - alpha[..., None]) * bg + aW), 0, 255).astype(np.uint8)
    out = wm_matte.invert(img, alpha, aW, np.zeros(inside.shape, bool), inside)
    assert out.dtype == np.uint8 and out.shape == img.shape
    assert np.abs(out.astype(np.float32) - bg)[inside > 0].max() <= 1.5, "đảo pha trộn phải ra đúng nền"
    assert (out[inside == 0] == img[inside == 0]).all(), "ngoài vùng watermark giữ nguyên"


def test_fit_recovers_alpha_and_color_from_true_backgrounds():
    rng = np.random.default_rng(0)
    h, w, n = 12, 20, 80
    inside = np.zeros((h, w), np.uint8); inside[3:9, 4:16] = 1
    a_true = np.zeros((h, w), np.float32); a_true[4:8, 5:15] = 0.3
    color = np.array([240, 230, 220], np.float32)
    B = rng.uniform(0, 255, (n, h, w, 3)).astype(np.float32)
    I = (1 - a_true[..., None]) * B + a_true[..., None] * color
    alpha, aW = wm_matte.fit(I, B, np.ones((n, h, w), np.uint8), inside)
    assert alpha.dtype == np.float32 and aW.shape == (h, w, 3)
    assert np.abs(alpha - a_true * inside).max() < 1e-3
    assert np.abs(aW - a_true[..., None] * color * inside[..., None]).max() < 0.5


def test_choose_needs_at_least_one_db_gain():
    assert wm_matte.choose(27.4, 26.5) == "inpaint"
    assert wm_matte.choose(27.5, 26.5) == "invert"
    assert wm_matte.choose(20.0, 26.5) == "inpaint"


def test_cut_marks_outside_frame_invalid():
    fr = np.arange(20 * 30 * 3, dtype=np.uint32).reshape(20, 30, 3).astype(np.uint8)
    p, v = wm_matte.cut(fr, -2, 15, 8, 6)
    assert p.shape == (8, 6, 3) and v.shape == (8, 6)
    assert v[:5, 2:].all() and not v[:, :2].any() and not v[5:].any()
    assert (p[0, 2] == fr[15, 0]).all() and (p[0, 0] == fr[15, 0]).all(), "ngoài khung lấy mép gần nhất"


def test_geometry_inside_is_padded_mask_and_ring_surrounds_it():
    mask = np.zeros((10, 16), np.uint8); mask[2:8, 3:13] = 255
    inside, ring = wm_matte.geometry(mask)
    assert inside.shape == (10 + 2 * wm_matte.PAD, 16 + 2 * wm_matte.PAD)
    assert inside.sum() == (mask > 0).sum() and not (inside & ring).any()
    assert ring[wm_matte.PAD + 2 - 1, wm_matte.PAD + 5] == 1, "vành sát ngay ngoài mask"


# ── Video tổng hợp: nền tĩnh từng đoạn 200 frame, watermark bán trong suốt trôi ~1 px/frame ──
# Ghi KHÔNG NÉN (chuỗi ảnh PNG, cv2.VideoCapture đọc tuần tự được): test chỉ kiểm thuật toán; ảnh hưởng của codec
# do PSNR trên video thật chấm. Nén mp4v làm α lệch ~0,048 (nhiễu codec co hệ số hồi quy) nên không dùng ở đây.
_W, _H, _N = 320, 240, 600


def _glyph():
    """(α, màu BGR) của watermark 'WM@12': chữ trắng viền tối, độ đậm đều 0,3 trên cả chữ + viền."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize("WM@12", font, 0.9, 2)
    h, w = th + base + 16, tw + 16
    outline, fill = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    cv2.putText(outline, "WM@12", (8, th + 8), font, 0.9, 255, 6)
    cv2.putText(fill, "WM@12", (8, th + 8), font, 0.9, 255, 2)
    alpha = (outline > 0).astype(np.float32) * 0.3
    color = np.where((fill > 0)[..., None], 245.0, 30.0) * np.ones(3, np.float32)
    return alpha, color.astype(np.float32)


def _truth(t):
    tri = lambda u: 1 - abs((u % 2) - 1)          # sóng tam giác 0..1
    return np.array([90 + 140 * tri(t / 133), 70 + 100 * tri(t / 97)])


def _make_video(path, static=True):
    """path: mẫu tên file kiểu printf (vd .../f_%04d.png), frame từ 0.
    static: 3 nền tĩnh, mỗi nền 200 frame; không thì nền trôi 2 px/frame (không có cảnh tĩnh).
    → (track như wm_track.json, mask như wm_mask.png, α thật của watermark)."""
    rng = np.random.default_rng(3)
    texs = []
    for _ in range(3):
        t = cv2.GaussianBlur(rng.integers(0, 256, (_H + 20, _W + 2 * _N + 20, 3)).astype(np.float32), (0, 0), 8)
        texs.append(cv2.normalize(t, None, 50, 200, cv2.NORM_MINMAX))
    alpha, color = _glyph()
    gh, gw = alpha.shape
    xy = []
    for t in range(_N):
        off = 0 if static else 2 * t
        bg = texs[(t // 200) % 3][10:10 + _H, off:off + _W].copy()
        cx, cy = _truth(t)
        x0, y0 = int(round(cx - gw / 2)), int(round(cy - gh / 2))
        roi = bg[y0:y0 + gh, x0:x0 + gw]
        bg[y0:y0 + gh, x0:x0 + gw] = (1 - alpha[..., None]) * roi + alpha[..., None] * color
        utils.imwrite_any(path % t, np.clip(np.round(bg), 0, 255).astype(np.uint8))
        xy.append([x0 + gw // 2, y0 + gh // 2])
    mask = cv2.dilate((alpha > 0).astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
    mask = cv2.copyMakeBorder(mask, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)   # như wm_mask: mẫu + lề 4 px
    track = {"found": True, "total_frames": _N, "fps": 25.0, "start": 0, "xy": xy, "evidence": "1" * _N,
             "size": [gw, gh]}
    return track, mask, alpha


@pytest.fixture(scope="module")
def static_video():
    """Chuỗi PNG nền tĩnh ghi 1 lần cho các test dùng chung (chỉ đọc). → (mẫu tên file, track, mask, α thật)."""
    d = tempfile.mkdtemp(prefix="matte ")      # ASCII: VideoCapture không mở được chuỗi ảnh trong thư mục Unicode
    try:
        path = os.path.join(d, "f_%04d.png")
        yield (path, *_make_video(path, static=True))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _alpha_error(rec, track, mask, a_true):
    """Sai lệch TB của α ước lượng so với α thật, đặt vào ô như lúc frame 0 của track CHƯA nhiễu (chỉ trong inside)."""
    inside, _ = wm_matte.geometry(mask)
    gh, gw = a_true.shape
    X0, Y0 = wm_matte.corner(track["xy"][0], mask.shape)
    g = _truth(0)
    ox, oy = int(round(g[0] - gw / 2)) - X0, int(round(g[1] - gh / 2)) - Y0
    truth = np.zeros(inside.shape, np.float32)
    truth[oy:oy + gh, ox:ox + gw] = a_true
    return float(np.abs(rec["alpha"] - truth)[inside > 0].mean())


def test_build_on_static_scenes_recovers_alpha_and_chooses_invert(static_video):
    path, track, mask, a_true = static_video
    rec = wm_matte.build(path, track, mask)
    assert rec["pairs"] >= wm_matte.MIN_PAIRS, rec["reason"]
    assert rec["method"] == "invert", rec["reason"]
    assert rec["psnr_invert"] > rec["psnr_inpaint"]
    err = _alpha_error(rec, track, mask, a_true)
    assert err < 0.03, f"α ước lượng lệch TB {err:.3f}"


def test_build_with_template_realigns_jittered_track(static_video):
    """Đường production: track lệch ±2 px từng frame, mẫu gradient căn lại ±REFINE_R → vẫn ra đúng α."""
    path, track, mask, a_true = static_video
    alpha, color = _glyph()
    tpl = wm_track._grad(cv2.cvtColor(np.round(alpha[..., None] * color).astype(np.uint8), cv2.COLOR_BGR2GRAY))
    jit = np.random.default_rng(5).integers(-2, 3, (len(track["xy"]), 2))
    jittered = dict(track, xy=[[x + int(dx), y + int(dy)] for (x, y), (dx, dy) in zip(track["xy"], jit)])
    rec = wm_matte.build(path, jittered, mask, tpl)
    assert rec["pairs"] >= wm_matte.MIN_PAIRS, rec["reason"]
    assert rec["method"] == "invert", rec["reason"]
    err = _alpha_error(rec, track, mask, a_true)
    assert err < 0.03, f"α ước lượng lệch TB {err:.3f}"


def test_build_without_static_scenes_falls_back_to_inpaint():
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "f_%04d.png")
        track, mask, _ = _make_video(path, static=False)
        rec = wm_matte.build(path, track, mask)
        assert rec["method"] == "inpaint" and rec["pairs"] < wm_matte.MIN_PAIRS
        assert "cảnh tĩnh" in rec["reason"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_plan_pairs_only_where_watermark_has_moved_away():
    track = {"start": 10, "evidence": "1" * 400, "xy": [[100 + i, 50] for i in range(400)]}   # trôi 1 px/frame
    plan = wm_matte.plan(track, (20, 120))                                                # rộng 120 px
    assert plan and all(len(js) <= wm_matte.TRIES for _, js in plan)
    for k, js in plan:
        for j in js:
            assert abs(track["xy"][j - 10][0] - track["xy"][k - 10][0]) >= 120
            assert 10 <= j < 410
    assert len(plan) <= wm_matte.MAX_CAND


def test_save_load_roundtrip_and_rejects_stale_or_broken():
    d = tempfile.mkdtemp(prefix="matte 修仙 ")
    try:
        mask = np.zeros((10, 16), np.uint8); mask[2:8, 3:13] = 255
        inside, _ = wm_matte.geometry(mask)
        tp = os.path.join(d, "wm_track.json")
        utils.write_json_atomic(tp, {"found": True, "start": 5, "xy": [[1, 2]] * 50})
        rec = {"method": "invert", "reason": "x", "pairs": 70, "psnr_invert": 30.0, "psnr_inpaint": 25.0,
               "track_start": 5, "track_len": 50, "mask_shape": [10, 16], "pad": wm_matte.PAD,
               "alpha": (inside * 0.3).astype(np.float32), "aW": np.zeros(inside.shape + (3,), np.float32),
               "edge": np.zeros(inside.shape, bool), "tpl": np.ones((4, 8), np.float32)}
        p = os.path.join(d, "wm_matte.npz")
        wm_matte.save(p, rec)
        m = wm_matte.load(p, tp, mask)
        assert m is not None and m["alpha"].dtype == np.float32 and abs(float(m["alpha"].max()) - 0.3) < 1e-3
        assert m["tpl"].shape == (4, 8) and (m["inside"] == inside).all()
        utils.write_json_atomic(tp, {"found": True, "start": 6, "xy": [[1, 2]] * 50})
        assert wm_matte.load(p, tp, mask) is None, "dò lại (track khác) → bỏ lớp cũ"
        utils.write_json_atomic(tp, {"found": True, "start": 5, "xy": [[1, 2]] * 50})
        wm_matte.save(p, dict(rec, alpha=np.zeros((5, 5), np.float32)))
        assert wm_matte.load(p, tp, mask) is None, "α lệch cỡ ô hiện tại → bỏ lớp cũ"
        wm_matte.save(p, dict(rec, pad=wm_matte.PAD + 1))
        assert wm_matte.load(p, tp, mask) is None, "PAD khác → bỏ lớp cũ"
        wm_matte.save(p, {"method": "inpaint", "reason": "thiếu cảnh tĩnh", "pairs": 3})
        assert wm_matte.load(p, tp, mask) is None, "method inpaint → render dùng inpaint"
        with open(p, "wb") as f:
            f.write(b"PK hong")
        assert wm_matte.load(p, tp, mask) is None, "file hỏng → không làm sập render"
        open(p, "wb").close()
        assert wm_matte.load(p, tp, mask) is None, "file rỗng 0 byte (EOFError) → không làm sập render"
        noisy = np.random.default_rng(0).uniform(0, 80, inside.shape + (3,)).astype(np.float32)   # để có thân nén lớn
        wm_matte.save(p, dict(rec, aW=noisy))
        assert wm_matte.load(p, tp, mask) is not None
        with open(p, "rb") as f:
            raw = f.read()
        for off in range(len(raw) // 2 - 40, len(raw) // 2 + 40):     # lật 4 byte giữa file: hỏng thân nén (CRC / zlib)
            bad = bytearray(raw)
            for i in range(4):
                bad[off + i] ^= 0xFF
            with open(p, "wb") as f:
                f.write(bad)
            assert wm_matte.load(p, tp, mask) is None, f"lật byte ở {off} → không làm sập render"
        assert wm_matte.load(os.path.join(d, "khong_co.npz"), tp, mask) is None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _matte_and_frame(center, bg_seed=1):
    """Lớp watermark α = 0,3 (màu 230) trên mask 10×16; frame 60×80 có watermark pha trộn tại center."""
    rng = np.random.default_rng(bg_seed)
    mask = np.zeros((10, 16), np.uint8); mask[2:8, 3:13] = 255
    inside, _ = wm_matte.geometry(mask)
    ph, pw = inside.shape
    alpha = (inside * 0.3).astype(np.float32)
    aW = alpha[..., None] * 230.0 * np.ones(3, np.float32)
    matte = {"alpha": alpha, "aW": aW, "edge": np.zeros(inside.shape, bool), "tpl": None, "inside": inside}
    bg = rng.integers(0, 256, (60, 80, 3)).astype(np.float32)
    x0, y0 = wm_matte.corner(center, mask.shape)
    ys, xs = slice(max(0, y0), min(60, y0 + ph)), slice(max(0, x0), min(80, x0 + pw))
    a = alpha[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0][..., None]
    frame = bg.copy()
    frame[ys, xs] = (1 - a) * bg[ys, xs] + a * 230.0
    region = np.zeros((60, 80), bool); region[ys, xs] = True
    ins = np.zeros((60, 80), bool)
    ins[ys, xs] = inside[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] > 0
    return mask, matte, bg, np.clip(np.round(frame), 0, 255).astype(np.uint8), region, ins


def test_apply_inverts_inside_frame_and_leaves_rest_untouched():
    mask, matte, bg, frame, region, ins = _matte_and_frame((75, 55))     # ô lòi ra mép phải + dưới
    before = frame.copy()
    out = wm_matte.apply(frame, (75, 55), matte, mask.shape)
    assert out.shape == (60, 80, 3)
    assert (out[~region] == before[~region]).all(), "ngoài ô không đổi"
    assert np.abs(out[ins].astype(np.float32) - bg[ins]).max() <= 2.0, "phần ô trong khung: khôi phục nền"


def test_remove_wm_uses_matte_when_given():
    import script as s
    mask, matte, bg, frame, _, ins = _matte_and_frame((40, 30))
    a = s.remove_wm(frame.copy(), (40, 30), mask, matte=matte)
    b = wm_matte.apply(frame.copy(), (40, 30), matte, mask.shape)
    assert (a == b).all()
    c = s.remove_wm(frame.copy(), (40, 30), mask)
    assert not (c == a).all(), "không có lớp → inpaint như cũ"
    err = lambda x: np.abs(x.astype(np.float32) - bg)[ins].mean()
    assert err(a) < err(c), "có lớp → gần nền thật hơn inpaint"


def test_job_has_wm_matte_path():
    import jobs
    d = tempfile.mkdtemp(prefix="job 修仙 ")
    orig = jobs.JOBS_DIR
    try:
        jobs.JOBS_DIR = os.path.join(d, "jobs")
        v = os.path.join(d, "v.mp4")
        with open(v, "wb") as f:
            f.write(os.urandom(64))
        job = jobs.open_job(v, create=True)
        assert job.wm_matte == os.path.join(job.dir, "wm_matte.npz")
    finally:
        jobs.JOBS_DIR = orig
        shutil.rmtree(d, ignore_errors=True)


def test_apply_realigns_template_within_refine_radius_on_odd_mask():
    """Mask cỡ LẺ + mẫu: lệch track ±REFINE_R theo cả x và y đều phải căn lại đúng ô (cửa sổ khớp đối xứng như collect)."""
    alpha, color = _glyph()
    h, w = (n - (n % 2 == 0) for n in alpha.shape)                    # cắt về cỡ lẻ
    alpha, color = alpha[:h, :w], color[:h, :w]
    mask = cv2.dilate((alpha > 0).astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
    assert mask.shape[0] % 2 == 1 and mask.shape[1] % 2 == 1
    inside, _ = wm_matte.geometry(mask)
    P, ph, pw = wm_matte.PAD, *inside.shape
    a_t = cv2.copyMakeBorder(alpha, P, P, P, P, cv2.BORDER_CONSTANT, value=0)
    aW = a_t[..., None] * cv2.copyMakeBorder(color, P, P, P, P, cv2.BORDER_CONSTANT, value=0)
    tpl = wm_track._grad(cv2.cvtColor(np.round(alpha[..., None] * color).astype(np.uint8), cv2.COLOR_BGR2GRAY))
    matte = {"alpha": a_t, "aW": aW, "edge": np.zeros(inside.shape, bool), "tpl": tpl, "inside": inside}

    rng = np.random.default_rng(7)
    bg = cv2.normalize(cv2.GaussianBlur(rng.integers(0, 256, (120, 200, 3)).astype(np.float32), (0, 0), 8),
                       None, 50, 200, cv2.NORM_MINMAX)
    true_c = (100, 60)
    x0, y0 = wm_matte.corner(true_c, mask.shape)
    frame = bg.copy()
    frame[y0:y0 + ph, x0:x0 + pw] = (1 - a_t[..., None]) * bg[y0:y0 + ph, x0:x0 + pw] + aW
    frame = np.clip(np.round(frame), 0, 255).astype(np.uint8)

    ref = wm_matte.apply(frame.copy(), true_c, matte, mask.shape)
    area = (slice(y0, y0 + ph), slice(x0, x0 + pw))
    assert np.abs(ref[area].astype(np.float32) - bg[area]).max() <= 2.0, "đúng tâm → khôi phục nền"
    R = wm_matte.REFINE_R
    for dy in (-R, 0, R):
        for dx in (-R, 0, R):
            out = wm_matte.apply(frame.copy(), (true_c[0] + dx, true_c[1] + dy), matte, mask.shape)
            d = np.abs(out[area].astype(np.float32) - ref[area]).max()
            assert d <= 2.0, f"track lệch ({dx:+d}, {dy:+d}) px không căn lại đúng ô: sai {d}"
