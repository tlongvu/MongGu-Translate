# Tách lớp watermark + xóa bằng đảo pha trộn — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xóa watermark chạy bằng cách tách lớp watermark (độ đậm α + màu) từ nhiều frame rồi đảo pha trộn, khôi phục nền thật thay vì inpaint làm nhòe — tự chấm điểm và chỉ dùng khi thắng inpaint.

**Architecture:** Module mới `wm_matte.py` (thuần numpy/OpenCV): chọn cặp (frame có watermark, frame khác khi watermark đã dời và cảnh tĩnh) từ vị trí đã bám → đọc video 1 lượt → hồi quy từng điểm ra α, α·màu → chấm PSNR trên cặp giữ riêng → `jobs/<video>/wm_matte.npz`. `wm_track.run()` gọi nó ngay sau khi bám (bước 2/4); `script.remove_wm()` dùng lớp đó khi có, không thì inpaint như cũ.

**Tech Stack:** Python 3.10, numpy, OpenCV (đã có sẵn). Test: pytest + video tổng hợp (mp4v), không gọi API.

**Spec:** `docs/superpowers/specs/2026-10-16-tach-lop-watermark-design.md`

## Global Constraints

- Không thêm thư viện; chỉ numpy / OpenCV có sẵn. Không mô hình AI.
- Hằng số (spec mục 4): `PAD = 10`, `RING = 12`, `REFINE_R = 3`, `MIN_SCORE = 0.5`, `GAPS = (75, -75, 125, -125, 200, -200, 300, -300)`, `MAX_CAND = 600`, `TRIES = 2`, `STATIC_DIFF = 3.0`, `MIN_PAIRS = 60`, `MAX_PAIRS = 400`, `ALPHA_MAX = 0.95`, `OPAQUE = 0.9`, `EDGE_FRAC = 0.5`, `MIN_GAIN_DB = 1.0`, `HOLDOUT = 0.2` (ít nhất 12 cặp chấm), chia cặp bằng `np.random.default_rng(0)`.
- **Không bao giờ tệ hơn hiện nay:** lỗi / thiếu cặp / không thắng inpaint ≥ 1 dB → `method = "inpaint"`, render như cũ. Lỗi tách lớp không làm hỏng bước 2/4, file `wm_matte.npz` hỏng / cũ không làm sập render.
- Sau bước 2/4 luôn có `wm_matte.npz` (kể cả khi không thấy watermark / tắt `WM_TRACK`, với `method = "inpaint"` + lý do).
- Test offline: `python -m pytest tests -q`. Không gọi API.
- Windows: đường dẫn Unicode → `utils.imwrite_any` / `utils.imread_gray`. File `.py` đang có dùng CRLF: sửa bằng công cụ Edit, **không** dùng `sed -i` (làm hỏng kiểu xuống dòng cả file).
- Commit: tiếng Việt, kết thúc bằng dòng `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (dùng 2 cờ `-m`). Nhánh `nang-cap`.

## File Structure

| File | Việc |
|---|---|
| `wm_matte.py` (mới) | Toàn bộ tách lớp: hình học ô, hồi quy, đảo pha trộn, chọn cặp, đọc video, chấm điểm, ghi / đọc `wm_matte.npz`, áp lớp khi render |
| `utils.py` | Thêm `inpaint_roi()` (chuyển từ `script._inpaint_roi`, để `wm_matte` dùng chung mà không import `script`) |
| `script.py` | Dùng `utils.inpaint_roi`; `remove_wm(..., matte=None)`; nạp lớp ở `main`, truyền xuống worker |
| `jobs.py` | Thêm `job.wm_matte` |
| `wm_track.py` | `run()` tách lớp sau khi bám; `run_matte_only()`; cờ `--matte-only` |
| `pipeline.py` | `--skip-existing`: có `wm_track.json` mà thiếu `wm_matte.npz` → chỉ tách lớp |
| `tests/test_wm_matte.py` (mới) | Test đơn vị + video tổng hợp cho `wm_matte` |
| `tests/test_wm.py` | Mở rộng test `run` + test pipeline bỏ qua |
| `README.md` | 1 gạch đầu dòng ở mục "Xóa watermark chạy" |

Import: `wm_matte` import `wm_track._local_match` ở đầu file; `wm_track` chỉ import `wm_matte` **bên trong hàm** (tránh vòng lặp import). `script` import cả hai (đã import `wm_track`).

---

### Task 1: Toán của lớp watermark (`wm_matte.py` phần lõi) + `utils.inpaint_roi`

**Files:**
- Create: `wm_matte.py`
- Modify: `utils.py` (thêm hàm cuối file), `script.py:322-335` (bỏ `_inpaint_roi`), `script.py:351`, `script.py:982`, `script.py:51` (import)
- Test: `tests/test_wm_matte.py` (mới)

**Interfaces:**
- Produces:
  - `utils.inpaint_roi(roi_frame: np.ndarray uint8 BGR, roi_mask: np.ndarray uint8) -> np.ndarray` — đúng hành vi cũ của `script._inpaint_roi` (theo `INPAINT_METHOD`, `INPAINT_RADIUS` trong config).
  - `wm_matte.geometry(mask) -> (inside, ring)`: 2 mảng uint8 0/1 cỡ `(mh + 2*PAD, mw + 2*PAD)`.
  - `wm_matte.cut(frame, x0, y0, ph, pw) -> (patch uint8 (ph, pw, 3), valid uint8 (ph, pw))`.
  - `wm_matte.corner(center, mask_shape) -> (x0, y0)` góc trái trên của ô.
  - `wm_matte.fit(I, B, V, inside) -> (alpha float32 (h, w), aW float32 (h, w, 3))`.
  - `wm_matte.edge_band(alpha, aW, inside) -> bool (h, w)`.
  - `wm_matte.invert(patch, alpha, aW, edge, inside) -> uint8 (h, w, 3)`.
  - `wm_matte.psnr(x, ref, m) -> float`; `wm_matte.choose(p_invert, p_inpaint) -> "invert" | "inpaint"`.

- [ ] **Step 1: Viết test (sẽ fail)** — tạo `tests/test_wm_matte.py`:

```python
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils  # noqa: E402
import wm_matte  # noqa: E402


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
```

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_wm_matte.py -q`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'wm_matte'`

- [ ] **Step 3: Chuyển `_inpaint_roi` sang `utils.inpaint_roi`**

Thêm vào cuối `utils.py`:

```python
def inpaint_roi(roi_frame, roi_mask):
    """Xóa chữ / watermark trong ROI theo INPAINT_METHOD (config): telea | ns | fill (blur nhanh).
    Dùng chung cho render (script.py) và chấm điểm lớp watermark (wm_matte.py)."""
    import cv2
    from config import INPAINT_METHOD, INPAINT_RADIUS
    method = INPAINT_METHOD.lower()
    if method == "fill":
        # Siêu nhanh: làm mờ mạnh ROI rồi chỉ dán phần mask (không tái dựng cấu trúc).
        blurred = cv2.GaussianBlur(roi_frame, (0, 0), sigmaX=max(INPAINT_RADIUS * 2, 6))
        out = roi_frame.copy()
        m = roi_mask > 0
        out[m] = blurred[m]
        return out
    if method == "ns":
        return cv2.inpaint(roi_frame, roi_mask, INPAINT_RADIUS, cv2.INPAINT_NS)
    return cv2.inpaint(roi_frame, roi_mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
```

Trong `script.py`: xóa hẳn hàm `_inpaint_roi` (dòng 322–335); đổi dòng import `from utils import imread_gray` thành `from utils import imread_gray, inpaint_roi`; đổi 2 chỗ gọi `_inpaint_roi(` thành `inpaint_roi(` (trong `remove_wm` và chỗ inpaint chữ OCR ~dòng 982). Với `INPAINT_RADIUS` và `INPAINT_METHOD`: nếu `script.py` còn dùng ở chỗ khác thì giữ import, không thì bỏ khỏi khối `from config import (...)` (kiểm bằng `grep -n "INPAINT_RADIUS\|INPAINT_METHOD" script.py`).

- [ ] **Step 4: Viết `wm_matte.py` phần lõi**

```python
"""
Tách lớp watermark (độ đậm α + màu) từ nhiều frame rồi xóa bằng đảo pha trộn
(spec docs/superpowers/specs/2026-10-16-tach-lop-watermark-design.md).

Mô hình: ảnh = (1 − α)·nền + α·màu ở từng điểm. Nền thật lấy từ frame khác — watermark đã dời đi, cảnh đứng yên
quanh đó — rồi hồi quy tuyến tính từng điểm ra α và α·màu. Xóa: nền = (ảnh − α·màu) / (1 − α), làm sạch dải mép
nét chữ. Tự chấm điểm trên cặp giữ riêng: chỉ dùng khi hơn inpaint ≥ MIN_GAIN_DB, không thì render giữ inpaint.
"""

import json
import os
import zipfile

import cv2
import numpy as np

from utils import inpaint_roi
from wm_track import _local_match

PAD = 10                  # ô = mask watermark nới 10 px mỗi phía
RING = 12                 # vành quanh mask để xét cảnh đứng yên
REFINE_R = 3              # căn lại vị trí ±3 px bằng khớp mẫu
MIN_SCORE = 0.5           # điểm khớp tại chỗ tối thiểu
GAPS = (75, -75, 125, -125, 200, -200, 300, -300)   # j = k ± 3–12 s (25 fps)
MAX_CAND = 600            # số frame k ứng viên tối đa (rải đều)
TRIES = 2                 # mỗi k giữ tối đa 2 frame j đã dời watermark (đọc 1 lượt: ~100 MB ô trong RAM)
STATIC_DIFF = 3.0         # sai khác TB trên vành < 3 mức xám → cảnh đứng yên
MIN_PAIRS, MAX_PAIRS = 60, 400
ALPHA_MAX = 0.95          # kẹp α khi đảo (không chia cho số quá nhỏ)
OPAQUE = 0.9              # α ≥ 0,9: gần đục → inpaint sau khi đảo
EDGE_FRAC = 0.5           # dải mép: gradient ảnh màu watermark ≥ 50% cực đại
MIN_GAIN_DB = 1.0         # đảo pha trộn phải hơn inpaint ≥ 1 dB mới được dùng
HOLDOUT = 0.2             # 20% cặp giữ riêng để chấm


def geometry(mask):
    """→ (inside, ring) cỡ ô (mask + 2·PAD), uint8 0/1: inside = vùng mask, ring = vành RING px quanh nó."""
    inside = cv2.copyMakeBorder((mask > 0).astype(np.uint8), PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0)
    k = 2 * RING + 1
    ring = cv2.dilate(inside, np.ones((k, k), np.uint8)) - inside
    return inside, ring


def corner(center, mask_shape):
    """Góc trái trên của ô khi tâm watermark ở center. Làm tròn .5 luôn lên (floor + 0,5): round() của Python làm
    tròn .5 về số chẵn → tâm nguyên với nửa bề rộng lẻ sẽ lệch ±1 px giữa frame chẵn / lẻ, làm nhòe hồi quy."""
    mh, mw = mask_shape
    return int(np.floor(center[0] - mw / 2 + 0.5)) - PAD, int(np.floor(center[1] - mh / 2 + 0.5)) - PAD


def cut(frame, x0, y0, ph, pw):
    """Ô (ph, pw) góc trái trên (x0, y0). Phần ngoài khung lấy điểm mép gần nhất và đánh dấu không hợp lệ.
    → (ô uint8, valid uint8 0/1)."""
    fh, fw = frame.shape[:2]
    ys, xs = np.arange(y0, y0 + ph), np.arange(x0, x0 + pw)
    patch = frame[np.ix_(np.clip(ys, 0, fh - 1), np.clip(xs, 0, fw - 1))]
    valid = (((ys >= 0) & (ys < fh))[:, None] & ((xs >= 0) & (xs < fw))[None, :]).astype(np.uint8)
    return patch, valid


def fit(I, B, V, inside):
    """Hồi quy từng điểm, từng kênh: ảnh = hệ_số·nền + chặn (trọng số valid).
    α = 1 − hệ_số (TB 3 kênh, kẹp [0, ALPHA_MAX], 0 ngoài inside); aW = α·màu = TB(ảnh) − (1 − α)·TB(nền).
    I, B: (n, h, w, 3); V: (n, h, w) 0/1."""
    I = np.asarray(I, np.float32)
    B = np.asarray(B, np.float32)
    M = np.asarray(V, np.float32)[..., None]
    nw = M.sum(0) + 1e-6
    bm, im = (B * M).sum(0) / nw, (I * M).sum(0) / nw
    cov = (M * (B - bm) * (I - im)).sum(0) / nw
    var = (M * (B - bm) ** 2).sum(0) / nw + 1e-6
    alpha = (np.clip(1 - (cov / var).mean(2), 0, ALPHA_MAX) * (inside > 0)).astype(np.float32)
    aW = ((im - (1 - alpha[..., None]) * bm) * (inside > 0)[..., None]).astype(np.float32)
    return alpha, aW


def edge_band(alpha, aW, inside):
    """Dải mép nét chữ: gradient (Sobel) của ảnh màu watermark (xám) ≥ EDGE_FRAC × cực đại, trong inside."""
    color = np.clip(aW / np.maximum(alpha, 1e-3)[..., None], 0, 255).astype(np.uint8)
    g = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    mag *= (inside > 0) & (alpha > 0.02)
    top = float(mag.max())
    return mag >= EDGE_FRAC * top if top > 0 else np.zeros(alpha.shape, bool)


def invert(patch, alpha, aW, edge, inside):
    """Đảo pha trộn trong inside → làm sạch dải mép (trung vị 5×5) → điểm gần đục (α ≥ OPAQUE) inpaint. → uint8."""
    p = patch.astype(np.float32)
    a = np.minimum(alpha, ALPHA_MAX)[..., None]
    out = np.where((inside > 0)[..., None], (p - aW) / (1 - a), p)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if edge.any():
        med = cv2.medianBlur(out, 5)
        out[edge] = med[edge]
    hard = ((alpha >= OPAQUE) & (inside > 0)).astype(np.uint8) * 255
    if hard.any():
        out = cv2.inpaint(out, hard, 3, cv2.INPAINT_TELEA)
    return out


def psnr(x, ref, m):
    """PSNR (dB) của x so với ref trên các điểm m > 0."""
    e = ((x.astype(np.float32) - ref.astype(np.float32)) ** 2)[m > 0].mean()
    return float(10 * np.log10(255.0 ** 2 / max(float(e), 1e-6)))


def choose(p_invert, p_inpaint):
    """Chỉ dùng đảo pha trộn khi hơn inpaint ≥ MIN_GAIN_DB."""
    return "invert" if p_invert >= p_inpaint + MIN_GAIN_DB else "inpaint"
```

- [ ] **Step 5: Chạy test, xác nhận qua**

Run: `python -m pytest tests/test_wm_matte.py -q`
Expected: `5 passed`

Run: `python -m pytest tests -q`
Expected: toàn bộ qua (đổi `_inpaint_roi` → `inpaint_roi` không đổi hành vi).

- [ ] **Step 6: Commit**

```bash
git add wm_matte.py utils.py script.py tests/test_wm_matte.py
git commit -m "feat(wm): wm_matte — hồi quy lớp watermark, đảo pha trộn, chấm PSNR; inpaint_roi chuyển sang utils" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Thu cặp nền thật + `build` + ghi / đọc `wm_matte.npz`

**Files:**
- Modify: `wm_matte.py` (thêm cuối file)
- Test: `tests/test_wm_matte.py` (thêm)

**Interfaces:**
- Consumes (Task 1): `geometry`, `corner`, `cut`, `fit`, `edge_band`, `invert`, `psnr`, `choose`, `utils.inpaint_roi`, `wm_track._local_match(gray, center, tpl, r) -> (np.array([x, y]), score)`.
- Produces:
  - `wm_matte.plan(track: dict, mask_shape) -> list[tuple[int, list[int]]]` — `track` là nội dung `wm_track.json` (`start`, `xy`, `evidence`).
  - `wm_matte.collect(video_path, track, mask, tpl=None) -> (I uint8 (n, ph, pw, 3), B uint8 (n, ph, pw, 3), V uint8 (n, ph, pw))`.
  - `wm_matte.build(video_path, track, mask, tpl=None) -> dict` với khóa: `method` ("invert"|"inpaint"), `reason` (str), `pairs` (int), `psnr_invert`, `psnr_inpaint` (float|None), `track_start`, `track_len`, `mask_shape` (list), và khi đủ cặp: `alpha`, `aW`, `edge`, `tpl`.
  - `wm_matte.save(path, rec) -> None`.
  - `wm_matte.load(path, track_path, mask) -> dict | None` với khóa `alpha` (float32), `aW` (float32), `edge` (bool), `tpl` (ndarray|None), `inside` (uint8).

- [ ] **Step 1: Viết test (sẽ fail)** — thêm vào `tests/test_wm_matte.py`:

```python
# ── Video tổng hợp: nền tĩnh từng đoạn 200 frame, watermark bán trong suốt trôi ~1 px/frame ──
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
    """static: 3 nền tĩnh, mỗi nền 200 frame; không thì nền trôi 2 px/frame (không có cảnh tĩnh).
    → (track như wm_track.json, mask như wm_mask.png, α thật của watermark)."""
    rng = np.random.default_rng(3)
    texs = []
    for _ in range(3):
        t = cv2.GaussianBlur(rng.integers(0, 256, (_H + 20, _W + 2 * _N + 20, 3)).astype(np.float32), (0, 0), 8)
        texs.append(cv2.normalize(t, None, 50, 200, cv2.NORM_MINMAX))
    alpha, color = _glyph()
    gh, gw = alpha.shape
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (_W, _H))
    xy = []
    for t in range(_N):
        off = 0 if static else 2 * t
        bg = texs[(t // 200) % 3][10:10 + _H, off:off + _W].copy()
        cx, cy = _truth(t)
        x0, y0 = int(round(cx - gw / 2)), int(round(cy - gh / 2))
        roi = bg[y0:y0 + gh, x0:x0 + gw]
        bg[y0:y0 + gh, x0:x0 + gw] = (1 - alpha[..., None]) * roi + alpha[..., None] * color
        vw.write(np.clip(bg, 0, 255).astype(np.uint8))
        xy.append([x0 + gw // 2, y0 + gh // 2])
    vw.release()
    mask = cv2.dilate((alpha > 0).astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
    mask = cv2.copyMakeBorder(mask, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)   # như wm_mask: mẫu + lề 4 px
    track = {"found": True, "total_frames": _N, "fps": 25.0, "start": 0, "xy": xy, "evidence": "1" * _N,
             "size": [gw, gh]}
    return track, mask, alpha


def test_build_on_static_scenes_recovers_alpha_and_chooses_invert():
    d = tempfile.mkdtemp(prefix="matte 修仙 ")
    try:
        path = os.path.join(d, "static.mp4")
        track, mask, a_true = _make_video(path, static=True)
        rec = wm_matte.build(path, track, mask)
        assert rec["pairs"] >= wm_matte.MIN_PAIRS, rec["reason"]
        assert rec["method"] == "invert", rec["reason"]
        assert rec["psnr_invert"] > rec["psnr_inpaint"]
        inside, _ = wm_matte.geometry(mask)
        gh, gw = a_true.shape
        X0, Y0 = wm_matte.corner(track["xy"][0], mask.shape)          # chỗ α thật nằm trong ô (từ frame 0)
        g = _truth(0)
        ox, oy = int(round(g[0] - gw / 2)) - X0, int(round(g[1] - gh / 2)) - Y0
        truth = np.zeros(inside.shape, np.float32)
        truth[oy:oy + gh, ox:ox + gw] = a_true
        err = np.abs(rec["alpha"] - truth)[inside > 0].mean()
        assert err < 0.03, f"α ước lượng lệch TB {err:.3f}"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_build_without_static_scenes_falls_back_to_inpaint():
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "moving.mp4")
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
               "track_start": 5, "track_len": 50, "mask_shape": [10, 16],
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
        wm_matte.save(p, {"method": "inpaint", "reason": "thiếu cảnh tĩnh", "pairs": 3})
        assert wm_matte.load(p, tp, mask) is None, "method inpaint → render dùng inpaint"
        with open(p, "wb") as f:
            f.write(b"PK hong")
        assert wm_matte.load(p, tp, mask) is None, "file hỏng → không làm sập render"
        assert wm_matte.load(os.path.join(d, "khong_co.npz"), tp, mask) is None
    finally:
        shutil.rmtree(d, ignore_errors=True)
```

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_wm_matte.py -q`
Expected: 4 test mới FAIL — `AttributeError: module 'wm_matte' has no attribute 'build'` (và `plan`, `save`).

- [ ] **Step 3: Viết phần thu cặp + build + save/load** — thêm vào cuối `wm_matte.py`:

```python
def plan(track, mask_shape):
    """Chọn trước các cặp chỉ từ vị trí đã bám (không đọc video): k = frame có bằng chứng, rải đều ≤ MAX_CAND;
    mỗi k giữ ≤ TRIES frame j = k + GAPS (theo thứ tự) mà watermark đã rời chỗ. → [(k, [j, …])]."""
    s0, xy, ev = int(track["start"]), track["xy"], track["evidence"]
    mh, mw = mask_shape
    ks = [s0 + i for i, c in enumerate(ev) if c == "1"]
    if len(ks) > MAX_CAND:
        ks = [ks[i] for i in np.linspace(0, len(ks) - 1, MAX_CAND).astype(int)]
    out = []
    for k in ks:
        xk, yk = xy[k - s0]
        js = []
        for g in GAPS:
            j = k + g
            if 0 <= j - s0 < len(xy):
                xj, yj = xy[j - s0]
                if abs(xj - xk) >= mw or abs(yj - yk) >= mh:
                    js.append(j)
                    if len(js) == TRIES:
                        break
        if js:
            out.append((k, js))
    return out


def collect(video_path, track, mask, tpl=None):
    """Đọc video MỘT lượt tuần tự, cắt ô ở các frame k (căn lại ±REFINE_R bằng tpl nếu có) và j (đúng chỗ đó),
    giữ cặp có cảnh đứng yên trên vành. → (I, B, V) uint8: (n, ph, pw, 3) ×2 và (n, ph, pw)."""
    s0, xy = int(track["start"]), track["xy"]
    mh, mw = mask.shape
    ph, pw, R = mh + 2 * PAD, mw + 2 * PAD, REFINE_R
    _, ring = geometry(mask)
    pairs = plan(track, mask.shape)
    need = {}                                       # frame → [(vai trò, chỉ số cặp)]
    for i, (k, js) in enumerate(pairs):
        need.setdefault(k, []).append(("k", i))
        for j in js:
            need.setdefault(j, []).append(("j", i))
    big_k, off, big_j = {}, {}, {}
    cap = cv2.VideoCapture(video_path)
    last = max(need) if need else -1
    f = 0
    while f <= last:
        if f in need:
            ok, frame = cap.read()
        else:
            ok, frame = cap.grab(), None
        if not ok:
            break
        for role, i in need.get(f, ()):
            k = pairs[i][0]
            x0, y0 = corner(xy[k - s0], mask.shape)
            crop = cut(frame, x0 - R, y0 - R, ph + 2 * R, pw + 2 * R)   # nới R để còn căn lại
            if role == "j":
                big_j.setdefault(i, {})[f] = crop
                continue
            dx = dy = 0
            if tpl is not None:
                c = np.array([R + PAD + mw / 2, R + PAD + mh / 2], np.float32)
                q, sc = _local_match(cv2.cvtColor(crop[0], cv2.COLOR_BGR2GRAY), c, tpl, R)
                if sc < MIN_SCORE:
                    continue
                dx, dy = int(np.clip(round(q[0] - c[0]), -R, R)), int(np.clip(round(q[1] - c[1]), -R, R))
            big_k[i], off[i] = crop, (dx, dy)
        f += 1
    cap.release()
    I, B, V = [], [], []
    for i, (kp, kv) in big_k.items():
        dx, dy = off[i]
        sl = (slice(R + dy, R + dy + ph), slice(R + dx, R + dx + pw))
        a, va = kp[sl], kv[sl]
        for j in pairs[i][1]:                       # thử theo thứ tự GAPS
            if j not in big_j.get(i, {}):
                continue
            jp, jv = big_j[i][j]
            b, vb = jp[sl], jv[sl]
            m = (ring > 0) & (va > 0) & (vb > 0)
            if m.any() and np.abs(a.astype(np.int16) - b.astype(np.int16))[m].mean() < STATIC_DIFF:
                I.append(a); B.append(b); V.append(va & vb)
                break
    if len(I) > MAX_PAIRS:
        keep = np.linspace(0, len(I) - 1, MAX_PAIRS).astype(int)
        I, B, V = [I[t] for t in keep], [B[t] for t in keep], [V[t] for t in keep]
    if not I:
        return (np.zeros((0, ph, pw, 3), np.uint8), np.zeros((0, ph, pw, 3), np.uint8),
                np.zeros((0, ph, pw), np.uint8))
    return np.stack(I), np.stack(B), np.stack(V)


def build(video_path, track, mask, tpl=None):
    """Cả bước tách lớp cho 1 video → bản ghi cho wm_matte.npz (method "invert" | "inpaint" + lý do)."""
    inside, _ = geometry(mask)
    rec = {"method": "inpaint", "pairs": 0, "psnr_invert": None, "psnr_inpaint": None, "pad": PAD,
           "track_start": int(track["start"]), "track_len": len(track["xy"]), "mask_shape": list(mask.shape)}
    I, B, V = collect(video_path, track, mask, tpl)
    n = rec["pairs"] = len(I)
    if n < MIN_PAIRS:
        rec["reason"] = f"thiếu cảnh tĩnh: {n} cặp nền thật (cần ≥ {MIN_PAIRS})"
        return rec
    order = np.random.default_rng(0).permutation(n)
    nt = max(int(round(n * HOLDOUT)), 12)
    te, tr = order[:nt], order[nt:]
    alpha, aW = fit(I[tr], B[tr], V[tr], inside)
    edge = edge_band(alpha, aW, inside)
    p_inv = float(np.mean([psnr(invert(I[t], alpha, aW, edge, inside), B[t], inside * V[t]) for t in te]))
    p_inp = float(np.mean([psnr(inpaint_roi(I[t], inside * 255), B[t], inside * V[t]) for t in te]))
    rec.update(psnr_invert=round(p_inv, 2), psnr_inpaint=round(p_inp, 2), method=choose(p_inv, p_inp),
               reason=f"{n} cặp nền thật; đảo pha trộn {p_inv:.1f} dB, inpaint {p_inp:.1f} dB",
               alpha=alpha, aW=aW, edge=edge, tpl=tpl)
    return rec


_ARRAYS = ("alpha", "aW", "edge", "tpl")


def save(path, rec):
    """Ghi wm_matte.npz: các mảng (α, α·màu dạng float16) + meta JSON. Qua file tạm + os.replace."""
    meta = {k: v for k, v in rec.items() if k not in _ARRAYS}
    arrays = {k: rec[k] for k in _ARRAYS if rec.get(k) is not None}
    for k in ("alpha", "aW"):
        if k in arrays:
            arrays[k] = arrays[k].astype(np.float16)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)
    os.replace(tmp, path)


def load(path, track_path, mask):
    """→ lớp watermark để render dùng ({alpha, aW, edge, tpl, inside}), hoặc None khi: chưa có file,
    method inpaint, không khớp vết bám hiện tại (dò lại sau khi tách lớp), hay file hỏng (cảnh báo, không làm sập)."""
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("method") != "invert":
                return None
            with open(track_path, encoding="utf-8") as f:
                t = json.load(f)
            if (meta["track_start"], meta["track_len"], list(meta["mask_shape"])) != \
                    (t["start"], len(t["xy"]), list(mask.shape)):
                print("  [WARN] wm_matte.npz không khớp vết bám hiện tại (đã dò lại?) — xóa watermark bằng inpaint. "
                      "Chạy lại: python pipeline.py --input <video> --only wm")
                return None
            m = {"alpha": z["alpha"].astype(np.float32), "aW": z["aW"].astype(np.float32),
                 "edge": z["edge"].astype(bool), "tpl": z["tpl"] if "tpl" in z.files else None}
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as e:
        print(f"  [WARN] {path} hỏng ({e!r}) — xóa watermark bằng inpaint.")
        return None
    m["inside"] = geometry(mask)[0]
    return m
```

- [ ] **Step 4: Chạy test, xác nhận qua**

Run: `python -m pytest tests/test_wm_matte.py -q`
Expected: `9 passed`. Nếu test α (< 0,03) không qua vì nén mp4v, **báo lại kèm số đo**, không tự nới ngưỡng.

- [ ] **Step 5: Commit**

```bash
git add wm_matte.py tests/test_wm_matte.py
git commit -m "feat(wm): thu cặp nền thật 1 lượt đọc, chấm điểm chọn cách xóa, ghi/đọc wm_matte.npz" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Render dùng lớp watermark

**Files:**
- Modify: `wm_matte.py` (thêm `apply`), `jobs.py:76-78` (thêm `wm_matte`), `script.py` (`remove_wm`, `process_chunk` args ~dòng 800-805 và chỗ gọi ~873, `main` ~dòng 1295 và `worker_args` ~dòng 1345-1352, import)
- Test: `tests/test_wm_matte.py` (thêm)

**Interfaces:**
- Consumes (Task 2): `wm_matte.load(path, track_path, mask)`, `geometry`, `corner`, `cut`, `invert`, `REFINE_R`, `MIN_SCORE`, `PAD`.
- Produces:
  - `wm_matte.apply(frame, center, matte, mask_shape) -> frame` (sửa tại chỗ, trả lại frame).
  - `script.remove_wm(frame, center, mask, pad=10, matte=None)` — `matte` là dict từ `wm_matte.load`.
  - `job.wm_matte` = `jobs/<tên>/wm_matte.npz`.

- [ ] **Step 1: Viết test (sẽ fail)** — thêm vào `tests/test_wm_matte.py`:

```python
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
    mask, matte, _, frame, _, _ = _matte_and_frame((40, 30))
    a = s.remove_wm(frame.copy(), (40, 30), mask, matte=matte)
    b = wm_matte.apply(frame.copy(), (40, 30), matte, mask.shape)
    assert (a == b).all()
    c = s.remove_wm(frame.copy(), (40, 30), mask)
    assert not (c == a).all(), "không có lớp → inpaint như cũ"


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
```

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_wm_matte.py -q`
Expected: 3 test mới FAIL (`apply` chưa có; `remove_wm()` không nhận `matte`; job chưa có `wm_matte`).

- [ ] **Step 3: Thêm `apply` vào cuối `wm_matte.py`**

```python
def apply(frame, center, matte, mask_shape):
    """Xóa watermark ở frame (tâm center) bằng lớp đã tách: căn lại ±REFINE_R bằng tpl (nếu có), đảo pha trộn,
    chỉ ghi phần ô nằm trong khung. Sửa frame tại chỗ."""
    mh, mw = mask_shape
    ph, pw, R = mh + 2 * PAD, mw + 2 * PAD, REFINE_R
    c = np.asarray(center, np.float32)
    if matte.get("tpl") is not None:
        x0, y0 = corner(c, mask_shape)
        o = np.array([x0 - R, y0 - R], np.float32)
        big, _ = cut(frame, x0 - R, y0 - R, ph + 2 * R, pw + 2 * R)
        q, sc = _local_match(cv2.cvtColor(big, cv2.COLOR_BGR2GRAY), c - o, matte["tpl"], R)
        if sc >= MIN_SCORE:
            c = q + o
    x0, y0 = corner(c, mask_shape)
    patch, _ = cut(frame, x0, y0, ph, pw)
    out = invert(patch, matte["alpha"], matte["aW"], matte["edge"], matte["inside"])
    fh, fw = frame.shape[:2]
    ys0, xs0, ys1, xs1 = max(0, y0), max(0, x0), min(fh, y0 + ph), min(fw, x0 + pw)
    if ys1 > ys0 and xs1 > xs0:
        frame[ys0:ys1, xs0:xs1] = out[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    return frame
```

- [ ] **Step 4: `jobs.py` — thêm đường dẫn** (trong `SimpleNamespace(...)` của `open_job`, ngay sau `wm_preview=...`):

```python
        wm_matte=os.path.join(d, "wm_matte.npz"),
```

- [ ] **Step 5: `script.py` — nối lớp watermark vào render**

Import (cạnh `from wm_track import load_track`):

```python
import wm_matte
```

`remove_wm` — thêm tham số và nhánh đầu hàm (giữ nguyên phần inpaint phía sau):

```python
def remove_wm(frame, center, mask, pad=10, matte=None):
    """Xóa watermark chạy tại center. Có lớp watermark (wm_matte.load) → đảo pha trộn (khôi phục nền thật);
    không thì đặt mask rồi inpaint theo INPAINT_METHOD. ROI rộng hơn mask `pad` px; sát / lố mép khung thì xén.
    Sửa frame tại chỗ."""
    if matte is not None:
        return wm_matte.apply(frame, center, matte, mask.shape)
```

`process_chunk` — thêm phần tử cuối vào tuple tham số và dùng nó:

```python
     wm_pos, wm_mask, wm_matte_rec) = args
```

```python
        if wm_mask is not None and frame_num in wm_pos:     # watermark chạy (wm_track.py): xóa ở mọi frame
            frame = remove_wm(frame, wm_pos[frame_num], wm_mask, matte=wm_matte_rec)
```

`main` — ngay sau `wm_pos, wm_mask = load_wm_track(...)`, thay khối in hiện tại bằng:

```python
    wm_matte_rec = wm_matte.load(job.wm_matte, job.wm_track, wm_mask) if wm_mask is not None else None
    if wm_mask is not None:
        how = "đảo pha trộn theo lớp watermark" if wm_matte_rec is not None else "inpaint"
        print(f"  Watermark chạy : xóa ở {len(wm_pos)} frame ({how})")
```

`worker_args` — thêm `wm_matte_rec` vào cuối mỗi tuple:

```python
         chunk_wm[i], wm_mask, wm_matte_rec)
```

- [ ] **Step 6: Chạy test, xác nhận qua**

Run: `python -m pytest tests -q`
Expected: toàn bộ qua (test cũ `test_remove_wm_touches_only_mask_and_clips_at_edges` vẫn qua vì `matte=None`).

- [ ] **Step 7: Commit**

```bash
git add wm_matte.py jobs.py script.py tests/test_wm_matte.py
git commit -m "feat(wm): render xóa watermark bằng lớp đã tách (đảo pha trộn) khi có wm_matte.npz" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Tách lớp trong bước 2/4 + `--matte-only` + pipeline bỏ qua + README

**Files:**
- Modify: `wm_track.py` (`run`, thêm `_build_matte`, `run_matte_only`, `main`, import `imread_gray`), `pipeline.py:105-108` và `pipeline.py:123-125`, `README.md` (mục "Xóa watermark chạy")
- Test: `tests/test_wm.py` (sửa `test_run_writes_track_mask_preview_and_can_be_disabled`, thêm test pipeline)

**Interfaces:**
- Consumes (Task 2, 3): `wm_matte.build(video, track, mask, tpl)`, `wm_matte.save(path, rec)`, `job.wm_matte`; `build_template(video, wm) -> (tpl, tpl_small, scale, mask) | None`, `find_moving_wm(ocr_data)`, `working_video(job, ocr_data, video_path)`.
- Produces:
  - `wm_track.run(video_path) -> int` giờ luôn ghi thêm `job.wm_matte`.
  - `wm_track.run_matte_only(video_path) -> int` (0 = xong, 1 = thiếu job / track / OCR / mask).
  - `wm_track.py --matte-only`; `pipeline.py --skip-existing` truyền `--matte-only` khi có `wm_track.json` mà thiếu `wm_matte.npz`.

- [ ] **Step 1: Viết test (sẽ fail)**

Trong `tests/test_wm.py`, thêm helper ngay trên `test_run_writes_track_mask_preview_and_can_be_disabled`:

```python
def _matte_meta(path):
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["meta"]))
```

và sửa thân test đó thành:

```python
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
        meta = _matte_meta(job.wm_matte)                  # nền video tổng hợp trôi 1 px/frame → thiếu cảnh tĩnh
        assert meta["method"] == "inpaint" and "cảnh tĩnh" in meta["reason"], meta
        os.remove(job.wm_matte)
        assert wm_track.run_matte_only(video) == 0 and os.path.exists(job.wm_matte), "chỉ tách lớp từ track có sẵn"
        wm_track.WM_TRACK = False
        assert wm_track.run(video) == 0
        assert json.load(open(job.wm_track, encoding="utf-8"))["found"] is False
        assert _matte_meta(job.wm_matte)["method"] == "inpaint", "tắt WM_TRACK vẫn ghi wm_matte.npz (để bỏ qua lần sau)"
        os.remove(job.ocr_data)
        assert wm_track.run(video) == 1, "chưa có OCR → mã 1"
    finally:
        jobs.JOBS_DIR, wm_track.WM_TRACK = orig
        shutil.rmtree(d, ignore_errors=True)
```

Thêm test mới (cạnh `test_optional_step_failure_does_not_stop_pipeline`):

```python
def test_pipeline_skip_existing_builds_only_missing_matte():
    import pipeline
    d = tempfile.mkdtemp(prefix="pipe 修仙 ")
    orig = (jobs.JOBS_DIR, pipeline.run_step, sys.argv)
    calls = []
    try:
        jobs.JOBS_DIR = os.path.join(d, "jobs")
        video = os.path.join(d, "v.mp4")
        with open(video, "wb") as f:
            f.write(os.urandom(64))
        job = jobs.open_job(video, create=True)
        utils.write_json_atomic(job.ocr_data, {"complete": True})
        utils.write_json_atomic(job.wm_track, {"found": False})
        pipeline.run_step = lambda title, script, extra, required=True: calls.append(extra) or 0.0
        sys.argv = ["pipeline.py", "--input", video, "--only", "wm", "--skip-existing"]
        pipeline.main()
        assert calls == [["--input", video, "--matte-only"]], calls
        open(job.wm_matte, "wb").close()
        calls.clear()
        pipeline.main()
        assert calls == [], "đã có cả wm_track.json lẫn wm_matte.npz → bỏ qua bước 2"
    finally:
        jobs.JOBS_DIR, pipeline.run_step, sys.argv = orig
        shutil.rmtree(d, ignore_errors=True)
```

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `python -m pytest tests/test_wm.py -q -k "run_writes_track or skip_existing_builds"`
Expected: 2 FAIL (`wm_matte.npz` chưa được ghi / chưa có `run_matte_only`; pipeline chưa truyền `--matte-only`).

- [ ] **Step 3: `wm_track.py`**

Import: đổi `from utils import imwrite_any, write_json_atomic` thành `from utils import imread_gray, imwrite_any, write_json_atomic`.

Thêm 2 hàm ngay trên `def run(`:

```python
def _build_matte(job, video, ocr_data, track, mask):
    """Tách lớp watermark (wm_matte.py) → job.wm_matte. Bước phụ: lỗi gì cũng chỉ làm render quay về inpaint."""
    import wm_matte                       # import muộn: wm_matte dùng _local_match của module này
    try:
        built = build_template(video, find_moving_wm(ocr_data))
        rec = wm_matte.build(video, track, mask, built[0] if built else None)
    except Exception as e:                # ponytail: bắt rộng vì là bước phụ; lý do vẫn in ra log để sửa
        rec = {"method": "inpaint", "reason": f"lỗi tách lớp: {e!r}", "pairs": 0}
    wm_matte.save(job.wm_matte, rec)
    print(f"  [WM] Lớp watermark: {rec['reason']} → "
          f"{'xóa bằng đảo pha trộn' if rec['method'] == 'invert' else 'giữ inpaint'}")
    return rec


def _no_matte(job, reason):
    """Ghi wm_matte.npz method inpaint (để --skip-existing biết bước 2/4 đã xong)."""
    import wm_matte
    wm_matte.save(job.wm_matte, {"method": "inpaint", "reason": reason, "pairs": 0})


def run_matte_only(video_path: str) -> int:
    """Đã dò watermark (wm_track.json) mà chưa tách lớp → chỉ tách lớp (pipeline --skip-existing).
    Mã 0 khi xong; 1 khi thiếu job / track / OCR / mask."""
    try:
        job = open_job(video_path)
        with open(job.wm_track, encoding="utf-8") as f:
            track = json.load(f)
        with open(job.ocr_data, encoding="utf-8") as f:
            ocr_data = json.load(f)
    except (JobError, OSError, ValueError) as e:
        print(f"  [WM][ERROR] {e}")
        return 1
    if not track.get("found"):
        _no_matte(job, "không có watermark chạy")
        return 0
    mask = imread_gray(job.wm_mask)
    if mask is None:
        print(f"  [WM][ERROR] Thiếu {job.wm_mask} — chạy lại: python pipeline.py --input <video> --only wm")
        return 1
    _build_matte(job, working_video(job, ocr_data, video_path), ocr_data, track, mask)
    return 0
```

Trong `run()`:
- nhánh `if not WM_TRACK:` — thêm `_no_matte(job, "tắt trong config (WM_TRACK = False)")` ngay trước `return 0` của nhánh đó;
- thay hai dòng cuối `write_json_atomic(job.wm_track, result)` / `return 0` bằng:

```python
    write_json_atomic(job.wm_track, result)
    if result["found"]:
        _build_matte(job, video, ocr_data, result, mask)
    else:
        _no_matte(job, result["reason"])
    return 0
```

`main()`:

```python
def main():
    ap = argparse.ArgumentParser(description="Dò watermark chạy → jobs/<tên>/wm_track.json + wm_matte.npz (render dùng để xóa)")
    ap.add_argument("--input", "-i", type=str, default=VIDEO_PATH, help="Đường dẫn video gốc")
    ap.add_argument("--matte-only", action="store_true", help="Chỉ tách lớp watermark (đã có wm_track.json)")
    a = ap.parse_args()
    sys.exit(run_matte_only(a.input) if a.matte_only else run(a.input))
```

- [ ] **Step 4: `pipeline.py`**

Ngay trên `if args.skip_existing:` thêm `wm_extra = []`. Thay khối bỏ qua bước watermark:

```python
        if run_wm and not run_ocr and os.path.exists(job.wm_track):   # OCR vừa chạy lại → phải dò lại
            if os.path.exists(job.wm_matte):
                print("  [SKIP] Đã dò watermark chạy + tách lớp watermark cho video này → bỏ qua")
                run_wm = False
            else:
                print("  [SKIP] Đã dò watermark chạy → chỉ tách lớp watermark")
                wm_extra = ["--matte-only"]
```

và lời gọi bước 2:

```python
        run_step("Bước 2/4 — Dò watermark chạy", "wm_track.py", ["--input", args.input] + wm_extra,
                 required=(args.only == "wm"))
```

- [ ] **Step 5: `README.md`** — trong mục "## Xóa watermark chạy", thêm sau dòng "4. render xóa theo **hình nét chữ** ở mọi frame.":

```markdown
5. **tách lớp watermark** (độ đậm + màu) từ những đoạn cảnh đứng yên, chấm điểm so với nền thật: thắng inpaint ≥ 1 dB thì render khôi phục nền bằng **đảo pha trộn** (giữ chi tiết, không nhòe), không thì giữ inpaint. Log: `[WM] Lớp watermark: … → xóa bằng đảo pha trộn | giữ inpaint`; kết quả ở `jobs/<tên>/wm_matte.npz`.
```

- [ ] **Step 6: Chạy test, xác nhận qua**

Run: `python -m pytest tests -q`
Expected: toàn bộ qua.

- [ ] **Step 7: Commit**

```bash
git add wm_track.py pipeline.py README.md tests/test_wm.py
git commit -m "feat(wm): tách lớp watermark trong bước 2/4, --matte-only, pipeline chỉ tách lớp cho job cũ" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Kiểm thật trên video 36 phút + ghi kết quả vào spec

Video gốc của job `jobs/douyinsaver.com-7688371023913291051/` không còn trên máy (chỉ còn bản 25 fps trong job), nên không chạy được `pipeline.py` theo đường dẫn gốc; kiểm bằng script gọi thẳng `wm_matte` trên dữ liệu của job.

**Files:**
- Create (nháp, không commit): `<scratchpad>/verify_matte.py`
- Modify: `docs/superpowers/specs/2026-10-16-tach-lop-watermark-design.md` (mục 9)

- [ ] **Step 1: Chạy tách lớp trên job thật, đo thời gian**

```python
# verify_matte.py — chạy từ thư mục repo: python <scratchpad>/verify_matte.py
import glob, json, sys, time
sys.path.insert(0, ".")
import cv2, numpy as np
import utils, wm_matte, wm_track, script as s
J = "jobs/douyinsaver.com-7688371023913291051"
V = glob.glob(J + "/*_25fps.mp4")[0]
track = json.load(open(J + "/wm_track.json", encoding="utf-8"))
ocr = json.load(open(J + "/ocr_data.json", encoding="utf-8"))
mask = utils.imread_gray(J + "/wm_mask.png")
t0 = time.time()
tpl = wm_track.build_template(V, wm_track.find_moving_wm(ocr))[0]
rec = wm_matte.build(V, track, mask, tpl)
dt = time.time() - t0
wm_matte.save(J + "/wm_matte.npz", rec)
print(rec["reason"], "| method", rec["method"], f"| {dt:.0f}s cho video {track['total_frames'] / 25 / 60:.1f} phút")
```

Expected: `method invert`; `psnr_invert ≥ 30` và `psnr_invert − psnr_inpaint ≥ 5`; thời gian ≤ 1/4 thời lượng (≤ 546 s).

- [ ] **Step 2: Ảnh so sánh trước / sau ở ≥ 6 thời điểm + tốc độ xóa**

```python
m = wm_matte.load(J + "/wm_matte.npz", J + "/wm_track.json", mask)
cap = cv2.VideoCapture(V); rows = []; t_inv = t_inp = 0.0
pos, _ = wm_track.load_track(J + "/wm_track.json", track["total_frames"])
frames = sorted(pos)[:: max(1, len(pos) // 8)][:8]
for f in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, fr = cap.read()
    t = time.time(); a = s.remove_wm(fr.copy(), pos[f], mask, matte=m); t_inv += time.time() - t
    t = time.time(); b = s.remove_wm(fr.copy(), pos[f], mask); t_inp += time.time() - t
    x0, y0 = wm_matte.corner(pos[f], mask.shape); ph, pw = mask.shape[0] + 40, mask.shape[1] + 40
    crop = lambda img: wm_matte.cut(img, x0 - 10, y0 - 10, ph, pw)[0]
    rows.append(np.hstack([crop(fr), crop(b), crop(a)]))
utils.imwrite_any("<scratchpad>/matte_truoc_sau.png", np.vstack(rows))
print(f"xóa/frame: đảo pha trộn {t_inv / len(frames) * 1000:.1f} ms, inpaint {t_inp / len(frames) * 1000:.1f} ms")
```

Expected: cột 3 (đảo pha trộn) không còn mảng nhòe / bóng ma rõ; xóa/frame của đảo pha trộn ≤ 1,1 × inpaint (render không chậm hơn quá 10%). Xem ảnh bằng mắt trước khi kết luận.

- [ ] **Step 3: Ghi mục 9 của spec** — thay dòng `(Điền khi làm xong.)` bằng số đo thật: số cặp, PSNR hai cách, thời gian tách lớp, ms/frame, nhận xét ảnh trước/sau (kèm điểm còn yếu nếu có).

- [ ] **Step 4: Commit + push**

```bash
git add docs/superpowers/specs/2026-10-16-tach-lop-watermark-design.md
git commit -m "docs(wm): kết quả kiểm thật tách lớp watermark trên video 36 phút" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```
