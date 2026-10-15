# Xóa watermark chạy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pipeline tự nhận watermark chạy trong mỗi video, bám vị trí nó ở mọi frame, và render xóa nó theo hình nét chữ. Hết cảnh watermark lúc bị xóa lúc không.

**Architecture:**
- Thêm bước mới `wm_track.py`, chạy giữa OCR và dịch.
  - Đầu vào: `ocr_data.json` + video làm việc của job.
  - Tự nhận chữ watermark, dựng mẫu gradient, dò toàn khung mọi frame, chọn mốc chắc, lấp khoảng bằng nội suy có dò lại.
  - Đầu ra: `jobs/<tên>/wm_track.json`, `wm_mask.png`, `wm_preview.png`.
- `script.py` đọc track và inpaint mask ở từng frame, trong cùng lần encode.
- Lỗi ở bước này không chặn pipeline.

**Tech Stack:** Python 3.10, OpenCV 4.6 (`cv2`), numpy. Không thêm thư viện.

**Spec:** `docs/superpowers/specs/2026-09-27-xoa-watermark-chay-design.md`

## Global Constraints

- Không thêm phụ thuộc mới; chỉ dùng `cv2`, `numpy` và thư viện chuẩn.
- Windows: đường dẫn có thể chứa tiếng Việt / chữ Trung. Đọc và ghi ảnh bằng `utils.imwrite_any` / `utils.imread_gray`, **không** dùng `cv2.imwrite` / `cv2.imread`.
- Test chạy offline, không gọi API. Viết theo kiểu hàm `test_*` + runner ở cuối file. Chạy bằng `python -m pytest tests/ -q`.
- Trước khi sửa file, backup vào `_backup_<mô-tả>_<timestamp>/` (thói quen của user; thư mục này đã được gitignore).
- Mỗi task một commit trên nhánh `nang-cap`. Message tiếng Việt, dòng cuối: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Bước dò watermark lỗi thì **không** chặn pipeline; các bước khác vẫn dừng ngay khi lỗi như cũ.
- Ngưỡng lấy nguyên văn từ spec:

  | Nhóm | Ngưỡng |
  |---|---|
  | Tự nhận | chữ ≥ 4 ký tự; ≥ 10 lần đọc; có mặt ≥ 40% số đoạn 10 s; tản ≥ 5% đường chéo khung; bề rộng dao động ≤ 0,15 |
  | Dựng mẫu | lần đọc đủ lệch ≤ 8% bề rộng trung vị; tối đa 30 lần; cần ≥ 5 |
  | Mốc chắc | điểm ≥ 0,45; ≥ 3 láng giềng trong ±1 s, trong tầm ≤ 100 px/s + 6 px |
  | Cổng năng lượng | 35% gradient trung bình của mẫu |
  | Lấp khoảng | dò lại ±24 px, nhận khi điểm ≥ 0,30; lệch > 8 px là đổi hướng |
  | Số mốc tối thiểu | 20 |
  | Mask | gradient ≥ 25% cực đại, đóng lỗ 3×3, nới 3 px |
  | Độ phân giải dò | nửa độ phân giải, trừ khi mẫu nửa hẹp hơn 48 px |

## File Structure

| File | Việc | Trách nhiệm |
|---|---|---|
| `utils.py` | sửa | + `imwrite_any`, `imread_gray` (ảnh với đường dẫn Unicode) |
| `jobs.py` | sửa | `open_job` + `wm_track`/`wm_mask`/`wm_preview`; + `working_video()` |
| `wm_track.py` | **tạo** | tự nhận → mẫu → dò → mốc → lấp khoảng → ghi file; CLI `--input` |
| `config.py` | sửa | + `WM_TRACK = True` |
| `script.py` | sửa | dùng `working_video`; + `remove_wm`, `load_wm_track`; truyền track cho worker |
| `pipeline.py` | sửa | bước "Dò watermark chạy" (không bắt buộc), `--only wm`, `--skip-existing` |
| `tests/wm_synth.py` | **tạo** | video tổng hợp + `ocr_data` giả cho test |
| `tests/test_wm.py` | **tạo** | test tự nhận, bám, `run`, xóa khi render, bước không bắt buộc |
| `tests/test_jobs.py`, `tests/test_core.py` | sửa | test đường dẫn job / video làm việc; test ảnh Unicode |
| `experiments/wm_remove.py` | chuyển | từ gốc (bị `wm_track.py` thay) |
| `README.md`, `experiments/README.md` | sửa | mục "Xóa watermark chạy", bảng dữ liệu, bảng code |

---

### Task 1: Ảnh với đường dẫn Unicode, đường dẫn job, video làm việc

**Files:**
- Modify: `utils.py` (thêm 2 hàm cuối file, sửa docstring đầu file)
- Modify: `jobs.py` (`open_job` return; thêm `working_video` cuối file)
- Modify: `script.py:50` (import) và `script.py:1218-1233` (chọn video làm việc)
- Test: `tests/test_jobs.py`, `tests/test_core.py`

**Interfaces:**
- Produces:
  - `utils.imwrite_any(path: str, img: np.ndarray) -> None` (lỗi encode thì `OSError`);
  - `utils.imread_gray(path: str) -> np.ndarray | None`;
  - `job.wm_track`, `job.wm_mask`, `job.wm_preview` (đường dẫn trong `job.dir`);
  - `jobs.working_video(job, ocr_data: dict, input_path: str) -> str`.

- [ ] **Step 1: Backup**

```powershell
$b = "_backup_wm-task1_" + (Get-Date -Format "yyyyMMdd_HHmmss"); New-Item -ItemType Directory -Force "$b\tests" | Out-Null
Copy-Item utils.py, jobs.py, script.py "$b\"; Copy-Item tests\test_jobs.py, tests\test_core.py "$b\tests\"
```

- [ ] **Step 2: Viết test (fail)**

Thêm vào `tests/test_jobs.py`, trước khối `if __name__ == "__main__":`:

```python
def test_open_job_has_watermark_paths_and_working_video():
    d = _setup()
    try:
        v = _video(d, "a", "tap1.mp4", b"video-1")
        j = jobs.open_job(v, create=True)
        assert j.wm_track == os.path.join(j.dir, "wm_track.json")
        assert j.wm_mask == os.path.join(j.dir, "wm_mask.png")
        assert j.wm_preview == os.path.join(j.dir, "wm_preview.png")
        # OCR không convert fps → đọc thẳng video gốc; có convert → bản trong job
        assert jobs.working_video(j, {"video": "tap1.mp4", "original_video": "tap1.mp4"}, v) == v
        assert jobs.working_video(j, {"video": "tap1_25fps.mp4", "original_video": "tap1.mp4"}, v) == \
            os.path.join(j.dir, "tap1_25fps.mp4")
        assert jobs.working_video(j, {}, v) == v
    finally:
        _cleanup(d)
```

Thêm `import numpy as np` vào khối import đầu `tests/test_core.py` (sau `import tempfile`). Rồi thêm test trước khối `if __name__ == "__main__":`:

```python
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
```

- [ ] **Step 3: Chạy, xác nhận fail**

Run: `python -m pytest tests/test_jobs.py tests/test_core.py -q -k "watermark_paths or unicode_path"`
Expected: 2 failed (`AttributeError: ... 'wm_track'` và `... has no attribute 'imwrite_any'`)

- [ ] **Step 4: Cài đặt**

`utils.py`: sửa docstring đầu file thành:

```python
"""
Monggu Translate - Tiện ích dùng chung.

- write_json_atomic(): ghi JSON qua file tạm + os.replace, lỗi giữa chừng không hỏng file cũ.
- imwrite_any() / imread_gray(): ghi/đọc ảnh với đường dẫn có dấu / chữ Trung (cv2.imwrite/imread không làm được trên Windows).
"""
```

và thêm cuối file:

```python
def imwrite_any(path: str, img) -> None:
    """Ghi ảnh (định dạng theo đuôi file). cv2.imwrite không ghi được đường dẫn có dấu / chữ Trung trên Windows
    → encode trong RAM rồi ghi bằng numpy."""
    import cv2
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    if not ok:
        raise OSError(f"Không encode được ảnh: {path}")
    buf.tofile(path)


def imread_gray(path: str):
    """Đọc ảnh xám (đường dẫn Unicode được). Thiếu file / hỏng → None."""
    import cv2
    import numpy as np
    try:
        data = np.fromfile(path, np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None
```

`jobs.py`: trong `open_job`, thay khối `return SimpleNamespace(...)` bằng:

```python
    return SimpleNamespace(
        name=name, dir=d, series=meta.get("series"),
        ocr_data=os.path.join(d, "ocr_data.json"),
        lines=os.path.join(d, "lines.json"),
        output=os.path.join(d, f"{name}_vi.mp4"),
        story=os.path.join(shared, "story.json"),
        wm_track=os.path.join(d, "wm_track.json"),
        wm_mask=os.path.join(d, "wm_mask.png"),
        wm_preview=os.path.join(d, "wm_preview.png"),
    )
```

và thêm cuối `jobs.py`:

```python
def working_video(job, ocr_data: dict, input_path: str) -> str:
    """Video mà chỉ số frame trong ocr_data tham chiếu tới: bản convert fps trong job nếu OCR đã convert,
    không thì chính video gốc. Render và wm_track đọc frame từ đây (tiếng luôn lấy từ video gốc)."""
    converted = ocr_data.get("video", "")
    if converted and converted != ocr_data.get("original_video", input_path):
        return os.path.join(job.dir, converted)
    return input_path
```

`script.py`: dòng 50 thành `from jobs import open_job, JobError, working_video`. Thay đoạn `script.py:1218-1233`, bắt đầu từ `# Nếu OCR đã convert video (vd: 60fps → 25fps), dùng video đã convert để render` tới hết `        audio_source = args.input`, bằng:

```python
    # OCR đã convert fps (vd 60 → 25) thì render trên bản convert trong job: frame index trong ocr_data là của nó.
    # Tiếng luôn lấy từ video gốc (bản convert đã bỏ tiếng).
    video_path = working_video(job, ocr_data, args.input)
    audio_source = args.input
    if video_path != args.input and not os.path.exists(video_path):
        print(f"  [ERROR] Thiếu video đã convert fps: {video_path} (bị xóa?) — chạy lại OCR để tạo lại.")
        sys.exit(1)
```

- [ ] **Step 5: Chạy toàn bộ test**

Run: `python -m pytest tests/ -q`
Expected: tất cả pass (54 passed)

- [ ] **Step 6: Commit**

```powershell
git add utils.py jobs.py script.py tests/test_jobs.py tests/test_core.py
git commit -m "refactor: đường dẫn watermark trong job, working_video dùng chung, ảnh với đường dẫn Unicode" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Tự nhận watermark chạy từ `ocr_data`

**Files:**
- Create: `wm_track.py`
- Test: `tests/test_wm.py` (tạo mới)

**Interfaces:**
- Produces: `wm_track.find_moving_wm(ocr_data: dict) -> dict | None`
  - Trả về `{"text": str, "reads": [(frame:int, cx:float, cy:float, w:float, h:float, conf:float), ...]}`, trong đó `reads` sắp theo frame.
  - `ocr_data` cần các khóa `fps`, `total_frames`, `width`, `height`, `frame_detections` (`{"<frame>": [{"text", "box", "confidence"}]}`).

- [ ] **Step 1: Viết test (fail)**

Tạo `tests/test_wm.py`:

```python
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
    assert len(wm["reads"]) == 150 and wm["reads"] == sorted(wm["reads"])
    no_wm = {f: [d for d in ds if not d["text"].startswith("原创@")] for f, ds in fd.items()}
    assert wm_track.find_moving_wm(_ocr(no_wm)) is None, "nhãn / watermark tĩnh / thoại không được nhận là watermark chạy"


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
```

- [ ] **Step 2: Chạy, xác nhận fail**

Run: `python -m pytest tests/test_wm.py -q`
Expected: FAIL/ERROR với `ModuleNotFoundError: No module named 'wm_track'`

- [ ] **Step 3: Cài đặt**

Tạo `wm_track.py`:

```python
"""
Monggu Translate - Dò watermark CHẠY (bước giữa OCR và dịch; pipeline.py tự gọi).

Watermark chạy = chữ ID nhỏ bán trong suốt trôi khắp khung hình (vd "原创@哈加密"), mỗi kênh một chữ.
OCR chỉ đọc được nó ở vài % số frame → render cũ xóa lúc có lúc không (nhấp nháy). Bước này:
  1. tự nhận chữ watermark từ ocr_data: có mặt suốt video, vị trí tản rộng, cỡ box không đổi;
  2. dựng mẫu nét chữ (trung vị gradient các lần OCR đọc đủ, đã căn) + mask hình nét chữ;
  3. dò mẫu trên toàn khung MỌI frame → mốc chắc (điểm cao + nhất quán với láng giềng);
  4. lấp khoảng giữa các mốc: nội suy thẳng + dò lại tại chỗ; lệch khỏi đường thẳng → chỗ đổi hướng.
Ra: jobs/<tên>/wm_track.json (vị trí từng frame), wm_mask.png, wm_preview.png. Render đọc để xóa ở mọi frame.

Chạy tay:  python wm_track.py --input video.mp4     (cần OCR trước)
"""

from collections import defaultdict

import numpy as np

# ── Tự nhận watermark chạy (số liệu chốt ngưỡng: spec 2026-09-27 mục 4.1) ──
MIN_TEXT_LEN = 4          # chữ watermark ≥ 4 ký tự
MIN_READS = 10            # xuất hiện ≥ 10 lần trong ocr_data
MIN_PRESENCE = 0.40       # có mặt ≥ 40% số đoạn 10 giây (watermark 57–78%; nhãn nhân vật ≤ 21%)
MIN_SPREAD_FRAC = 0.05    # tâm box tản ≥ 5% đường chéo khung (watermark tĩnh tản 4–6 px)
MAX_WIDTH_CV = 0.15       # bề rộng box dao động ≤ 15% (watermark có cỡ cố định)


def find_moving_wm(ocr_data: dict):
    """Chữ watermark chạy trong ocr_data, hoặc None.
    → {"text": str, "reads": [(frame, cx, cy, w, h, conf)] mọi lần OCR đọc đúng chữ đó, sắp theo frame}."""
    fps, n = ocr_data["fps"], ocr_data["total_frames"]
    win = max(1, int(round(10 * fps)))
    n_win = max(1, -(-n // win))
    min_spread = MIN_SPREAD_FRAC * float(np.hypot(ocr_data["width"], ocr_data["height"]))
    groups = defaultdict(list)
    for fr, dets in ocr_data["frame_detections"].items():
        for d in dets:
            t = "".join(d["text"].split())
            if len(t) < MIN_TEXT_LEN:
                continue
            xs = [p[0] for p in d["box"]]
            ys = [p[1] for p in d["box"]]
            groups[t].append((int(fr), (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                              max(xs) - min(xs), max(ys) - min(ys), float(d.get("confidence", 1.0))))
    best = None
    for t, reads in groups.items():
        if len(reads) < MIN_READS:
            continue
        a = np.array(reads)
        presence = len({int(f) // win for f in a[:, 0]}) / n_win
        spread = float(np.hypot(a[:, 1].std(), a[:, 2].std()))
        width_cv = float(a[:, 3].std() / max(a[:, 3].mean(), 1e-6))
        if presence >= MIN_PRESENCE and spread >= min_spread and width_cv <= MAX_WIDTH_CV:
            if best is None or len(reads) > len(best["reads"]):
                best = {"text": t, "reads": sorted(reads)}
    return best
```

- [ ] **Step 4: Chạy test**

Run: `python -m pytest tests/test_wm.py -q`
Expected: 1 passed

- [ ] **Step 5: Commit**

```powershell
git add wm_track.py tests/test_wm.py
git commit -m "feat(wm): tự nhận watermark chạy từ ocr_data (có mặt suốt video, vị trí tản, cỡ ổn định)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Mẫu, dò toàn khung, mốc chắc, lấp khoảng (`track_watermark`)

**Files:**
- Modify: `wm_track.py` (import + hằng + hàm, thêm sau `find_moving_wm`)
- Create: `tests/wm_synth.py`
- Test: `tests/test_wm.py`

**Interfaces:**
- Consumes: `find_moving_wm(ocr_data)` (Task 2)
- Produces:
  - `wm_track.track_watermark(video_path: str, ocr_data: dict) -> tuple[dict, np.ndarray | None]`:
    - thấy watermark: `({"found": True, "text", "size": [w, h], "total_frames", "fps", "start": int, "xy": [[x, y], ...], "evidence": "0101…", "anchors": int, "seconds": float}, mask_uint8)`;
    - không thấy: `({"found": False, "reason": str}, None)`.

    `xy` chứa tâm watermark (int) ở từng frame từ `start` tới mốc cuối. `evidence[i] == "1"` nghĩa là frame `start+i` có bằng chứng tại chỗ. Tâm mask trùng tâm trong `xy`.
  - `tests/wm_synth.py`:
    - `make_video(path) -> (centers: np.ndarray[N, 2], (gw, gh))`;
    - `fake_ocr(centers, size, video_name) -> dict`;
    - hằng `N`, `FAINT`.

- [ ] **Step 1: Tạo video tổng hợp cho test**

Tạo `tests/wm_synth.py`:

```python
"""Video tổng hợp + ocr_data giả cho test wm_track: chữ bán trong suốt (lõi trắng, viền tối) trôi trên nền
có vân và trôi theo, có một đoạn gần như vô hình, đổi hướng giữa chừng, và một mảng nền trơn (nơi NCC dễ khớp nhầm)."""
import cv2
import numpy as np

FPS, N, W, H = 25.0, 240, 480, 270
FAINT = range(40, 80)      # đoạn gần như vô hình (độ mờ 0.08), nằm trên đoạn thẳng
TURN = 150                 # đổi hướng (lúc watermark còn rõ)


def truth(t):
    """Tâm watermark ở frame t: thẳng tới TURN rồi rẽ."""
    if t <= TURN:
        return np.array([60 + 1.6 * t, 60 + 0.7 * t])
    return truth(TURN) + np.array([-1.5, 0.5]) * (t - TURN)


def glyph():
    """(alpha viền+lõi, lõi) của chữ 'WM@12', cắt sát."""
    font, sc = cv2.FONT_HERSHEY_SIMPLEX, 0.9
    (tw, th), base = cv2.getTextSize("WM@12", font, sc, 2)
    edge = np.zeros((th + base + 16, tw + 16), np.uint8)
    fill = edge.copy()
    cv2.putText(edge, "WM@12", (8, th + 8), font, sc, 255, 6)
    cv2.putText(fill, "WM@12", (8, th + 8), font, sc, 255, 2)
    ys, xs = np.nonzero(edge)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    return edge[sl].astype(np.float32) / 255, fill[sl].astype(np.float32) / 255


def make_video(path):
    """Ghi video mp4v; trả về (tâm thật từng frame, (rộng, cao) của chữ)."""
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur(rng.integers(0, 256, (H + 40, W + N + 40)).astype(np.float32), (0, 0), 2)
    tex = cv2.normalize(tex, None, 40, 200, cv2.NORM_MINMAX)
    edge, fill = glyph()
    gh, gw = edge.shape
    ov = np.where(fill > 0.5, 255.0, 25.0)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    centers = []
    for t in range(N):
        bg = tex[20:20 + H, t:t + W].copy()           # nền trôi 1 px/frame
        bg[10:80, 330:470] = 200.0                    # mảng trơn
        c = truth(t)
        x0, y0 = int(round(c[0] - gw / 2)), int(round(c[1] - gh / 2))
        a = (0.08 if t in FAINT else 0.6) * edge
        roi = bg[y0:y0 + gh, x0:x0 + gw]
        bg[y0:y0 + gh, x0:x0 + gw] = (1 - a) * roi + a * ov
        centers.append((x0 + gw / 2, y0 + gh / 2))
        vw.write(cv2.cvtColor(np.clip(bg, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    vw.release()
    return np.array(centers), (gw, gh)


def fake_ocr(centers, size, video_name):
    """ocr_data giả: 'WM@12' đọc đủ mỗi 8 frame (trừ đoạn mờ) + 1 câu thoại đứng yên."""
    gw, gh = size
    fd = {}
    for t in range(0, N, 8):
        if t in FAINT:
            continue
        cx, cy = centers[t]
        x0, y0 = cx - gw / 2, cy - gh / 2
        fd[str(t)] = [{"text": "WM@12", "confidence": 0.95,
                       "box": [[x0, y0], [x0 + gw, y0], [x0 + gw, y0 + gh], [x0, y0 + gh]]}]
    for t in range(100, 160, 4):
        fd.setdefault(str(t), []).append({"text": "你好世界啊", "confidence": 0.99,
                                         "box": [[180, 230], [300, 230], [300, 260], [180, 260]]})
    return {"video": video_name, "original_video": video_name, "fps": FPS, "width": W, "height": H,
            "total_frames": N, "ocr_interval_frames": 1, "frame_detections": fd, "text_segments": [],
            "complete": True}
```

- [ ] **Step 2: Viết test (fail)**

Trong `tests/test_wm.py`, thêm `import wm_synth  # noqa: E402` ngay sau `import wm_track  # noqa: E402`. Thêm 2 test trước khối `if __name__ == "__main__":`:

```python
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
```

- [ ] **Step 3: Chạy, xác nhận fail**

Run: `python -m pytest tests/test_wm.py -q`
Expected: 2 failed (`AttributeError: module 'wm_track' has no attribute 'track_watermark'`)

- [ ] **Step 4: Cài đặt**

Trong `wm_track.py`, thay khối import:

```python
from collections import defaultdict

import numpy as np
```

bằng:

```python
import time
from collections import defaultdict

import cv2
import numpy as np
```

Thêm ngay dưới khối hằng "Tự nhận watermark chạy":

```python
# ── Mẫu + bám (số liệu: spec mục 2, 4.2, 4.3) ──
FULL_READ_TOL = 0.08      # lần OCR "đọc đủ chữ": bề rộng lệch ≤ 8% trung vị
MAX_TEMPLATE_READS = 30   # dựng mẫu từ tối đa 30 lần đọc đủ, rải đều theo thời gian
MIN_TEMPLATE_READS = 5
SURE = 0.45               # điểm khớp toàn khung đủ chắc để làm mốc
LOCAL_OK = 0.30           # điểm khớp tại chỗ (quanh điểm nội suy) chấp nhận được
MAX_SPEED = 100           # px/giây — watermark đo được trôi 8–50 px/s
ENERGY_GATE = 0.35        # cửa sổ có gradient TB < 35% của mẫu = nền trơn → NCC khuếch đại nhiễu, không tin
MIN_ANCHORS = 20
LOCAL_R = 24              # bán kính dò lại quanh điểm nội suy (px, full-res)
SPLIT_DEV = 8             # vị trí dò lại lệch khỏi đường thẳng > 8 px → chỗ đổi hướng
MIN_SMALL_W = 48          # mẫu nửa độ phân giải hẹp hơn thế → dò toàn khung ở độ phân giải gốc
_PAD, _MARGIN = 8, 4      # lề để căn patch (±8 px) / lề giữ quanh mẫu cho mask (4 px)
```

Thêm sau hàm `find_moving_wm`:

```python
def _grad(gray):
    """Độ lớn gradient (Sobel) — bám theo nét chữ, không phụ thuộc nền sáng hay tối."""
    g = gray.astype(np.float32)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def _ncc(img, tpl):
    r = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    return np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)


def _gated_ncc(g, tpl):
    """NCC của mẫu trên ảnh gradient g. Vị trí nền trơn (gradient TB trong cửa sổ < ENERGY_GATE × của mẫu) = -1:
    ở đó NCC tự khuếch đại nhiễu thành 'giống chữ' (khớp nhầm duy nhất gặp trên video thật)."""
    th, tw = tpl.shape
    r = _ncc(g, tpl)
    energy = cv2.boxFilter(g, -1, (tw, th), anchor=(0, 0), normalize=True)[:r.shape[0], :r.shape[1]]
    r[energy < ENERGY_GATE * float(tpl.mean())] = -1.0
    return r


def _median_patch(grads, centers, tw, th):
    """Cắt patch gradient quanh các tâm, căn thẳng hàng (2 lượt, ±_PAD) rồi lấy trung vị: nền mỗi frame một
    khác nên trung vị chỉ giữ lại nét chữ. → mảng (th+2*_MARGIN, tw+2*_MARGIN), hoặc None nếu quá ít patch."""
    patches = []
    for g, (cx, cy) in zip(grads, centers):
        x0, y0 = int(round(cx - tw / 2)) - _PAD, int(round(cy - th / 2)) - _PAD
        if x0 >= 0 and y0 >= 0 and x0 + tw + 2 * _PAD <= g.shape[1] and y0 + th + 2 * _PAD <= g.shape[0]:
            patches.append(g[y0:y0 + th + 2 * _PAD, x0:x0 + tw + 2 * _PAD])
    if len(patches) < MIN_TEMPLATE_READS:
        return None
    ref = patches[0][_PAD:-_PAD, _PAD:-_PAD]
    big = None
    for _ in range(2):   # lượt 2 căn theo trung vị của lượt 1 (không phụ thuộc patch đầu có tốt không)
        stack = []
        for p in patches:
            _, _, _, (dx, dy) = cv2.minMaxLoc(_ncc(p, ref))
            if _MARGIN <= dx <= 2 * _PAD - _MARGIN and _MARGIN <= dy <= 2 * _PAD - _MARGIN:
                stack.append(p[dy - _MARGIN:dy + th + _MARGIN, dx - _MARGIN:dx + tw + _MARGIN])
        if len(stack) < MIN_TEMPLATE_READS:
            return None
        big = np.median(np.stack(stack), axis=0).astype(np.float32)
        ref = big[_MARGIN:-_MARGIN, _MARGIN:-_MARGIN]
    return big


def build_template(video_path, wm):
    """→ (mẫu full-res, mẫu để dò toàn khung, tỉ lệ dò, mask uint8 nét chữ), hoặc None nếu quá ít lần đọc đủ."""
    reads = wm["reads"]
    w_med = float(np.median([r[3] for r in reads]))
    h_med = float(np.median([r[4] for r in reads]))
    full = [r for r in reads if abs(r[3] - w_med) <= FULL_READ_TOL * w_med]
    if len(full) < MIN_TEMPLATE_READS:
        return None
    groups = np.array_split(np.arange(len(full)), min(MAX_TEMPLATE_READS, len(full)))
    picked = [max((full[i] for i in g), key=lambda r: r[5]) for g in groups if len(g)]
    tw, th = int(round(w_med)), int(round(h_med))
    scale = 0.5 if tw * 0.5 >= MIN_SMALL_W else 1.0
    stw, sth = int(round(tw * scale)), int(round(th * scale))
    cap = cv2.VideoCapture(video_path)
    g_full, g_small, centers = [], [], []
    for f, cx, cy, *_ in picked:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g_full.append(_grad(gray))
        g_small.append(_grad(cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
                       if scale != 1.0 else g_full[-1])
        centers.append((cx, cy))
    cap.release()
    big = _median_patch(g_full, centers, tw, th)
    big_s = _median_patch(g_small, [(x * scale, y * scale) for x, y in centers], stw, sth)
    if big is None or big_s is None:
        return None
    mask = (big >= 0.25 * big.max()).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))      # nới 3 px
    return big[_MARGIN:-_MARGIN, _MARGIN:-_MARGIN], big_s[_MARGIN:-_MARGIN, _MARGIN:-_MARGIN], scale, mask


def detect_all(video_path, tpl_small, scale, n_frames):
    """Lượt 1: dò toàn khung mọi frame (tỉ lệ scale) → (pos (n,2) tâm full-res, score (n,)); frame không đọc được: -1."""
    th, tw = tpl_small.shape
    pos = np.zeros((n_frames, 2), np.float32)
    score = np.full(n_frames, -1.0, np.float32)
    cap = cv2.VideoCapture(video_path)
    for f in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if scale != 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        g = _grad(gray)
        if g.shape[0] < th or g.shape[1] < tw:
            break
        _, s, _, (x, y) = cv2.minMaxLoc(_gated_ncc(g, tpl_small))
        pos[f] = ((x + tw / 2) / scale, (y + th / 2) / scale)
        score[f] = s
    cap.release()
    return pos, score


def pick_anchors(pos, score, fps):
    """Mốc chắc: điểm ≥ SURE và có ≥ 3 đỉnh chắc khác trong ±1 s nằm trong tầm di chuyển (loại khớp nhầm lẻ)."""
    sure = np.flatnonzero(score >= SURE)
    win = max(1, int(round(fps)))
    vmax = MAX_SPEED / fps
    keep = []
    for k, i in enumerate(sure):
        lo, hi = np.searchsorted(sure, [i - win, i + win + 1])
        nb = np.concatenate([sure[lo:k], sure[k + 1:hi]])
        if len(nb) < 3:
            continue
        d = np.hypot(*(pos[nb] - pos[i]).T)
        if np.sum(d <= vmax * np.abs(nb - i) + 6) >= 3:
            keep.append(int(i))
    return keep


def _local_match(gray, center, tpl, r):
    """Khớp mẫu full-res trong cửa sổ ±r quanh center → (tâm tìm được, điểm)."""
    th, tw = tpl.shape
    h, w = gray.shape
    x0 = int(np.clip(round(center[0] - tw / 2) - r, 0, max(0, w - tw - 2 * r)))
    y0 = int(np.clip(round(center[1] - th / 2) - r, 0, max(0, h - th - 2 * r)))
    g = _grad(gray[y0:y0 + th + 2 * r, x0:x0 + tw + 2 * r])
    if g.shape[0] < th or g.shape[1] < tw:
        return np.asarray(center, np.float32), -1.0
    _, s, _, (mx, my) = cv2.minMaxLoc(_gated_ncc(g, tpl))
    return np.array([x0 + mx + tw / 2, y0 + my + th / 2], np.float32), float(s)


def fill_gaps(video_path, anchors, pos, tpl):
    """Lượt 2: vị trí mọi frame trong [mốc đầu, mốc cuối]. Giữa 2 mốc: nội suy thẳng rồi dò lại tại chỗ;
    vị trí dò được lệch khỏi đường thẳng > SPLIT_DEV → chỗ đổi hướng: thành mốc mới và nội suy lại các frame
    không có bằng chứng từ mốc trước tới đây. → (xy (k,2), evidence (k,) bool) bắt đầu từ anchors[0]."""
    first, last = anchors[0], anchors[-1]
    xy = np.zeros((last - first + 1, 2), np.float32)
    ev = np.zeros(last - first + 1, bool)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    nxt, a, pa = 0, first, pos[first]
    for f in range(first, last + 1):
        ok, frame = cap.read()
        if not ok:   # video ngắn hơn dữ liệu: giữ vị trí cuối
            xy[f - first:] = xy[f - first - 1] if f > first else pos[first]
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        i = f - first
        if f == anchors[nxt]:
            q, s = _local_match(gray, pos[f], tpl, 6)
            xy[i], ev[i] = (q if s >= LOCAL_OK else pos[f]), True
            a, pa, nxt = f, xy[i].copy(), nxt + 1
            continue
        b = anchors[nxt]
        p = pa + (f - a) / (b - a) * (pos[b] - pa)
        q, s = _local_match(gray, p, tpl, LOCAL_R)
        if s < LOCAL_OK:
            xy[i] = p
            continue
        xy[i], ev[i] = q, True
        if np.hypot(*(q - p)) > SPLIT_DEV:
            for g in range(a + 1, f):
                if not ev[g - first]:
                    xy[g - first] = pa + (g - a) / (f - a) * (q - pa)
            a, pa = f, q.copy()
    cap.release()
    return xy, ev


def track_watermark(video_path, ocr_data):
    """Toàn bộ bước dò cho một video → (kết quả cho wm_track.json, mask uint8 hoặc None)."""
    t0 = time.time()
    wm = find_moving_wm(ocr_data)
    if wm is None:
        return {"found": False, "reason": "không thấy chữ nào giống watermark chạy trong OCR"}, None
    built = build_template(video_path, wm)
    if built is None:
        return {"found": False,
                "reason": f"'{wm['text']}': dưới {MIN_TEMPLATE_READS} lần OCR đọc đủ chữ để dựng mẫu"}, None
    tpl, tpl_small, scale, mask = built
    pos, score = detect_all(video_path, tpl_small, scale, ocr_data["total_frames"])
    anchors = pick_anchors(pos, score, ocr_data["fps"])
    if len(anchors) < MIN_ANCHORS:
        return {"found": False,
                "reason": f"'{wm['text']}': chỉ {len(anchors)} mốc chắc (cần ≥ {MIN_ANCHORS})"}, None
    xy, ev = fill_gaps(video_path, anchors, pos, tpl)
    return {"found": True, "text": wm["text"], "size": [int(tpl.shape[1]), int(tpl.shape[0])],
            "total_frames": int(ocr_data["total_frames"]), "fps": float(ocr_data["fps"]),
            "start": int(anchors[0]), "xy": [[int(round(x)), int(round(y))] for x, y in xy],
            "evidence": "".join("1" if e else "0" for e in ev), "anchors": len(anchors),
            "seconds": round(time.time() - t0, 1)}, mask
```

- [ ] **Step 5: Chạy test**

Run: `python -m pytest tests/test_wm.py -q`
Expected: 3 passed (bản nháp cùng code: lệch lớn nhất 2,2 px, 100% frame có bằng chứng lệch ≤ 3 px)

- [ ] **Step 6: Commit**

```powershell
git add wm_track.py tests/wm_synth.py tests/test_wm.py
git commit -m "feat(wm): bám watermark chạy mọi frame (mẫu gradient, dò toàn khung, mốc chắc, nội suy có dò lại)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `run()`, CLI, ảnh xem nhanh, `WM_TRACK`

**Files:**
- Modify: `config.py` (thêm mục `WM_TRACK` trước mục `# TIKTOK PASS MODE`)
- Modify: `wm_track.py` (import, `write_preview`, `run`, `main`)
- Test: `tests/test_wm.py`

**Interfaces:**
- Consumes:
  - `track_watermark` (Task 3);
  - `open_job`, `working_video`, `job.wm_track` / `wm_mask` / `wm_preview`, `imwrite_any`, `write_json_atomic` (Task 1).
- Produces:
  - `wm_track.run(video_path: str) -> int`:
    - trả 0 khi đã ghi `wm_track.json` (kể cả `found: false`);
    - trả 1 khi thiếu job/OCR hoặc không mở được video.
  - `wm_track.write_preview(video_path, result, out_path, n=6) -> bool`.
  - CLI `python wm_track.py --input <video>`.
  - `config.WM_TRACK: bool`.

- [ ] **Step 0: Backup `config.py`**

```powershell
$b = "_backup_wm-task4_" + (Get-Date -Format "yyyyMMdd_HHmmss"); New-Item -ItemType Directory -Force $b | Out-Null; Copy-Item config.py "$b\"
```

- [ ] **Step 1: Viết test (fail)**

Trong `tests/test_wm.py`:
- thêm `import json` vào khối import chuẩn;
- sau `import wm_synth  # noqa: E402`, thêm:

```python
import jobs  # noqa: E402
import utils  # noqa: E402
```

Thêm test trước khối `if __name__ == "__main__":`:

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
        wm_track.WM_TRACK = False
        assert wm_track.run(video) == 0
        assert json.load(open(job.wm_track, encoding="utf-8"))["found"] is False
        os.remove(job.ocr_data)
        assert wm_track.run(video) == 1, "chưa có OCR → mã 1"
    finally:
        jobs.JOBS_DIR, wm_track.WM_TRACK = orig
        shutil.rmtree(d, ignore_errors=True)
```

- [ ] **Step 2: Chạy, xác nhận fail**

Run: `python -m pytest tests/test_wm.py -q -k run_writes`
Expected: FAIL (`AttributeError: module 'wm_track' has no attribute 'WM_TRACK'`)

- [ ] **Step 3: Cài đặt**

`config.py`: chèn ngay trước dòng `# ──────────────────────────────────────────────` của mục `# TIKTOK PASS MODE (Bypass detection)`:

```python
# ──────────────────────────────────────────────
# XÓA WATERMARK CHẠY (wm_track.py — pipeline tự chạy sau OCR)
# ──────────────────────────────────────────────
WM_TRACK = True                  # False = không dò/xóa watermark chạy (tắt cho 1 video qua MONGGU_CONFIG)

```

`wm_track.py`: thay khối import (từ `import time` tới `import numpy as np`) bằng:

```python
import argparse
import json
import os
import sys
import time
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):   # in tiếng Việt khi chạy tay trên Windows
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

from config import VIDEO_PATH, WM_TRACK
from jobs import JobError, open_job, working_video
from utils import imwrite_any, write_json_atomic
```

Thêm cuối `wm_track.py`:

```python
def write_preview(video_path, result, out_path, n=6):
    """n frame rải đều trong khoảng bám: vùng quanh watermark + khung (xanh = có bằng chứng, vàng = nội suy)
    → ảnh ghép 3 cột để user liếc là biết bám đúng hay sai. → False nếu không đọc được frame nào."""
    w, h = result["size"]
    start, xy, ev = result["start"], result["xy"], result["evidence"]
    cw, ch = 3 * w, 5 * h
    cap = cv2.VideoCapture(video_path)
    tiles = []
    for i in np.linspace(0, len(xy) - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start + i)
        ok, frame = cap.read()
        if not ok:
            continue
        x, y = xy[i]
        color = (0, 200, 0) if ev[i] == "1" else (0, 220, 255)
        cv2.rectangle(frame, (int(x - w / 2 - 4), int(y - h / 2 - 4)), (int(x + w / 2 + 4), int(y + h / 2 + 4)), color, 2)
        big = cv2.copyMakeBorder(frame, ch, ch, cw, cw, cv2.BORDER_CONSTANT)
        tile = big[int(y) + ch - ch // 2:int(y) + ch + ch // 2, int(x) + cw - cw // 2:int(x) + cw + cw // 2]
        tile = cv2.resize(tile, (480, int(480 * tile.shape[0] / tile.shape[1])))
        cv2.putText(tile, f"frame {start + i}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(tile)
    cap.release()
    if not tiles:
        return False
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    imwrite_any(out_path, np.vstack([np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles), 3)]))
    return True


def run(video_path: str) -> int:
    """Dò watermark chạy cho video → jobs/<tên>/wm_track.json (+ wm_mask.png, wm_preview.png).
    Mã 0 cả khi không thấy / bỏ qua (ghi found: false + lý do); 1 khi thiếu job/OCR hoặc không mở được video."""
    try:
        job = open_job(video_path)
    except JobError as e:
        print(f"  [WM][ERROR] {e}")
        return 1
    if not os.path.exists(job.ocr_data):
        print(f"  [WM][ERROR] Chưa có OCR cho {job.name} — chạy OCR trước: python ocr_pass.py --input \"{video_path}\"")
        return 1
    if not WM_TRACK:
        write_json_atomic(job.wm_track, {"found": False, "reason": "tắt trong config (WM_TRACK = False)"})
        print("  [WM] Tắt trong config (WM_TRACK = False) → không xóa watermark chạy.")
        return 0
    with open(job.ocr_data, encoding="utf-8") as f:
        ocr_data = json.load(f)
    video = working_video(job, ocr_data, video_path)
    cap = cv2.VideoCapture(video)
    opened = cap.isOpened()
    cap.release()
    if not opened:
        print(f"  [WM][ERROR] Không mở được video: {video}")
        return 1
    result, mask = track_watermark(video, ocr_data)
    if result["found"]:
        imwrite_any(job.wm_mask, mask)          # ghi mask trước json: render thấy found thì mask đã có
        write_preview(video, result, job.wm_preview)
        n = len(result["xy"])
        print(f"  [WM] Thấy watermark chạy '{result['text']}' ({result['size'][0]}×{result['size'][1]}) — "
              f"bám {n} frame ({result['evidence'].count('1') / n:.0%} có bằng chứng, {result['anchors']} mốc), "
              f"{result['seconds']}s")
        print(f"  [WM] Xem nhanh vị trí bám: {job.wm_preview}")
    else:
        print(f"  [WM] Không xóa watermark chạy: {result['reason']}")
    write_json_atomic(job.wm_track, result)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Dò watermark chạy → jobs/<tên>/wm_track.json (render dùng để xóa)")
    ap.add_argument("--input", "-i", type=str, default=VIDEO_PATH, help="Đường dẫn video gốc")
    sys.exit(run(ap.parse_args().input))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Chạy test**

Run: `python -m pytest tests/ -q`
Expected: tất cả pass

- [ ] **Step 5: Commit**

```powershell
git add config.py wm_track.py tests/test_wm.py
git commit -m "feat(wm): wm_track.py chạy được (CLI, wm_track.json + mask + ảnh xem nhanh), tắt bằng WM_TRACK" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Render xóa watermark ở mọi frame

**Files:**
- Modify: `script.py`
  - import (`from utils import imread_gray`);
  - thêm `remove_wm`, `load_wm_track` sau `_inpaint_roi` (khoảng dòng 333);
  - `process_chunk`: unpack tham số + xóa ngay sau khi đọc frame;
  - `main`: nạp track, chia theo chunk, `worker_args`.
- Test: `tests/test_wm.py`

**Interfaces:**
- Consumes:
  - `job.wm_track` / `job.wm_mask`, `utils.imread_gray` (Task 1);
  - định dạng `wm_track.json` (Task 3/4).
- Produces:
  - `script.remove_wm(frame, center, mask, pad=10) -> frame`: sửa tại chỗ, chỉ đổi pixel thuộc mask.
  - `script.load_wm_track(job, total_frames: int) -> (dict[int, (float, float)], np.ndarray | None)`.

- [ ] **Step 1: Backup `script.py`**

```powershell
$b = "_backup_wm-task5_" + (Get-Date -Format "yyyyMMdd_HHmmss"); New-Item -ItemType Directory -Force $b | Out-Null; Copy-Item script.py "$b\"
```

- [ ] **Step 2: Viết test (fail)**

Trong `tests/test_wm.py`:
- thêm `from types import SimpleNamespace` vào khối import chuẩn;
- sau `import utils  # noqa: E402`, thêm `import script as s  # noqa: E402`.

Thêm test trước khối `if __name__ == "__main__":`:

```python
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
```

- [ ] **Step 3: Chạy, xác nhận fail**

Run: `python -m pytest tests/test_wm.py -q -k "remove_wm or load_wm_track"`
Expected: 2 failed (`AttributeError: module 'script' has no attribute 'remove_wm'`)

- [ ] **Step 4: Cài đặt**

`script.py`: sau dòng `from jobs import open_job, JobError, working_video`, thêm `from utils import imread_gray`.

Thêm ngay sau hàm `_inpaint_roi`:

```python
def remove_wm(frame, center, mask, pad=10):
    """Xóa watermark chạy: đặt mask (tâm = center) lên frame rồi inpaint theo INPAINT_METHOD.
    ROI rộng hơn mask `pad` px để inpaint có ngữ cảnh; mask sát / lố mép khung thì xén. Sửa frame tại chỗ."""
    mh, mw = mask.shape
    fh, fw = frame.shape[:2]
    x0, y0 = int(round(center[0] - mw / 2)), int(round(center[1] - mh / 2))
    rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
    rx1, ry1 = min(fw, x0 + mw + pad), min(fh, y0 + mh + pad)
    mx0, my0 = max(x0, rx0), max(y0, ry0)
    mx1, my1 = min(x0 + mw, rx1), min(y0 + mh, ry1)
    if mx1 <= mx0 or my1 <= my0:
        return frame
    roi_mask = np.zeros((ry1 - ry0, rx1 - rx0), np.uint8)
    roi_mask[my0 - ry0:my1 - ry0, mx0 - rx0:mx1 - rx0] = mask[my0 - y0:my1 - y0, mx0 - x0:mx1 - x0]
    frame[ry0:ry1, rx0:rx1] = _inpaint_roi(frame[ry0:ry1, rx0:rx1], roi_mask)
    return frame


def load_wm_track(job, total_frames: int):
    """Vị trí watermark chạy ({frame: (x, y)}) + mask từ wm_track.py; ({}, None) nếu không có / không dùng được."""
    if not os.path.exists(job.wm_track):
        return {}, None
    try:
        with open(job.wm_track, encoding="utf-8") as f:
            t = json.load(f)
    except ValueError as e:
        print(f"  [WARN] {job.wm_track} hỏng ({e}) — bỏ qua xóa watermark chạy.")
        return {}, None
    if not t.get("found"):
        return {}, None
    if t.get("total_frames") != total_frames:
        print("  [WARN] wm_track.json không khớp video (OCR đã chạy lại?) — bỏ qua xóa watermark chạy. "
              "Chạy lại: python pipeline.py --input <video> --only wm")
        return {}, None
    mask = imread_gray(job.wm_mask)
    if mask is None:
        print(f"  [WARN] Thiếu {job.wm_mask} — bỏ qua xóa watermark chạy.")
        return {}, None
    start = int(t["start"])
    return {start + i: (float(x), float(y)) for i, (x, y) in enumerate(t["xy"])}, mask
```

Trong `process_chunk`, đổi phần unpack tham số thành:

```python
    (chunk_idx, start_frame, end_frame,
     video_path, temp_output, font_path,
     fps, width, height,
     ocr_interval_frames, translation_map,
     frame_detections_for_chunk, text_segments_for_chunk, vi_for_seg,
     wm_pos, wm_mask) = args
```

và đổi đầu vòng lặp frame (`for frame_num in range(start_frame, end_frame):`) thành:

```python
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        if wm_mask is not None and frame_num in wm_pos:     # watermark chạy (wm_track.py): xóa ở mọi frame
            frame = remove_wm(frame, wm_pos[frame_num], wm_mask)
```

Trong `main`, ngay trước dòng `output_video = job.output`, thêm:

```python
    # Watermark chạy (wm_track.py): vị trí ở từng frame + mask nét chữ → xóa ở mọi frame
    wm_pos, wm_mask = load_wm_track(job, ocr_data["total_frames"])
    if wm_mask is not None:
        print(f"  Watermark chạy : xóa ở {len(wm_pos)} frame")
```

Ngay trước dòng `print(f"\n[3/4] Rendering {num_workers} chunks in parallel...")`, thêm:

```python
    chunk_wm = [{f: p for f, p in wm_pos.items() if s <= f < e} for s, e in chunks]
```

và đổi `worker_args` thành:

```python
    worker_args = [
        (i, chunks[i][0], chunks[i][1],
         video_path, temp_files[i],
         FONT_PATH, fps, width, height,
         ocr_interval_frames, translation_map,
         chunk_detections[i], chunk_segments[i], vi_for_seg,
         chunk_wm[i], wm_mask)
        for i in range(num_workers)
    ]
```

- [ ] **Step 5: Chạy test**

Run: `python -m pytest tests/ -q`
Expected: tất cả pass

- [ ] **Step 6: Commit**

```powershell
git add script.py tests/test_wm.py
git commit -m "feat(render): xóa watermark chạy ở mọi frame theo wm_track.json (mask nét chữ, bỏ qua track cũ)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Pipeline chạy bước dò watermark (không bắt buộc)

**Files:**
- Modify: `pipeline.py` (docstring, `run_step`, `main`)
- Test: `tests/test_wm.py`

**Interfaces:**
- Consumes: CLI `wm_track.py --input` (Task 4), `job.wm_track` (Task 1).
- Produces:
  - `pipeline.run_step(title, script, extra_args, required=True) -> float`: bước không bắt buộc mà lỗi thì cảnh báo và trả thời gian, không thoát;
  - `--only wm`.

- [ ] **Step 0: Backup `pipeline.py`**

```powershell
$b = "_backup_wm-task6_" + (Get-Date -Format "yyyyMMdd_HHmmss"); New-Item -ItemType Directory -Force $b | Out-Null; Copy-Item pipeline.py "$b\"
```

- [ ] **Step 1: Viết test (fail)**

Thêm vào `tests/test_wm.py` trước khối `if __name__ == "__main__":`:

```python
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
```

- [ ] **Step 2: Chạy, xác nhận fail**

Run: `python -m pytest tests/test_wm.py -q -k optional_step`
Expected: FAIL (`TypeError: run_step() got an unexpected keyword argument 'required'`)

- [ ] **Step 3: Cài đặt**

`pipeline.py`:
1. Docstring dòng 3 thành: `Chạy toàn bộ quy trình bằng 1 lệnh: OCR → dò watermark chạy → Dịch → Render.`. Thêm dòng usage `    python pipeline.py --input video.mp4 --only wm           # chỉ dò lại watermark chạy` sau dòng `--only ocr`. Thay 2 dòng cuối docstring (`Mỗi pass chạy như một tiến trình con; nếu một pass lỗi (returncode != 0),` / `pipeline dừng ngay (fail-fast) để tránh render bằng dữ liệu hỏng.`) bằng:

```
Mỗi bước chạy như một tiến trình con; bước nào lỗi (returncode != 0) thì pipeline dừng ngay (fail-fast)
để tránh render bằng dữ liệu hỏng — trừ bước dò watermark chạy: lỗi thì chỉ cảnh báo, vẫn dịch và render.
```

2. Thay hàm `run_step` bằng:

```python
def run_step(title: str, script: str, extra_args: list, required: bool = True) -> float:
    """Chạy 1 bước bằng tiến trình con. Trả về thời gian chạy (s).
    Bước bắt buộc lỗi → dừng pipeline; bước không bắt buộc (dò watermark) lỗi → cảnh báo rồi chạy tiếp."""
    cmd = [sys.executable, os.path.join(_HERE, script)] + extra_args
    print("\n" + "═" * 60)
    print(f"  ▶ {title}")
    print(f"    {' '.join(cmd)}")
    print("═" * 60)
    # Ép subprocess dùng UTF-8 stdout để in tiếng Việt qua pipe không lỗi cp1252
    child_env = dict(os.environ, PYTHONIOENCODING="utf-8")
    t0 = time.time()
    result = subprocess.run(cmd, env=child_env)
    elapsed = time.time() - t0
    if result.returncode != 0:
        if not required:
            print(f"\n  ⚠ '{title}' lỗi (exit code {result.returncode}) — bỏ qua bước này, chạy tiếp.")
            return elapsed
        print(f"\n  ✗ '{title}' THẤT BẠI (exit code {result.returncode}). Dừng pipeline.")
        sys.exit(result.returncode)
    print(f"  ✓ '{title}' xong sau {elapsed:.1f}s")
    return elapsed
```

3. Trong `main`:
   - `ArgumentParser(description=...)` thành `"Monggu Translate - Pipeline (OCR → watermark → Dịch → Render)"`;
   - `choices=["all", "ocr", "translate", "render"]` thành `choices=["all", "ocr", "wm", "translate", "render"]`;
   - sau `run_ocr = args.only in ("all", "ocr")`, thêm `run_wm = args.only in ("all", "wm")`;
   - trong khối `if args.skip_existing:`, ngay trước `if run_tr and not run_ocr and os.path.exists(job.lines):`, thêm:

```python
        if run_wm and not run_ocr and os.path.exists(job.wm_track):   # OCR vừa chạy lại → phải dò lại
            print("  [SKIP] Đã dò watermark chạy cho video này → bỏ qua")
            run_wm = False
```

   - thay đoạn từ dòng `        run_step("Pass 1/3 — OCR Scan", "ocr_pass.py", ocr_args)` tới hết khối `if run_tr:` (dòng `run_step("Pass 2/3 — Translation", ...)`) bằng đoạn dưới (phần dựng `ocr_args` phía trên giữ nguyên):

```python
        run_step("Bước 1/4 — OCR", "ocr_pass.py", ocr_args)

    if run_wm:
        run_step("Bước 2/4 — Dò watermark chạy", "wm_track.py", ["--input", args.input], required=False)

    if run_tr:
        run_step("Bước 3/4 — Dịch", "translate_pass.py", ["--input", args.input])
```

   - trong khối `if run_rd:`, đổi `run_step("Pass 3/3 — Render Video", "script.py", rd_args)` thành `run_step("Bước 4/4 — Render", "script.py", rd_args)`.

- [ ] **Step 4: Chạy test + kiểm CLI**

Run: `python -m pytest tests/ -q`
Expected: tất cả pass

Run: `python pipeline.py --help`
Expected: có `{all,ocr,wm,translate,render}`

- [ ] **Step 5: Commit**

```powershell
git add pipeline.py tests/test_wm.py
git commit -m "feat(pipeline): bước dò watermark chạy sau OCR (lỗi chỉ cảnh báo), --only wm, --skip-existing" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Dọn dẹp + tài liệu

**Files:**
- Move: `wm_remove.py` → `experiments/wm_remove.py`
- Modify: `experiments/README.md`, `README.md`

- [ ] **Step 0: Backup README**

```powershell
$b = "_backup_wm-task7_" + (Get-Date -Format "yyyyMMdd_HHmmss"); New-Item -ItemType Directory -Force "$b\experiments" | Out-Null; Copy-Item README.md "$b\"; Copy-Item experiments\README.md "$b\experiments\"
```

- [ ] **Step 1: Chuyển `wm_remove.py`**

```powershell
git mv wm_remove.py experiments/wm_remove.py
```

`experiments/README.md`: thêm dòng vào bảng, ngay trên dòng `wm_ocr_probe.py`:

```
| `wm_remove.py` | Xóa watermark chạy bằng OCR dày mọi frame + nội suy (chậm, còn sót). Đã được `wm_track.py` thay. |
```

- [ ] **Step 2: README**

`README.md`:
- dòng 3 thành: `Dịch phụ đề video Trung Quốc sang tiếng Việt: **OCR → xóa watermark chạy → dịch theo cốt truyện, đúng nhân vật → render lại video**.`;
- trong khối lệnh "Dịch một video", thêm dòng `python pipeline.py --input video.mp4 --only wm                     # chỉ dò lại watermark chạy` sau dòng `--only render`;
- dòng "Bước nào lỗi thì pipeline dừng ngay…" thành: `Bước nào lỗi thì pipeline dừng ngay với thông báo rõ ràng, không âm thầm xuất video sai. Riêng bước dò watermark chạy lỗi thì chỉ cảnh báo, vẫn dịch và render.`;
- bảng "Dữ liệu của mỗi video": thêm sau dòng `ocr_data.json`:

```
| `jobs/<tên>/wm_track.json` | Watermark chạy: chữ, vị trí ở từng frame. `found: false` kèm lý do nếu không có / bỏ qua. |
| `jobs/<tên>/wm_mask.png`, `wm_preview.png` | Mask nét chữ dùng để xóa; ảnh xem nhanh vị trí bám. |
```

- thêm mục mới ngay trước `## Cache`:

```markdown
## Xóa watermark chạy

Watermark chạy là chữ ID nhỏ, mờ, trôi khắp khung hình (vd "原创@哈加密"). OCR chỉ đọc được nó ở vài % số frame, nên trước đây video ra bị lúc xóa lúc không. Bước `wm_track.py` (pipeline tự chạy sau OCR):

1. tự nhận chữ watermark từ kết quả OCR: chữ có mặt suốt video, vị trí thay đổi dần, cỡ không đổi (không cần khai báo chữ);
2. dựng mẫu nét chữ từ những lần OCR đọc đủ;
3. dò mẫu trên mọi frame, nối các lần dò chắc chắn thành quỹ đạo, lấp đoạn mờ bằng nội suy;
4. render xóa theo **hình nét chữ** ở mọi frame.

- **Kiểm:** log `[WM] Thấy watermark chạy '…' — bám N frame (…% có bằng chứng)`, và ảnh `jobs/<tên>/wm_preview.png` (khung xanh = dò thấy, vàng = nội suy) phải nằm đúng lên chữ.
- **Chạy lại riêng bước này:** `--only wm`, rồi `--only render`.
- **Tắt cho một video:** tạo file JSON `{"WM_TRACK": false}`, đặt biến môi trường `MONGGU_CONFIG` trỏ tới file đó, rồi chạy `--only wm` và `--only render`.
- **Giới hạn:**
  - mỗi video 1 watermark chạy;
  - watermark tĩnh vẫn dùng `SUPPRESS_TEXTS`;
  - nền nhiều vân có thể còn vệt nhòe nhẹ của inpaint.
```

- bảng "Code": thay dòng `| \`wm_remove.py\` | Xóa watermark (thử nghiệm, chạy tay). |` bằng `| \`wm_track.py\` | Dò watermark chạy (tự nhận, bám mọi frame) để render xóa. |`; sửa dòng `pipeline.py` thành `| \`pipeline.py\` | Chạy OCR → dò watermark → dịch → render; dừng ngay khi một bước lỗi (trừ bước dò watermark). |`.

- [ ] **Step 3: Kiểm**

Run: `python -m pytest tests/ -q`
Expected: tất cả pass.

Run: `git grep -n "wm_remove" -- ":!docs" ":!experiments"`
Expected: không ra dòng nào (ngoài `docs/` và `experiments/`, không còn chỗ nào nhắc `wm_remove.py`).

- [ ] **Step 4: Commit**

```powershell
git add -A experiments/ README.md
git commit -m "docs: mục xóa watermark chạy trong README; wm_remove.py vào experiments (wm_track.py thay)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Kiểm chứng trên dữ liệu thật (không gọi API, không commit code)

Dùng job nháp đã có OCR đầy đủ của `newvideo` ở `<scratch>\wm_ocrfull` (cấu hình `cfg.json` trỏ `JOBS_DIR` vào đó). `<scratch>` là thư mục scratchpad của phiên.

- [ ] **Step 1: Chạy bước dò**

```powershell
$env:MONGGU_CONFIG = "<scratch>\wm_ocrfull\cfg.json"; python wm_track.py --input newvideo.mp4; Remove-Item Env:MONGGU_CONFIG
```

Expected: `[WM] Thấy watermark chạy '原创@哈加密' (194×48) — bám ~3000 frame (~42% có bằng chứng…)`.

- [ ] **Step 2: So với 19 mốc OCR full-res**

```powershell
python -c "import json,numpy as np; t=json.load(open(r'<scratch>\wm_ocrfull\jobs\newvideo\wm_track.json',encoding='utf-8')); A=json.load(open('wm_anchors.json',encoding='utf-8'))['anchors']; xy=np.array(t['xy']); e=[float(np.hypot(*(xy[round(a['t']*25)-t['start']]-(a['cx'],a['cy'])))) for a in A if a['w']>=180 and 0<=round(a['t']*25)-t['start']<len(xy)]; print(len(e),'mốc | median %.1f | max %.1f px' % (np.median(e), max(e)))"
```

Expected: `19 mốc`, max ≤ 10 px (bản nháp: median 3,6, max 7,8).

- [ ] **Step 3: Tự nhận trên hihi**

```powershell
python -c "import json,wm_track; print(wm_track.find_moving_wm(json.load(open('ocr_data.json',encoding='utf-8')))['text'])"
```

Expected: `原创@哈加密`

- [ ] **Step 4: Render thử 20 s, trước / sau**

- Tạo `lines.json` rỗng bản dịch cho job nháp: mỗi segment một dòng, `vi` là `""`, đủ để render mà không gọi API.
- Render `--test` 2 lần:
  - có `wm_track.json`: bản "sau";
  - đổi tên `wm_track.json` đi: bản "trước".

  Nhớ copy video ra của lần đầu sang chỗ khác trước khi render lần 2.
- Trích frame 200, 300, 400 của cả hai bản, ghép thành `wm_truoc_sau.png` rồi gửi user bằng `SendUserFile`.

```powershell
$env:MONGGU_CONFIG = "<scratch>\wm_ocrfull\cfg.json"
python -c "import json; d=json.load(open(r'<scratch>\wm_ocrfull\jobs\newvideo\ocr_data.json',encoding='utf-8')); json.dump([{'id':i,'start':int(s['start_frame']),'end':int(s.get('end_frame',s['start_frame'])),'vi':''} for i,s in enumerate(d['text_segments'])], open(r'<scratch>\wm_ocrfull\jobs\newvideo\lines.json','w',encoding='utf-8'))"
python script.py --input newvideo.mp4 --test
Copy-Item "<scratch>\wm_ocrfull\jobs\newvideo\newvideo_vi.mp4" "<scratch>\wm_sau.mp4"
Rename-Item "<scratch>\wm_ocrfull\jobs\newvideo\wm_track.json" "wm_track.off.json"
python script.py --input newvideo.mp4 --test
Copy-Item "<scratch>\wm_ocrfull\jobs\newvideo\newvideo_vi.mp4" "<scratch>\wm_truoc.mp4"
Rename-Item "<scratch>\wm_ocrfull\jobs\newvideo\wm_track.off.json" "wm_track.json"
Remove-Item Env:MONGGU_CONFIG
ffmpeg -v error -y -i "<scratch>\wm_truoc.mp4" -i "<scratch>\wm_sau.mp4" -filter_complex "[0]select='eq(n\,200)+eq(n\,300)+eq(n\,400)',scale=720:-1,tile=1x3[a];[1]select='eq(n\,200)+eq(n\,300)+eq(n\,400)',scale=720:-1,tile=1x3[b];[a][b]hstack" -frames:v 1 "<scratch>\wm_truoc_sau.png"
```

Expected: cột trái (trước) còn thấy "原创@哈加密" ở ít nhất một frame; cột phải (sau) không còn.

- [ ] **Step 5: Báo user** kết quả đo, ảnh trước/sau và thời gian chạy. Hỏi có muốn chạy thật trên video mới không.

---

## Self-Review (đã làm khi viết plan)

- **Spec coverage:**
  - mục 3 (kiến trúc): Task 1, 4, 5, 6;
  - 4.1: Task 2; 4.2–4.4: Task 3–4; 4.5: Task 5;
  - mục 5 (lỗi / an toàn: `found: false`, không chặn pipeline, track cũ): Task 3, 4, 5, 6;
  - mục 6 (test + dữ liệu thật): Task 2–6, Task 8;
  - mục 7 (dọn dẹp / tài liệu): Task 7.
- **Kiểu dữ liệu thống nhất:**
  - `track_watermark` → `(dict, mask)`; `run` → `int`;
  - `load_wm_track` → `(dict, mask|None)`; `remove_wm(frame, center, mask)`;
  - `job.wm_track` / `wm_mask` / `wm_preview` dùng giống nhau ở Task 1, 4, 5, 6.
- **Code lõi Task 3** là đúng bản nháp đã chạy:
  - video tổng hợp: lệch lớn nhất 2,2 px;
  - `newvideo` thật: lệch lớn nhất 7,8 px so với 19 mốc OCR; 71 s.
