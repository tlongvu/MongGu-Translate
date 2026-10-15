# B2 — Thư mục job + series — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mỗi video có thư mục riêng `jobs/<tên>/` chứa mọi dữ liệu và video ra. Video thuộc series thì dùng chung hồ sơ nhân vật và glossary ở `series/<bộ>/`. Hết cảnh video sau đè video trước và lẫn ngữ cảnh giữa các phim.

**Architecture:**
- `jobs.open_job()` trả về đường dẫn của từng video.
- Dấu vân tay (kích thước + sha1 1MB đầu) chặn trường hợp hai video trùng tên dùng chung dữ liệu.
- Ba pass và `pipeline.py` lấy đường dẫn từ job, không còn dùng hằng số đường dẫn toàn cục.

**Tech Stack:** `hashlib`, `json`, `types.SimpleNamespace` (thư viện chuẩn).

**Spec:** `docs/superpowers/specs/2026-09-26-dich-theo-cot-truyen-design.md`, mục 8.

## Global Constraints

- Nhánh `nang-cap`. Backup vào `_backup_nang-cap-B2_<timestamp>/` trước khi sửa.
- **Không gọi API thật:** không chạy `translate_pass.py` hay `pipeline.py` với video thật có job.
  - Test dùng `ask_json` giả.
  - Render thử dùng dữ liệu OCR giả trong `JOBS_DIR` tạm.
- Không xóa dữ liệu cũ ở gốc project (`ocr_data.json`, `translation_map.json`, `glossary.json`, `character_profiles.json`). Chỉ thôi đọc chúng.
- Giữ phong cách code hiện có. Windows + PowerShell, `$env:PYTHONIOENCODING="utf-8"`. `SCRATCH` như các plan trước.

## Chuẩn bị

- [ ] Backup `config.py, jobs.py(chưa có), ocr_pass.py, translate_pass.py, script.py, pipeline.py, tests\test_core.py`. Chạy `python -m pytest tests -q` → `20 passed`.

---

### Task 1: `jobs.py` + config

**Files:**
- Create: `jobs.py`, `tests/test_jobs.py`
- Modify: `config.py` (thêm `JOBS_DIR`, `SERIES_DIR`), `.gitignore` (thêm `jobs/`)

**Interfaces — Produces:**
- `jobs.open_job(video_path: str, series: str = None, create: bool = False) -> SimpleNamespace`, gồm:
  - `name`, `dir`, `series`
  - `ocr_data`, `translation_map`, `output`
  - `profiles`, `glossary`
- `jobs.JobError(RuntimeError)`
- `jobs.fingerprint(video_path) -> str`

- [ ] **Step 1: config + gitignore.** Trong `config.py`, ngay sau dòng `LLM_CACHE_PATH = ...`, thêm:

```python
# Dữ liệu theo video (jobs.py): jobs/<tên-video>/…; series dùng chung hồ sơ: series/<tên-bộ>/…
JOBS_DIR = os.path.join(_SCRIPT_DIR, "jobs")
SERIES_DIR = os.path.join(_SCRIPT_DIR, "series")
```

`.gitignore`: thêm `jobs/` sau `cache/`. `series/` **vẫn được theo dõi** vì đó là hồ sơ nhỏ, người dùng sửa tay.

- [ ] **Step 2: Test (fail)** — `tests/test_jobs.py`:

```python
"""
Test thư mục job (jobs.py) — chỉ dùng file giả trong thư mục tạm.
Chạy:  python -m pytest tests/   hoặc   python tests/test_jobs.py
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jobs  # noqa: E402


def _setup():
    d = tempfile.mkdtemp()
    jobs.JOBS_DIR = os.path.join(d, "jobs")
    jobs.SERIES_DIR = os.path.join(d, "series")
    return d


def _video(d, sub, name, content):
    os.makedirs(os.path.join(d, sub), exist_ok=True)
    p = os.path.join(d, sub, name)
    with open(p, "wb") as f:
        f.write(content)
    return p


def test_open_job_create_and_reopen():
    d = _setup()
    try:
        v = _video(d, "a", "tap1.mp4", b"video-1")
        try:
            jobs.open_job(v)
            raise AssertionError("Chưa có job mà không create phải lỗi")
        except jobs.JobError:
            pass
        j = jobs.open_job(v, create=True)
        assert j.dir == os.path.join(jobs.JOBS_DIR, "tap1")
        assert j.ocr_data == os.path.join(j.dir, "ocr_data.json")
        assert j.output == os.path.join(j.dir, "tap1_vi.mp4")
        assert os.path.dirname(j.profiles) == j.dir, "Video lẻ: hồ sơ nằm trong job"
        assert jobs.open_job(v).dir == j.dir, "Mở lại cùng video → cùng job"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_open_job_rejects_other_video_with_same_name():
    d = _setup()
    try:
        jobs.open_job(_video(d, "a", "tap1.mp4", b"video-1"), create=True)
        try:
            jobs.open_job(_video(d, "b", "tap1.mp4", b"video khac hoan toan"), create=True)
            raise AssertionError("Video khác trùng tên phải bị chặn")
        except jobs.JobError:
            pass
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_series_shares_profiles_between_episodes():
    d = _setup()
    try:
        j1 = jobs.open_job(_video(d, "a", "daosiep1.mp4", b"ep1"), series="daosi", create=True)
        j2 = jobs.open_job(_video(d, "a", "daosiep2.mp4", b"ep2"), series="daosi", create=True)
        assert j1.profiles == j2.profiles == os.path.join(jobs.SERIES_DIR, "daosi", "character_profiles.json")
        assert jobs.open_job(os.path.join(d, "a", "daosiep1.mp4")).series == "daosi", "Series được nhớ trong job.json"
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
```

Run: `python tests/test_jobs.py` → `ModuleNotFoundError: No module named 'jobs'`

- [ ] **Step 3: `jobs.py`**

```python
"""
Monggu Translate - Thư mục job: mỗi video một thư mục jobs/<tên>/ chứa mọi dữ liệu của video đó.
Video thuộc series thì hồ sơ nhân vật + glossary nằm ở series/<bộ>/, dùng chung giữa các tập.
"""

import hashlib
import json
import os
from types import SimpleNamespace

from config import JOBS_DIR, SERIES_DIR
from utils import write_json_atomic


class JobError(RuntimeError):
    """Job không dùng được: chưa có, hoặc đang thuộc một video khác trùng tên."""


def fingerprint(video_path: str) -> str:
    """Dấu vân tay rẻ: kích thước + sha1 của 1MB đầu — đủ phân biệt hai video khác nhau trùng tên."""
    h = hashlib.sha1()
    with open(video_path, "rb") as f:
        h.update(f.read(1 << 20))
    return f"{os.path.getsize(video_path)}-{h.hexdigest()[:16]}"


def open_job(video_path: str, series: str = None, create: bool = False) -> SimpleNamespace:
    """Đường dẫn dữ liệu của một video. create=True (OCR/pipeline): tạo job nếu chưa có.
    series: tên bộ; video lẻ để None. Truyền series khác lần trước → cập nhật job.json."""
    if not os.path.isfile(video_path):
        raise JobError(f"Không thấy video: {video_path}")
    name = os.path.splitext(os.path.basename(video_path))[0]
    d = os.path.join(JOBS_DIR, name)
    meta_path = os.path.join(d, "job.json")
    fp = fingerprint(video_path)
    changed = False
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("fingerprint") != fp:
            raise JobError(f"jobs/{name}/ đang là của video khác ({meta.get('video')}). "
                           f"Đổi tên video, hoặc xóa thư mục job cũ nếu không cần nữa.")
    elif create:
        meta, changed = {"video": os.path.abspath(video_path), "fingerprint": fp, "series": None}, True
    else:
        raise JobError(f"Chưa có dữ liệu cho {os.path.basename(video_path)} — hãy chạy OCR trước "
                       f"(python ocr_pass.py --input \"{video_path}\").")
    if series and series != meta.get("series"):
        if meta.get("series"):
            print(f"  [JOB] Đổi series của {name}: {meta['series']} → {series}")
        meta["series"], changed = series, True
    if changed:
        os.makedirs(d, exist_ok=True)
        write_json_atomic(meta_path, meta)

    shared = os.path.join(SERIES_DIR, meta["series"]) if meta.get("series") else d
    os.makedirs(shared, exist_ok=True)
    return SimpleNamespace(
        name=name, dir=d, series=meta.get("series"),
        ocr_data=os.path.join(d, "ocr_data.json"),
        translation_map=os.path.join(d, "translation_map.json"),
        output=os.path.join(d, f"{name}_vi.mp4"),
        profiles=os.path.join(shared, "character_profiles.json"),
        glossary=os.path.join(shared, "glossary.json"),
    )
```

- [ ] **Step 4:** `python tests/test_jobs.py` → `3/3`; `python -m pytest tests -q` → `23 passed`

- [ ] **Step 5: Commit**: `feat(jobs): mỗi video một thư mục jobs/<tên>/, series dùng chung hồ sơ` (+ Co-Authored-By)

---

### Task 2: OCR ghi vào job

**Files:** Modify `ocr_pass.py`

**Interfaces:**
- `run_ocr_pass(video_path, test_mode=False, test_frames=300, series=None)`
- `convert_video_fps(input_path, target_fps, out_dir)`
- `ocr_data` có thêm `"complete": bool`, `"test_mode": bool`

- [ ] **Step 1: Import.** Thêm `from jobs import open_job, JobError` sau `from utils import write_json_atomic`. Bỏ `OCR_DATA_PATH` khỏi `from config import (...)`.

- [ ] **Step 2: Video convert nằm trong job.** Sửa hàm convert:
  - thay dòng `def convert_video_fps(input_path: str, target_fps: int) -> str:` bằng `def convert_video_fps(input_path: str, target_fps: int, out_dir: str) -> str:`;
  - thay 2 dòng sau:

```python
    base, ext = os.path.splitext(input_path)
    converted_path = f"{base}_{target_fps}fps{ext}"
```

bằng:

```python
    base, ext = os.path.splitext(os.path.basename(input_path))
    converted_path = os.path.join(out_dir, f"{base}_{target_fps}fps{ext}")
```

- [ ] **Step 3: Checkpoint.**
  - Chữ ký thành `def _save_intermediate_checkpoint(data_path, test_mode, video_path, original_video_path, fps, width, height, total_frames, ocr_interval_frames, unique_texts, frame_detections):`.
  - Trong dict, thêm `"complete": False,` và `"test_mode": test_mode,` ngay trước dòng `"unique_texts": unique_texts,`.
  - `write_json_atomic(OCR_DATA_PATH, ocr_data)` thành `write_json_atomic(data_path, ocr_data)`.
  - Chỗ gọi: `_save_intermediate_checkpoint(` → thêm `data_path, test_mode,` làm hai đối số đầu.

- [ ] **Step 4: `run_ocr_pass`.**
  - Chữ ký thành `def run_ocr_pass(video_path: str, test_mode: bool = False, test_frames: int = 300, series: str = None):`.
  - Ngay trước dòng `# ── Phân tích video ──`, chèn:

```python
    try:
        job = open_job(video_path, series, create=True)
    except JobError as e:
        print(f"  [ERROR] {e}")
        return None
    data_path = job.ocr_data
```

  - `convert_video_fps(video_path, TARGET_FPS)` → `convert_video_fps(video_path, TARGET_FPS, job.dir)`.
  - Mọi `OCR_DATA_PATH` còn lại trong hàm (kiểm tra resume, ghi cuối, 2 dòng in `Output`) → `data_path`.
  - Trong dict `ocr_data` cuối, thêm `"complete": not interrupted,` và `"test_mode": test_mode,` ngay trước `"unique_texts": unique_texts,`.
  - Docstring: `Output: jobs/<tên>/ocr_data.json. Trả về ocr_data, hoặc None nếu không mở được video / bị Ctrl+C.`

- [ ] **Step 5: CLI.**
  - Thêm `parser.add_argument("--series", "-s", type=str, default=None, help="Tên bộ (series) nếu video là một tập")`.
  - Gọi `run_ocr_pass(args.input, test_mode=args.test, test_frames=TEST_FRAMES, series=args.series)`.

- [ ] **Step 6: Kiểm chứng.**
  - `Select-String -Path ocr_pass.py -Pattern 'OCR_DATA_PATH'` → không còn.
  - `python -m pytest tests -q` → `23 passed`.
  - `python ocr_pass.py --input khong_co.mp4; $LASTEXITCODE` → `[ERROR] Không thấy video: khong_co.mp4`, `1`.

- [ ] **Step 7: Commit**: `feat(ocr): ghi ocr_data + video convert vào thư mục job; đánh dấu complete/test_mode` (+ Co-Authored-By)

---

### Task 3: Dịch đọc/ghi trong job; bỏ prompt viết cứng

**Files:** Modify `translate_pass.py`, `config.py`, `tests/test_core.py`

**Interfaces:**
- `run_translation(video_path: str) -> bool`
- `load_ocr_data(path)`, `load_glossary(path)`
- `infer_character_profiles(transcript, glossary, profiles_path)`
- `translate_contextual(text_segments, normalized_mapping, texts_to_translate, glossary, profiles_path)`

- [ ] **Step 1: Test (sửa cho chữ ký mới — fail).** Trong `tests/test_core.py`:
  - thêm `import jobs  # noqa: E402` cạnh các import module project;
  - thay `test_run_translation_fails_without_input` bằng:

```python
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
```

  - trong `_run_contextual`, thay:

```python
    orig_ask, orig_prof = tp.ask_json, tp.CHAR_PROFILES_PATH
    tmp = tempfile.mkdtemp()
    try:
        tp.ask_json = fake
        tp.CHAR_PROFILES_PATH = os.path.join(tmp, "profiles.json")
        return tp.translate_contextual(segs, norm, texts, {})
    finally:
        tp.ask_json, tp.CHAR_PROFILES_PATH = orig_ask, orig_prof
        shutil.rmtree(tmp, ignore_errors=True)
```

bằng:

```python
    orig_ask = tp.ask_json
    tmp = tempfile.mkdtemp()
    try:
        tp.ask_json = fake
        return tp.translate_contextual(segs, norm, texts, {}, os.path.join(tmp, "profiles.json"))
    finally:
        tp.ask_json = orig_ask
        shutil.rmtree(tmp, ignore_errors=True)
```

Run: `python -m pytest tests/test_core.py -q -k "contextual or run_translation"` → FAIL

- [ ] **Step 2: Import + loader.**
  - `from config import (...)` còn `TRANSLATION_GENRE, TRANSLATION_SYSTEM_PROMPT, NORMALIZATION_SYSTEM_PROMPT, VIDEO_PATH`.
  - Thêm `from jobs import open_job, JobError`.
  - `load_glossary()` → `load_glossary(path: str)`; `load_ocr_data()` → `load_ocr_data(path: str)`: trong thân hàm, `GLOSSARY_PATH` / `OCR_DATA_PATH` → `path`. Câu nhắc khi thiếu `glossary.json`: in `"  (Chưa có glossary riêng cho video/series này)"` thay cho `[WARN]`.
  - Xóa dòng `CHAR_PROFILES_PATH = os.path.join(os.path.dirname(GLOSSARY_PATH), "character_profiles.json")`.

- [ ] **Step 3: Hồ sơ theo job/series.**
  - `infer_character_profiles(transcript, glossary)` → `infer_character_profiles(transcript, glossary, profiles_path)`: mọi `CHAR_PROFILES_PATH` trong hàm → `profiles_path`.
  - Sửa comment trộn thành `# Trộn với hồ sơ đã có của video/series này (ưu tiên bản người dùng đã chỉnh)`.
  - `translate_contextual(..., glossary)` → thêm tham số cuối `profiles_path`, truyền vào `infer_character_profiles(transcript, glossary, profiles_path)`.

- [ ] **Step 4: `run_translation(video_path)`.** Thay phần đầu hàm, từ `ocr_data = load_ocr_data()` tới hết khối `if not unique_texts: ... return False`, bằng:

```python
    try:
        job = open_job(video_path)
    except JobError as e:
        print(f"  [ERROR] {e}")
        return False
    ocr_data = load_ocr_data(job.ocr_data)
    if not ocr_data:
        return False
    if ocr_data.get("complete") is False:
        print("  [ERROR] OCR của video này chưa chạy xong (bị ngắt?) — chạy tiếp ocr_pass.py trước.")
        return False

    glossary = load_glossary(job.glossary)
    unique_texts = ocr_data.get("unique_texts", [])

    if not unique_texts:
        print("  [ERROR] Không có text nào để dịch!")
        return False
```

Tiếp theo:
- `translate_contextual(text_segments, normalized_mapping, texts_to_translate, glossary)` → thêm `, job.profiles`;
- `write_json_atomic(TRANSLATION_MAP_PATH, translation_map)` → `write_json_atomic(job.translation_map, translation_map)`;
- dòng in `Output     : {TRANSLATION_MAP_PATH}` → `{job.translation_map}`;
- chữ ký `def run_translation():` → `def run_translation(video_path: str):`.

`__main__`:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monggu Translate - Translation Pass")
    parser.add_argument("--input", "-i", type=str, default=VIDEO_PATH, help="Đường dẫn video (đã chạy OCR)")
    args = parser.parse_args()
    sys.exit(0 if run_translation(args.input) else 1)
```

- [ ] **Step 5: Prompt không còn bối cảnh viết cứng.** Trong `config.py`:
  - xóa 4 dòng từ `Ngữ cảnh video: Lời kể chuyện hồi ký...` tới `- Chỉ xưng là "Con" khi...`, cùng dòng trống sau chúng;
  - xóa đoạn cuối gồm 3 dòng `Tôi sẽ gửi một JSON array...` / `Hãy trả về một JSON array...` / `CHỈ trả về JSON array, không thêm giải thích.` (kèm dòng trống trước chúng), để chuỗi kết thúc ở quy tắc 5 bằng `"""`;
  - `TRANSLATION_GENRE = "tình cảm gia đình / hài hước tuổi thơ"` → `TRANSLATION_GENRE = "phim ngắn Trung Quốc"`;
  - xóa dòng `GLOSSARY_PATH = ...`;
  - sửa comment trên `TRANSLATION_GENRE` thành `# Prompt dịch (dùng cho mọi LLM qua llm.ask_json). Định dạng đầu ra do Phase 1 tự nối thêm.`

- [ ] **Step 6: Kiểm chứng.**
  - `Select-String -Path translate_pass.py -Pattern 'OCR_DATA_PATH|TRANSLATION_MAP_PATH|GLOSSARY_PATH|CHAR_PROFILES_PATH'` → không còn.
  - `python -c "import config; p=config.TRANSLATION_SYSTEM_PROMPT; assert 'hồi ký' not in p and 'JSON array' not in p; print('prompt OK')"`.
  - `python -m pytest tests -q` → `23 passed`.

- [ ] **Step 7: Commit**: `feat(dịch): đọc/ghi trong thư mục job; hồ sơ + glossary theo video/series; bỏ prompt viết cứng của video cũ` (+ Co-Authored-By)

---

### Task 4: Render đọc/ghi trong job

**Files:** Modify `script.py`

- [ ] **Step 1: Import.**
  - Bỏ `OUTPUT_PREFIX`, `OCR_DATA_PATH`, `TRANSLATION_MAP_PATH` khỏi `from config import (...)`.
  - Thêm `from jobs import open_job, JobError` sau khối đó.
  - Xóa hàm `get_next_output_path`. Nếu sau đó `glob` hoặc `re` không còn chỗ nào dùng (kiểm tra bằng `Select-String`), xóa luôn import tương ứng.

- [ ] **Step 2: Concat list trong thư mục output, UTF-8.** Trong `concatenate_videos`:
  - thay `list_file = "_concat_list.txt"` bằng `list_file = os.path.join(os.path.dirname(os.path.abspath(output_path)), "_concat_list.txt")`;
  - thay `with open(list_file, "w") as f:` bằng `with open(list_file, "w", encoding="utf-8") as f:`, vì ffmpeg đọc file concat theo UTF-8 và tên job có thể có dấu.

- [ ] **Step 3: Nạp dữ liệu từ job.** Thay khối từ `if not os.path.exists(OCR_DATA_PATH):` tới hết `translation_map = json.load(f)` bằng:

```python
    try:
        job = open_job(args.input)
    except JobError as e:
        print(f"  [ERROR] {e}")
        sys.exit(1)
    if not os.path.exists(job.ocr_data):
        print(f"  [ERROR] Chưa có OCR cho {job.name} — hãy chạy: python ocr_pass.py --input \"{args.input}\"")
        sys.exit(1)
    if not os.path.exists(job.translation_map):
        print(f"  [ERROR] Chưa có bản dịch cho {job.name} — hãy chạy: python translate_pass.py --input \"{args.input}\"")
        sys.exit(1)

    with open(job.ocr_data, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)
    if ocr_data.get("complete") is False:
        print("  [ERROR] OCR của video này chưa chạy xong (bị ngắt?) — chạy tiếp OCR rồi dịch lại.")
        sys.exit(1)

    with open(job.translation_map, "r", encoding="utf-8") as f:
        translation_map = json.load(f)
```

- [ ] **Step 4: Video convert, output, file tạm trong job.**
  - Thay 2 dòng:

```python
    # ocr_pass ghi video đã convert CẠNH video gốc (chỉ lưu basename) → tìm ở đó, không phải thư mục hiện tại
    converted_path = os.path.join(os.path.dirname(os.path.abspath(args.input)), converted_video)
```

bằng:

```python
    # ocr_pass ghi video đã convert vào thư mục job (chỉ lưu basename)
    converted_path = os.path.join(job.dir, converted_video)
```

  - `output_video = get_next_output_path(OUTPUT_PREFIX)` → `output_video = job.output`.
  - `temp_files = [f"_temp_chunk_{i}.mp4" for i in range(num_workers)]` → `temp_files = [os.path.join(job.dir, f"_temp_chunk_{i}.mp4") for i in range(num_workers)]`.
  - Dọn file tạm cả khi worker lỗi. Thay:

```python
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(process_chunk, worker_args)
```

bằng:

```python
    try:
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(process_chunk, worker_args)
    except BaseException:
        for tf in temp_files:  # worker lỗi → dọn chunk dở rồi báo lỗi
            if os.path.exists(tf):
                os.remove(tf)
        raise
```

- [ ] **Step 5: Kiểm chứng.**
  - `Select-String -Path script.py -Pattern 'OCR_DATA_PATH|TRANSLATION_MAP_PATH|OUTPUT_PREFIX|get_next_output_path'` → không còn.
  - `python -m pytest tests -q` → `23 passed`.
  - `python script.py --input test.mp4 --test; $LASTEXITCODE` → `[ERROR] Chưa có dữ liệu cho test.mp4 — hãy chạy OCR trước ...`, `1`.
  - `git status --short` → không tạo `jobs/test`, vì `create=False`.
- [ ] **Step 6: Commit**: `feat(render): đọc/ghi trong thư mục job; file tạm trong job, luôn được dọn; concat list UTF-8` (+ Co-Authored-By)

---

### Task 5: Pipeline + dọn config + render thử trọn vẹn

**Files:** Modify `pipeline.py`, `config.py`

- [ ] **Step 1: `pipeline.py`.**
  - `from config import OCR_DATA_PATH, TRANSLATION_MAP_PATH, VIDEO_PATH` → `from config import VIDEO_PATH`.
  - Thêm `import json` và `from jobs import open_job, JobError`.
  - Thêm tham số `--series`, cùng mẫu với `ocr_pass`.
  - Docstring có thêm dòng usage: `python pipeline.py --input daosiep4.mp4 --series daosi   # tập của một bộ: dùng chung hồ sơ nhân vật`.
  - Ngay sau các dòng in thông tin đầu, chèn:

```python
    try:
        job = open_job(args.input, args.series, create=True)
    except JobError as e:
        print(f"  [ERROR] {e}")
        sys.exit(1)
    print(f"  Job     : {job.dir}" + (f"  (series: {job.series})" if job.series else ""))
```

  - Khối `--skip-existing` đổi thành:

```python
    if args.skip_existing:
        if run_ocr and os.path.exists(job.ocr_data):
            with open(job.ocr_data, encoding="utf-8") as f:
                if json.load(f).get("complete", True):
                    print("  [SKIP] OCR đã xong cho video này → bỏ qua OCR")
                    run_ocr = False
        if run_tr and os.path.exists(job.translation_map):
            print("  [SKIP] Đã có bản dịch cho video này → bỏ qua dịch")
            run_tr = False
```

  - Khi gọi OCR: `ocr_args = ["--input", args.input]`, rồi thêm `if args.series: ocr_args += ["--series", args.series]`.
  - Khi gọi dịch: `run_step("Pass 2/3 — Translation", "translate_pass.py", ["--input", args.input])`.
  - Cuối pipeline, in thêm `print(f"  Video ra: {job.output}")`.

- [ ] **Step 2: Dọn config.** Xóa các dòng `OCR_DATA_PATH = ...`, `TRANSLATION_MAP_PATH = ...` và `OUTPUT_PREFIX = "output_final"`. Kiểm tra: `Select-String -Path *.py,tests\*.py -Pattern 'OCR_DATA_PATH|TRANSLATION_MAP_PATH|OUTPUT_PREFIX|GLOSSARY_PATH'` → không còn.

- [ ] **Step 3: Render thử trọn vẹn trên job tạm.** Viết lại `$SCRATCH\smoke_render.py`:

```python
"""Render thử trọn vẹn (không API) theo cấu trúc job: JOBS_DIR tạm, dữ liệu OCR giả KHỚP test.mp4."""
import json, os, shutil, subprocess, sys
PROJ = r"D:\monggutranslate\test-gpu copy"
work = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke_jobs")
shutil.rmtree(work, ignore_errors=True)
os.makedirs(work)
override = os.path.join(work, "override.json")
json.dump({"JOBS_DIR": os.path.join(work, "jobs"), "SERIES_DIR": os.path.join(work, "series")},
          open(override, "w", encoding="utf-8"))
os.environ["MONGGU_CONFIG"] = override          # để import jobs/ocr_pass trong script này cũng trỏ vào thư mục tạm
sys.path.insert(0, PROJ)
import jobs, ocr_pass

video = os.path.join(PROJ, "test.mp4")
job = jobs.open_job(video, create=True)
box = [[200, 1000], [520, 1000], [520, 1050], [200, 1050]]
fd = {str(f): [{"text": "你好世界", "box": box, "confidence": 0.99}] for f in range(10, 60)}
ocr = {"video": "test.mp4", "original_video": "test.mp4", "fps": 25.0, "width": 720, "height": 1280,
       "total_frames": 2482, "ocr_interval_frames": 1, "complete": True, "test_mode": False,
       "unique_texts": ["你好世界"], "frame_detections": fd,
       "text_segments": ocr_pass.build_text_segments(fd, 25.0, 1)}
json.dump(ocr, open(job.ocr_data, "w", encoding="utf-8"), ensure_ascii=False)
json.dump({"你好世界": "Xin chào thế giới"}, open(job.translation_map, "w", encoding="utf-8"), ensure_ascii=False)
env = dict(os.environ, PYTHONIOENCODING="utf-8")
r = subprocess.run([sys.executable, os.path.join(PROJ, "script.py"), "--input", video, "--test", "--workers", "2"],
                   cwd=work, env=env)
print("EXIT", r.returncode, "| job:", sorted(os.listdir(job.dir)), "| cwd:", sorted(os.listdir(work)))
```

Run: `python "$SCRATCH\smoke_render.py"`. Kết quả mong đợi:
- `EXIT 0`;
- job gồm `job.json`, `ocr_data.json`, `test_vi.mp4`, `translation_map.json`, và **không** còn `_temp_chunk_*` hay `_concat_list.txt`;
- `ffprobe` trên `test_vi.mp4` cho `aac`, 500 frame;
- trích 1 frame ở giây 1.4 thấy chữ "Xin chào thế giới".

- [ ] **Step 4:** `python -m pytest tests -q` → `23 passed`. `git status --short` → chỉ các file dự kiến.

- [ ] **Step 5: Commit**: `feat(pipeline): --series, --skip-existing theo job; bỏ đường dẫn dữ liệu toàn cục khỏi config` (+ Co-Authored-By)
