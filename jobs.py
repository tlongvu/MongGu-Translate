"""
Monggu Translate - Thư mục job: mỗi video một thư mục jobs/<tên>/ chứa mọi dữ liệu của video đó.
Video thuộc series thì hồ sơ truyện (story.json) nằm ở series/<bộ>/, dùng chung giữa các tập.
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
    if series is not None:
        series = series.strip()
        if not series or os.path.basename(series) != series or any(c in series for c in ':?*<>|"'):
            raise JobError(f"Tên series không hợp lệ: {series!r} — chỉ dùng chữ, số, dấu cách, '-', '_'")
    name = os.path.splitext(os.path.basename(video_path))[0]
    d = os.path.join(JOBS_DIR, name)
    meta_path = os.path.join(d, "job.json")
    fp = fingerprint(video_path)
    changed = False
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if not isinstance(meta, dict):
                raise ValueError("không phải JSON object")
        except (OSError, ValueError) as e:
            raise JobError(f"jobs/{name}/job.json hỏng ({e}) — sửa hoặc xóa file đó rồi chạy lại OCR.")
        if meta.get("fingerprint") != fp:
            if meta.get("video") == os.path.abspath(video_path):
                raise JobError(f"{video_path} đã thay đổi so với lúc OCR (bị encode / cắt lại?) — "
                               f"xóa thư mục jobs/{name}/ để làm lại từ đầu.")
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
        lines=os.path.join(d, "lines.json"),
        output=os.path.join(d, f"{name}_vi.mp4"),
        story=os.path.join(shared, "story.json"),
    )
