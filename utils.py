"""
Monggu Translate - Tiện ích dùng chung: logging, progress bar, ghi JSON an toàn.

- get_logger(): logger có timestamp + level, thay dần cho print() rải rác.
- progress_iter(): dùng tqdm nếu cài, không thì trả về iterable gốc (no-op).
- write_json_atomic(): ghi JSON qua file tạm + os.replace, lỗi giữa chừng không hỏng file cũ.
"""

import contextlib
import json
import logging
import os
import shutil
import sys
import tempfile


def get_logger(name: str = "monggu", level: int = logging.INFO) -> logging.Logger:
    """Trả về logger đã cấu hình (idempotent — không gắn handler trùng)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def progress_iter(iterable, total=None, desc: str = ""):
    """Bọc iterable bằng tqdm nếu khả dụng; nếu không thì trả về nguyên bản."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc)
    except ImportError:
        return iterable


def write_json_atomic(path: str, obj) -> None:
    """Ghi JSON an toàn: ghi ra file tạm cùng thư mục rồi os.replace.
    Lỗi giữa chừng (mất điện, Ctrl+C, object không serialize được) → file cũ còn nguyên."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)),
                               prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(tmp, path)
        except PermissionError:
            # Windows: file đích đang được chương trình khác mở → replace bị chặn,
            # nhưng ghi đè nội dung vẫn được. Giữ dữ liệu quan trọng hơn tính atomic.
            shutil.copyfile(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            os.remove(tmp)
