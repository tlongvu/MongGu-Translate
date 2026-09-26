"""
Monggu Translate - Tiện ích dùng chung: logging + progress bar.

- get_logger(): logger có timestamp + level, thay dần cho print() rải rác.
- progress_iter(): dùng tqdm nếu cài, không thì trả về iterable gốc (no-op).
"""

import logging
import sys


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
