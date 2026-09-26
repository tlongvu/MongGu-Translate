"""
Monggu Translate - Tiện ích dùng chung.

- write_json_atomic(): ghi JSON qua file tạm + os.replace, lỗi giữa chừng không hỏng file cũ.
"""

import contextlib
import json
import os
import shutil
import tempfile


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
