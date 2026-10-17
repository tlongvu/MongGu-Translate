"""
Monggu Translate - Tiện ích dùng chung.

- write_json_atomic(): ghi JSON qua file tạm + os.replace, lỗi giữa chừng không hỏng file cũ.
- imwrite_any() / imread_gray(): ghi/đọc ảnh với đường dẫn có dấu / chữ Trung (cv2.imwrite/imread không làm được trên Windows).
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
