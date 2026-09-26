"""
Monggu Translate - Định nghĩa cấu trúc dữ liệu (TypedDict).

Mục đích: tài liệu hóa + hỗ trợ IDE/type-checker bắt lỗi sai key.
Runtime vẫn là dict thuần (TypedDict không thêm chi phí), nên có thể import
và dùng làm annotation ở các pass mà không ảnh hưởng hiệu năng.

Ví dụ:
    from schemas import Detection, Segment
    def ocr_frame(...) -> list[Detection]: ...
"""

from typing import List, TypedDict


class Detection(TypedDict):
    """1 cụm text OCR trên 1 frame (đơn vị trong ocr_data['frame_detections'])."""
    text: str
    box: List[List[float]]   # 4 điểm [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    confidence: float


class KeyframeBBox(TypedDict):
    """bbox của 1 text tại 1 frame cụ thể (để render interpolate vị trí)."""
    frame: int
    bbox: List[float]        # [x1, y1, x2, y2]


class Segment(TypedDict):
    """1 lần xuất hiện liên tục của 1 câu (đơn vị trong ocr_data['text_segments'])."""
    text: str
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    bbox: List[float]                    # union bbox (fallback)
    keyframe_bboxes: List[KeyframeBBox]


class RenderEntry(TypedDict):
    """Entry phụ đề đã giải mã sẵn sàng vẽ trong render loop."""
    box: List[List[float]]
    text: str
    vi_text: str
    x: int
    y: int
    box_height: int
    box_width: int


class RenderJob(TypedDict):
    """1 cụm chữ đã chuẩn bị cho thuật toán chống đè (_layout_subtitle_blocks)."""
    text: str
    cx: float
    max_w: int
    size: int
    y: int
    orig_y: int
