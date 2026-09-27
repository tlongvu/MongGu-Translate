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
