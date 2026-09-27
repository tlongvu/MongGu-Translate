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
