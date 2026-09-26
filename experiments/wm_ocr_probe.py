"""
wm_ocr_probe.py — Chẩn đoán: PaddleOCR có "thấy" watermark mờ-chạy không?

Chạy trên máy có PaddleOCR + GPU (cùng môi trường pipeline).

    python wm_ocr_probe.py --input newvideo_25fps.mp4
    python wm_ocr_probe.py --input newvideo_25fps.mp4 --start 4 --end 8 --n 30
    python wm_ocr_probe.py --input newvideo_25fps.mp4 --wm "原创@哈加吧"

Nó sẽ:
  1. Trích N frame trong khoảng [--start, --end] giây (mặc định quét cả video).
  2. Chạy OCR 2 lần mỗi frame: full-res và downscale theo --resize (mặc định 0.5,
     đúng như pipeline) — để xem downscale có làm mất chữ mờ không.
  3. KHÔNG lọc confidence — in TẤT CẢ detection: text | confidence | tâm box.
  4. Đánh dấu [WM] mọi detection khớp một phần chuỗi watermark.
  5. Lưu frame chú thích vào _wm_probe_out/ (xanh = text thường, đỏ = khớp WM).
  6. In tổng kết: WM bị bắt ở bao nhiêu frame, dải confidence, full-res vs downscale.

Đọc kết quả:
  - Nếu [WM] xuất hiện đều với conf >= ~0.3  -> sửa pipeline (hạ ngưỡng cho mẫu WM
    + thêm vào SUPPRESS_TEXTS) là xoá được, tự động bám theo nó di chuyển.
  - Nếu [WM] gần như không bao giờ xuất hiện -> OCR không thấy, phải dùng
    template-matching/ProPainter (báo lại để mình làm tiếp).
"""

import os
import sys
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["FLAGS_use_mkldnn"] = "0"

import cv2
import numpy as np

# Lấy cấu hình OCR giống pipeline nếu có; nếu không, dùng mặc định an toàn.
try:
    from config import OCR_LANG, USE_GPU, OCR_RESIZE_FACTOR
except Exception:
    OCR_LANG, USE_GPU, OCR_RESIZE_FACTOR = "ch", True, 0.5

from paddleocr import PaddleOCR


def norm(s: str) -> str:
    return "".join(s.split()).lower()


def wm_hit(text: str, wm: str) -> bool:
    """Khớp 'mềm': trùng cả chuỗi, hoặc chứa cụm con đặc trưng của watermark."""
    t = norm(text)
    w = norm(wm)
    if not t:
        return False
    if w in t or t in w:
        return True
    # các cụm con đặc trưng (chỉnh theo watermark của bạn)
    for piece in ["原创", "哈加吧", "哈加", "加吧", "@哈", "原创@"]:
        if piece and norm(piece) in t:
            return True
    return False


def run_ocr(engine, img):
    """Trả list (text, conf, box[4 điểm]) — KHÔNG lọc gì."""
    out = []
    try:
        result = engine.ocr(img, cls=True)
    except Exception as e:
        print("   ! OCR lỗi:", e)
        return out
    if not result or not result[0]:
        return out
    for line in result[0]:
        box, (text, conf) = line[0], line[1]
        out.append((text, float(conf), box))
    return out


def draw(img, dets, wm, scale=1.0):
    vis = img.copy()
    for text, conf, box in dets:
        pts = np.array([[int(p[0] / scale), int(p[1] / scale)] for p in box], np.int32)
        hit = wm_hit(text, wm)
        color = (0, 0, 255) if hit else (0, 200, 0)
        cv2.polylines(vis, [pts], True, color, 2 if hit else 1)
        if hit:
            x, y = pts[0]
            cv2.putText(vis, f"{conf:.2f}", (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--wm", default="原创@哈加吧", help="nội dung watermark cần dò")
    ap.add_argument("--start", type=float, default=0.0, help="giây bắt đầu")
    ap.add_argument("--end", type=float, default=-1.0, help="giây kết thúc (-1 = hết video)")
    ap.add_argument("--n", type=int, default=24, help="số frame lấy mẫu")
    ap.add_argument("--resize", type=float, default=OCR_RESIZE_FACTOR,
                    help="hệ số downscale để so sánh (giống pipeline)")
    ap.add_argument("--outdir", default="_wm_probe_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("Không mở được video:", args.input)
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps if fps else 0
    end = args.end if args.end > 0 else dur
    start = max(0.0, args.start)
    times = np.linspace(start, max(start, end - 0.01), args.n)
    print(f"Video: {args.input}  fps={fps:.2f}  dur={dur:.1f}s  frames={total}")
    print(f"Lấy {args.n} mẫu trong [{start:.1f}, {end:.1f}]s | watermark='{args.wm}'")
    print(f"OCR lang={OCR_LANG} gpu={USE_GPU} | so full-res vs downscale x{args.resize}\n")

    print("Khởi tạo PaddleOCR...")
    engine = PaddleOCR(use_angle_cls=True, lang=OCR_LANG, show_log=False, use_gpu=USE_GPU)

    stat = {
        "full": {"frames_with_wm": 0, "confs": []},
        "small": {"frames_with_wm": 0, "confs": []},
    }
    frames_done = 0
    anchors = []          # [{t, frame_idx, conf, text, box, cx, cy, w, h}]
    patch_dir = os.path.join(args.outdir, "patches")
    os.makedirs(patch_dir, exist_ok=True)

    for i, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        frames_done += 1
        print(f"\n=== frame {i:02d}  t={t:.2f}s ===")

        # --- full-res ---
        dets_full = run_ocr(engine, frame)
        wm_full = [d for d in dets_full if wm_hit(d[0], args.wm)]
        for text, conf, box in dets_full:
            tag = " [WM]" if wm_hit(text, args.wm) else ""
            cx = int(np.mean([p[0] for p in box]))
            cy = int(np.mean([p[1] for p in box]))
            print(f"   full  {conf:5.2f}  ({cx:4d},{cy:4d})  {text!r}{tag}")
        if wm_full:
            stat["full"]["frames_with_wm"] += 1
            stat["full"]["confs"] += [d[1] for d in wm_full]
            # lưu anchor + cắt patch watermark thật (để dựng template chuẩn)
            for text, conf, box in wm_full:
                xs = [p[0] for p in box]; ys = [p[1] for p in box]
                x0, y0 = int(min(xs)), int(min(ys))
                x1, y1 = int(max(xs)), int(max(ys))
                anchors.append({
                    "t": round(float(t), 3), "frame_idx": i, "conf": round(conf, 3),
                    "text": text, "box": [[float(p[0]), float(p[1])] for p in box],
                    "cx": (x0 + x1) // 2, "cy": (y0 + y1) // 2,
                    "w": x1 - x0, "h": y1 - y0,
                })
                pad = 6
                patch = frame[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
                if patch.size:
                    cv2.imwrite(os.path.join(patch_dir,
                                f"anchor_t{t:.2f}_c{conf:.2f}.png"), patch)

        # --- downscale (giống pipeline) ---
        small = cv2.resize(frame, (0, 0), fx=args.resize, fy=args.resize)
        dets_small = run_ocr(engine, small)
        wm_small = [d for d in dets_small if wm_hit(d[0], args.wm)]
        for text, conf, box in wm_small:
            cx = int(np.mean([p[0] for p in box]) / args.resize)
            cy = int(np.mean([p[1] for p in box]) / args.resize)
            print(f"   small {conf:5.2f}  ({cx:4d},{cy:4d})  {text!r} [WM]")
        if wm_small:
            stat["small"]["frames_with_wm"] += 1
            stat["small"]["confs"] += [d[1] for d in wm_small]

        # lưu ảnh chú thích full-res (vẽ cả box small đã scale về full)
        vis = draw(frame, dets_full, args.wm, scale=1.0)
        vis = draw(vis, wm_small, args.wm, scale=args.resize)
        cv2.imwrite(os.path.join(args.outdir, f"probe_{i:02d}_t{t:.1f}.png"), vis)

    cap.release()

    def summ(key):
        s = stat[key]
        confs = s["confs"]
        if confs:
            return (f"{key:5s}: thấy WM ở {s['frames_with_wm']}/{frames_done} frame | "
                    f"conf min={min(confs):.2f} max={max(confs):.2f} mean={np.mean(confs):.2f}")
        return f"{key:5s}: KHÔNG thấy watermark ở frame nào."

    # ghi anchors.json (mình đọc trực tiếp được trong cùng thư mục)
    import json
    with open("wm_anchors.json", "w", encoding="utf-8") as f:
        json.dump({"input": args.input, "wm": args.wm, "fps": fps,
                   "total_frames": total, "anchors": anchors}, f,
                  ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("TỔNG KẾT")
    print(summ("full"))
    print(summ("small"))
    print(f"Ảnh chú thích đã lưu trong: {args.outdir}/")
    print(f"Anchors: {len(anchors)} cái -> wm_anchors.json ; patch -> {patch_dir}/")
    print("ANCHORS (t | cx,cy | w x h | conf):")
    for a in anchors:
        print(f"   t={a['t']:6.2f}  ({a['cx']:4d},{a['cy']:4d})  {a['w']:3d}x{a['h']:3d}  conf={a['conf']:.2f}")
    print("=" * 60)
    print("Đã ghi wm_anchors.json + patches. Báo mình 'xong' để mình đọc & dựng tracker.")


if __name__ == "__main__":
    main()
