"""
wm_remove.py — Xoá watermark mờ-CHẠY khỏi video (pre-pass trước pipeline dịch).

Ý tưởng (đã kiểm chứng trên newvideo_25fps.mp4):
  * PaddleOCR ở FULL-RES đọc được watermark với độ tin cậy cao, nhưng chỉ ~20%
    số frame (khi nó trôi qua nền dễ đọc). 80% còn lại bị trượt.
  * Watermark có KÍCH THƯỚC cố định và QUỸ ĐẠO trôi mượt.
  => Chạy OCR dày (mỗi frame / mỗi 2 frame), gom mọi lần đọc khớp chuỗi watermark
     làm "anchor", rồi NỘI SUY + làm mượt để có vị trí ở MỌI frame; với khoảng
     trống dài thì "quét hành lang" giữa 2 anchor cho chắc. Sau đó tạo mask và xoá.

Cách dùng:
    # 1) Dò + tạo mask + xoá bằng inpaint (mặc định, không cần GPU thêm)
    python wm_remove.py --input newvideo_25fps.mp4 --output newvideo_clean.mp4

    # 2) Chất lượng cao nhất: xoá bằng ProPainter (cần cài, dùng GPU)
    python wm_remove.py --input newvideo_25fps.mp4 --output newvideo_clean.mp4 \
        --backend propainter --propainter /path/to/ProPainter

    # 3) Chỉ dò + xuất video overlay để kiểm tra mask (không xoá)
    python wm_remove.py --input newvideo_25fps.mp4 --debug --only-detect

Tham số hay dùng:
    --wm "原创@哈加吧"   nội dung watermark (mặc định)
    --every 1            OCR mỗi N frame (1 = mọi frame, chính xác nhất; 2-3 = nhanh hơn)
    --dilate 12          nới mask quanh chữ (px) cho chắc
    --gap-sweep 0.4      khoảng trống (giây) dài hơn mức này thì quét cả hành lang
    --debug              xuất thêm video overlay + vài frame mẫu vào _wm_dbg/

Sau khi có video sạch, đưa nó vào pipeline dịch như bình thường:
    python pipeline.py --input newvideo_clean.mp4
"""

import os
import sys
import argparse
import json
import subprocess
import tempfile
import shutil

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["FLAGS_use_mkldnn"] = "0"

import cv2
import numpy as np

try:
    from config import OCR_LANG, USE_GPU
except Exception:
    OCR_LANG, USE_GPU = "ch", True


# ────────────────────────────── khớp watermark ──────────────────────────────
WM_PIECES = ["原创", "哈加吧", "哈加", "加吧", "@哈", "原创@", "哈加密", "创@哈"]


def _norm(s):
    return "".join(s.split()).lower()


def wm_hit(text, wm):
    t, w = _norm(text), _norm(wm)
    if not t:
        return False
    if w in t or (len(t) >= 4 and t in w):
        return True
    return any(_norm(p) in t for p in WM_PIECES)


# ────────────────────────────── tiện ích video ──────────────────────────────
def ffprobe_info(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
         "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True).stdout
    j = json.loads(out)
    st = j["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    return int(st["width"]), int(st["height"]), fps


# ────────────────────────────── PASS 1: dò anchor ───────────────────────────
def detect_anchors(path, wm, every, min_w):
    from paddleocr import PaddleOCR
    print(f"[1/3] OCR full-res mỗi {every} frame (lang={OCR_LANG}, gpu={USE_GPU})...")
    eng = PaddleOCR(use_angle_cls=True, lang=OCR_LANG, show_log=False, use_gpu=USE_GPU)
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    anchors = {}      # frame_idx -> (cx, cy, w, h)
    sizes = []
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % every == 0:
            try:
                res = eng.ocr(frame, cls=True)
            except Exception:
                res = None
            if res and res[0]:
                best = None
                for line in res[0]:
                    box, (text, conf) = line[0], line[1]
                    if not wm_hit(text, wm):
                        continue
                    xs = [p[0] for p in box]; ys = [p[1] for p in box]
                    w = max(xs) - min(xs); h = max(ys) - min(ys)
                    # ưu tiên lần đọc đầy đủ, tin cậy cao
                    score = conf * (1.0 if w >= min_w else 0.5)
                    if best is None or score > best[0]:
                        cx = (min(xs) + max(xs)) / 2.0
                        cy = (min(ys) + max(ys)) / 2.0
                        best = (score, cx, cy, w, h, conf)
                if best:
                    _, cx, cy, w, h, conf = best
                    anchors[fi] = (cx, cy, w, h)
                    if w >= min_w:
                        sizes.append((w, h))
            if fi % (every * 50) == 0:
                print(f"    frame {fi}/{total}  anchors={len(anchors)}", end="\r")
        fi += 1
    cap.release()
    print(f"\n    Tổng anchor: {len(anchors)}  (trên {fi} frame)")
    if not anchors:
        print("    !! Không thấy watermark nào. Kiểm tra lại --wm hoặc OCR.")
        sys.exit(2)
    if sizes:
        mw = int(np.median([s[0] for s in sizes]))
        mh = int(np.median([s[1] for s in sizes]))
    else:
        mw, mh = 200, 48
    print(f"    Kích thước watermark (trung vị): {mw} x {mh}")
    return anchors, fi, (mw, mh)


# ────────────────── PASS 2: quỹ đạo + mask cho mọi frame ─────────────────────
def build_masks(anchors, n_frames, wm_size, W, H, dilate, gap_sweep_frames):
    af = sorted(anchors.keys())
    axs = np.array([anchors[f][0] for f in af])
    ays = np.array([anchors[f][1] for f in af])
    af = np.array(af)
    TW, TH = wm_size
    half_w, half_h = TW / 2 + dilate, TH / 2 + dilate

    # nội suy vị trí cho mọi frame
    allf = np.arange(n_frames)
    cx = np.interp(allf, af, axs)
    cy = np.interp(allf, af, ays)
    # làm mượt nhẹ (cửa sổ 5) để bớt giật
    k = 5
    ker = np.ones(k) / k
    cx = np.convolve(cx, ker, mode="same")
    cy = np.convolve(cy, ker, mode="same")

    # đánh dấu frame nằm trong "khoảng trống dài" -> cần quét hành lang
    gap = np.zeros(n_frames, bool)
    for i in range(len(af) - 1):
        if af[i + 1] - af[i] > gap_sweep_frames:
            gap[af[i]:af[i + 1] + 1] = True

    def mask_for(fi):
        m = np.zeros((H, W), np.uint8)
        x, y = cx[fi], cy[fi]
        cv2.rectangle(m, (int(x - half_w), int(y - half_h)),
                      (int(x + half_w), int(y + half_h)), 255, -1)
        if gap[fi]:
            # quét hành lang: nối box dọc theo đoạn nội suy quanh frame này
            for dd in range(-6, 7):
                j = min(max(fi + dd, 0), n_frames - 1)
                cv2.rectangle(m, (int(cx[j] - half_w), int(cy[j] - half_h)),
                              (int(cx[j] + half_w), int(cy[j] + half_h)), 255, -1)
        return m

    return mask_for, (cx, cy)


# ────────────────────────────── PASS 3: xoá ─────────────────────────────────
def remove_inpaint(path, out_path, mask_for, debug_dir=None, radius=6):
    print("[3/3] Xoá bằng cv2.inpaint (TELEA)...")
    W, H, fps = ffprobe_info(path)
    cap = cv2.VideoCapture(path)
    tmp = out_path + ".noaudio.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    dbg = None
    if debug_dir:
        dbg = cv2.VideoWriter(os.path.join(debug_dir, "overlay.mp4"),
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        m = mask_for(fi)
        clean = cv2.inpaint(frame, m, radius, cv2.INPAINT_TELEA)
        vw.write(clean)
        if dbg is not None:
            ov = frame.copy()
            ov[m > 0] = (0, 0, 255)
            ov = cv2.addWeighted(frame, 0.6, ov, 0.4, 0)
            dbg.write(ov)
            if fi % 200 == 0:
                cv2.imwrite(os.path.join(debug_dir, f"f{fi:05d}.png"),
                            np.hstack([frame, clean]))
        fi += 1
    cap.release(); vw.release()
    if dbg is not None:
        dbg.release()
    mux_audio(tmp, path, out_path)
    os.remove(tmp)
    print(f"    Xong -> {out_path}")


def remove_propainter(path, out_path, mask_for, n_frames, propainter_dir, debug_dir=None):
    print("[3/3] Xoá bằng ProPainter (GPU, flow-guided)...")
    if not propainter_dir or not os.path.isdir(propainter_dir):
        print("    !! Chưa chỉ --propainter <thư mục ProPainter đã clone>.")
        print("       Cài: git clone https://github.com/sczhou/ProPainter")
        print("            cd ProPainter && pip install -r requirements.txt")
        sys.exit(3)
    W, H, fps = ffprobe_info(path)
    work = tempfile.mkdtemp(prefix="wm_pp_")
    fdir = os.path.join(work, "frames"); mdir = os.path.join(work, "masks")
    os.makedirs(fdir); os.makedirs(mdir)
    print("    Xuất frame + mask...")
    cap = cv2.VideoCapture(path); fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(fdir, f"{fi:05d}.png"), frame)
        cv2.imwrite(os.path.join(mdir, f"{fi:05d}.png"), mask_for(fi))
        fi += 1
    cap.release()
    # gọi ProPainter (đọc thư mục frame + thư mục mask)
    cmd = [sys.executable, os.path.join(propainter_dir, "inference_propainter.py"),
           "--video", fdir, "--mask", mdir, "--output", work,
           "--save_frames", "--fp16"]
    print("    Chạy:", " ".join(cmd))
    subprocess.run(cmd, cwd=propainter_dir, check=True)
    # ProPainter ghi kết quả; tìm thư mục frame đã sửa rồi ghép lại
    # (tùy phiên bản; mặc định nằm trong <output>/<tên>/frames hoặc inpaint_out)
    cand = []
    for root, _, files in os.walk(work):
        if any(f.endswith(".png") for f in files) and root not in (fdir, mdir):
            cand.append(root)
    if not cand:
        print("    !! Không thấy frame kết quả của ProPainter trong", work)
        sys.exit(4)
    res_dir = max(cand, key=lambda d: len([f for f in os.listdir(d) if f.endswith('.png')]))
    tmp = out_path + ".noaudio.mp4"
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i",
                    os.path.join(res_dir, "%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", tmp], check=True)
    mux_audio(tmp, path, out_path)
    os.remove(tmp); shutil.rmtree(work, ignore_errors=True)
    print(f"    Xong -> {out_path}")


def mux_audio(video_noaudio, src_with_audio, out_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_noaudio, "-i", src_with_audio,
         "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac",
         "-shortest", out_path],
        capture_output=True)
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        shutil.copy(video_noaudio, out_path)   # nguồn không có audio


# ────────────────────────────────── main ────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--wm", default="原创@哈加吧")
    ap.add_argument("--every", type=int, default=1, help="OCR mỗi N frame")
    ap.add_argument("--min-w", type=int, default=140, help="bề rộng tối thiểu coi là đọc đầy đủ")
    ap.add_argument("--dilate", type=int, default=12)
    ap.add_argument("--gap-sweep", type=float, default=0.4, help="giây; gap dài hơn thì quét hành lang")
    ap.add_argument("--backend", choices=["inpaint", "propainter"], default="inpaint")
    ap.add_argument("--propainter", default=None, help="thư mục ProPainter đã clone")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--only-detect", action="store_true", help="chỉ dò + overlay, không xoá")
    ap.add_argument("--cache", default=None, help="đọc/ghi anchors json (bỏ qua OCR nếu đã có)")
    args = ap.parse_args()

    out = args.output or os.path.splitext(args.input)[0] + "_clean.mp4"
    W, H, fps = ffprobe_info(args.input)
    gap_sweep_frames = max(1, int(args.gap_sweep * fps))

    # PASS 1 (có cache)
    if args.cache and os.path.exists(args.cache):
        print(f"[1/3] Dùng cache anchors: {args.cache}")
        c = json.load(open(args.cache, encoding="utf-8"))
        anchors = {int(k): tuple(v) for k, v in c["anchors"].items()}
        n_frames = c["n_frames"]; wm_size = tuple(c["wm_size"])
    else:
        anchors, n_frames, wm_size = detect_anchors(args.input, args.wm, args.every, args.min_w)
        if args.cache:
            json.dump({"anchors": {str(k): list(v) for k, v in anchors.items()},
                       "n_frames": n_frames, "wm_size": list(wm_size)},
                      open(args.cache, "w", encoding="utf-8"))

    # PASS 2
    print("[2/3] Dựng quỹ đạo + mask cho mọi frame...")
    mask_for, _ = build_masks(anchors, n_frames, wm_size, W, H,
                              args.dilate, gap_sweep_frames)

    dbg_dir = None
    if args.debug or args.only_detect:
        dbg_dir = "_wm_dbg"; os.makedirs(dbg_dir, exist_ok=True)

    if args.only_detect:
        print("[3/3] --only-detect: chỉ xuất overlay để kiểm tra mask...")
        cap = cv2.VideoCapture(args.input)
        vw = cv2.VideoWriter(os.path.join(dbg_dir, "overlay.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            m = mask_for(fi)
            ov = frame.copy(); ov[m > 0] = (0, 0, 255)
            vw.write(cv2.addWeighted(frame, 0.6, ov, 0.4, 0))
            fi += 1
        cap.release(); vw.release()
        print(f"    Xong -> {dbg_dir}/overlay.mp4  (kiểm tra mask đỏ có ôm đúng watermark không)")
        return

    # PASS 3
    if args.backend == "propainter":
        remove_propainter(args.input, out, mask_for, n_frames, args.propainter, dbg_dir)
    else:
        remove_inpaint(args.input, out, mask_for, dbg_dir)


if __name__ == "__main__":
    main()
