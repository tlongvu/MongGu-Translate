"""
Monggu Translate - Dò watermark CHẠY (bước giữa OCR và dịch; pipeline.py tự gọi).

Watermark chạy = chữ ID nhỏ bán trong suốt trôi khắp khung hình (vd "原创@哈加密"), mỗi kênh một chữ.
OCR chỉ đọc được nó ở vài % số frame → render cũ xóa lúc có lúc không (nhấp nháy). Bước này:
  1. tự nhận chữ watermark từ ocr_data: có mặt suốt video, vị trí tản rộng, cỡ box không đổi;
  2. dựng mẫu nét chữ (trung vị gradient các lần OCR đọc đủ, đã căn) + mask hình nét chữ;
  3. dò mẫu trên toàn khung MỌI frame → nối đỉnh thành đoạn quỹ đạo → mốc = chuỗi đoạn dài nhất nối được bằng
     chuyển động khả thi; đoạn nằm xa chỗ OCR đọc đúng chữ watermark cùng lúc (chữ tĩnh khớp nhầm) bị loại;
  4. lấp khoảng giữa các mốc: nội suy Hermite (khớp vận tốc 2 đầu) + dò lại tại chỗ; lệch → chỗ đổi hướng.
Ra: jobs/<tên>/wm_track.json (vị trí từng frame), wm_mask.png, wm_preview.png. Render đọc để xóa ở mọi frame.

Chạy tay:  python wm_track.py --input video.mp4     (cần OCR trước)
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):   # in tiếng Việt khi chạy tay trên Windows
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

from config import SUPPRESS_TEXTS, VIDEO_PATH, WM_TRACK
from jobs import JobError, open_job, working_video
from utils import imwrite_any, write_json_atomic

# ── Tự nhận watermark chạy (số liệu chốt ngưỡng: spec 2026-09-27 mục 4.1) ──
MIN_TEXT_LEN = 3          # chữ watermark ≥ 3 ký tự (kênh "粥咪咪"); ngắn hơn dễ trùng nhãn tên nhân vật
MIN_READS = 10            # xuất hiện ≥ 10 lần trong ocr_data
MIN_PRESENCE = 0.40       # có mặt ≥ 40% số đoạn 10 giây (watermark 57–78%; nhãn nhân vật ≤ 21%)
MIN_SPREAD_FRAC = 0.05    # tâm box tản ≥ 5% đường chéo khung (watermark tĩnh tản 4–6 px)
MAX_WIDTH_CV = 0.15       # bề rộng box dao động ≤ 15% (watermark có cỡ cố định)
LOW_PRESENCE = 0.10       # chữ có dấu hiệu watermark (_wm_like): watermark mờ, OCR hiếm đọc được (明月猫meme: 14–17%)
_WM_LIKE = re.compile(r"meme|原创|^原文|^《.+》$", re.I)
_KNOWN_WM = {"".join(s.split()) for s in SUPPRESS_TEXTS}
PARTNER_MIN_READS = 3     # dòng chạy kèm watermark: ≥ 3 lần đọc chung frame với watermark
PARTNER_JITTER = 8        # ... độ lệch so với watermark dao động ≤ 8 px (cùng một khối)

# ── Mẫu + bám (số liệu: spec mục 2, 4.2, 4.3) ──
FULL_READ_TOL = 0.08      # lần OCR "đọc đủ chữ": bề rộng lệch ≤ 8% trung vị
MAX_TEMPLATE_READS = 30   # dựng mẫu từ tối đa 30 lần đọc đủ, rải đều theo thời gian
MIN_TEMPLATE_READS = 5
SURE = 0.45               # điểm khớp toàn khung đủ chắc để làm mốc
WEAK = 0.30               # đỉnh yếu: chỉ dùng khi thuộc một đoạn quỹ đạo có ít nhất 1 đỉnh chắc
MIN_TRACKLET = 3          # đoạn quỹ đạo ngắn hơn 3 đỉnh = nhiễu
VETO_PX = 30              # OCR đọc đúng chữ watermark trong lúc đoạn đang chạy mà cách đoạn > 30 px → đoạn khớp nhầm
LOCAL_OK = 0.30           # điểm khớp tại chỗ (quanh điểm nội suy) chấp nhận được
MAX_SPEED = 100           # px/giây — watermark đo được trôi 8–50 px/s
ENERGY_GATE = 0.35        # cửa sổ có gradient TB < 35% của mẫu = nền trơn → NCC khuếch đại nhiễu, không tin
MIN_ANCHORS = 20
LOCAL_R = 24              # bán kính dò lại quanh điểm nội suy (px, full-res)
SPLIT_DEV = 8             # vị trí dò lại lệch khỏi đường dự đoán > 8 px → chỗ đổi hướng
MIN_SMALL_W = 48          # mẫu nửa độ phân giải hẹp hơn thế → dò toàn khung ở độ phân giải gốc
MAX_GUESS = 25            # chỉ dùng (xóa / lọc dòng) vị trí cách frame có bằng chứng ≤ 25 frame: video 奇异之修
                          # (watermark mờ, 20% có bằng chứng) đoán xa hơn thì lệch vài chục px → inpaint làm nhòe hình
_PAD, _MARGIN = 8, 4      # lề để căn patch (±8 px) / lề giữ quanh mẫu cho mask (4 px)

# ── Dòng OCR nằm trên watermark: bước dịch bỏ luôn, dù model chuẩn hóa giữ lại (vd "粥米咪" → "Châu Mễ Mễ") ──
LINE_MARGIN = 10          # nới khung watermark ±10 px (bám lệch ≤ ~10 px, box OCR rung vài px)
LINE_INSIDE = 0.6         # ≥ 60% diện tích box dòng nằm trong khung → keyframe đó là watermark


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
        min_presence = LOW_PRESENCE if (_WM_LIKE.search(t) or t in _KNOWN_WM) else MIN_PRESENCE
        if presence >= min_presence and spread >= min_spread and width_cv <= MAX_WIDTH_CV:
            if best is None or len(reads) > len(best["reads"]):
                best = {"text": t, "reads": sorted(reads)}
    return best and _with_partners(best, groups, min_spread)


def _with_partners(wm, groups, min_spread):
    """Dòng chữ khác luôn chạy kèm watermark ở độ lệch cố định (vd "《奇异之修》" nằm trên "明月猫meme") → cùng một
    khối: các lần đọc đủ chữ của watermark đổi thành hộp bao cả khối, để dựng mẫu / bám / xóa / lọc dòng theo khối.
    Không có dòng nào chạy kèm → giữ nguyên."""
    w_med = float(np.median([r[3] for r in wm["reads"]]))
    full = {r[0]: r for r in wm["reads"] if abs(r[3] - w_med) <= FULL_READ_TOL * w_med}
    mh = float(np.median([r[4] for r in full.values()]))
    box, names = [-w_med / 2, -mh / 2, w_med / 2, mh / 2], [wm["text"]]   # khối, tương đối so với tâm watermark
    for t, rs in groups.items():
        rel = [(r[1] - full[r[0]][1], r[2] - full[r[0]][2], r[3], r[4], full[r[0]][1], full[r[0]][2])
               for r in rs if r[0] in full]
        # mảnh đọc thiếu ("明月猫") / đọc gộp của chính watermark: so theo chữ — hộp OCR cao hơn nét chữ nhiều nên
        # 2 dòng thật cũng chồng nhau quá nửa (video 奇异之修: hộp cao ~76 px, 2 dòng cách nhau 28 px)
        if t in wm["text"] or wm["text"] in t or len(rel) < PARTNER_MIN_READS:
            continue
        a = np.array(rel)
        if a[:, :2].std(axis=0).max() > PARTNER_JITTER or np.hypot(*a[:, 4:].std(axis=0)) < min_spread:
            continue                        # độ lệch không cố định, hoặc chưa thấy chúng cùng di chuyển
        dx, dy, pw, ph = np.median(a[:, :4], axis=0)
        box =[min(box[0], dx - pw / 2), min(box[1], dy - ph / 2), max(box[2], dx + pw / 2), max(box[3], dy + ph / 2)]
        names.append(t)
    if len(names) == 1:
        return wm
    ox, oy, bw, bh = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]
    return {"text": " + ".join(names),
            "reads": [(f, cx + ox, cy + oy, bw, bh, c) for f, cx, cy, _, _, c in sorted(full.values())]}


def _grad(gray):
    """Độ lớn gradient (Sobel) — bám theo nét chữ, không phụ thuộc nền sáng hay tối."""
    g = gray.astype(np.float32)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def _ncc(img, tpl):
    r = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    return np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)


def _gated_ncc(g, tpl):
    """NCC của mẫu trên ảnh gradient g. Vị trí nền trơn (gradient TB trong cửa sổ < ENERGY_GATE × của mẫu) = -1:
    ở đó NCC tự khuếch đại nhiễu thành 'giống chữ' (khớp nhầm duy nhất gặp trên video thật)."""
    th, tw = tpl.shape
    r = _ncc(g, tpl)
    energy = cv2.boxFilter(g, -1, (tw, th), anchor=(0, 0), normalize=True)[:r.shape[0], :r.shape[1]]
    r[energy < ENERGY_GATE * float(tpl.mean())] = -1.0
    return r


def _median_patch(grads, centers, tw, th):
    """Cắt patch gradient quanh các tâm, căn thẳng hàng (2 lượt, ±_PAD) rồi lấy trung vị: nền mỗi frame một
    khác nên trung vị chỉ giữ lại nét chữ. → mảng (th+2*_MARGIN, tw+2*_MARGIN), hoặc None nếu quá ít patch."""
    patches = []
    for g, (cx, cy) in zip(grads, centers):
        # Watermark hay chạy sát mép khung (kênh 粥咪咪): đệm _PAD px "không có nét" quanh khung để lề căn không
        # bắt hộp chữ phải cách mép — chỉ cần hộp chữ nằm trong khung. Tọa độ trong g đệm = tọa độ gốc + _PAD.
        g = cv2.copyMakeBorder(g, _PAD, _PAD, _PAD, _PAD, cv2.BORDER_CONSTANT, value=0)
        x0, y0 = int(round(cx - tw / 2)), int(round(cy - th / 2))
        if x0 >= 0 and y0 >= 0 and x0 + tw + 2 * _PAD <= g.shape[1] and y0 + th + 2 * _PAD <= g.shape[0]:
            patches.append(g[y0:y0 + th + 2 * _PAD, x0:x0 + tw + 2 * _PAD])
    if len(patches) < MIN_TEMPLATE_READS:
        return None
    ref = patches[0][_PAD:-_PAD, _PAD:-_PAD]
    big = None
    for _ in range(2):   # lượt 2 căn theo trung vị của lượt 1 (không phụ thuộc patch đầu có tốt không)
        stack = []
        for p in patches:
            _, _, _, (dx, dy) = cv2.minMaxLoc(_ncc(p, ref))
            if _MARGIN <= dx <= 2 * _PAD - _MARGIN and _MARGIN <= dy <= 2 * _PAD - _MARGIN:
                stack.append(p[dy - _MARGIN:dy + th + _MARGIN, dx - _MARGIN:dx + tw + _MARGIN])
        if len(stack) < MIN_TEMPLATE_READS:
            return None
        big = np.median(np.stack(stack), axis=0).astype(np.float32)
        ref = big[_MARGIN:-_MARGIN, _MARGIN:-_MARGIN]
    return big


def build_template(video_path, wm):
    """→ (mẫu full-res, mẫu để dò toàn khung, tỉ lệ dò, mask uint8 nét chữ), hoặc None nếu quá ít lần đọc đủ."""
    reads = wm["reads"]
    w_med = float(np.median([r[3] for r in reads]))
    h_med = float(np.median([r[4] for r in reads]))
    full = [r for r in reads if abs(r[3] - w_med) <= FULL_READ_TOL * w_med]
    if len(full) < MIN_TEMPLATE_READS:
        return None
    groups = np.array_split(np.arange(len(full)), min(MAX_TEMPLATE_READS, len(full)))
    picked = [max((full[i] for i in g), key=lambda r: r[5]) for g in groups if len(g)]
    tw, th = int(round(w_med)), int(round(h_med))
    scale = 0.5 if tw * 0.5 >= MIN_SMALL_W else 1.0
    stw, sth = int(round(tw * scale)), int(round(th * scale))
    cap = cv2.VideoCapture(video_path)
    g_full, g_small, centers = [], [], []
    for f, cx, cy, *_ in picked:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g_full.append(_grad(gray))
        g_small.append(_grad(cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
                       if scale != 1.0 else g_full[-1])
        centers.append((cx, cy))
    cap.release()
    big = _median_patch(g_full, centers, tw, th)
    big_s = _median_patch(g_small, [(x * scale, y * scale) for x, y in centers], stw, sth)
    if big is None or big_s is None:
        return None
    mask = (big >= 0.25 * big.max()).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))      # nới 3 px
    return big[_MARGIN:-_MARGIN, _MARGIN:-_MARGIN], big_s[_MARGIN:-_MARGIN, _MARGIN:-_MARGIN], scale, mask


def detect_all(video_path, tpl_small, scale, n_frames):
    """Lượt 1: dò toàn khung mọi frame (tỉ lệ scale) → (pos (n,2) tâm full-res, score (n,)); frame không đọc được: -1."""
    th, tw = tpl_small.shape
    pos = np.zeros((n_frames, 2), np.float32)
    score = np.full(n_frames, -1.0, np.float32)
    cap = cv2.VideoCapture(video_path)
    for f in range(n_frames):
        if f and f % max(1, -(-n_frames // 10)) == 0:   # bước này ~0,5× thời lượng video: báo tiến độ mỗi 10%
            print(f"  [WM] lượt 1/2 (dò toàn khung): {100 * f // n_frames}%", flush=True)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if scale != 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # đệm nửa mẫu "không có nét" quanh khung: watermark chạy sát / lòi ra mép (kênh 粥咪咪 ở mép dưới) vẫn khớp được
        g = cv2.copyMakeBorder(_grad(gray), th // 2, th // 2, tw // 2, tw // 2, cv2.BORDER_CONSTANT, value=0)
        _, s, _, (x, y) = cv2.minMaxLoc(_gated_ncc(g, tpl_small))
        pos[f] = ((x - tw // 2 + tw / 2) / scale, (y - th // 2 + th / 2) / scale)
        score[f] = s
    cap.release()
    return pos, score


def _tracklets(pos, score, fps):
    """Nối các đỉnh ≥ WEAK thành đoạn quỹ đạo: mỗi đỉnh nối vào đoạn đang mở gần nhất mà nó với tới được
    (cách ≤ 1 s, dời ≤ MAX_SPEED); không với tới đoạn nào thì mở đoạn mới. Nhiều đoạn sống song song được
    (watermark thật và chữ tĩnh khớp nhầm xen kẽ nhau từng frame)."""
    cand = np.flatnonzero(score >= WEAK)
    win, vmax = max(1, int(round(fps))), MAX_SPEED / fps
    active, done = [], []
    for i in cand:
        still = []
        for t in active:
            (still if i - t[-1] <= win else done).append(t)
        active = still
        best, best_d = None, None
        for t in active:
            d = float(np.hypot(*(pos[i] - pos[t[-1]])))
            if d <= vmax * (i - t[-1]) + 6 and (best_d is None or d < best_d):
                best, best_d = t, d
        if best is None:
            active.append([int(i)])
        else:
            best.append(int(i))
    return done + active


def _ocr_dist(t, pos, rf, rxy):
    """Khoảng cách từ đoạn t (nội suy tại frame đọc) tới các lần OCR đọc đúng chữ watermark trong khoảng thời gian
    của đoạn. rf/rxy: frame/tâm các lần đọc, sắp theo frame."""
    lo, hi = np.searchsorted(rf, [t[0], t[-1] + 1])
    tf = np.asarray(t, float)
    return np.hypot(np.interp(rf[lo:hi], tf, pos[t, 0]) - rxy[lo:hi, 0],
                    np.interp(rf[lo:hi], tf, pos[t, 1]) - rxy[lo:hi, 1])


def pick_anchors(pos, score, fps, reads=()):
    """Mốc = các đỉnh của chuỗi đoạn quỹ đạo dài nhất (tổng số đỉnh) nối được với nhau bằng chuyển động khả thi.
    Chỉ xét đoạn ≥ MIN_TRACKLET đỉnh, có ít nhất 1 đỉnh chắc (≥ SURE), và không bị OCR phủ nhận: trong lúc đoạn chạy,
    OCR đọc đúng chữ watermark (reads) ở chỗ cách đoạn > VETO_PX → đoạn là chữ tĩnh / nhãn khớp nhầm, kể cả khi nó
    với tới được quỹ đạo. Watermark NHẢY (kênh 粥咪咪 chạy hết mép thì hiện lại ở đầu kia) → cho nối qua bước nhảy
    không khả thi tới đoạn được OCR xác nhận (có lần đọc đúng chữ ngay trên đoạn). Còn hở: vật khớp nhầm xuất hiện lúc
    OCR không đọc được watermark lần nào mà vẫn nối được vào quỹ đạo thì có thể được chọn nếu làm chuỗi thêm đỉnh."""
    vmax = MAX_SPEED / fps
    if reads:   # chỉ tin tâm của lần đọc đủ chữ: hộp rộng / hẹp khác thường = đọc gộp chữ bên cạnh / bị cắt → tâm lệch
        w_med = float(np.median([r[3] for r in reads]))
        reads = [r for r in reads if abs(r[3] - w_med) <= FULL_READ_TOL * w_med]
    rf = np.array([r[0] for r in reads], int)
    rxy = np.array([(r[1], r[2]) for r in reads], float).reshape(-1, 2)
    segs = []
    for t in _tracklets(pos, score, fps):
        if len(t) >= MIN_TRACKLET and max(score[i] for i in t) >= SURE:
            dist = _ocr_dist(t, pos, rf, rxy)
            if not (dist > VETO_PX).any():
                segs.append((t, dist.size > 0))                      # (đoạn, được OCR xác nhận)
    if not segs:
        return []
    segs.sort(key=lambda s: s[0][0])
    confirmed = [c for _, c in segs]
    segs = [t for t, _ in segs]
    ends = [(t[0], t[-1], pos[t[0]], pos[t[-1]]) for t in segs]
    total, prev = [len(t) for t in segs], [-1] * len(segs)
    for j in range(len(segs)):   # ponytail: O(số đoạn²), vài nghìn đoạn vẫn vài giây; cần nhanh hơn thì giới hạn cửa sổ i
        b, pb = ends[j][0], ends[j][2]
        for i in range(j):
            a, pa = ends[i][1], ends[i][3]
            if b > a and total[i] + len(segs[j]) > total[j] and \
                    (confirmed[j] or math.hypot(pb[0] - pa[0], pb[1] - pa[1]) <= vmax * (b - a) + 6):
                total[j], prev[j] = total[i] + len(segs[j]), i
    j, chain = int(np.argmax(total)), []
    while j >= 0:
        chain = segs[j] + chain
        j = prev[j]
    return chain


def _velocity(frames, pts, lo, hi, min_span):
    """Vận tốc (px/frame) = hệ số góc bình phương tối thiểu của các mốc có frame trong [lo, hi];
    None nếu < 2 mốc hoặc chúng trải < min_span frame (quá ngắn → vận tốc nhiễu, nhân lên qua khoảng dài thì văng)."""
    m = (frames >= lo) & (frames <= hi)
    if m.sum() < 2 or np.ptp(frames[m]) < min_span:
        return None
    f = frames[m].astype(np.float64)
    return np.array([np.polyfit(f, pts[m, 0], 1)[0], np.polyfit(f, pts[m, 1], 1)[0]], np.float32)


def _hermite(pa, va, pb, vb, a, b, f):
    """Nội suy Hermite bậc 3 giữa (frame a, vị trí pa, vận tốc va) và (b, pb, vb): khớp cả vị trí lẫn vận tốc
    2 đầu → đi theo đoạn cong trong khoảng mờ (nội suy thẳng lệch ~60 px ở newvideo frame 300).
    Vận tốc 2 đầu = dây cung → đúng nội suy thẳng."""
    t, s = b - a, (f - a) / (b - a)
    return ((2 * s**3 - 3 * s**2 + 1) * pa + (s**3 - 2 * s**2 + s) * t * va
            + (-2 * s**3 + 3 * s**2) * pb + (s**3 - s**2) * t * vb)


def _grad_crop(gray, x0, y0, cw, ch):
    """Gradient của vùng gray[y0:y0+ch, x0:x0+cw]; phần nằm ngoài khung = 0 (không có nét)."""
    out = np.zeros((ch, cw), np.float32)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(gray.shape[1], x0 + cw), min(gray.shape[0], y0 + ch)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = _grad(gray[sy0:sy1, sx0:sx1])
    return out


def _local_match(gray, center, tpl, r):
    """Khớp mẫu full-res trong cửa sổ ±r quanh center → (tâm tìm được, điểm). Cửa sổ được lòi ra ngoài khung
    (phần ngoài = không có nét) để bám watermark chạy sát / lòi ra mép."""
    th, tw = tpl.shape
    x0 = int(round(center[0] - tw / 2)) - r
    y0 = int(round(center[1] - th / 2)) - r
    _, s, _, (mx, my) = cv2.minMaxLoc(_gated_ncc(_grad_crop(gray, x0, y0, tw + 2 * r, th + 2 * r), tpl))
    return np.array([x0 + mx + tw / 2, y0 + my + th / 2], np.float32), float(s)


def fill_gaps(video_path, anchors, pos, tpl, fps):
    """Lượt 2: vị trí mọi frame trong [mốc đầu, mốc cuối]. Giữa 2 mốc: nội suy Hermite (vận tốc 2 đầu ước từ các
    mốc trong 1 s quanh đó; thiếu thì dây cung) rồi dò lại tại chỗ; vị trí dò được lệch khỏi đường dự đoán
    > SPLIT_DEV → chỗ đổi hướng: thành mốc mới, các frame không có bằng chứng từ mốc trước tới đây nội suy lại thẳng,
    phần sau đi thẳng tới mốc kế. 2 mốc cách nhau một bước NHẢY (xa hơn tốc độ tối đa cho phép): đứng yên ở mốc trước
    tới mốc sau, không nội suy quét ngang khung. → (xy (k,2), evidence (k,) bool) bắt đầu từ anchors[0]."""
    first, last = anchors[0], anchors[-1]
    af = np.asarray(anchors)
    ap = pos[af]
    win, vmax = max(2, int(round(fps))), MAX_SPEED / fps
    xy = np.zeros((last - first + 1, 2), np.float32)
    ev = np.zeros(last - first + 1, bool)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    nxt, a, pa, va, vb, jump = 0, first, pos[first], None, None, False
    for f in range(first, last + 1):
        if f > first and (f - first) % max(1, -(-(last - first + 1) // 10)) == 0:
            print(f"  [WM] lượt 2/2 (lấp khoảng): {100 * (f - first) // (last - first + 1)}%", flush=True)
        ok, frame = cap.read()
        if not ok:   # video ngắn hơn dữ liệu: giữ vị trí cuối
            xy[f - first:] = xy[f - first - 1] if f > first else pos[first]
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        i = f - first
        if f == anchors[nxt]:
            q, s = _local_match(gray, pos[f], tpl, 6)
            xy[i], ev[i] = (q if s >= LOCAL_OK else pos[f]), True
            a, pa, nxt = f, xy[i].copy(), nxt + 1
            if nxt < len(anchors):
                b = anchors[nxt]
                chord = (pos[b] - pa) / (b - a)
                va = _velocity(af, ap, a - win, a, win // 2)
                vb = _velocity(af, ap, b, b + win, win // 2)
                va, vb = (chord if va is None else va), (chord if vb is None else vb)
                jump = np.hypot(*(pos[b] - pa)) > vmax * (b - a) + 6
            continue
        if jump:
            xy[i] = pa
            continue
        b = anchors[nxt]
        p = np.clip(_hermite(pa, va, pos[b], vb, a, b, f), 0, (gray.shape[1] - 1, gray.shape[0] - 1))
        q, s = _local_match(gray, p, tpl, LOCAL_R)
        if s < LOCAL_OK:
            xy[i] = p
            continue
        xy[i], ev[i] = q, True
        if np.hypot(*(q - p)) > SPLIT_DEV:
            for g in range(a + 1, f):
                if not ev[g - first]:
                    xy[g - first] = pa + (g - a) / (f - a) * (q - pa)
            a, pa = f, q.copy()
            va = vb = (pos[b] - pa) / (b - a)
    cap.release()
    return xy, ev


def track_watermark(video_path, ocr_data):
    """Toàn bộ bước dò cho một video → (kết quả cho wm_track.json, mask uint8 hoặc None)."""
    t0 = time.time()
    wm = find_moving_wm(ocr_data)
    if wm is None:
        return {"found": False, "reason": "không thấy chữ nào giống watermark chạy trong OCR"}, None
    built = build_template(video_path, wm)
    if built is None:
        return {"found": False,
                "reason": f"'{wm['text']}': dưới {MIN_TEMPLATE_READS} lần OCR đọc đủ chữ để dựng mẫu"}, None
    tpl, tpl_small, scale, mask = built
    pos, score = detect_all(video_path, tpl_small, scale, ocr_data["total_frames"])
    anchors = pick_anchors(pos, score, ocr_data["fps"], wm["reads"])
    if len(anchors) < MIN_ANCHORS:
        return {"found": False,
                "reason": f"'{wm['text']}': chỉ {len(anchors)} mốc chắc (cần ≥ {MIN_ANCHORS})"}, None
    xy, ev = fill_gaps(video_path, anchors, pos, tpl, ocr_data["fps"])
    return {"found": True, "text": wm["text"], "size": [int(tpl.shape[1]), int(tpl.shape[0])],
            "total_frames": int(ocr_data["total_frames"]), "fps": float(ocr_data["fps"]),
            "start": int(anchors[0]), "xy": [[int(round(x)), int(round(y))] for x, y in xy],
            "evidence": "".join("1" if e else "0" for e in ev), "anchors": len(anchors),
            "seconds": round(time.time() - t0, 1)}, mask


def load_track(path: str, total_frames: int):
    """wm_track.json → ({frame: (x, y)}, (w, h) hoặc None nếu thiếu size); ({}, None) nếu chưa có / không thấy
    watermark / không khớp video (OCR đã chạy lại) / hỏng (sửa tay sai dạng — cảnh báo, không làm sập bước sau)."""
    if not os.path.exists(path):
        return {}, None
    try:
        with open(path, encoding="utf-8") as f:
            t = json.load(f)
        if not t.get("found"):
            return {}, None
        if t.get("total_frames") != total_frames:
            print("  [WARN] wm_track.json không khớp video (OCR đã chạy lại?) — bỏ qua watermark chạy. "
                  "Chạy lại: python pipeline.py --input <video> --only wm")
            return {}, None
        start = int(t["start"])
        pos = {start + i: (float(x), float(y)) for i, (x, y) in enumerate(t["xy"])}
        ev = t.get("evidence")
        if isinstance(ev, str) and len(ev) == len(pos) and "1" in ev:   # bỏ vị trí đoán quá xa bằng chứng
            e = np.frombuffer(ev.encode(), np.uint8) == ord("1")
            near = np.convolve(e, np.ones(2 * MAX_GUESS + 1), "same") > 0
            pos = {f: p for (f, p), ok in zip(pos.items(), near) if ok}
        size = (float(t["size"][0]), float(t["size"][1])) if "size" in t else None
    except (ValueError, KeyError, TypeError, AttributeError, IndexError) as e:
        print(f"  [WARN] {path} hỏng ({e!r}) — bỏ qua watermark chạy.")
        return {}, None
    return pos, size


def on_watermark(seg: dict, pos: dict, size) -> bool:
    """Dòng OCR (text segment của ocr_data) nằm gọn trong khung watermark ở ≥ nửa số keyframe có vị trí → chính là
    watermark, OCR đọc sai thành chữ gì cũng vậy. Thoại bị watermark đè / dính vào thì rộng hơn khung nhiều, còn nhãn
    đứng yên chỉ bị watermark lướt qua vài keyframe → giữ."""
    hw, hh = size[0] / 2 + LINE_MARGIN, size[1] / 2 + LINE_MARGIN
    hits = seen = 0
    for k in seg.get("keyframe_bboxes") or []:
        c = pos.get(k["frame"])
        if c is None:
            continue
        x0, y0, x1, y1 = k["bbox"]
        ix = min(x1, c[0] + hw) - max(x0, c[0] - hw)
        iy = min(y1, c[1] + hh) - max(y0, c[1] - hh)
        seen += 1
        hits += ix > 0 and iy > 0 and ix * iy >= LINE_INSIDE * (x1 - x0) * (y1 - y0)
    return seen > 0 and 2 * hits >= seen


def write_preview(video_path, result, out_path, n=6):
    """n frame rải đều trong khoảng bám: vùng quanh watermark + khung (xanh = có bằng chứng, vàng = nội suy)
    → ảnh ghép 3 cột để user liếc là biết bám đúng hay sai. → False nếu không đọc được frame nào."""
    w, h = result["size"]
    start, xy, ev = result["start"], result["xy"], result["evidence"]
    cw, ch = 3 * w, 5 * h
    cap = cv2.VideoCapture(video_path)
    tiles = []
    for i in np.linspace(0, len(xy) - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start + i)
        ok, frame = cap.read()
        if not ok:
            continue
        x, y = xy[i]
        color = (0, 200, 0) if ev[i] == "1" else (0, 220, 255)
        cv2.rectangle(frame, (int(x - w / 2 - 4), int(y - h / 2 - 4)), (int(x + w / 2 + 4), int(y + h / 2 + 4)), color, 2)
        big = cv2.copyMakeBorder(frame, ch, ch, cw, cw, cv2.BORDER_CONSTANT)
        tile = big[int(y) + ch - ch // 2:int(y) + ch + ch // 2, int(x) + cw - cw // 2:int(x) + cw + cw // 2]
        tile = cv2.resize(tile, (480, int(480 * tile.shape[0] / tile.shape[1])))
        cv2.putText(tile, f"frame {start + i}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(tile)
    cap.release()
    if not tiles:
        return False
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    imwrite_any(out_path, np.vstack([np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles), 3)]))
    return True


def run(video_path: str) -> int:
    """Dò watermark chạy cho video → jobs/<tên>/wm_track.json (+ wm_mask.png, wm_preview.png).
    Mã 0 cả khi không thấy / bỏ qua (ghi found: false + lý do); 1 khi thiếu job/OCR hoặc không mở được video."""
    try:
        job = open_job(video_path)
    except JobError as e:
        print(f"  [WM][ERROR] {e}")
        return 1
    if not os.path.exists(job.ocr_data):
        print(f"  [WM][ERROR] Chưa có OCR cho {job.name} — chạy OCR trước: python ocr_pass.py --input \"{video_path}\"")
        return 1
    if not WM_TRACK:
        write_json_atomic(job.wm_track, {"found": False, "reason": "tắt trong config (WM_TRACK = False)"})
        print("  [WM] Tắt trong config (WM_TRACK = False) → không xóa watermark chạy.")
        return 0
    with open(job.ocr_data, encoding="utf-8") as f:
        ocr_data = json.load(f)
    video = working_video(job, ocr_data, video_path)
    cap = cv2.VideoCapture(video)
    opened = cap.isOpened()
    cap.release()
    if not opened:
        print(f"  [WM][ERROR] Không mở được video: {video}")
        return 1
    result, mask = track_watermark(video, ocr_data)
    if result["found"]:
        imwrite_any(job.wm_mask, mask)          # ghi mask trước json: render thấy found thì mask đã có
        write_preview(video, result, job.wm_preview)
        n = len(result["xy"])
        print(f"  [WM] Thấy watermark chạy '{result['text']}' ({result['size'][0]}×{result['size'][1]}) — "
              f"bám {n} frame ({result['evidence'].count('1') / n:.0%} có bằng chứng, {result['anchors']} mốc), "
              f"{result['seconds']}s")
        print(f"  [WM] Xem nhanh vị trí bám: {job.wm_preview}")
    else:
        print(f"  [WM] Không xóa watermark chạy: {result['reason']}")
    write_json_atomic(job.wm_track, result)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Dò watermark chạy → jobs/<tên>/wm_track.json (render dùng để xóa)")
    ap.add_argument("--input", "-i", type=str, default=VIDEO_PATH, help="Đường dẫn video gốc")
    sys.exit(run(ap.parse_args().input))


if __name__ == "__main__":
    main()
