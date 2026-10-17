"""
Tách lớp watermark (độ đậm α + màu) từ nhiều frame rồi xóa bằng đảo pha trộn
(spec docs/superpowers/specs/2026-10-16-tach-lop-watermark-design.md).

Mô hình: ảnh = (1 − α)·nền + α·màu ở từng điểm. Nền thật lấy từ frame khác — watermark đã dời đi, cảnh đứng yên
quanh đó — rồi hồi quy tuyến tính từng điểm ra α và α·màu. Xóa: nền = (ảnh − α·màu) / (1 − α), làm sạch dải mép
nét chữ. Tự chấm điểm trên cặp giữ riêng: chỉ dùng khi hơn inpaint ≥ MIN_GAIN_DB, không thì render giữ inpaint.
"""

import json
import os

import cv2
import numpy as np

from utils import inpaint_roi
from wm_track import _local_match

PAD = 10                  # ô = mask watermark nới 10 px mỗi phía
RING = 12                 # vành quanh mask để xét cảnh đứng yên
REFINE_R = 3              # căn lại vị trí ±3 px bằng khớp mẫu
MIN_SCORE = 0.5           # điểm khớp tại chỗ tối thiểu
GAPS = (75, -75, 125, -125, 200, -200, 300, -300)   # j = k ± 3–12 s (25 fps)
MAX_CAND = 600            # số frame k ứng viên tối đa (rải đều)
TRIES = 2                 # mỗi k giữ tối đa 2 frame j đã dời watermark (đọc 1 lượt: ~100 MB ô trong RAM)
STATIC_DIFF = 3.0         # sai khác TB trên vành < 3 mức xám → cảnh đứng yên
MIN_PAIRS, MAX_PAIRS = 60, 400
ALPHA_MAX = 0.95          # kẹp α khi đảo (không chia cho số quá nhỏ)
OPAQUE = 0.9              # α ≥ 0,9: gần đục → inpaint sau khi đảo
EDGE_FRAC = 0.5           # dải mép: gradient ảnh màu watermark ≥ 50% cực đại
MIN_GAIN_DB = 1.0         # đảo pha trộn phải hơn inpaint ≥ 1 dB mới được dùng
HOLDOUT = 0.2             # 20% cặp giữ riêng để chấm


def geometry(mask):
    """→ (inside, ring) cỡ ô (mask + 2·PAD), uint8 0/1: inside = vùng mask, ring = vành RING px quanh nó."""
    inside = cv2.copyMakeBorder((mask > 0).astype(np.uint8), PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0)
    k = 2 * RING + 1
    ring = cv2.dilate(inside, np.ones((k, k), np.uint8)) - inside
    return inside, ring


def corner(center, mask_shape):
    """Góc trái trên của ô khi tâm watermark ở center. Làm tròn .5 luôn lên (floor + 0,5): round() của Python làm
    tròn .5 về số chẵn → tâm nguyên với nửa bề rộng lẻ sẽ lệch ±1 px giữa frame chẵn / lẻ, làm nhòe hồi quy."""
    mh, mw = mask_shape
    return int(np.floor(center[0] - mw / 2 + 0.5)) - PAD, int(np.floor(center[1] - mh / 2 + 0.5)) - PAD


def cut(frame, x0, y0, ph, pw):
    """Ô (ph, pw) góc trái trên (x0, y0). Phần ngoài khung lấy điểm mép gần nhất và đánh dấu không hợp lệ.
    → (ô uint8, valid uint8 0/1)."""
    fh, fw = frame.shape[:2]
    ys, xs = np.arange(y0, y0 + ph), np.arange(x0, x0 + pw)
    patch = frame[np.ix_(np.clip(ys, 0, fh - 1), np.clip(xs, 0, fw - 1))]
    valid = (((ys >= 0) & (ys < fh))[:, None] & ((xs >= 0) & (xs < fw))[None, :]).astype(np.uint8)
    return patch, valid


def fit(I, B, V, inside):
    """Hồi quy từng điểm, từng kênh: ảnh = hệ_số·nền + chặn (trọng số valid).
    α = 1 − hệ_số (TB 3 kênh, kẹp [0, ALPHA_MAX], 0 ngoài inside); aW = α·màu = TB(ảnh) − (1 − α)·TB(nền).
    I, B: (n, h, w, 3); V: (n, h, w) 0/1."""
    I = np.asarray(I, np.float32)
    B = np.asarray(B, np.float32)
    M = np.asarray(V, np.float32)[..., None]
    nw = M.sum(0) + 1e-6
    bm, im = (B * M).sum(0) / nw, (I * M).sum(0) / nw
    cov = (M * (B - bm) * (I - im)).sum(0) / nw
    var = (M * (B - bm) ** 2).sum(0) / nw + 1e-6
    alpha = (np.clip(1 - (cov / var).mean(2), 0, ALPHA_MAX) * (inside > 0)).astype(np.float32)
    aW = ((im - (1 - alpha[..., None]) * bm) * (inside > 0)[..., None]).astype(np.float32)
    return alpha, aW


def edge_band(alpha, aW, inside):
    """Dải mép nét chữ: gradient (Sobel) của ảnh màu watermark (xám) ≥ EDGE_FRAC × cực đại, trong inside."""
    color = np.clip(aW / np.maximum(alpha, 1e-3)[..., None], 0, 255).astype(np.uint8)
    g = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    mag *= (inside > 0) & (alpha > 0.02)
    top = float(mag.max())
    return mag >= EDGE_FRAC * top if top > 0 else np.zeros(alpha.shape, bool)


def invert(patch, alpha, aW, edge, inside):
    """Đảo pha trộn trong inside → làm sạch dải mép (trung vị 5×5) → điểm gần đục (α ≥ OPAQUE) inpaint. → uint8."""
    p = patch.astype(np.float32)
    a = np.minimum(alpha, ALPHA_MAX)[..., None]
    out = np.where((inside > 0)[..., None], (p - aW) / (1 - a), p)
    out = np.clip(np.round(out), 0, 255).astype(np.uint8)  # làm tròn, không cắt: cắt lệch nền xuống 0,5 mức
    if edge.any():
        med = cv2.medianBlur(out, 5)
        out[edge] = med[edge]
    hard = ((alpha >= OPAQUE) & (inside > 0)).astype(np.uint8) * 255
    if hard.any():
        out = cv2.inpaint(out, hard, 3, cv2.INPAINT_TELEA)
    return out


def psnr(x, ref, m):
    """PSNR (dB) của x so với ref trên các điểm m > 0."""
    e = ((x.astype(np.float32) - ref.astype(np.float32)) ** 2)[m > 0].mean()
    return float(10 * np.log10(255.0 ** 2 / max(float(e), 1e-6)))


def choose(p_invert, p_inpaint):
    """Chỉ dùng đảo pha trộn khi hơn inpaint ≥ MIN_GAIN_DB."""
    return "invert" if p_invert >= p_inpaint + MIN_GAIN_DB else "inpaint"


def plan(track, mask_shape):
    """Chọn trước các cặp chỉ từ vị trí đã bám (không đọc video): k = frame có bằng chứng, rải đều ≤ MAX_CAND;
    mỗi k giữ ≤ TRIES frame j = k + GAPS (theo thứ tự) mà watermark đã rời chỗ. → [(k, [j, …])]."""
    s0, xy, ev = int(track["start"]), track["xy"], track["evidence"]
    mh, mw = mask_shape
    ks = [s0 + i for i, c in enumerate(ev) if c == "1"]
    if len(ks) > MAX_CAND:
        ks = [ks[i] for i in np.linspace(0, len(ks) - 1, MAX_CAND).astype(int)]
    out = []
    for k in ks:
        xk, yk = xy[k - s0]
        js = []
        for g in GAPS:
            j = k + g
            if 0 <= j - s0 < len(xy):
                xj, yj = xy[j - s0]
                if abs(xj - xk) >= mw or abs(yj - yk) >= mh:
                    js.append(j)
                    if len(js) == TRIES:
                        break
        if js:
            out.append((k, js))
    return out


def collect(video_path, track, mask, tpl=None):
    """Đọc video MỘT lượt tuần tự, cắt ô ở các frame k (căn lại ±REFINE_R bằng tpl nếu có) và j (đúng chỗ đó),
    giữ cặp có cảnh đứng yên trên vành. → (I, B, V) uint8: (n, ph, pw, 3) ×2 và (n, ph, pw)."""
    s0, xy = int(track["start"]), track["xy"]
    mh, mw = mask.shape
    ph, pw, R = mh + 2 * PAD, mw + 2 * PAD, REFINE_R
    _, ring = geometry(mask)
    pairs = plan(track, mask.shape)
    need = {}                                       # frame → [(vai trò, chỉ số cặp)]
    for i, (k, js) in enumerate(pairs):
        need.setdefault(k, []).append(("k", i))
        for j in js:
            need.setdefault(j, []).append(("j", i))
    big_k, off, big_j = {}, {}, {}
    cap = cv2.VideoCapture(video_path)
    last = max(need) if need else -1
    f = 0
    while f <= last:
        if f in need:
            ok, frame = cap.read()
        else:
            ok, frame = cap.grab(), None
        if not ok:
            break
        for role, i in need.get(f, ()):
            k = pairs[i][0]
            x0, y0 = corner(xy[k - s0], mask.shape)
            crop = cut(frame, x0 - R, y0 - R, ph + 2 * R, pw + 2 * R)   # nới R để còn căn lại
            if role == "j":
                big_j.setdefault(i, {})[f] = crop
                continue
            dx = dy = 0
            if tpl is not None:
                c = np.array([R + PAD + mw / 2, R + PAD + mh / 2], np.float32)
                q, sc = _local_match(cv2.cvtColor(crop[0], cv2.COLOR_BGR2GRAY), c, tpl, R)
                if sc < MIN_SCORE:
                    continue
                dx, dy = int(np.clip(round(q[0] - c[0]), -R, R)), int(np.clip(round(q[1] - c[1]), -R, R))
            big_k[i], off[i] = crop, (dx, dy)
        f += 1
    cap.release()
    I, B, V = [], [], []
    for i, (kp, kv) in big_k.items():
        dx, dy = off[i]
        sl = (slice(R + dy, R + dy + ph), slice(R + dx, R + dx + pw))
        a, va = kp[sl], kv[sl]
        for j in pairs[i][1]:                       # thử theo thứ tự GAPS
            if j not in big_j.get(i, {}):
                continue
            jp, jv = big_j[i][j]
            b, vb = jp[sl], jv[sl]
            m = (ring > 0) & (va > 0) & (vb > 0)
            if m.any() and np.abs(a.astype(np.int16) - b.astype(np.int16))[m].mean() < STATIC_DIFF:
                I.append(a); B.append(b); V.append(va & vb)
                break
    if len(I) > MAX_PAIRS:
        keep = np.linspace(0, len(I) - 1, MAX_PAIRS).astype(int)
        I, B, V = [I[t] for t in keep], [B[t] for t in keep], [V[t] for t in keep]
    if not I:
        return (np.zeros((0, ph, pw, 3), np.uint8), np.zeros((0, ph, pw, 3), np.uint8),
                np.zeros((0, ph, pw), np.uint8))
    return np.stack(I), np.stack(B), np.stack(V)


def build(video_path, track, mask, tpl=None):
    """Cả bước tách lớp cho 1 video → bản ghi cho wm_matte.npz (method "invert" | "inpaint" + lý do)."""
    inside, _ = geometry(mask)
    rec = {"method": "inpaint", "pairs": 0, "psnr_invert": None, "psnr_inpaint": None, "pad": PAD,
           "track_start": int(track["start"]), "track_len": len(track["xy"]), "mask_shape": list(mask.shape)}
    I, B, V = collect(video_path, track, mask, tpl)
    n = rec["pairs"] = len(I)
    if n < MIN_PAIRS:
        rec["reason"] = f"thiếu cảnh tĩnh: {n} cặp nền thật (cần ≥ {MIN_PAIRS})"
        return rec
    order = np.random.default_rng(0).permutation(n)
    nt = max(int(round(n * HOLDOUT)), 12)
    te, tr = order[:nt], order[nt:]
    alpha, aW = fit(I[tr], B[tr], V[tr], inside)
    edge = edge_band(alpha, aW, inside)
    p_inv = float(np.mean([psnr(invert(I[t], alpha, aW, edge, inside), B[t], inside * V[t]) for t in te]))
    p_inp = float(np.mean([psnr(inpaint_roi(I[t], inside * 255), B[t], inside * V[t]) for t in te]))
    rec.update(psnr_invert=round(p_inv, 2), psnr_inpaint=round(p_inp, 2), method=choose(p_inv, p_inp),
               reason=f"{n} cặp nền thật; đảo pha trộn {p_inv:.1f} dB, inpaint {p_inp:.1f} dB",
               alpha=alpha, aW=aW, edge=edge, tpl=tpl)
    return rec


_ARRAYS = ("alpha", "aW", "edge", "tpl")


def save(path, rec):
    """Ghi wm_matte.npz: các mảng (α, α·màu dạng float16) + meta JSON. Qua file tạm + os.replace."""
    meta = {k: v for k, v in rec.items() if k not in _ARRAYS}
    arrays = {k: rec[k] for k in _ARRAYS if rec.get(k) is not None}
    for k in ("alpha", "aW"):
        if k in arrays:
            arrays[k] = arrays[k].astype(np.float16)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)
    os.replace(tmp, path)


def load(path, track_path, mask):
    """→ lớp watermark để render dùng ({alpha, aW, edge, tpl, inside}), hoặc None khi: chưa có file,
    method inpaint, không khớp vết bám / cỡ ô hiện tại (dò lại sau khi tách lớp), hay file hỏng (cảnh báo, không làm sập)."""
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("method") != "invert":
                return None
            with open(track_path, encoding="utf-8") as f:
                t = json.load(f)
            m = {"alpha": z["alpha"].astype(np.float32), "aW": z["aW"].astype(np.float32),
                 "edge": z["edge"].astype(bool), "tpl": z["tpl"] if "tpl" in z.files else None,
                 "inside": geometry(mask)[0]}
            if (meta["track_start"], meta["track_len"], list(meta["mask_shape"]), meta.get("pad"), m["alpha"].shape) != \
                    (t["start"], len(t["xy"]), list(mask.shape), PAD, m["inside"].shape):
                print("  [WARN] wm_matte.npz không khớp vết bám hiện tại (đã dò lại?) — xóa watermark bằng inpaint. "
                      "Chạy lại: python pipeline.py --input <video> --only wm")
                return None
    except Exception as e:  # file hỏng kiểu gì cũng không được làm sập render
        print(f"  [WARN] {path} hỏng ({e!r}) — xóa watermark bằng inpaint.")
        return None
    return m


def apply(frame, center, matte, mask_shape):
    """Xóa watermark ở frame (tâm center) bằng lớp đã tách: căn lại ±REFINE_R bằng tpl (nếu có), đảo pha trộn,
    chỉ ghi phần ô nằm trong khung. Sửa frame tại chỗ."""
    mh, mw = mask_shape
    ph, pw, R = mh + 2 * PAD, mw + 2 * PAD, REFINE_R
    c = np.asarray(center, np.float32)
    if matte.get("tpl") is not None:
        x0, y0 = corner(c, mask_shape)
        o = np.array([x0 - R, y0 - R], np.float32)
        big, _ = cut(frame, x0 - R, y0 - R, ph + 2 * R, pw + 2 * R)
        q, sc = _local_match(cv2.cvtColor(big, cv2.COLOR_BGR2GRAY), c - o, matte["tpl"], R)
        if sc >= MIN_SCORE:
            c = q + o
    x0, y0 = corner(c, mask_shape)
    patch, _ = cut(frame, x0, y0, ph, pw)
    out = invert(patch, matte["alpha"], matte["aW"], matte["edge"], matte["inside"])
    fh, fw = frame.shape[:2]
    ys0, xs0, ys1, xs1 = max(0, y0), max(0, x0), min(fh, y0 + ph), min(fw, x0 + pw)
    if ys1 > ys0 and xs1 > xs0:
        frame[ys0:ys1, xs0:xs1] = out[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    return frame
