"""
Tách lớp watermark (độ đậm α + màu) từ nhiều frame rồi xóa bằng đảo pha trộn
(spec docs/superpowers/specs/2026-10-16-tach-lop-watermark-design.md).

Mô hình: ảnh = (1 − α)·nền + α·màu ở từng điểm. Nền thật lấy từ frame khác — watermark đã dời đi, cảnh đứng yên
quanh đó — rồi hồi quy tuyến tính từng điểm ra α và α·màu. Xóa: nền = (ảnh − α·màu) / (1 − α), làm sạch dải mép
nét chữ. Tự chấm điểm trên cặp giữ riêng: chỉ dùng khi hơn inpaint ≥ MIN_GAIN_DB, không thì render giữ inpaint.
"""

import json
import os
import zipfile

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
