"""
Test tách lớp watermark + xóa bằng đảo pha trộn (wm_matte.py) — dữ liệu tổng hợp, không gọi API.
Chạy:  python -m pytest tests/test_wm_matte.py -q
"""

import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils  # noqa: E402
import wm_matte  # noqa: E402


def test_invert_recovers_background_when_alpha_and_color_known():
    rng = np.random.default_rng(0)
    inside = np.zeros((16, 24), np.uint8); inside[3:13, 4:20] = 1
    alpha = (inside * 0.3).astype(np.float32)
    color = np.array([240, 230, 220], np.float32)
    aW = alpha[..., None] * color
    bg = rng.uniform(0, 255, (16, 24, 3)).astype(np.float32)
    img = np.clip(np.round((1 - alpha[..., None]) * bg + aW), 0, 255).astype(np.uint8)
    out = wm_matte.invert(img, alpha, aW, np.zeros(inside.shape, bool), inside)
    assert out.dtype == np.uint8 and out.shape == img.shape
    assert np.abs(out.astype(np.float32) - bg)[inside > 0].max() <= 1.5, "đảo pha trộn phải ra đúng nền"
    assert (out[inside == 0] == img[inside == 0]).all(), "ngoài vùng watermark giữ nguyên"


def test_fit_recovers_alpha_and_color_from_true_backgrounds():
    rng = np.random.default_rng(0)
    h, w, n = 12, 20, 80
    inside = np.zeros((h, w), np.uint8); inside[3:9, 4:16] = 1
    a_true = np.zeros((h, w), np.float32); a_true[4:8, 5:15] = 0.3
    color = np.array([240, 230, 220], np.float32)
    B = rng.uniform(0, 255, (n, h, w, 3)).astype(np.float32)
    I = (1 - a_true[..., None]) * B + a_true[..., None] * color
    alpha, aW = wm_matte.fit(I, B, np.ones((n, h, w), np.uint8), inside)
    assert alpha.dtype == np.float32 and aW.shape == (h, w, 3)
    assert np.abs(alpha - a_true * inside).max() < 1e-3
    assert np.abs(aW - a_true[..., None] * color * inside[..., None]).max() < 0.5


def test_choose_needs_at_least_one_db_gain():
    assert wm_matte.choose(27.4, 26.5) == "inpaint"
    assert wm_matte.choose(27.5, 26.5) == "invert"
    assert wm_matte.choose(20.0, 26.5) == "inpaint"


def test_cut_marks_outside_frame_invalid():
    fr = np.arange(20 * 30 * 3, dtype=np.uint32).reshape(20, 30, 3).astype(np.uint8)
    p, v = wm_matte.cut(fr, -2, 15, 8, 6)
    assert p.shape == (8, 6, 3) and v.shape == (8, 6)
    assert v[:5, 2:].all() and not v[:, :2].any() and not v[5:].any()
    assert (p[0, 2] == fr[15, 0]).all() and (p[0, 0] == fr[15, 0]).all(), "ngoài khung lấy mép gần nhất"


def test_geometry_inside_is_padded_mask_and_ring_surrounds_it():
    mask = np.zeros((10, 16), np.uint8); mask[2:8, 3:13] = 255
    inside, ring = wm_matte.geometry(mask)
    assert inside.shape == (10 + 2 * wm_matte.PAD, 16 + 2 * wm_matte.PAD)
    assert inside.sum() == (mask > 0).sum() and not (inside & ring).any()
    assert ring[wm_matte.PAD + 2 - 1, wm_matte.PAD + 5] == 1, "vành sát ngay ngoài mask"
