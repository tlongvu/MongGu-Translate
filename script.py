"""
Monggu Translate - Pass 3: Render Video (GPU Accelerated)
Load jobs/<tên>/ocr_data.json + lines.json → inpaint + draw tiếng Việt (theo từng lần xuất hiện) → xuất video
Sử dụng NVENC (encode), NVDEC (decode), PyTorch CUDA (resize/blur/cvtColor)

Usage:
    python script.py                     # Render video mặc định
    python script.py --input my.mp4      # Render video chỉ định
    python script.py --test              # Test mode (giới hạn frames)

Yêu cầu: Đã chạy ocr_pass.py và translate_pass.py trước.
"""

import sys

# Đảm bảo console stdout/stderr in được tiếng Việt Unicode trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import contextlib
import time
import os
import json
import multiprocessing as mp
import subprocess
from functools import lru_cache

try:
    import torch
    HAS_TORCH_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH_CUDA = False

from config import (
    VIDEO_PATH, TEST_MODE, TEST_FRAMES,
    NUM_WORKERS, FONT_PATH, TEXT_COLOR, OUTLINE_COLOR, OUTLINE_WIDTH,
    INPAINT_RADIUS, CROP_TOP_RATIO, SUPPRESS_TEXTS, AUTO_SUPPRESS_LATIN_ONLY,
    TIKTOK_PASS_MODE, PASS_ZOOM_RATIO, PASS_SHARPEN, PASS_COLOR_ADJUST,
    PASS_PORTRAIT_916, SUPPRESS_QR_CODES,
    USE_GPU, USE_NVENC, USE_NVDEC, USE_TORCH_CUDA, GPU_MAX_LOAD,
    INPAINT_METHOD,
    SUBTITLE_BG_ENABLE, SUBTITLE_BG_COLOR, SUBTITLE_BG_ALPHA,
    SUBTITLE_BG_PAD, SUBTITLE_BG_RADIUS,
)
from jobs import open_job, JobError, working_video
from utils import imread_gray
from wm_track import load_track


# ──────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────

def lines_mismatch(lines: list, text_segments: list) -> bool:
    """lines.json phải được dịch từ ĐÚNG ocr_data hiện tại: đủ dòng, mỗi dòng cùng khung thời gian với segment
    cùng id. OCR chạy lại sau khi dịch → số thứ tự segment xê dịch → không chặn thì cả video lệch phụ đề."""
    if len(lines) != len(text_segments):
        return True
    for l in lines:
        i = l.get("id") if isinstance(l, dict) else None
        if not isinstance(i, int) or not 0 <= i < len(text_segments):
            return True
        seg = text_segments[i]
        if (l.get("start"), l.get("end")) != (int(seg.get("start_frame", 0)),
                                              int(seg.get("end_frame", seg.get("start_frame", 0)))):
            return True
    return False


def video_mismatch(ocr_data: dict, width: int, height: int, fps: float, n_frames: int):
    """So video sắp render với ocr_data. Trả về câu mô tả chỗ lệch, hoặc None nếu khớp.
    Video DÀI hơn dữ liệu vẫn cho qua (dữ liệu OCR --test chỉ phủ đoạn đầu)."""
    if (width, height) != (ocr_data["width"], ocr_data["height"]):
        return f"kích thước {width}x{height} ≠ {ocr_data['width']}x{ocr_data['height']}"
    if abs(fps - ocr_data["fps"]) > 0.5:
        return f"fps {fps:.2f} ≠ {ocr_data['fps']:.2f}"
    if 0 < n_frames < ocr_data["total_frames"] - ocr_data["fps"]:
        return f"video chỉ có {n_frames} frame, dữ liệu OCR cần {ocr_data['total_frames']}"
    return None


@lru_cache(maxsize=128)
def get_font(font_path, size):
    """Load font (đã cache theo (font_path, size) để không tạo lại mỗi lần vẽ)."""
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()


def wrap_text(text, font, max_width):
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    if not words:
        return [text]
    lines, current = [], ""
    for w in words:
        test = (current + " " + w).strip()
        bbox = font.getbbox(test)
        if bbox[2] - bbox[0] > max_width and current:
            lines.append(current)
            current = w
        else:
            current = test
    if current:
        lines.append(current)
    return lines or [text]


@lru_cache(maxsize=4096)
def _resolve_layout_cached(text, font_path, preferred_size, max_width, min_size):
    """Lõi tính cỡ chữ + word-wrap, ĐÃ CACHE theo (text, font, size, max_width).
    Cùng một câu phụ đề lặp lại qua hàng chục frame → chỉ tính 1 lần."""
    size = preferred_size
    font = get_font(font_path, size)

    words = text.split()
    while size > min_size and words:
        max_word_w = max(font.getbbox(w)[2] - font.getbbox(w)[0] for w in words)
        if max_word_w <= max_width:
            break
        size -= 2
        font = get_font(font_path, size)

    lines = tuple(wrap_text(text, font, max_width))
    return size, lines


def _resolve_font_and_lines(text, font_path, preferred_size, max_width, min_size=22):
    """
    Chọn cỡ chữ phù hợp (thu nhỏ nếu một TỪ đơn dài tràn `max_width`) rồi word-wrap.
    Trả về (font, size, lines). Dùng chung cho cả tính chiều cao lẫn vẽ → đảm bảo
    chiều cao tính trước khớp tuyệt đối với chiều cao vẽ thực tế.
    """
    size, lines = _resolve_layout_cached(text, font_path, preferred_size, max_width, min_size)
    return get_font(font_path, size), size, list(lines)


def get_text_render_box(text, font_path, preferred_size, max_width,
                        min_size=22, line_spacing=6):
    """Trả về (width, height, size) của cụm chữ sau word-wrap — dùng vẽ nền plate."""
    if not text:
        return 0, 0, preferred_size
    font, size, lines = _resolve_font_and_lines(text, font_path, preferred_size,
                                                max_width, min_size)
    if not lines:
        return 0, 0, size
    width = max((font.getbbox(line)[2] - font.getbbox(line)[0]) for line in lines)
    height = len(lines) * _line_advance(font, line_spacing)
    return width, height, size


def _line_advance(font, line_spacing):
    """Khoảng cách dọc giữa 2 dòng (ascent + descent + line_spacing) — ổn định,
    không phụ thuộc nội dung từng dòng nên việc xếp chồng cụm chữ rất nhất quán."""
    ascent, descent = font.getmetrics()
    return ascent + descent + line_spacing


def get_text_render_height(text, font_path, preferred_size, max_width,
                           min_size=22, line_spacing=6):
    """
    Tính TRƯỚC chiều cao (pixel) của cụm chữ dịch sau khi word-wrap, KHÔNG vẽ.
    = số dòng thực tế * (chiều cao font + line_spacing).
    """
    if not text:
        return 0
    font, _size, lines = _resolve_font_and_lines(text, font_path, preferred_size,
                                                 max_width, min_size)
    if not lines:
        return 0
    return len(lines) * _line_advance(font, line_spacing)


def smart_draw_text_on_pil(draw, text, cx, y, font_path, preferred_size, max_width,
                          text_color, outline_color, outline_width,
                          img_h, img_w, min_size=22, line_spacing=6, align="center"):
    """
    Wrap-first text rendering — vẽ trực tiếp lên PIL Draw object.
    Căn giữa hoặc căn trái tùy thuộc vào tham số `align`.
    Trả về chiều cao (pixel) đã vẽ thực tế để caller có thể stack cụm chữ kế tiếp.
    """
    cx = max(0, min(cx, img_w - 1))
    y = max(0, min(y, img_h - 1))

    font, size, lines = _resolve_font_and_lines(text, font_path, preferred_size,
                                                max_width, min_size)
    advance = _line_advance(font, line_spacing)

    cur_y = y
    drawn = 0
    for line in lines:
        if cur_y >= img_h:
            break

        line_w = font.getbbox(line)[2] - font.getbbox(line)[0]
        if align == "center":
            draw_x = cx - line_w / 2
            # Giới hạn tọa độ để không bị tràn viền màn hình (cách mép ít nhất 10px)
            draw_x = max(10, min(draw_x, img_w - line_w - 10))
        else:
            draw_x = max(0, min(cx, img_w - line_w - 10))

        draw.text((draw_x, cur_y), line, font=font, fill=text_color,
                  stroke_width=outline_width, stroke_fill=outline_color)
        cur_y += advance
        drawn += 1

    return drawn * advance


def _layout_subtitle_blocks(jobs, img_h, font_path, margin=8,
                            line_spacing=12, min_size=22):
    """
    Điều chỉnh tọa độ Y động để CHỐNG ĐÈ CHỮ giữa các cụm phụ đề trong 1 frame.

    jobs: list dict, mỗi cụm có {'text','cx','y','orig_y','max_w','size'}.
    Thuật toán:
      1. Sort theo Y tăng dần (trên → dưới).
      2. Tính chiều cao thực của từng cụm sau word-wrap.
      3. Đẩy xuống: nếu cụm i+1 bắt đầu trước (đáy cụm i + margin) → đẩy nó xuống.
      4. Nếu tổng tràn quá đáy màn hình → thu nhỏ font dần (mỗi vòng -2) rồi xếp lại.
      5. Vẫn tràn ở cỡ min → đẩy ngược (push-back) từ dưới lên cho vừa khung.
    Trả về jobs đã cập nhật {'y','size','height'} (giữ thứ tự trên→dưới).
    """
    if not jobs:
        return jobs

    jobs = sorted(jobs, key=lambda j: j['y'])

    def recompute_heights():
        for j in jobs:
            j['height'] = get_text_render_height(
                j['text'], font_path, j['size'], j['max_w'],
                min_size=min_size, line_spacing=line_spacing,
            )

    def stack_down():
        """Đẩy các cụm xuống để không đè nhau; trả về tọa độ đáy của cụm cuối."""
        prev_bottom = None
        for j in jobs:
            j['y'] = j['orig_y']
            if prev_bottom is not None and j['y'] < prev_bottom + margin:
                j['y'] = prev_bottom + margin
            prev_bottom = j['y'] + j['height']
        return prev_bottom

    recompute_heights()
    bottom = stack_down()

    # Thu nhỏ font dần đến khi vừa khung (chỉ khi còn dư địa > min_size)
    attempts = 0
    while bottom is not None and bottom > img_h and attempts < 16:
        shrunk = False
        for j in jobs:
            if j['size'] > min_size:
                j['size'] = max(min_size, j['size'] - 2)
                shrunk = True
        if not shrunk:
            break
        recompute_heights()
        bottom = stack_down()
        attempts += 1

    # Vẫn tràn dù đã ở cỡ nhỏ nhất → đẩy ngược từ dưới lên cho khít khung hình
    if bottom is not None and bottom > img_h:
        next_top = img_h
        for j in reversed(jobs):
            max_y = next_top - j['height'] - margin
            if j['y'] > max_y:
                j['y'] = max_y
            next_top = j['y']
        # Cụm trên cùng không được âm
        if jobs and jobs[0]['y'] < 0:
            jobs[0]['y'] = 0

    return jobs



def _draw_subtitle_backgrounds(pil_img, jobs, font_path, line_spacing, min_size,
                               img_w, img_h):
    """
    Vẽ hộp nền mờ bán trong suốt (bo góc) phía sau mỗi cụm phụ đề để tăng độ đọc
    trên nền sáng/nhiễu. Trả về ảnh RGB mới đã ghép nền.
    """
    base = pil_img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    r, g, b = SUBTITLE_BG_COLOR
    fill = (r, g, b, SUBTITLE_BG_ALPHA)
    pad = SUBTITLE_BG_PAD

    drew_any = False
    for j in jobs:
        w, h, _size = get_text_render_box(
            j['text'], font_path, j['size'], j['max_w'],
            min_size=min_size, line_spacing=line_spacing,
        )
        if w <= 0 or h <= 0:
            continue
        cx = j['cx']
        x0 = max(0, cx - w / 2 - pad)
        x1 = min(img_w, cx + w / 2 + pad)
        y0 = max(0, j['y'] - pad)
        y1 = min(img_h, j['y'] + h + pad)
        if x1 <= x0 or y1 <= y0:
            continue
        odraw.rounded_rectangle([x0, y0, x1, y1], radius=SUBTITLE_BG_RADIUS, fill=fill)
        drew_any = True

    if not drew_any:
        return pil_img
    return Image.alpha_composite(base, overlay).convert("RGB")


def _inpaint_roi(roi_frame, roi_mask):
    """Xóa text gốc trong ROI theo INPAINT_METHOD: telea | ns | fill (blur nhanh)."""
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


def remove_wm(frame, center, mask, pad=10):
    """Xóa watermark chạy: đặt mask (tâm = center) lên frame rồi inpaint theo INPAINT_METHOD.
    ROI rộng hơn mask `pad` px để inpaint có ngữ cảnh; mask sát / lố mép khung thì xén. Sửa frame tại chỗ."""
    mh, mw = mask.shape
    fh, fw = frame.shape[:2]
    x0, y0 = int(round(center[0] - mw / 2)), int(round(center[1] - mh / 2))
    rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
    rx1, ry1 = min(fw, x0 + mw + pad), min(fh, y0 + mh + pad)
    mx0, my0 = max(x0, rx0), max(y0, ry0)
    mx1, my1 = min(x0 + mw, rx1), min(y0 + mh, ry1)
    if mx1 <= mx0 or my1 <= my0:
        return frame
    roi_mask = np.zeros((ry1 - ry0, rx1 - rx0), np.uint8)
    roi_mask[my0 - ry0:my1 - ry0, mx0 - rx0:mx1 - rx0] = mask[my0 - y0:my1 - y0, mx0 - x0:mx1 - x0]
    frame[ry0:ry1, rx0:rx1] = _inpaint_roi(frame[ry0:ry1, rx0:rx1], roi_mask)
    return frame


def load_wm_track(job, total_frames: int):
    """Vị trí watermark chạy ({frame: (x, y)}) + mask từ wm_track.py; ({}, None) nếu không có / không dùng được."""
    pos, _ = load_track(job.wm_track, total_frames)
    if not pos:
        return {}, None
    mask = imread_gray(job.wm_mask)
    if mask is None:
        print(f"  [WARN] Thiếu {job.wm_mask} — bỏ qua xóa watermark chạy.")
        return {}, None
    return pos, mask


def _gpu_resize(frame, new_w, new_h):
    """Resize frame trên GPU bằng PyTorch CUDA."""
    if USE_GPU and USE_TORCH_CUDA and HAS_TORCH_CUDA:
        t = torch.from_numpy(frame).cuda().permute(2, 0, 1).unsqueeze(0).float()
        t = torch.nn.functional.interpolate(t, size=(new_h, new_w), mode='bilinear', align_corners=False)
        return t.squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _gpu_gaussian_blur(frame, ksize, sigma=0):
    """GaussianBlur trên GPU bằng PyTorch CUDA."""
    if USE_GPU and USE_TORCH_CUDA and HAS_TORCH_CUDA:
        k = ksize[0] if isinstance(ksize, tuple) else ksize
        if k <= 0:
            k = int(6 * sigma + 1) | 1  # Ensure odd
        padding = k // 2
        t = torch.from_numpy(frame).cuda().permute(2, 0, 1).unsqueeze(0).float()
        # Tạo 1D Gaussian kernel
        coords = torch.arange(k, dtype=torch.float32, device='cuda') - k // 2
        g = torch.exp(-coords**2 / (2 * max(sigma, 1.0)**2))
        g = g / g.sum()
        # Separable: blur X rồi blur Y (nhanh hơn 2D conv)
        kernel_x = g.view(1, 1, 1, k).expand(3, 1, 1, k)
        kernel_y = g.view(1, 1, k, 1).expand(3, 1, k, 1)
        t = torch.nn.functional.conv2d(t, kernel_x, padding=(0, padding), groups=3)
        t = torch.nn.functional.conv2d(t, kernel_y, padding=(padding, 0), groups=3)
        return t.squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
    return cv2.GaussianBlur(frame, ksize, sigma)


def apply_pass_transformations(frame):
    """
    Apply visual transformations to change frame hash and improve quality.
    Sử dụng GPU (PyTorch CUDA) cho resize/blur nếu có.
    """
    h, w = frame.shape[:2]

    # 1. Zoom & Crop
    if PASS_ZOOM_RATIO > 1.0:
        new_w = int(w * PASS_ZOOM_RATIO)
        new_h = int(h * PASS_ZOOM_RATIO)
        zoomed = _gpu_resize(frame, new_w, new_h)
        
        # Crop to center
        x1 = (new_w - w) // 2
        y1 = (new_h - h) // 2
        frame = zoomed[y1:y1+h, x1:x1+w]

    # 2. Sharpening (Unsharp Mask)
    if PASS_SHARPEN:
        gaussian_blur = _gpu_gaussian_blur(frame, (0, 0), 3)
        frame = cv2.addWeighted(frame, 1.5, gaussian_blur, -0.5, 0)

    # 3. Color Adjustment (Slight shift) — GPU via PyTorch
    if PASS_COLOR_ADJUST:
        if USE_GPU and USE_TORCH_CUDA and HAS_TORCH_CUDA:
            t = torch.from_numpy(frame).cuda().float()
            # BGR→HSV approximation: adjust saturation/brightness trực tiếp
            t[:, :, 1] = (t[:, :, 1] * 1.05).clamp(0, 255)  # Saturation-ish +5%
            t[:, :, 2] = (t[:, :, 2] + 2).clamp(0, 255)      # Brightness +2
            frame = t.byte().cpu().numpy()
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * 1.05
            hsv[:, :, 2] = hsv[:, :, 2] + 2
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
            frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return frame


def apply_916_background(frame, target_w=1080, target_h=1920):
    """
    Wrap frame in a 9:16 portrait canvas with a blurred background.
    """
    orig_h, orig_w = frame.shape[:2]
    
    # Scale content to fit width of target (keeping aspect ratio)
    scale = target_w / orig_w
    content_w = target_w
    content_h = int(orig_h * scale)
    
    # If content_h still taller than target_h, scale down by height instead
    if content_h > target_h:
        scale = target_h / orig_h
        content_h = target_h
        content_w = int(orig_w * scale)
    
    content_resized = _gpu_resize(frame, content_w, content_h)
    
    # Create background: Zoom-fill the 9:16 canvas and blur heavily
    bg_scale = max(target_w / orig_w, target_h / orig_h)
    bg_w = int(orig_w * bg_scale)
    bg_h = int(orig_h * bg_scale)
    bg = _gpu_resize(frame, bg_w, bg_h)
    
    # Center crop background to target size
    x1 = (bg_w - target_w) // 2
    y1 = (bg_h - target_h) // 2
    bg_cropped = bg[y1:y1+target_h, x1:x1+target_w]
    
    # Blur background (GPU nếu có)
    bg_blurred = _gpu_gaussian_blur(bg_cropped, (51, 51), 0)
    # Darken background slightly to emphasize foreground
    bg_blurred = cv2.convertScaleAbs(bg_blurred, alpha=0.6, beta=0)
    
    # Place foreground content in the center
    y_off = (target_h - content_h) // 2
    x_off = (target_w - content_w) // 2
    bg_blurred[y_off:y_off+content_h, x_off:x_off+content_w] = content_resized
    
    return bg_blurred, (x_off, y_off, scale)


def interpolate_bbox(keyframe_bboxes: list, frame_num: int) -> list:
    """
    Nội suy tuyến tính bbox tại frame_num từ danh sách keyframe_bboxes.
    keyframe_bboxes: [{'frame': int, 'bbox': [x1,y1,x2,y2]}, ...]
    Returns: [x1, y1, x2, y2] (float)
    """
    if not keyframe_bboxes:
        return None

    # Sort theo frame
    kfbs = sorted(keyframe_bboxes, key=lambda k: k['frame'])

    # Trước keyframe đầu tiên → dùng bbox đầu
    if frame_num <= kfbs[0]['frame']:
        return kfbs[0]['bbox']

    # Sau keyframe cuối → dùng bbox cuối
    if frame_num >= kfbs[-1]['frame']:
        return kfbs[-1]['bbox']

    # Tìm 2 keyframe bao quanh frame_num và interpolate tuyến tính
    for i in range(len(kfbs) - 1):
        f0, bb0 = kfbs[i]['frame'], kfbs[i]['bbox']
        f1, bb1 = kfbs[i + 1]['frame'], kfbs[i + 1]['bbox']
        if f0 <= frame_num <= f1:
            t = (frame_num - f0) / max(f1 - f0, 1)
            return [
                bb0[0] + t * (bb1[0] - bb0[0]),  # x1
                bb0[1] + t * (bb1[1] - bb0[1]),  # y1
                bb0[2] + t * (bb1[2] - bb0[2]),  # x2
                bb0[3] + t * (bb1[3] - bb0[3]),  # y2
            ]

    return kfbs[-1]['bbox']


def calculate_iou(boxA, boxB):
    """Tính Intersection over Union (IoU) giữa 2 box dạng [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def filter_overlapping_boxes(detections, iou_threshold=0.6):
    """Loại bỏ các box bị chồng lấn nhau quá nhiều (giữ lại box to hơn/confidence cao hơn)."""
    if not detections:
        return detections
    
    # Chuyển đổi box sang dạng chuẩn [x1, y1, x2, y2]
    processed = []
    for d in detections:
        box = d['box']
        pts = np.array(box, dtype=np.int32)
        x1, y1, w, h = cv2.boundingRect(pts)
        processed.append({
            'orig': d,
            'rect': [x1, y1, x1 + w, y1 + h],
            'area': w * h,
            'keep': True
        })
        
    # Sắp xếp theo diện tích giảm dần để ưu tiên giữ box to
    processed = sorted(processed, key=lambda x: x['area'], reverse=True)
    
    for i in range(len(processed)):
        if not processed[i]['keep']: continue
        for j in range(i + 1, len(processed)):
            if not processed[j]['keep']: continue
            iou = calculate_iou(processed[i]['rect'], processed[j]['rect'])
            if iou > iou_threshold:
                processed[j]['keep'] = False
                
    return [p['orig'] for p in processed if p['keep']]


def merge_nearby_detections(detections, y_threshold=5, x_gap_threshold=80):
    """
    Merge text blocks within y_threshold px of each other (left-to-right), 
    ONLY IF they are close horizontally (gap <= x_gap_threshold).
    Deduplicates identical text to prevent "A  A" repetition.
    Strictly prevents vertical line merging (Y threshold = 5px).
    """
    if not detections:
        return detections
    
    # 0. Lọc nhiễu trước khi gộp
    detections = filter_overlapping_boxes(detections)
    
    # 1. Group into lines based on Y - Cực kỳ khắt khe (chỉ lệch max 8px)
    sorted_dets = sorted(detections, key=lambda d: d['y'])
    lines = []
    used = [False] * len(sorted_dets)
    for i, det in enumerate(sorted_dets):
        if used[i]: continue
        used[i] = True
        line = [det]
        for j, other in enumerate(sorted_dets):
            if used[j]: continue
            # Nếu lệch Y quá 8px -> Coi như dòng khác, không gộp
            if abs(other['y'] - det['y']) <= y_threshold:
                line.append(other)
                used[j] = True
        lines.append(line)
        
    # 2. Merge horizontally within each line
    merged = []
    for line in lines:
        line_sorted = sorted(line, key=lambda d: d['x'])
        current_group = [line_sorted[0]]
        
        for curr in line_sorted[1:]:
            prev = current_group[-1]
            prev_x = prev['box'][0][0]
            prev_w = prev['box'][1][0] - prev_x
            
            gap = curr['x'] - (prev_x + prev_w)
            # Gap nhỏ thì gộp ngang, tuyệt đối không gộp dọc
            if gap <= x_gap_threshold:
                current_group.append(curr)
            else:
                texts = []
                for g in current_group:
                    txt = g.get('vi_text', g['text']).strip()
                    if txt and txt not in texts:
                        texts.append(txt)
                
                combined = "  ".join(texts)
                new_det = dict(current_group[0])
                new_det['vi_text'] = combined
                new_det['y'] = int(sum(g['y'] for g in current_group) / len(current_group))
                new_det['box_height'] = max(g['box_height'] for g in current_group)
                
                # Tính toán tổng chiều rộng của khối đã gộp
                min_x = new_det['x']
                max_x2 = max(g['x'] + g.get('box_width', 0) for g in current_group)
                new_det['box_width'] = int(max_x2 - min_x)
                
                merged.append(new_det)
                
                current_group = [curr]
                
        # Chốt nhóm cuối
        if current_group:
            texts = []
            for g in current_group:
                txt = g.get('vi_text', g['text']).strip()
                if txt and txt not in texts:
                    texts.append(txt)
                    
            combined = "  ".join(texts)
            new_det = dict(current_group[0])
            new_det['vi_text'] = combined
            new_det['y'] = int(sum(g['y'] for g in current_group) / len(current_group))
            new_det['box_height'] = max(g['box_height'] for g in current_group)
            
            # Tính toán tổng chiều rộng của khối đã gộp
            min_x = new_det['x']
            max_x2 = max(g['x'] + g.get('box_width', 0) for g in current_group)
            new_det['box_width'] = int(max_x2 - min_x)
            
            merged.append(new_det)
            
    return merged



# ──────────────────────────────────────────────
# WORKER FUNCTION (Render pass - no OCR, no API)
# ──────────────────────────────────────────────

def _build_segment_index(text_segments: list, start_frame: int, end_frame: int,
                          translation_map: dict, extend_frames: int = 5,
                          ocr_interval_frames: int = 0, vi_for_seg: dict = None) -> dict:
    """
    Build frame→[{bbox, text, vi_text, box_height}] index từ text_segments.
    Mỗi frame nhận bbox được interpolate tuyến tính từ keyframe_bboxes.
    Tự động giới hạn/cắt phần mở rộng thời gian nếu có các segment chồng lấn không gian
    nhưng khác nhau về thời gian (tránh đè phụ đề ở các phân cảnh xuất hiện text mới).
    """
    index = {}  # frame_num → list of segment hits
    extend_start = max(extend_frames, ocr_interval_frames)

    # 1. Khởi tạo giới hạn mở rộng mặc định cho từng segment
    seg_limits = {}  # id(seg) -> [sf, ef]
    for seg in text_segments:
        sf = max(0, seg['start_frame'] - extend_start)
        ef = seg['end_frame'] + extend_frames
        seg_limits[id(seg)] = [sf, ef]

    # Helper kiểm tra chồng lấn không gian (Bounding Box giao nhau)
    def boxes_intersect(boxA, boxB):
        return not (boxA[2] < boxB[0] or boxA[0] > boxB[2] or
                    boxA[3] < boxB[1] or boxA[1] > boxB[3])

    # 2. Quét chéo để phát hiện và xử lý các segment chồng lấn không gian liên tiếp nhau
    for i in range(len(text_segments)):
        segA = text_segments[i]
        boxA = segA['bbox']
        for j in range(i + 1, len(text_segments)):
            segB = text_segments[j]
            boxB = segB['bbox']

            if boxes_intersect(boxA, boxB):
                # Xác định thứ tự thời gian xuất hiện
                if segA['end_frame'] < segB['start_frame']:
                    first_seg, second_seg = segA, segB
                elif segB['end_frame'] < segA['start_frame']:
                    first_seg, second_seg = segB, segA
                else:
                    # Giao nhau sẵn trong gốc -> Giữ nguyên
                    continue

                gap = second_seg['start_frame'] - first_seg['end_frame']
                limits_first = seg_limits[id(first_seg)]
                limits_second = seg_limits[id(second_seg)]

                # Nếu khoảng mở rộng làm chúng đè lên nhau
                if limits_first[1] >= limits_second[0]:
                    if gap <= 1:
                        # Ranh giới quá sát, không cho mở rộng ở điểm chạm
                        limits_first[1] = min(limits_first[1], first_seg['end_frame'])
                        limits_second[0] = max(limits_second[0], second_seg['start_frame'])
                    else:
                        # Chia đôi khoảng trống còn lại cho hai bên (chỉ thu hẹp, không nới lại)
                        ext_first = (gap - 1) // 2
                        ext_second = gap - 1 - ext_first
                        limits_first[1] = min(limits_first[1], first_seg['end_frame'] + ext_first)
                        limits_second[0] = max(limits_second[0], second_seg['start_frame'] - ext_second)

    # 3. Tiến hành phân bổ segment vào index của từng frame dựa trên giới hạn đã tối ưu
    for seg in text_segments:
        sf, ef = seg_limits[id(seg)]
        # Chỉ xử lý segment có giao với chunk này
        if ef < start_frame or sf >= end_frame:
            continue

        zh_text = seg['text']
        # Bản dịch theo TỪNG LẦN XUẤT HIỆN (id segment); không có id → tra theo chữ (tương thích)
        vi_text = (vi_for_seg or {}).get(seg.get('id'), translation_map.get(zh_text, ""))
        keyframe_bboxes = seg.get('keyframe_bboxes', [])

        for fn in range(max(sf, start_frame), min(ef + 1, end_frame)):
            if keyframe_bboxes:
                bbox = interpolate_bbox(keyframe_bboxes, fn)
            else:
                bbox = seg['bbox']

            x1, y1, x2, y2 = bbox
            box_height = max(int(y2 - y1), 20)
            box_width = max(int(x2 - x1), 10)
            box_4pt = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            entry = {
                'box': box_4pt,
                'text': zh_text,
                'vi_text': vi_text,
                'x': int(x1),
                'y': int(y1),
                'box_height': box_height,
                'box_width': box_width,
            }
            index.setdefault(fn, []).append(entry)

    return index


def entries_overlap(entryA: dict, entryB: dict) -> bool:
    """Kiểm tra xem 2 entry phụ đề có chồng lấn không gian đáng kể (>20% IoU) hay không."""
    xA1, yA1 = entryA['x'], entryA['y']
    xA2, yA2 = xA1 + entryA['box_width'], yA1 + entryA['box_height']
    
    xB1, yB1 = entryB['x'], entryB['y']
    xB2, yB2 = xB1 + entryB['box_width'], yB1 + entryB['box_height']
    
    xI1 = max(xA1, xB1)
    yI1 = max(yA1, yB1)
    xI2 = min(xA2, xB2)
    yI2 = min(yA2, yB2)
    
    inter_area = max(0, xI2 - xI1) * max(0, yI2 - yI1)
    if inter_area <= 0:
        return False
        
    areaA = (xA2 - xA1) * (yA2 - yA1)
    areaB = (xB2 - xB1) * (yB2 - yB1)
    union_area = areaA + areaB - inter_area
    if union_area <= 0:
        return False
        
    iou = inter_area / float(union_area)
    return iou > 0.20


def process_chunk(args):
    """
    Worker: đọc frames → tra cứu OCR data + translation map → inpaint + draw → ghi video.
    KHÔNG gọi OCR, KHÔNG gọi API dịch.
    """
    (chunk_idx, start_frame, end_frame,
     video_path, temp_output, font_path,
     fps, width, height,
     ocr_interval_frames, translation_map,
     frame_detections_for_chunk, text_segments_for_chunk, vi_for_seg,
     wm_pos, wm_mask) = args

    gpu_mode = USE_GPU and USE_NVENC
    print(f"  [Worker {chunk_idx}] Render {'(GPU+NVENC) ' if gpu_mode else ''}{'(Pass Mode) ' if TIKTOK_PASS_MODE else ''}frames {start_frame}-{end_frame}...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[Worker {chunk_idx}] Không mở được video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Nếu chạy 9:16 nền mờ, output resolution thay đổi
    out_w, out_h = width, height
    if TIKTOK_PASS_MODE and PASS_PORTRAIT_916:
        out_w, out_h = 1080, 1920

    # ── Video Writer: NVENC (GPU) hoặc cv2 fallback ──
    ffmpeg_proc = None
    out = None
    if gpu_mode:
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{out_w}x{out_h}', '-r', str(fps),
            '-i', '-',
            '-c:v', 'h264_nvenc',
            '-pix_fmt', 'yuv420p',     # Force correct color conversion
            '-preset', 'p4',           # Balanced speed/quality
            '-rc', 'vbr',              # Variable bitrate
            '-cq', '23',               # Quality level
            '-gpu', '0',
            temp_output
        ]
        try:
            ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        except FileNotFoundError:
            print(f"  [Worker {chunk_idx}] WARN: ffmpeg not found, fallback cv2")
            gpu_mode = False
    
    if not gpu_mode:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (out_w, out_h))

    # QR Detector (chỉ tạo 1 lần trong worker)
    qr_detector = cv2.QRCodeDetector() if (TIKTOK_PASS_MODE and SUPPRESS_QR_CODES) else None

    # ── Build segment index: frame → [{bbox, text, vi_text}] ──
    # Đây là lookup chính xác: biết frame nào cần inpaint vùng nào.
    # Truyền ocr_interval_frames để extend ngược về trước 1 OCR interval,
    # tránh text Trung nhấp nháy trước khi inpaint kịp bắt đầu.
    segment_index = _build_segment_index(
        text_segments_for_chunk, start_frame, end_frame, translation_map,
        ocr_interval_frames=ocr_interval_frames, vi_for_seg=vi_for_seg,
    )

    # Fallback: keyframe-based lookup (cho các frame không có segment)
    sorted_keyframes = sorted([int(k) for k in frame_detections_for_chunk.keys()])

    local_idx = 0
    total_chunk = end_frame - start_frame
    t0 = time.time()

    # Cache bbox thực tế gần nhất của từng text (cho Priority 2 gap-fill)
    text_last_seen = {}  # zh_text → entry dict (box, vi_text, ...)

    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        if wm_mask is not None and frame_num in wm_pos:     # watermark chạy (wm_track.py): xóa ở mọi frame
            frame = remove_wm(frame, wm_pos[frame_num], wm_mask)

        # ── Priority 1: Actual keyframe (exact bbox, chính xác nhất) ──
        frame_str = str(frame_num)
        raw_detections = []
        detected_texts = set()
        if frame_str in frame_detections_for_chunk:
            for det in frame_detections_for_chunk[frame_str]:
                box = det['box']
                x = int(box[0][0])
                y = int(box[0][1])
                box_height = int(box[3][1] - box[0][1]) if len(box) > 3 else 30
                box_width = int(box[1][0] - box[0][0]) if len(box) > 1 else 100
                zh_text = det['text']
                vi_text = translation_map.get(zh_text, "")
                entry = {
                    'box': box, 'text': zh_text, 'vi_text': vi_text,
                    'x': x, 'y': y, 'box_height': box_height,
                    'box_width': box_width,
                }
                raw_detections.append(entry)
                text_last_seen[zh_text] = entry  # Cập nhật cache

        # ── Priority 2: Segment detections (Gap-fill đã được gom cụm mượt mà hơn) ──
        segment_detections = []
        if frame_num in segment_index:
            for seg_entry in segment_index[frame_num]:
                seg_text = seg_entry['text']
                segment_detections.append(seg_entry)
                text_last_seen[seg_text] = seg_entry

        # ── Trộn & Lọc bỏ trùng lấn không gian (Bảo vệ hiệu ứng typewriter tránh đè chữ) ──
        cached_detections = []
        
        # Thêm toàn bộ các segment detections trước (ưu tiên câu đầy đủ và ổn định)
        for seg_entry in segment_detections:
            cached_detections.append(seg_entry)
            detected_texts.add(seg_entry['text'])
            
        # Thêm các raw detections của keyframe hiện tại nếu vị trí của chúng không bị trùng lấn với segment nào
        for raw_entry in raw_detections:
            has_overlap = False
            for seg_entry in segment_detections:
                if entries_overlap(raw_entry, seg_entry):
                    has_overlap = True
                    break
            if not has_overlap:
                if raw_entry['text'] not in detected_texts:
                    cached_detections.append(raw_entry)
                    detected_texts.add(raw_entry['text'])

        # ── Priority 3: Frame không có detection nào → keyframe gần nhất trong 30 frames ──
        if not detected_texts:
            nearest_key = None
            for kf in sorted_keyframes:
                if kf <= frame_num:
                    nearest_key = kf
                else:
                    break
            if nearest_key is not None and (frame_num - nearest_key) <= max(30, ocr_interval_frames * 5):
                for det in frame_detections_for_chunk.get(str(nearest_key), []):
                    box = det['box']
                    x = int(box[0][0])
                    y = int(box[0][1])
                    box_height = int(box[3][1] - box[0][1]) if len(box) > 3 else 30
                    box_width = int(box[1][0] - box[0][0]) if len(box) > 1 else 100
                    zh_text = det['text']
                    vi_text = translation_map.get(zh_text, "")
                    cached_detections.append({
                        'box': box, 'text': zh_text, 'vi_text': vi_text,
                        'x': x, 'y': y, 'box_height': box_height,
                        'box_width': box_width,
                    })

        # Inpaint + draw
        if cached_detections:
            # Inpaint TẤT CẢ text cục bộ (Smart Local Inpaint) để tăng tốc độ gấp 10-20 lần
            for d in cached_detections:
                box = d['box']
                pts = np.array(box, dtype=np.int32)
                x, y, w, h = cv2.boundingRect(pts)
                
                # Mở rộng ROI đủ lớn để chứa mask phóng to (1.7 + 6px) VÀ vùng quét (INPAINT_RADIUS)
                exp = int(max(w, h) * 0.4) + INPAINT_RADIUS + 10
                
                y1 = max(0, y - exp)
                y2 = min(frame.shape[0], y + h + exp)
                x1 = max(0, x - exp)
                x2 = min(frame.shape[1], x + w + exp)
                
                # Cắt ảnh nhỏ (ROI)
                roi_frame = frame[y1:y2, x1:x2]
                roi_mask = np.zeros(roi_frame.shape[:2], dtype=np.uint8)
                
                # Trỏ tọa độ polygon về ROI cục bộ
                local_pts = pts.copy()
                local_pts[:, 0] -= x1
                local_pts[:, 1] -= y1
                center = local_pts.mean(axis=0).astype(np.float32)
                
                # Phong to mask 1.2 lan de xoa sach ca vien den cua chu cu (giảm từ 1.4 → 1.2 cho nhanh hơn)
                pts_exp = (center + (local_pts.astype(np.float32) - center) * 1.2).astype(np.int32)
                pts_exp[:, 0] = np.clip(pts_exp[:, 0], 0, roi_mask.shape[1] - 1)
                pts_exp[:, 1] = np.clip(pts_exp[:, 1], 0, roi_mask.shape[0] - 1)
                
                cv2.fillPoly(roi_mask, [pts_exp], 255)
                
                # Inpaint SIEU NHANH tren tep anh be dan de vao anh to
                inpainted_roi = _inpaint_roi(roi_frame, roi_mask)
                frame[y1:y2, x1:x2] = inpainted_roi

        # ── TIKTOK PASS MODE: Transformations ──
        x_off, y_off, content_scale = 0, 0, 1.0
        if TIKTOK_PASS_MODE:
            # 1. Phát hiện và xóa mã QR
            if SUPPRESS_QR_CODES and qr_detector:
                retval, points = qr_detector.detect(frame)
                if retval:
                    for pts in points:
                        p = pts.astype(np.int32)
                        # Padding nhẹ vùng QR
                        p_center = p.mean(axis=0)
                        p_exp = (p_center + (p - p_center) * 1.2).astype(np.int32)
                        cv2.fillPoly(frame, [p_exp], (128, 128, 128)) # Tô xám vùng QR

            # 2. Biến đổi khung hình (Zoom, Sharpen, Color)
            frame = apply_pass_transformations(frame)

            # 3. Chế độ 9:16 Nền mờ
            if PASS_PORTRAIT_916:
                frame, (x_off, y_off, content_scale) = apply_916_background(frame, out_w, out_h)

        # Draw text — Batch PIL: chỉ convert BGR↔RGB 1 lần cho toàn bộ text trong frame
        if cached_detections:
            # Chỉ vẽ lại text KHÔNG bị suppress
            def _is_suppressed(text):
                for sup in SUPPRESS_TEXTS:
                    if sup in text or text in sup:
                        return True
                # Auto-suppress: text chỉ có Latin/số, không có chữ Trung → logo/watermark
                if AUTO_SUPPRESS_LATIN_ONLY and not any('\u4e00' <= c <= '\u9fff' for c in text):
                    return True
                return False

            non_suppressed = [d for d in cached_detections
                              if not _is_suppressed(d['text'])]
            drawable = merge_nearby_detections(non_suppressed)
            
            if drawable:
                # ── Bước 1: Build "render jobs" cho từng cụm chữ vẽ được ──
                LINE_SPACING = 12
                MARGIN = 8        # khoảng cách an toàn giữa 2 cụm chữ (px)
                MIN_SIZE = 22

                jobs = []
                
                for det in drawable:
                    vi_text = det.get('vi_text', det['text'])

                    # Guard: nếu vi_text rỗng hoặc vẫn chứa chữ Hán
                    # (translation thất bại / text chưa dịch) → bỏ qua vẽ
                    if not vi_text or any('\u4e00' <= c <= '\u9fff' for c in vi_text):
                        continue

                    # Tọa độ thực sau khi scale/offset 9:16
                    draw_x = int(det['x'] * content_scale + x_off)
                    draw_y = int(det['y'] * content_scale + y_off)
                    draw_box_h = int(det['box_height'] * content_scale)
                    draw_box_w = int(det.get('box_width', 100) * content_scale)

                    # Tọa độ X tâm để căn giữa chữ tiếng Việt
                    cx = draw_x + draw_box_w / 2

                    # Chiều rộng tối đa đối xứng qua tâm để tránh bị lệch và tràn mép trái/phải
                    max_w = int(min(cx, out_w - cx) * 2 - 30)
                    max_w = max(max_w, 150)
                    
                    # Cỡ chữ tỷ lệ theo frame mới
                    pref_size = max(MIN_SIZE, min(int(draw_box_h * 0.8), 36))

                    start_y = max(0, draw_y - 2)
                    jobs.append({
                        'text': vi_text, 'cx': cx, 'max_w': max_w,
                        'size': pref_size, 'y': start_y, 'orig_y': start_y,
                    })
                    
                    # (việc vẽ thực hiện sau khi đã layout xong toàn bộ jobs bên dưới)
                
                if jobs:
                    # ── Bước 2: Điều chỉnh tọa độ Y động để CHỐNG ĐÈ CHỮ ──
                    jobs = _layout_subtitle_blocks(
                        jobs, out_h, font_path,
                        margin=MARGIN, line_spacing=LINE_SPACING, min_size=MIN_SIZE,
                    )

                    # ── Bước 3: Convert 1 lần → (nền plate) → vẽ tất cả → convert lại ──
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    # Nền mờ bán trong suốt sau chữ (tăng độ đọc)
                    if SUBTITLE_BG_ENABLE:
                        pil_img = _draw_subtitle_backgrounds(
                            pil_img, jobs, font_path, LINE_SPACING, MIN_SIZE, out_w, out_h,
                        )

                    draw = ImageDraw.Draw(pil_img)

                    for j in jobs:
                        smart_draw_text_on_pil(
                            draw, j['text'], j['cx'], j['y'],
                            font_path, j['size'], j['max_w'],
                            TEXT_COLOR, OUTLINE_COLOR, OUTLINE_WIDTH,
                            out_h, out_w, min_size=MIN_SIZE,
                            line_spacing=LINE_SPACING, align="center",
                        )

                    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Ghi frame — NVENC pipe hoặc cv2
        if ffmpeg_proc and ffmpeg_proc.stdin:
            try:
                ffmpeg_proc.stdin.write(frame.tobytes())
            except OSError:
                raise RuntimeError(f"[Worker {chunk_idx}] ffmpeg/NVENC dừng giữa chừng (pipe vỡ) — xem lỗi ffmpeg in ở trên")
        elif out:
            out.write(frame)
        
        local_idx += 1

        if local_idx % 300 == 0:
            elapsed = time.time() - t0
            pct = local_idx / total_chunk * 100
            eta = elapsed / local_idx * (total_chunk - local_idx)
            print(f"  [Worker {chunk_idx}] {local_idx}/{total_chunk} ({pct:.0f}%) "
                  f"| {elapsed:.0f}s elapsed | ETA {eta:.0f}s")

    cap.release()
    if ffmpeg_proc:
        ffmpeg_proc.stdin.close()
        if ffmpeg_proc.wait() != 0:
            raise RuntimeError(f"[Worker {chunk_idx}] ffmpeg lỗi (mã {ffmpeg_proc.returncode}) khi encode {temp_output}")
    if out:
        out.release()
    elapsed = time.time() - t0
    print(f"  [Worker {chunk_idx}] Done in {elapsed:.0f}s -> {temp_output}")
    return chunk_idx, temp_output


# ──────────────────────────────────────────────
# CONCAT
# ──────────────────────────────────────────────

def concatenate_videos(temp_files, output_path):
    """Concatenate temp chunks using ffmpeg (fast stream copy), fallback to cv2."""
    # Đặt cạnh file output (thư mục job); UTF-8 vì ffmpeg đọc concat list theo UTF-8 và tên job có thể có dấu
    list_file = os.path.join(os.path.dirname(os.path.abspath(output_path)), "_concat_list.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for tf in temp_files:
            # Cú pháp concat của ffmpeg: dấu ' trong đường dẫn phải viết thành '\'' (tên video có dấu nháy)
            f.write("file '" + os.path.abspath(tf).replace("'", "'\\''") + "'\n")

    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
           "-i", list_file, "-c", "copy", output_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"  ffmpeg concat OK")
        else:
            raise RuntimeError(result.stderr)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"  [WARN] ffmpeg not available or failed: {e}")
        print("  Falling back to cv2 concat (slower)...")
        _cv2_concat(temp_files, output_path)
    finally:
        if os.path.exists(list_file):
            os.remove(list_file)


def _cv2_concat(temp_files, output_path):
    """Fallback: concat via cv2 frame by frame."""
    cap0 = cv2.VideoCapture(temp_files[0])
    fps = cap0.get(cv2.CAP_PROP_FPS)
    w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for tf in temp_files:
        cap = cv2.VideoCapture(tf)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
    out.release()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monggu Translate - Render Pass")
    parser.add_argument("--input", "-i", type=str, default=VIDEO_PATH, help="Đường dẫn video gốc")
    parser.add_argument("--test", action="store_true", default=TEST_MODE, help="Test mode")
    parser.add_argument("--workers", "-w", type=int, default=NUM_WORKERS, help="Số workers")
    parser.add_argument("--no-audio", action="store_true", default=False,
                        help="Không ghép audio (dùng khi muốn xử lý audio riêng bằng audio_tools.py)")
    args = parser.parse_args()

    print("=" * 55)
    print("  Monggu Translate - Pass 3: Render Video (GPU)")
    print("=" * 55)

    # ── GPU Status ──
    if USE_GPU:
        print(f"\n[GPU] NVENC encode: {'ON' if USE_NVENC else 'OFF'}")
        print(f"[GPU] NVDEC decode: {'ON' if USE_NVDEC else 'OFF'}")
        print(f"[GPU] CUDA ops   : {'ON' if USE_TORCH_CUDA and HAS_TORCH_CUDA else 'OFF'}")
        print(f"[GPU] Max load   : {GPU_MAX_LOAD*100:.0f}%")
        if HAS_TORCH_CUDA:
            print(f"[GPU] Device     : {torch.cuda.get_device_name(0)}")

    # ── Load pre-computed data ──
    print("\n[1/4] Loading OCR data + translation map...")

    try:
        job = open_job(args.input)
    except JobError as e:
        print(f"  [ERROR] {e}")
        sys.exit(1)
    if not os.path.exists(job.ocr_data):
        print(f"  [ERROR] Chưa có OCR cho {job.name} — hãy chạy: python ocr_pass.py --input \"{args.input}\"")
        sys.exit(1)
    if not os.path.exists(job.lines):
        print(f"  [ERROR] Chưa có bản dịch cho {job.name} — hãy chạy: python translate_pass.py --input \"{args.input}\"")
        sys.exit(1)

    with open(job.ocr_data, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)
    if ocr_data.get("complete") is False:
        print("  [ERROR] OCR của video này chưa chạy xong (bị ngắt?) — chạy tiếp OCR rồi dịch lại.")
        sys.exit(1)

    try:
        with open(job.lines, "r", encoding="utf-8") as f:
            lines = json.load(f)
    except ValueError as e:
        print(f"  [ERROR] {job.lines} sai cú pháp JSON ({e}) — sửa lại file rồi render lại.")
        sys.exit(1)

    frame_detections = ocr_data["frame_detections"]
    text_segments = ocr_data.get("text_segments", [])  # Có thể thiếu nếu ocr_data cũ

    # Bản dịch theo TỪNG LẦN XUẤT HIỆN: id segment → vi_sua (sửa tay) hoặc vi.
    # translation_map (chữ OCR thô → bản dịch lần xuất hiện đầu) chỉ dùng cho chữ thô không thuộc segment nào.
    if lines_mismatch(lines, text_segments):
        print(f"  [ERROR] lines.json không khớp ocr_data.json (OCR đã chạy lại sau khi dịch?) — "
              f"chạy lại: python translate_pass.py --input \"{args.input}\"")
        sys.exit(1)
    for i, seg in enumerate(text_segments):
        seg["id"] = i
    vi_for_seg, translation_map = {}, {}
    for l in sorted(lines, key=lambda l: l["start"]):
        vi = l["vi_sua"] if isinstance(l.get("vi_sua"), str) else (l.get("vi") or "")  # vi_sua "" = cố ý bỏ phụ đề
        vi_for_seg[l["id"]] = vi
        if l["id"] < len(text_segments):
            translation_map.setdefault(text_segments[l["id"]]["text"], vi)
    fps = ocr_data["fps"]
    width = ocr_data["width"]
    height = ocr_data["height"]
    total_frames = ocr_data["total_frames"]
    ocr_interval_frames = ocr_data["ocr_interval_frames"]

    if args.test:
        total_frames = min(total_frames, TEST_FRAMES)
        print(f"  [TEST MODE] Gioi han {TEST_FRAMES} frames")
    elif ocr_data.get("test_mode"):
        print(f"  [WARN] Dữ liệu OCR là của lần chạy --test (chỉ {total_frames} frame đầu) → video ra sẽ ngắn. "
              f"Chạy lại OCR không --test để có video đầy đủ.")

    print(f"  OCR detections : {len(frame_detections)} keyframes")
    print(f"  Text segments  : {len(text_segments)} tracked appearances")
    if not text_segments:
        print(f"  [WARN] Không có text_segments! Hãy chạy lại ocr_pass.py để tái tạo.")
    print(f"  Translations   : {len(lines)} dòng (lines.json)")
    print(f"  Video          : {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    # ── Analyze video (get path) ──
    print(f"\n[2/4] Chuan bi render...")

    # OCR đã convert fps (vd 60 → 25) thì render trên bản convert trong job: frame index trong ocr_data là của nó.
    # Tiếng luôn lấy từ video gốc (bản convert đã bỏ tiếng).
    video_path = working_video(job, ocr_data, args.input)
    audio_source = args.input
    if video_path != args.input and not os.path.exists(video_path):
        print(f"  [ERROR] Thiếu video đã convert fps: {video_path} (bị xóa?) — chạy lại OCR để tạo lại.")
        sys.exit(1)

    # Chặn render nhầm video: dữ liệu OCR phải khớp video sắp render
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Không mở được video: {video_path}")
        sys.exit(1)
    problem = video_mismatch(ocr_data, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                             cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    if problem:
        print(f"  [ERROR] Video '{video_path}' không khớp dữ liệu OCR: {problem}")
        print(f"  ocr_data.json là của '{ocr_data.get('original_video', '?')}' — hãy chạy lại OCR cho video này.")
        sys.exit(1)

    # Watermark chạy (wm_track.py): vị trí ở từng frame + mask nét chữ → xóa ở mọi frame
    wm_pos, wm_mask = load_wm_track(job, ocr_data["total_frames"])
    if wm_mask is not None:
        print(f"  Watermark chạy : xóa ở {len(wm_pos)} frame")

    output_video = job.output
    num_workers = args.workers

    print(f"  Input  : {video_path}")
    if video_path != audio_source:
        print(f"  Audio  : {audio_source}")
    print(f"  Output : {output_video}")
    print(f"  Workers: {num_workers}")

    # ── Split detections per chunk ──
    chunk_size = total_frames // num_workers
    chunks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = total_frames if i == num_workers - 1 else start + chunk_size
        chunks.append((start, end))

    temp_files = [os.path.join(job.dir, f"_temp_chunk_{i}.mp4") for i in range(num_workers)]

    # Split frame_detections per worker chunk
    chunk_detections = []
    chunk_segments = []
    for i, (start, end) in enumerate(chunks):
        chunk_det = {}
        for frame_str, dets in frame_detections.items():
            frame_idx = int(frame_str)
            if start <= frame_idx < end:
                chunk_det[frame_str] = dets
        chunk_detections.append(chunk_det)

        # Split text_segments: giữ các segment có giao với chunk
        chunk_seg = [
            seg for seg in text_segments
            if seg['end_frame'] >= start and seg['start_frame'] < end
        ]
        chunk_segments.append(chunk_seg)

    chunk_wm = [{f: p for f, p in wm_pos.items() if s <= f < e} for s, e in chunks]

    print(f"\n[3/4] Rendering {num_workers} chunks in parallel...")
    for i, (s, e) in enumerate(chunks):
        det_count = len(chunk_detections[i])
        seg_count = len(chunk_segments[i])
        print(f"  Chunk {i}: frames {s}-{e-1} ({e-s} frames, {det_count} keyframes, {seg_count} segments)")

    worker_args = [
        (i, chunks[i][0], chunks[i][1],
         video_path, temp_files[i],
         FONT_PATH, fps, width, height,
         ocr_interval_frames, translation_map,
         chunk_detections[i], chunk_segments[i], vi_for_seg,
         chunk_wm[i], wm_mask)
        for i in range(num_workers)
    ]

    wall_start = time.time()
    # File trung gian nằm trong job; video ra CHỈ được thay ở bước cuối (render lỗi giữa chừng không làm
    # mất video tốt của lần trước); luôn dọn file tạm, kể cả khi lỗi / Ctrl+C.
    staging = os.path.join(job.dir, "_render_tmp.mp4")
    staging_audio = os.path.join(job.dir, "_render_tmp_audio.mp4")

    def _publish(src, dst):
        try:
            os.replace(src, dst)
        except PermissionError:
            print(f"  [ERROR] Không ghi đè được {dst} — file đang mở ở chương trình khác (trình phát video?). "
                  f"Đóng lại rồi chạy lại render.")
            sys.exit(1)

    mux_ok = True
    try:
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(process_chunk, worker_args)
        print(f"\n  All workers finished in {time.time() - wall_start:.1f}s")

        # ── Concatenate ──
        print(f"\n[4/4] Concatenating -> {output_video}...")
        ordered_temps = [r[1] for r in sorted(results, key=lambda r: r[0])]
        concatenate_videos(ordered_temps, staging)

        # ── Mux audio từ video gốc ──
        if args.no_audio:
            print(f"\n[5/5] Bỏ qua mux audio (--no-audio)")
            print(f"  → Dùng audio_tools.py để xử lý audio riêng:")
            print(f"    python audio_tools.py extract {video_path}")
            print(f"    python audio_tools.py mux {output_video} <audio_file>")
            _publish(staging, output_video)
        else:
            print(f"\n[5/5] Mux audio tu video goc...")
            mux_cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", staging,              # Video đã render (không có audio)
                "-i", audio_source,         # Video gốc (lấy audio)
                "-c:v", "copy",             # Copy video stream (không re-encode)
                "-c:a", "aac",              # Encode audio AAC
                "-map", "0:v:0",            # Video từ file đầu
                "-map", "1:a:0?",           # Audio từ file gốc
                "-shortest",                # Cắt theo stream ngắn nhất
                staging_audio
            ]
            try:
                result = subprocess.run(mux_cmd, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    _publish(staging_audio, output_video)
                    print(f"  Audio mux OK")
                else:
                    print(f"  [ERROR] Mux audio thất bại: {result.stderr[:200]}")
                    mux_ok = False
            except Exception as e:
                print(f"  [ERROR] ffmpeg không khả dụng cho audio mux: {e}")
                mux_ok = False
            if not mux_ok:
                silent = os.path.splitext(output_video)[0] + "_khong_tieng.mp4"
                _publish(staging, silent)
                print(f"  Video (không tiếng) ở: {silent} — video ra của lần trước (nếu có) giữ nguyên")
    finally:
        for tf in temp_files + [staging, staging_audio]:
            with contextlib.suppress(OSError):
                os.remove(tf)

    if not mux_ok:
        sys.exit(1)

    total_time = time.time() - wall_start
    print(f"\n{'=' * 55}")
    print(f"  Done!")
    print(f"  Output         : {output_video}")
    print(f"  Total wall time: {total_time:.1f}s")
    print(f"  Avg speed      : {total_frames / total_time:.1f} fps (wall clock)")
    print(f"{'=' * 55}")