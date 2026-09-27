"""
Monggu Translate - Pass 1: OCR toàn bộ video
Quét OCR → dedup → xuất ocr_data.json

Usage:
    python ocr_pass.py                  # OCR video mặc định (config.VIDEO_PATH)
    python ocr_pass.py --input my.mp4   # OCR video chỉ định
    python ocr_pass.py --test           # Chỉ xử lý TEST_FRAMES frames
"""

import os
import sys

# Đảm bảo console stdout/stderr in được tiếng Việt Unicode trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

os.environ["FLAGS_use_mkldnn"] = "0"

import cv2
import numpy as np
import time
import json
import argparse
import subprocess
from difflib import SequenceMatcher
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

from config import (
    VIDEO_PATH, TEST_MODE, TEST_FRAMES,
    OCR_LANG, OCR_INTERVAL_SEC, OCR_CONFIDENCE_THRESHOLD,
    OCR_RESIZE_FACTOR, CROP_TOP_RATIO, WATERMARK_PATTERNS,
    SUPPRESS_TEXTS, TARGET_FPS,
    USE_GPU,
    OCR_SCENE_DIFF, OCR_SCENE_DIFF_THRESHOLD, OCR_SCENE_DIFF_WIDTH, OCR_MAX_SKIP_SEC,
)
from utils import write_json_atomic
from jobs import open_job, JobError


def calculate_iou(boxA, boxB):
    """Tính Intersection over Union (IoU) giữa 2 box dạng [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    denominator = float(boxAArea + boxBArea - interArea)
    if denominator <= 0:
        return 0.0
    return interArea / denominator


def build_text_segments(frame_detections: dict, fps: float, ocr_interval_frames: int) -> list:
    """
    Từ frame_detections, group các keyframe liên tiếp có cùng hoặc tương đồng text
    và vị trí địa lý thành các segment.
    """
    if not frame_detections or fps <= 0:
        return []

    # Sắp xếp keyframes theo thứ tự
    sorted_keys = sorted(int(k) for k in frame_detections.keys())

    # active: list of segment dicts
    active: list = []
    finished: list = []

    # Khoảng "gap" tối đa dựa theo FPS của video (khoảng 0.4s) để tránh bị trượt do OCR miss
    max_gap_frames = max(ocr_interval_frames * 5, int(fps * 0.4), 12)

    for kf in sorted_keys:
        dets = frame_detections[str(kf)]
        updated_segs = set()

        # Cập nhật active segments
        for det in dets:
            txt = det['text']
            box = det['box']
            # Tính bounding rect của box
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            brect = [min(xs), min(ys), max(xs), max(ys)]  # [x1, y1, x2, y2]

            matched_seg = None
            for seg in active:
                if id(seg) in updated_segs:
                    continue

                last_bbox = seg['bboxes'][-1]
                iou = calculate_iou(last_bbox, brect)

                is_match = False
                # 1. Khớp tuyệt đối và có tương tác vị trí tối thiểu
                if seg['text'] == txt and iou > 0.1:
                    is_match = True
                # 2. Khớp tương đối (fuzzy) và có độ chồng chéo trung bình
                elif texts_are_similar(seg['text'], txt, 0.7) and iou > 0.3:
                    is_match = True
                # KHÔNG gộp chỉ vì trùng vị trí: câu phụ đề nối tiếp luôn ở cùng chỗ, gộp theo vị trí
                # làm câu sau "nuốt" câu trước (đo trên dữ liệu thật: -90% frame vẽ sai câu).

                if is_match:
                    matched_seg = seg
                    break

            if matched_seg:
                matched_seg['last_frame'] = kf
                matched_seg['bboxes'].append(brect)
                matched_seg['keyframes'].append(kf)

                # Chọn đại diện text tốt nhất (ưu tiên chữ Trung và độ dài)
                def count_chinese(s):
                    return sum(1 for c in s if '\u4e00' <= c <= '\u9fff')

                if count_chinese(txt) > count_chinese(matched_seg['text']):
                    matched_seg['text'] = txt
                elif count_chinese(txt) == count_chinese(matched_seg['text']) and len(txt) > len(matched_seg['text']):
                    matched_seg['text'] = txt

                updated_segs.add(id(matched_seg))
            else:
                # Tạo segment mới
                new_seg = {
                    'text': txt,
                    'start_frame': kf,
                    'last_frame': kf,
                    'bboxes': [brect],
                    'keyframes': [kf]
                }
                active.append(new_seg)
                updated_segs.add(id(new_seg))

        # Đóng các segment đã hết hoạt động lâu hơn max_gap_frames
        still_active = []
        for seg in active:
            if kf - seg['last_frame'] > max_gap_frames:
                finished.append(_make_segment(seg['text'], seg, fps))
            else:
                still_active.append(seg)
        active = still_active

    # Close tất cả segment còn lại
    for seg in active:
        finished.append(_make_segment(seg['text'], seg, fps))

    return finished


def _make_segment(text: str, seg: dict, fps: float) -> dict:
    """Tạo dict segment từ accumulated data."""
    bboxes = seg['bboxes']
    keyframes = seg.get('keyframes', [])  # có thể thiếu nếu data cũ
    # Merge tất cả bbox thành 1 union bbox (dùng làm fallback)
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    start_f = seg['start_frame']
    end_f = seg['last_frame']
    return {
        'text': text,
        'start_frame': start_f,
        'end_frame': end_f,
        'start_sec': round(start_f / fps, 3),
        'end_sec': round(end_f / fps, 3),
        'bbox': [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],  # union bbox, fallback
        # Danh sách (frame_idx, bbox) để render pass interpolate vị trí chính xác
        'keyframe_bboxes': [
            {'frame': kf, 'bbox': [round(bb[0], 1), round(bb[1], 1),
                                   round(bb[2], 1), round(bb[3], 1)]}
            for kf, bb in zip(keyframes, bboxes)
        ],
    }


def is_watermark(text: str) -> bool:
    """Kiểm tra text có phải watermark/UI element không.

    - SUPPRESS_TEXTS: OCR vẫn giữ lại (để render pass biết vị trí inpaint xóa),
      nhưng KHÔNG dịch. Vì vậy hàm này return False cho các text đó.
    - WATERMARK_PATTERNS: bỏ qua hoàn toàn (không OCR, không inpaint).
    """
    text_lower = text.lower()

    # 0. Ưu tiên: nếu text thuộc SUPPRESS_TEXTS → KHÔNG lọc ở đây!
    #    Render pass sẽ inpaint xóa nhưng không dịch.
    text_stripped = text.strip()
    for suppress in SUPPRESS_TEXTS:
        if suppress.lower() in text_lower or suppress in text_stripped:
            return False

    # 1. Exact-substring match với WATERMARK_PATTERNS
    for pattern in WATERMARK_PATTERNS:
        if pattern.lower() in text_lower:
            return True

    # Không lọc "chữ Trung lẫn > 30% Latin" nữa: đo trên 15 video thật, luật đó bỏ cả thoại chen tiếng lóng
    # ("就是这个feel") → để nguyên chữ Trung. Nhãn kiểu "猫meme" giờ được B3 lọc (chuẩn hóa/dịch coi là rác → xóa).
    return False


def texts_are_similar(text1: str, text2: str, threshold: float = 0.85) -> bool:
    """So sánh fuzzy 2 chuỗi text, return True nếu giống >= threshold."""
    if not text1 or not text2:
        return False
    return SequenceMatcher(None, text1, text2).ratio() >= threshold


def ocr_frame(ocr_engine, frame, height, width, crop_top_ratio, resize_factor):
    """
    Chạy OCR trên 1 frame, trả về list detections.
    Mỗi detection: {'text': str, 'box': [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], 'confidence': float}
    """
    crop_y_offset = int(height * crop_top_ratio)
    crop = frame[crop_y_offset:, 0:width]

    # Downscale để OCR nhanh hơn
    if resize_factor < 1.0:
        h_c, w_c = crop.shape[:2]
        small = cv2.resize(crop, (int(w_c * resize_factor), int(h_c * resize_factor)))
        result = ocr_engine.ocr(small, cls=True)
        # Scale tọa độ ngược lại
        if result and result[0]:
            for line in result[0]:
                for pt in line[0]:
                    pt[0] /= resize_factor
                    pt[1] /= resize_factor
    else:
        result = ocr_engine.ocr(crop, cls=True)

    detections = []
    if not result or not result[0]:
        return detections

    for line in result[0]:
        box, (text, confidence) = line[0], line[1]

        # Lọc confidence thấp
        if confidence < OCR_CONFIDENCE_THRESHOLD:
            continue

        # Lọc watermark
        if is_watermark(text):
            continue

        # Điều chỉnh tọa độ y theo crop offset
        adjusted_box = []
        for pt in box:
            adjusted_box.append([float(pt[0]), float(pt[1]) + crop_y_offset])

        detections.append({
            'text': text.strip(),
            'box': adjusted_box,
            'confidence': round(float(confidence), 4),
        })

    return detections


def _downscale_gray(frame, target_w: int):
    """Ảnh xám downscale để so sánh nhanh độ khác biệt giữa 2 frame."""
    h, w = frame.shape[:2]
    if w <= 0:
        return None
    scale = target_w / float(w)
    small = cv2.resize(frame, (target_w, max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def convert_video_fps(input_path: str, target_fps: int, out_dir: str) -> str:
    """Convert video to target FPS using ffmpeg NVENC. Returns path to converted video."""
    base, ext = os.path.splitext(os.path.basename(input_path))
    converted_path = os.path.join(out_dir, f"{base}_{target_fps}fps{ext}")

    # Check if already converted
    if os.path.exists(converted_path):
        cap = cv2.VideoCapture(converted_path)
        if cap.isOpened():
            existing_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if abs(existing_fps - target_fps) < 2:
                print(f"  Da ton tai: {converted_path} ({existing_fps:.0f}fps) -> bo qua convert")
                return converted_path
        else:
            cap.release()

    print(f"  Converting -> {target_fps}fps (NVENC)...")
    cmd = [
        'ffmpeg', '-y', '-hwaccel', 'cuda',
        '-i', input_path,
        '-vf', f'fps={target_fps}',
        '-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '18',
        '-an',
        converted_path
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  [WARN] NVENC failed, trying CPU libx264...")
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-vf', f'fps={target_fps}',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-an',
            converted_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  [ERROR] Convert that bai: {result.stderr[:200]}")
            return input_path

    elapsed = time.time() - t0
    print(f"  Convert OK -> {converted_path} ({elapsed:.1f}s)")
    return converted_path


def _save_intermediate_checkpoint(data_path, test_mode, video_path, original_video_path, fps, width, height, total_frames, ocr_interval_frames, unique_texts, frame_detections):
    """Lưu tiến trình OCR tạm thời (complete=False → dịch/render từ chối dùng)."""
    try:
        temp_segments = build_text_segments(frame_detections, fps, ocr_interval_frames)
        ocr_data = {
            "video": os.path.basename(video_path),
            "original_video": os.path.basename(original_video_path),
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "ocr_interval_frames": ocr_interval_frames,
            "complete": False,
            "test_mode": test_mode,
            "unique_texts": unique_texts,
            "frame_detections": frame_detections,
            "text_segments": temp_segments,
        }
        write_json_atomic(data_path, ocr_data)
        print(f"\n  [CHECKPOINT] Đã tự động lưu tiến trình: {len(frame_detections)} frames.")
    except Exception as e:
        print(f"\n  [WARN] Không thể lưu checkpoint: {e}")


def run_ocr_pass(video_path: str, test_mode: bool = False, test_frames: int = 300, series: str = None):
    """
    Pass 1: OCR toàn bộ video, thu thập text + vị trí.
    Output: jobs/<tên>/ocr_data.json. Trả về ocr_data, hoặc None nếu không mở được video / bị Ctrl+C.
    """
    print("=" * 55)
    print("  Monggu Translate - Pass 1: OCR Scan")
    print("=" * 55)

    try:
        job = open_job(video_path, series, create=True)
    except JobError as e:
        print(f"  [ERROR] {e}")
        return None
    data_path = job.ocr_data

    # ── Phân tích video ──
    print("\n[1/3] Phân tích video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Không mở được video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if test_mode:
        total_frames = min(total_frames, test_frames)
        print(f"  [TEST MODE] Giới hạn {test_frames} frames")

    duration = total_frames / fps if fps > 0 else 0

    print(f"  Resolution : {width}x{height}")
    print(f"  FPS        : {fps:.1f}")
    print(f"  Frames     : {total_frames}")
    print(f"  Duration   : {duration:.1f}s")

    # ── Convert FPS nếu cần ──
    original_video_path = video_path
    if TARGET_FPS > 0 and fps > TARGET_FPS + 1:
        print(f"\n  Video {fps:.0f}fps > target {TARGET_FPS}fps -> convert de giam workload")
        cap.release()
        video_path = convert_video_fps(video_path, TARGET_FPS, job.dir)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if test_mode:
            total_frames = min(total_frames, test_frames)
        duration = total_frames / fps if fps > 0 else 0
        print(f"  Converted  : {fps:.0f}fps, {total_frames} frames, {duration:.1f}s")

    ocr_interval_frames = max(1, int(fps * OCR_INTERVAL_SEC))
    print(f"  OCR mỗi    : {ocr_interval_frames} frames ({OCR_INTERVAL_SEC}s)")

    # Import lười: PaddleOCR nặng (~15s); để module import nhanh cho test/tool dùng build_text_segments
    import paddle
    from paddleocr import PaddleOCR

    # ── Khởi tạo PaddleOCR (1 engine per worker thread) ──
    # GPU chỉ khi Paddle THẬT SỰ có CUDA (bản paddlepaddle CPU lặng lẽ chạy CPU → log cũ "use_gpu=True" sai).
    # 1 worker cả khi chạy CPU: Paddle CPU đã tự đa luồng — đo trên máy 8 luồng, 500 frame: 1 worker 45.7s,
    # 4 worker 47.6s (kết quả y hệt) → nhiều engine chỉ tốn gấp 4 RAM. GPU (GTX 1650 4GB): 1 worker cho khỏi tràn VRAM.
    ocr_gpu = USE_GPU and paddle.device.is_compiled_with_cuda()
    current_num_workers = 1
    print(f"\n[2/3] Khởi tạo {current_num_workers} PaddleOCR workers (use_gpu={ocr_gpu})...")

    # Tạo 1 engine riêng cho mỗi worker để tránh GIL contention
    ocr_engines = []
    for i in range(current_num_workers):
        eng = PaddleOCR(use_angle_cls=True, lang=OCR_LANG, show_log=False, use_gpu=ocr_gpu)
        ocr_engines.append(eng)
        print(f"  Worker {i + 1}/{current_num_workers} sẵn sàng")

    # ── Quét OCR (multi-threaded producer-consumer) ──
    print(f"\n[3/3] Đang quét OCR toàn bộ video ({current_num_workers} threads)...")

    frame_detections = {}   # frame_idx -> list of detections
    unique_texts = []       # Danh sách text unique theo thứ tự xuất hiện
    seen_texts = set()      # Để track text đã thấy
    prev_frame_texts = []   # Text của frame trước (để dedup liên tiếp)
    start_frame = 0

    # ── Kiểm tra checkpoint cũ để resume ──
    if os.path.exists(data_path):
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            # Kiểm tra đúng video
            if (old_data.get("original_video") == os.path.basename(original_video_path) or
                old_data.get("video") == os.path.basename(video_path)):
                frame_detections = old_data.get("frame_detections", {})
                unique_texts = old_data.get("unique_texts", [])
                seen_texts = set(unique_texts)
                
                scanned_frames = [int(k) for k in frame_detections.keys()]
                if scanned_frames:
                    start_frame = max(scanned_frames) + ocr_interval_frames
                    print(f"  [RESUME] Phát hiện tiến trình cũ. Sẽ tiếp tục từ frame {start_frame}/{total_frames}...")
        except Exception as e:
            print(f"  [WARN] Không đọc được checkpoint cũ, bắt đầu mới: {e}")

    # Di chuyển con trỏ video nếu resume
    if start_frame > 0 and start_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
    else:
        frame_idx = 0

    # Queue: producer đẩy (frame_idx, frame), workers lấy ra OCR
    task_q = queue.Queue(maxsize=current_num_workers * 8)
    # Queue: workers đẩy (frame_idx, detections) về cho collector
    result_q = queue.Queue()

    _SENTINEL = None  # Tín hiệu kết thúc cho workers

    def worker_fn(worker_id: int):
        """Worker thread: lấy frame từ task_q, chạy OCR, đẩy kết quả vào result_q."""
        engine = ocr_engines[worker_id]
        while True:
            item = task_q.get()
            if item is _SENTINEL:
                task_q.task_done()
                break
            fidx, frm = item
            try:
                dets = ocr_frame(engine, frm, height, width, CROP_TOP_RATIO, OCR_RESIZE_FACTOR)
            except Exception as e:
                print(f"  [WARN] Worker {worker_id} lỗi frame {fidx}: {e}")
                dets = []
            result_q.put((fidx, dets))
            task_q.task_done()

    # Khởi động worker threads
    # Stack lớn cho luồng OCR: mã C++ của Paddle (predict_rec) đôi khi tràn stack mặc định của luồng phụ trên Windows
    # → "stack overflow / access violation" giết cả tiến trình giữa chừng (gặp ở video Douyin 36 phút, ~10-15 phút/lần).
    threading.stack_size(64 * 1024 * 1024)
    workers = []
    for wid in range(current_num_workers):
        t = threading.Thread(target=worker_fn, args=(wid,), daemon=True)
        t.start()
        workers.append(t)

    t0 = time.time()
    ocr_count = len(frame_detections)
    submitted = 0   # Số frame đã đẩy vào queue trong phiên này
    last_ckpt_submitted = 0  # Mốc `submitted` của lần lưu checkpoint gần nhất
    collected = 0   # Số kết quả đã thu thập trong phiên này
    skipped_diff = 0  # Số frame bị bỏ qua vì trùng lặp với frame trước
    interrupted = False

    # ── Scene-diff state: bỏ qua các frame gần như giống hệt frame OCR gần nhất ──
    # Giới hạn skip tối đa để keyframe vẫn đủ dày, segment không bị đứt đoạn.
    max_skip_frames = max(ocr_interval_frames, int(fps * OCR_MAX_SKIP_SEC)) if fps > 0 else ocr_interval_frames
    last_ocr_small = None      # ảnh xám downscale của frame đã OCR gần nhất
    last_submit_idx = -10 ** 9  # frame index đã submit gần nhất
    if OCR_SCENE_DIFF:
        print(f"  [SCENE-DIFF] Bật bỏ qua frame trùng (threshold={OCR_SCENE_DIFF_THRESHOLD}, "
              f"max skip={max_skip_frames} frames)")

    def collect_results(blocking: bool = False):
        """Thu thập kết quả từ result_q (non-blocking hoặc blocking cho tất cả)."""
        nonlocal ocr_count, collected
        while True:
            try:
                timeout = None if blocking and collected < submitted else 0
                fidx, dets = result_q.get(block=(timeout is None), timeout=timeout if timeout else 0.001)
                collected += 1
                if dets:
                    frame_detections[str(fidx)] = dets
                    ocr_count += 1
                    current_texts = [d['text'] for d in dets]
                    for text in current_texts:
                        if text not in seen_texts:
                            seen_texts.add(text)
                            unique_texts.append(text)
            except queue.Empty:
                break

    try:
        # Producer: đọc frame và đẩy vào task_q
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % ocr_interval_frames == 0:
                do_submit = True
                small = None
                if OCR_SCENE_DIFF:
                    small = _downscale_gray(frame, OCR_SCENE_DIFF_WIDTH)
                    if (small is not None and last_ocr_small is not None
                            and small.shape == last_ocr_small.shape):
                        diff = float(np.mean(cv2.absdiff(small, last_ocr_small)))
                        frames_since = frame_idx - last_submit_idx
                        # Trùng lặp + chưa skip quá lâu → bỏ qua OCR frame này
                        if diff < OCR_SCENE_DIFF_THRESHOLD and frames_since < max_skip_frames:
                            do_submit = False
                            skipped_diff += 1

                if do_submit:
                    task_q.put((frame_idx, frame))  # Block nếu queue đầy (back-pressure)
                    submitted += 1
                    last_submit_idx = frame_idx
                    if small is not None:
                        last_ocr_small = small

            # Thu thập kết quả sẵn có (non-blocking) để giải phóng result_q
            collect_results(blocking=False)

            frame_idx += 1

            # Progress log
            if frame_idx % 300 == 0:
                elapsed = time.time() - t0
                pct = frame_idx / total_frames * 100
                eta = elapsed / frame_idx * (total_frames - frame_idx) if frame_idx > 0 else 0
                print(f"  Frame {frame_idx}/{total_frames} ({pct:.0f}%) "
                      f"| {elapsed:.0f}s | ETA {eta:.0f}s "
                      f"| OCR scans: {ocr_count} | Unique texts: {len(unique_texts)}")

            # Tự động lưu tiến trình sau mỗi 1000 frame thực sự OCR trong phiên này.
            # (Không dùng `submitted % 1000`: frame bị scene-diff bỏ qua giữ nguyên submitted → lưu lặp file 24MB.)
            if submitted - last_ckpt_submitted >= 1000:
                last_ckpt_submitted = submitted
                collect_results(blocking=False)
                _save_intermediate_checkpoint(
                    data_path, test_mode,
                    video_path, original_video_path, fps, width, height, total_frames,
                    ocr_interval_frames, unique_texts, frame_detections
                )
    except KeyboardInterrupt:
        print("\n\n  [INFO] Nhận thấy Ctrl+C! Đang dừng các worker threads và lưu tiến trình...")
        interrupted = True

    cap.release()

    # Gửi sentinel để workers dừng
    for _ in range(current_num_workers):
        task_q.put(_SENTINEL)

    # Chờ tất cả workers xong và thu thập kết quả còn lại
    for w in workers:
        w.join()
    collect_results(blocking=True)

    elapsed = time.time() - t0

    # ── Build text segments ──
    print("\n  Building text segments (start/end time tracking)...")
    text_segments = build_text_segments(frame_detections, fps, ocr_interval_frames)
    print(f"  Segments: {len(text_segments)} text appearances tracked")
    for seg in text_segments[:8]:  # preview
        print(f"    [{seg['start_sec']:.1f}s → {seg['end_sec']:.1f}s] "
              f"bbox={[int(v) for v in seg['bbox']]}  \"{seg['text']}\"")
    if len(text_segments) > 8:
        print(f"    ... và {len(text_segments) - 8} segment nữa")

    # ── Xuất JSON ──
    ocr_data = {
        "video": os.path.basename(video_path),
        "original_video": os.path.basename(original_video_path),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "ocr_interval_frames": ocr_interval_frames,
        "complete": not interrupted,
        "test_mode": test_mode,
        "unique_texts": unique_texts,
        "frame_detections": frame_detections,
        "text_segments": text_segments,
    }

    write_json_atomic(data_path, ocr_data)

    if interrupted:
        print(f"\n{'=' * 55}")
        print(f"  ĐÃ LƯU TIẾN TRÌNH BỊ NGẮT QUÃNG!")
        print(f"  Frames scanned: {ocr_count} (Tổng cộng: {len(frame_detections)})")
        print(f"  Unique texts  : {len(unique_texts)}")
        print(f"  Output        : {data_path}")
        print(f"  Bạn có thể chạy lại script để tiếp tục quét từ frame {frame_idx}.")
        print(f"{'=' * 55}")
        return None  # Chưa xong → báo lỗi để pipeline không chạy tiếp với dữ liệu dở dang
    else:
        print(f"\n{'=' * 55}")
        print(f"  OCR Pass hoàn tất!")
        print(f"  Thời gian     : {elapsed:.1f}s")
        print(f"  Frames scanned: {ocr_count}")
        if OCR_SCENE_DIFF:
            print(f"  Frames skipped: {skipped_diff} (trùng lặp, tiết kiệm OCR)")
        print(f"  Unique texts  : {len(unique_texts)}")
        print(f"  Text segments : {len(text_segments)}")
        print(f"  Output        : {data_path}")
        print(f"{'=' * 55}")

    return ocr_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monggu Translate - OCR Pass")
    parser.add_argument("--input", "-i", type=str, default=VIDEO_PATH, help="Đường dẫn video")
    parser.add_argument("--test", action="store_true", default=TEST_MODE, help="Test mode (giới hạn frames)")
    parser.add_argument("--series", "-s", type=str, default=None, help="Tên bộ (series) nếu video là một tập")
    args = parser.parse_args()

    result = run_ocr_pass(args.input, test_mode=args.test, test_frames=TEST_FRAMES, series=args.series)
    sys.exit(0 if result else 1)
