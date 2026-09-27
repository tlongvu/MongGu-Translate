"""Video tổng hợp + ocr_data giả cho test wm_track: chữ bán trong suốt (lõi trắng, viền tối) trôi trên nền
có vân và trôi theo, có một đoạn gần như vô hình, đổi hướng giữa chừng, và một mảng nền trơn (nơi NCC dễ khớp nhầm)."""
import cv2
import numpy as np

FPS, N, W, H = 25.0, 240, 480, 270
FAINT = range(40, 80)      # đoạn gần như vô hình (độ mờ 0.08), nằm trên đoạn thẳng
TURN = 150                 # đổi hướng (lúc watermark còn rõ)
DISTRACT = range(45, 76)   # vật giống hệt chữ nhưng ĐỨNG YÊN, lúc watermark thật đang mờ (chỉ khi distractor=True)
DISTRACT_AT = (400, 220)


def truth(t):
    """Tâm watermark ở frame t: thẳng tới TURN rồi rẽ."""
    if t <= TURN:
        return np.array([60 + 1.6 * t, 60 + 0.7 * t])
    return truth(TURN) + np.array([-1.5, 0.5]) * (t - TURN)


def glyph():
    """(alpha viền+lõi, lõi) của chữ 'WM@12', cắt sát."""
    font, sc = cv2.FONT_HERSHEY_SIMPLEX, 0.9
    (tw, th), base = cv2.getTextSize("WM@12", font, sc, 2)
    edge = np.zeros((th + base + 16, tw + 16), np.uint8)
    fill = edge.copy()
    cv2.putText(edge, "WM@12", (8, th + 8), font, sc, 255, 6)
    cv2.putText(fill, "WM@12", (8, th + 8), font, sc, 255, 2)
    ys, xs = np.nonzero(edge)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    return edge[sl].astype(np.float32) / 255, fill[sl].astype(np.float32) / 255


def make_video(path, distractor=None, faint=FAINT, track=truth):
    """Ghi video mp4v; trả về (tâm thật từng frame, (rộng, cao) của chữ).
    faint: các frame chữ gần như vô hình. distractor = (các frame, (x, y)): thêm vật giống hệt chữ nhưng đứng yên.
    track: hàm frame → tâm chữ (mặc định đường thẳng rồi rẽ)."""
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur(rng.integers(0, 256, (H + 40, W + N + 40)).astype(np.float32), (0, 0), 2)
    tex = cv2.normalize(tex, None, 40, 200, cv2.NORM_MINMAX)
    edge, fill = glyph()
    gh, gw = edge.shape
    ov = np.where(fill > 0.5, 255.0, 25.0)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    centers = []
    for t in range(N):
        bg = tex[20:20 + H, t:t + W].copy()           # nền trôi 1 px/frame
        bg[10:80, 330:470] = 200.0                    # mảng trơn
        if distractor and t in distractor[0]:
            dx0, dy0 = int(distractor[1][0] - gw / 2), int(distractor[1][1] - gh / 2)
            droi = bg[dy0:dy0 + gh, dx0:dx0 + gw]
            bg[dy0:dy0 + gh, dx0:dx0 + gw] = (1 - 0.6 * edge) * droi + 0.6 * edge * ov
        c = track(t)
        x0, y0 = int(round(c[0] - gw / 2)), int(round(c[1] - gh / 2))
        a = (0.08 if t in faint else 0.6) * edge
        vx0, vy0, vx1, vy1 = max(0, x0), max(0, y0), min(W, x0 + gw), min(H, y0 + gh)   # phần chữ trong khung
        gs = (slice(vy0 - y0, vy1 - y0), slice(vx0 - x0, vx1 - x0))
        bg[vy0:vy1, vx0:vx1] = (1 - a[gs]) * bg[vy0:vy1, vx0:vx1] + a[gs] * ov[gs]
        centers.append((x0 + gw / 2, y0 + gh / 2))
        vw.write(cv2.cvtColor(np.clip(bg, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    vw.release()
    return np.array(centers), (gw, gh)


def fake_ocr(centers, size, video_name, faint=FAINT, extra=()):
    """ocr_data giả: 'WM@12' đọc đủ mỗi 8 frame (trừ đoạn mờ) + các frame trong extra (OCR hiếm hoi đọc được
    lúc mờ) + 1 câu thoại đứng yên."""
    gw, gh = size
    fd = {}
    for t in sorted({t for t in range(0, N, 8) if t not in faint} | set(extra)):
        cx, cy = centers[t]
        x0, y0 = max(0.0, cx - gw / 2), max(0.0, cy - gh / 2)            # như OCR thật: chỉ phần chữ trong khung
        x1, y1 = min(float(W), cx + gw / 2), min(float(H), cy + gh / 2)
        fd[str(t)] = [{"text": "WM@12", "confidence": 0.95,
                       "box": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}]
    for t in range(100, 160, 4):
        fd.setdefault(str(t), []).append({"text": "你好世界啊", "confidence": 0.99,
                                         "box": [[180, 230], [300, 230], [300, 260], [180, 260]]})
    return {"video": video_name, "original_video": video_name, "fps": FPS, "width": W, "height": H,
            "total_frames": N, "ocr_interval_frames": 1, "frame_detections": fd, "text_segments": [],
            "complete": True}
