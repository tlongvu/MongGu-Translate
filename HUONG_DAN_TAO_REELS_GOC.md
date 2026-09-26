# Hướng dẫn tạo Reels meme AI "nguyên bản" bằng open-source
### Tận dụng pipeline `Monggu Translate` sẵn có — GTX 1650 (4GB VRAM)

---

## 0. Vì sao đổi hướng này lại quan trọng

Kênh của bạn bị Facebook hạn chế đề xuất vì lỗi **"unoriginal content"** — lấy video người khác rồi dịch/lồng tiếng và đăng lại. Dù làm kỹ tới đâu, đây vẫn là *nội dung tái sử dụng*, nên sẽ tiếp tục bị phạt và có rủi ro bản quyền.

Hướng đi bền vững: **tự sinh ra hình ảnh, kịch bản và giọng nói của riêng bạn**. Lúc đó nội dung là "do bạn tạo", không còn dính lỗi tái sử dụng, và **bạn không cần dùng các chiêu lách như `TIKTOK_PASS_MODE`** nữa (nên tắt hẳn — xem mục 5).

Tin tốt: ~80% pipeline kỹ thuật bạn đã có (ffmpeg/NVENC, render phụ đề bằng Pillow, edge-tts, tách nhạc Demucs, API LLM). Chỉ cần thay khâu "lấy video gốc" bằng khâu "tự sinh hình ảnh".

---

## 1. Giới hạn phần cứng cần biết trước (RẤT QUAN TRỌNG)

GTX 1650 chỉ có **4GB VRAM**. Đây là yếu tố quyết định công cụ nào chạy được:

| Tác vụ | Chạy local trên 4GB? | Ghi chú |
|---|---|---|
| Sinh **ảnh** (SD 1.5) | ✅ Được | Chậm hơn card to, nhưng ổn |
| Sinh ảnh (SDXL / FLUX) | ⚠️ Rất chậm / hay tràn VRAM | Nên dùng cloud free |
| Sinh **video AI** (Wan, LTX, Hunyuan, CogVideoX) | ❌ Không | Tất cả cần **12GB+** VRAM |
| Animate ảnh tĩnh (Ken Burns/ffmpeg) | ✅ Được, rất nhẹ | Đây là cách chính cho meme reels |
| Lip-sync nhân vật nói (LivePortrait) | ⚠️ Tùy, nặng | Cân nhắc, hoặc dùng cloud |
| Lồng tiếng TTS | ✅ Được (CPU cũng chạy) | |

**Kết luận:** với 4GB, công thức reels meme thực tế là **ảnh AI tĩnh + chuyển động ffmpeg (zoom/pan, cắt cảnh) + lồng tiếng + chữ + nhạc** — chứ không phải "sinh video AI" từng giây. Đây cũng đúng là cách phần lớn reels meme mèo/chuối được làm.

---

## 2. Bộ công cụ open-source theo từng khâu

### Khâu 1 — Kịch bản & lời thoại (bạn đã có sẵn)
Dùng chính API LLM trong `config.py` (Gemini/DeepSeek/OpenAI) để **tự viết kịch bản hài** thay vì dịch. Ví dụ prompt: "Viết kịch bản reel 30 giây, 4 cảnh, nhân vật mèo béo tên Mỡ, tình huống đi xin việc, giọng hài hước tuổi teen, kèm lời thoại từng cảnh và gợi ý hình ảnh."

### Khâu 2 — Tạo hình ảnh nhân vật & cảnh
- **ComfyUI** — công cụ local mạnh nhất, tiết kiệm VRAM nhất (node-based). Phù hợp khi quen tay.
- **Forge (Stable Diffusion WebUI Forge)** — giao diện đơn giản, nhanh hơn AUTOMATIC1111 30–75%, dễ cài nhất → **khuyến nghị bắt đầu ở đây**.
- **Fooocus** — kiểu "Midjourney tại nhà", chạy được từ 4GB nhưng dùng SDXL nên sẽ chậm.

Với 4GB: dùng **model SD 1.5** (nhẹ) trong Forge/ComfyUI. Muốn chất lượng cao hơn thì sinh ảnh trên **cloud free** rồi tải về.

**Giữ nhân vật nhất quán giữa các cảnh** (rất quan trọng cho series meme): dùng cùng seed + LoRA nhân vật, hoặc IPAdapter/reference image trong ComfyUI.

### Khâu 3 — Tạo chuyển động (animation)
- **Cách chính (nhẹ, miễn phí):** ffmpeg tạo hiệu ứng **Ken Burns** (zoom/pan chậm) trên ảnh tĩnh + cắt cảnh theo nhịp thoại. Đủ sống động cho meme reel. (Code mẫu ở mục 4.)
- **Nhân vật biết "nói" (mấp máy môi):** [LivePortrait](https://github.com/KwaiVGI/LivePortrait) (chất lượng cao) hoặc [SadTalker](https://github.com/OpenTalker/SadTalker) (nhanh, dễ) — animate 1 ảnh khuôn mặt theo file audio. Lưu ý nặng VRAM, cân nhắc chạy cloud nếu 4GB tràn.

### Khâu 4 — Lồng tiếng tiếng Việt
- **Hiện tại bạn đang dùng `edge-tts`** (`test_edge.py`) — miễn phí, ổn, giữ nguyên cũng tốt cho bắt đầu.
- Nâng cấp giọng đặc trưng / nhân bản giọng (open-source):
  - [VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) — nhân bản giọng tức thì, chạy **real-time trên CPU**, 24kHz → hợp máy yếu.
  - [viet-tts (dangvansam)](https://huggingface.co/dangvansam/viet-tts) — toolkit TTS + voice cloning tiếng Việt.
  - [F5-TTS-Vietnamese](https://github.com/nguyenthienhy/F5-TTS-Vietnamese) / [bản ViVoice](https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice) — chất lượng cao.
  - [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — nhân bản giọng chỉ với ~1 phút mẫu, tạo "giọng riêng" cho nhân vật.

### Khâu 5 — Nhạc nền & hiệu ứng âm thanh (KHÔNG bản quyền)
Đây là điểm bạn từng phải dùng Demucs để gỡ nhạc bản quyền. Nay chủ động dùng nhạc free để khỏi lo:
- Thư viện âm thanh của Facebook/Meta (Sound Collection trong Creator Studio) — an toàn nhất cho Reels.
- Nhạc CC0/royalty-free, hoặc tự sinh bằng model nhạc open-source.

### Khâu 6 — Ghép video cuối (TÁI DÙNG code của bạn)
ffmpeg + NVENC + render phụ đề Pillow bạn đã có dùng lại được nguyên vẹn — chỉ đổi nguồn đầu vào.

---

## 3. Tích hợp vào pipeline `Monggu Translate` hiện có

Pipeline cũ (dịch lại) chạy theo chiều:
```
video gốc → OCR (PaddleOCR) → dịch (LLM) → render sub (Pillow) → mux audio (ffmpeg)
```

Pipeline mới (tự tạo) **đảo chiều phần đầu**, giữ nguyên phần cuối:
```
ý tưởng → kịch bản (LLM) → ảnh AI (ComfyUI/Forge) → chuyển động (ffmpeg Ken Burns)
        → lồng tiếng (edge-tts/viet-tts) → render chữ (Pillow của bạn) → ghép (ffmpeg/NVENC của bạn)
```

Các module dùng lại được ngay:
- **Render chữ/caption:** logic vẽ chữ vàng viền đen + hộp nền mờ trong `config.py` (`TEXT_COLOR`, `SUBTITLE_BG_*`, `FONT_PATH`) và code render → dùng để gắn caption meme.
- **Ghép tiếng vào video:** `audio_tools.py` lệnh `mux` dùng nguyên.
- **Encode 9:16:** dùng lại cấu hình NVENC + portrait 1080x1920.
- **Gọi LLM:** dùng lại phần đọc `GEMINI_API_KEYS` trong `config.py`.

### Bỏ những phần không còn cần
Với nội dung tự tạo, **không dùng**: PaddleOCR, bảng `SUPPRESS_TEXTS`/`WATERMARK_PATTERNS` (không còn watermark người khác để xóa), Demucs (không còn nhạc bản quyền để gỡ), và **`TIKTOK_PASS_MODE`** (xem mục 5).

---

## 4. Script khung mẫu — `make_reel.py`

Đặt cạnh các script hiện có. Đây là khung tối giản để bạn mở rộng, dùng đúng phong cách (ffmpeg + edge-tts) bạn đang có:

```python
"""
make_reel.py — Tạo 1 reel meme từ danh sách (ảnh, lời thoại).
Tái dùng: edge-tts (giống test_edge.py) + ffmpeg NVENC.
Cài: pip install edge-tts
"""
import asyncio, subprocess, os, edge_tts

VOICE = "vi-VN-NamMinhNeural"   # hoặc vi-VN-HoaiMyNeural (nữ)
W, H = 1080, 1920               # dọc 9:16

# Mỗi cảnh: ảnh AI đã sinh sẵn + lời thoại
SCENES = [
    {"img": "scene1.png", "line": "Hôm nay tôi đi xin việc."},
    {"img": "scene2.png", "line": "Sếp hỏi: em biết làm gì?"},
    {"img": "scene3.png", "line": "Tôi: dạ em biết... nằm ạ."},
]

async def tts(text, path):
    await edge_tts.Communicate(text, VOICE, rate="+8%").save(path)

def dur(path):  # độ dài audio (giây)
    out = subprocess.run(["ffprobe","-v","quiet","-show_entries",
        "format=duration","-of","csv=p=0", path], capture_output=True, text=True)
    return float(out.stdout.strip() or 3.0)

def make_clip(img, audio, out, d):
    # Ken Burns: zoom chậm; scale về khung 9:16; ghép tiếng; encode NVENC
    # Lưu ý: zoompan phải có s={W}x{H}, nếu không sẽ tự xuất 1280x720
    vf = (f"scale={W*2}:{H*2}:force_original_aspect_ratio=increase,"
          f"crop={W*2}:{H*2},"
          f"zoompan=z='min(zoom+0.0008,1.15)':d={int(d*25)}:fps=25:s={W}x{H}")
    subprocess.run(["ffmpeg","-y","-loop","1","-i",img,"-i",audio,
        "-vf",vf,"-t",f"{d}",
        "-c:v","h264_nvenc","-pix_fmt","yuv420p",
        "-c:a","aac","-b:a","192k","-shortest",out], check=True)

async def main():
    clips = []
    for i, s in enumerate(SCENES):
        a = f"_a{i}.mp3"; c = f"_c{i}.mp4"
        await tts(s["line"], a)
        make_clip(s["img"], a, c, dur(a) + 0.4)
        clips.append(c)
    with open("_list.txt","w",encoding="utf-8") as f:
        for c in clips: f.write(f"file '{c}'\n")
    # nối các cảnh
    subprocess.run(["ffmpeg","-y","-f","concat","-safe","0",
        "-i","_list.txt","-c","copy","reel_output.mp4"], check=True)
    print("✓ Xong: reel_output.mp4")

if __name__ == "__main__":
    asyncio.run(main())
```

> Bước tiếp theo để "đẹp" hơn: thêm caption (dùng code render chữ của bạn), thêm nhạc nền free (mix bằng `audio_tools.py`), thêm hiệu ứng chuyển cảnh. Nếu muốn nhân vật mấp máy môi, thay `make_clip` bằng đầu ra của LivePortrait/SadTalker.

---

## 5. Lưu ý quan trọng về `TIKTOK_PASS_MODE`

Trong `config.py` có khối `TIKTOK_PASS_MODE` (zoom/sharpen/đổi hash để "qua mặt phát hiện"). **Nên tắt hẳn (`False`) và bỏ dần.** Lý do:
- Khi nội dung là của chính bạn, **không cần** né tránh gì cả.
- Các chiêu đổi hash không bền — hệ thống Facebook ngày càng nhận diện theo nội dung, không chỉ theo hash. Càng dùng càng dễ bị đánh giá là tài khoản "spam/né luật", làm án phạt nặng thêm.
- Mình không hỗ trợ tối ưu phần né tránh này; thay vào đó tập trung làm nội dung gốc chất lượng — đó mới là thứ gỡ được án phạt.

---

## 6. Lộ trình bắt đầu (gợi ý)

1. **Tuần 1 — MVP nhỏ:** cài Forge, sinh 3–4 ảnh 1 nhân vật; chạy `make_reel.py` ra 1 reel 20–30s; đăng thử. Mục tiêu: thông toàn bộ luồng.
2. **Tuần 2 — nhất quán nhân vật:** tạo/ò tìm LoRA hoặc cố định seed + IPAdapter để nhân vật giống nhau qua các cảnh → dựng "series".
3. **Tuần 3 — nâng chất:** thêm caption động, nhạc free, thử LivePortrait cho cảnh nhân vật nói; nâng giọng lên viet-tts/GPT-SoVITS nếu muốn chất riêng.
4. **Liên tục:** đăng đều nội dung gốc vài tuần → Facebook tự đánh giá lại và khôi phục đề xuất.

---

## Nguồn tham khảo công cụ
- Image gen / VRAM thấp: BentoML, Awesome Agents, DevToolLab (xem link trong phần chat).
- Video gen & VRAM: Spheron, LTX, Pixazo.
- TTS tiếng Việt: VieNeu-TTS, viet-tts (dangvansam), F5-TTS-Vietnamese, GPT-SoVITS.
- Lip-sync ảnh: LivePortrait (KwaiVGI), SadTalker (OpenTalker).
