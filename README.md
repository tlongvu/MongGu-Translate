# Monggu Translate

Dịch phụ đề video Trung Quốc sang tiếng Việt: **OCR → xóa watermark chạy → dịch theo cốt truyện, đúng nhân vật → render lại video**.
Thư mục `story_engine/` làm reel gốc: viết kịch bản tay, hoặc `--idea "…"` để LLM viết kịch bản (dùng chung lõi `llm.py` và cache). Xem `story_engine/README.md`.

## Cài đặt

- Python 3.10, và **ffmpeg** trong PATH.
- `pip install -r requirements.txt`
- API key để trong **`D:\monggutranslate\.env`** (thư mục cha của project):
  - `GEMINI_API_KEY`, `GEMINI_API_KEY1`… `GEMINI_API_KEY19`: nhiều key Gemini, tự xoay vòng khi hết quota;
  - `DEEPSEEK_API_KEY`: dự phòng khi mọi key Gemini đều hết quota.

## Dịch một video

```
python pipeline.py --input D:\videos\tap1.mp4
python pipeline.py --input D:\videos\daosiep4.mp4 --series daosi   # tập của một bộ: dùng chung hồ sơ nhân vật
python pipeline.py --input video.mp4 --test                        # thử nhanh ~20s đầu (không lưu hồ sơ truyện)
python pipeline.py --input video.mp4 --skip-existing               # bỏ qua bước đã xong
python pipeline.py --input video.mp4 --only render                 # chỉ render lại
python pipeline.py --input video.mp4 --only wm                     # chỉ dò lại watermark chạy
```

**Video ra:** `jobs/<tên-video>/<tên-video>_vi.mp4`.

Bước nào lỗi thì pipeline dừng ngay với thông báo rõ ràng, không âm thầm xuất video sai. Riêng bước dò watermark chạy lỗi thì chỉ cảnh báo, vẫn dịch và render.

## Dữ liệu của mỗi video

| File | Nội dung |
|---|---|
| `jobs/<tên>/job.json` | Video gốc, dấu vân tay, series. Chặn trường hợp hai video trùng tên dùng chung dữ liệu. |
| `jobs/<tên>/ocr_data.json` | Kết quả OCR. `complete: false` nghĩa là OCR bị ngắt; chạy lại sẽ làm tiếp. |
| `jobs/<tên>/wm_track.json` | Watermark chạy: chữ, vị trí ở từng frame. `found: false` kèm lý do nếu không có / bỏ qua. |
| `jobs/<tên>/wm_mask.png`, `wm_preview.png` | Mask nét chữ dùng để xóa; ảnh xem nhanh vị trí bám. |
| `jobs/<tên>/lines.json` | Từng lần xuất hiện của từng câu: tiếng Trung, người nói (`noi`), người nghe (`nghe`), bản dịch (`vi`). |
| `jobs/<tên>/story.json` | Hồ sơ truyện của video lẻ. |
| `series/<bộ>/story.json` | Hồ sơ truyện dùng chung cho mọi tập của một bộ. |
| `cache/llm.sqlite` | Cache lời gọi LLM. Chạy lại không tốn API cho phần không đổi. |

## Dịch đúng nhân vật: cách hoạt động và cách sửa

1. **Hồ sơ truyện** (`story.json`) được lập từ toàn bộ lời thoại. Nó gồm:
   - tóm tắt;
   - nhân vật;
   - **bảng xưng hô theo cặp**: `"a>b"` nghĩa là khi a nói với b thì a xưng gì, gọi b là gì; `"a>*"` là cách xưng mặc định của a;
   - tên riêng;
   - nhãn tên hiện trên màn hình.
2. **Người nói / người nghe** được xác định cho từng câu, có dùng nhãn tên đang hiện trên màn hình làm gợi ý.
3. **Dịch từng lần xuất hiện:** xưng hô tra từ bảng theo cặp. Cùng một câu tiếng Trung, người nói khác thì cách xưng khác.
4. **Rà soát** xưng hô, chủ ngữ và ý theo mạch truyện.

**Khi thấy dịch sai:**

- **Sai xưng hô giữa hai nhân vật:** sửa cặp `"a>b"` trong `story.json`, rồi chạy lại bước dịch và render:
  ```
  python translate_pass.py --input video.mp4
  python script.py --input video.mp4
  ```
- **Nhầm người nói ở một câu:** sửa `noi` / `nghe` của dòng đó trong `lines.json`, rồi chạy lại hai lệnh trên.
- **Muốn tự sửa hẳn một câu:** ghi bản của bạn vào `vi_sua` của dòng đó, rồi chỉ cần render lại (`--only render`).

Mọi chỗ sửa tay đều được giữ khi chạy lại. Nhờ cache, chỉ những đoạn có liên quan mới phải gọi lại LLM.

## Xóa watermark chạy

Watermark chạy là chữ ID nhỏ, mờ, trôi khắp khung hình (vd "原创@哈加密"). OCR chỉ đọc được nó ở vài % số frame, nên trước đây video ra bị lúc xóa lúc không. Bước `wm_track.py` (pipeline tự chạy sau OCR):

1. tự nhận chữ watermark từ kết quả OCR: chữ có mặt suốt video, vị trí thay đổi dần, cỡ không đổi (không cần khai báo chữ);
2. dựng mẫu nét chữ từ những lần OCR đọc đủ;
3. dò mẫu trên mọi frame, nối các lần dò chắc chắn thành quỹ đạo, lấp đoạn mờ bằng nội suy;
4. render xóa theo **hình nét chữ** ở mọi frame.
5. **tách lớp watermark** (độ đậm + màu) từ những đoạn cảnh đứng yên, chấm điểm so với nền thật: thắng inpaint ≥ 1 dB thì render khôi phục nền bằng **đảo pha trộn** (giữ chi tiết, không nhòe), không thì giữ inpaint. Log: `[WM] Lớp watermark: … → xóa bằng đảo pha trộn | giữ inpaint`; kết quả ở `jobs/<tên>/wm_matte.npz`.

- **Kiểm:** log `[WM] Thấy watermark chạy '…' — bám N frame (…% có bằng chứng)`, và ảnh `jobs/<tên>/wm_preview.png` (khung xanh = dò thấy, vàng = nội suy) phải nằm đúng lên chữ.
- **Chạy lại riêng bước này:** `--only wm`, rồi `--only render`.
- **Tắt cho một video:** tạo file JSON `{"WM_TRACK": false}`, đặt biến môi trường `MONGGU_CONFIG` trỏ tới file đó, rồi chạy `--only wm` và `--only render`.
- **Thời gian:** thêm khoảng 0,5× thời lượng video (video 2 phút ≈ 1 phút, 30 phút ≈ 16 phút). Log in tiến độ mỗi 10%.
- **Giới hạn:**
  - mỗi video 1 watermark chạy;
  - watermark tĩnh vẫn dùng `SUPPRESS_TEXTS`;
  - nền nhiều vân có thể còn vệt nhòe nhẹ của inpaint.

## Cache

- Xóa `cache/llm.sqlite` để dịch lại từ đầu.
- Tăng `CACHE_VERSION` trong `llm.py` để bỏ toàn bộ cache cũ, ví dụ khi đổi model.

## Test

`python -m pytest tests -q`. Test không gọi API thật.

## Code

| File | Vai trò |
|---|---|
| `pipeline.py` | Chạy OCR → dò watermark → dịch → render; dừng ngay khi một bước lỗi (trừ bước dò watermark). |
| `ocr_pass.py` | OCR (PaddleOCR); gộp các lần nhận diện thành segment. |
| `translate_pass.py` | Hồ sơ truyện, người nói, dịch theo cặp xưng hô, rà soát. |
| `script.py` | Xóa chữ gốc, vẽ phụ đề tiếng Việt, ghép tiếng (NVENC). |
| `llm.py` | Gọi LLM (Gemini → DeepSeek), kèm cache. |
| `jobs.py` | Đường dẫn dữ liệu theo từng video / series. |
| `config.py` | Cấu hình. Có thể ghi đè bằng biến môi trường `MONGGU_CONFIG=<file.json>`. |
| `audio_tools.py` | Tách giọng / ghép tiếng (chạy tay). |
| `wm_track.py` | Dò watermark chạy (tự nhận, bám mọi frame) để render xóa. |

Thiết kế và kế hoạch nằm trong `docs/superpowers/`.
