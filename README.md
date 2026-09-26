# Monggu Translate

Dịch phụ đề video Trung Quốc sang tiếng Việt: **OCR → dịch theo cốt truyện, đúng nhân vật → render lại video**.
Thư mục `story_engine/` làm reel gốc (chưa gắn vào pipeline).

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
```

**Video ra:** `jobs/<tên-video>/<tên-video>_vi.mp4`.

Bước nào lỗi thì pipeline dừng ngay với thông báo rõ ràng, không âm thầm xuất video sai.

## Dữ liệu của mỗi video

| File | Nội dung |
|---|---|
| `jobs/<tên>/job.json` | Video gốc, dấu vân tay, series. Chặn trường hợp hai video trùng tên dùng chung dữ liệu. |
| `jobs/<tên>/ocr_data.json` | Kết quả OCR. `complete: false` nghĩa là OCR bị ngắt; chạy lại sẽ làm tiếp. |
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

## Cache

- Xóa `cache/llm.sqlite` để dịch lại từ đầu.
- Tăng `CACHE_VERSION` trong `llm.py` để bỏ toàn bộ cache cũ, ví dụ khi đổi model.

## Test

`python -m pytest tests -q`. Test không gọi API thật.

## Code

| File | Vai trò |
|---|---|
| `pipeline.py` | Chạy OCR → dịch → render; dừng ngay khi một bước lỗi. |
| `ocr_pass.py` | OCR (PaddleOCR); gộp các lần nhận diện thành segment. |
| `translate_pass.py` | Hồ sơ truyện, người nói, dịch theo cặp xưng hô, rà soát. |
| `script.py` | Xóa chữ gốc, vẽ phụ đề tiếng Việt, ghép tiếng (NVENC). |
| `llm.py` | Gọi LLM (Gemini → DeepSeek), kèm cache. |
| `jobs.py` | Đường dẫn dữ liệu theo từng video / series. |
| `config.py` | Cấu hình. Có thể ghi đè bằng biến môi trường `MONGGU_CONFIG=<file.json>`. |
| `audio_tools.py` | Tách giọng / ghép tiếng (chạy tay). |
| `wm_remove.py` | Xóa watermark (thử nghiệm, chạy tay). |

Thiết kế và kế hoạch nằm trong `docs/superpowers/`.
