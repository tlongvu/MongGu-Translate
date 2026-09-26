# Nâng cấp Monggu — thiết kế tổng thể + dự án con A

**Ngày:** 2026-09-26 · **Nhánh làm việc:** `nang-cap` · **Bản gốc:** `main` @ `3bd8555`

## 1. Mục tiêu

Hai sản phẩm chạy song song trên **một lõi chung**:

- **Dịch video** Trung→Việt (OCR → dịch → render). Đang dùng được nhưng có lỗi làm sai đầu ra.
- **Reel gốc** (`story_engine/`). Hướng đi để thoát án "unoriginal content"; hiện chưa chạy được trên Windows.

Lõi chung chỉ chứa thứ **cả hai** thực sự dùng: gọi LLM, vẽ chữ, ffmpeg/NVENC, TTS.

User giao việc qua `/loop` ("tiến hóa repo lên 1 bậc"). Claude tự làm lần lượt từng dự án con, mỗi bước một commit trên `nang-cap`; user ngắt hoặc đổi hướng bất cứ lúc nào.

## 2. Hiện trạng

- `pipeline.py` chạy 3 tiến trình con nối tiếp: `ocr_pass.py` → `ocr_data.json`; `translate_pass.py` → `translation_map.json`; `script.py` → `output_final_N.mp4`. Mọi file dữ liệu dùng đường dẫn cố định cạnh script.
- Chạy tay, ngoài pipeline: `wm_remove.py` (thử nghiệm), `audio_tools.py` (Demucs, mux), `story_engine/` (reel gốc), cùng khoảng 10 script thử nghiệm lẻ.
- Test: 7 test hàm thuần trong `tests/test_core.py`, chạy bằng pytest hoặc `python tests/test_core.py`.

## 3. Vấn đề đã kiểm chứng

Mọi mục đều đã đọc code. Mục gắn **[dữ liệu]** đã đối chiếu thêm với dữ liệu thật trong thư mục.

### Đầu ra sai, hoặc lỗi mà không báo

1. **[dữ liệu] Hiện sai câu phụ đề.** `build_text_segments` gộp hai câu khác nhau vào một segment chỉ vì cùng vị trí (luật 3, `ocr_pass.py:106`). Render lại bỏ câu thật đang có trên frame vì nó "chồng" lên segment (`script.py:858`). Ví dụ: frame 1025–1073 đang hiện "我要去修仙！" nhưng vẽ bản dịch của câu sau, "Để cái tên họ Bạch kia phải hối hận!".
2. **Pass thất bại vẫn thoát mã 0**, nên pipeline chạy tiếp bằng dữ liệu của video trước:
   - `translate_pass.py:1386/1393/1441` chỉ `return`;
   - `ocr_pass.py:381` và khi bị Ctrl+C;
   - render: NVENC vỡ pipe chỉ `break` (`script.py:1040`), mã lỗi ffmpeg bị bỏ qua (`:1058`), mux audio lỗi chỉ in WARN (`:1292`).
3. **Checkpoint OCR ghi lặp.** Điều kiện ở `ocr_pass.py:574` được xét ở mọi frame. Khi scene-diff bỏ qua frame, `submitted` đứng yên ở bội số của 1000, nên file 24MB bị dựng lại và ghi lại liên tục.
4. **Ghi JSON không atomic.** Nếu file cache hỏng giữa chừng thì `load_translation_cache` trả `{}`, và lần lưu sau ghi đè mất cả 6.892 bản dịch (`translate_pass.py:42-51`).

### Dữ liệu lẫn giữa các video

5. Prompt dịch viết cứng bối cảnh của một video cũ (`config.py:105-108`); `TRANSLATION_GENRE` cố định.
6. `glossary.json` dùng chung, đánh dấu "BẮT BUỘC", lẫn cả mục không phải tên riêng (`我→Tôi`, `丛林→rừng rậm`).
7. `character_profiles.json` chứa 66 nhân vật của khoảng 9 phim; hồ sơ cũ đè hồ sơ mới (`translate_pass.py:1141`).
8. Cache chỉ dùng câu tiếng Trung làm khóa (`你` luôn thành "Ngươi.", `你好` luôn thành "Chào con." ở mọi phim).
9. Đường dẫn cố định, nhận diện video bằng basename:
   - video sau đè dữ liệu video trước;
   - resume nhầm khi hai video trùng tên;
   - `--skip-existing` dùng dữ liệu cũ (`pipeline.py:80`);
   - render tìm video đã convert theo basename ở thư mục hiện tại, không thấy thì lệch khung (`script.py:1189`).
10. Lọc chữ quá rộng:
    - `SUPPRESS_TEXTS` so khớp chuỗi con (`script.py:956`), xóa luôn cả thoại thật;
    - `WATERMARK_PATTERNS` (`"关注"`, `"@"`…) bỏ thoại ngay từ bước OCR (`ocr_pass.py:210`).

### Trùng lặp và nợ kỹ thuật

11. **4 backend LLM trùng code.** Tổng 684 dòng, khoảng 380 dòng giống hệt nhau; chuỗi fallback copy 3 lần. Mỗi bản còn có lỗi riêng:
    - thiếu câu thì gán theo vị trí (`:517`);
    - một batch lỗi làm mất cả phần đã dịch;
    - một phản hồi JSON hỏng tắt Gemini cả phiên (`:1058`);
    - prompt vừa đòi array vừa đòi object.
12. Vẽ phụ đề, ghép audio, ffprobe, khởi tạo PaddleOCR: mỗi thứ có 2–3 bản copy.
13. `story_engine` không chạy trên Windows (font Linux ở `make_story_reel.py:19`), encode bằng CPU, kịch bản phải viết tay.
14. OCR chỉ chạy 1 luồng vì `USE_GPU=True`, trong khi Paddle đang cài bản CPU (`ocr_pass.py:420`).
15. Tài liệu và phụ thuộc lỗi thời:
    - `run_commands.txt` trỏ tới thư mục không còn tồn tại;
    - `requirements.txt` (thư mục cha) thiếu `google-genai`, `deep-translator`, thừa `asyncio`;
    - code chết: `TIKTOK_PASS_MODE`, `schemas.py`, `create_text_mask`…

## 4. Chia dự án con

| | Dự án con | Giải quyết | Vì sao đứng ở vị trí này |
|---|---|---|---|
| **A** | Đầu ra đúng, hoặc báo lỗi rõ | 1–4 | Lỗi người xem thấy ngay; diff nhỏ; kiểm chứng offline trên dữ liệu thật |
| **B** | Mỗi video một thư mục job | 5–10 | Sửa tận gốc việc lẫn dữ liệu; nền cho chạy hàng loạt |
| **C** | Lõi LLM chung | 11 | Xóa khoảng 450–500 dòng; reel gốc cần gọi LLM |
| **D** | Lõi media chung + reel gốc | 12–13 | Cần có C để viết kịch bản |
| **E** | Dọn dẹp, tài liệu, tăng tốc OCR | 14–15 | Làm cuối, khi cấu trúc đã ổn |

Mỗi dự án con đi đủ vòng spec → plan → code. Tài liệu này chi tiết A; spec của B–E viết khi tới lượt.

## 5. Dự án con A — Đầu ra đúng, hoặc báo lỗi rõ

### A1. Không gộp hai câu khác nhau chỉ vì cùng vị trí

**Sửa:** xóa luật 3 (`elif iou > 0.6`) ở `ocr_pass.py:105-107`. Luật 1 (cùng chữ) và luật 2 (chữ giống ≥ 0.7) vẫn giữ.

**Căn cứ:** mô phỏng đúng logic chọn segment của render trên `frame_detections` thật (10.295 keyframe). Bản sao của code hiện tại tái tạo khớp 100% 3.573 segment đã lưu.

| Luật 3 | Segment | Keyframe vẽ sai câu | Segment "lọt thỏm" (nguy cơ 2 dòng chồng nhau) |
|---|---|---|---|
| hiện tại (chỉ xét vị trí) | 3573 | 1291 | 14 (340 frame) |
| + chữ giống ≥ 0.4 | 3904 | 683 | 10 (342 frame) |
| + chữ giống ≥ 0.5 | 3977 | 658 | 12 (346 frame) |
| **bỏ hẳn** | 4427 | **132** | 43 (379 frame) |

Bỏ luật 3 giảm 90% số frame sai câu, đổi lại chỉ thêm 39 frame có nguy cơ chồng dòng. Các trường hợp đó đều là chữ biển hiệu bị OCR đọc sai (`搭筝堂`/`撩拿堂`). Hiệu ứng chữ hiện dần (typewriter) không bị ảnh hưởng: phần chữ đã hiện có IoU > 0.6 với cả câu thì luôn giống ≥ 0.75, nên luật 2 đã xử lý.

**Dữ liệu cũ:** `ocr_data.json` đã có vẫn chứa segment sai. Video mới tự đúng; video cũ muốn render lại thì chạy lại OCR.

### A2. Pass lỗi thì thoát mã khác 0

- **`ocr_pass.py`:** không mở được video, hoặc bị Ctrl+C (vẫn lưu tiến trình để resume) → exit 1.
- **`translate_pass.py`:** thiếu `ocr_data`, không có câu nào để dịch, hoặc mọi câu bị lọc là rác → exit 1.
- **`script.py`:**
  - kiểm tra ngay từ đầu, trên video thực sự đem render: kích thước khác `ocr_data`, fps lệch quá 0.5, hoặc video ngắn hơn dữ liệu quá 1 giây (video dài hơn thì vẫn cho qua, để dữ liệu OCR `--test` còn dùng được) → exit 1 kèm thông báo "video không khớp dữ liệu OCR";
  - worker: không mở được video, NVENC vỡ pipe, hoặc ffmpeg trả mã lỗi → raise (`pool.map` đẩy lỗi ra ngoài) → exit 1;
  - mux audio lỗi → exit 1, vẫn giữ file video không tiếng.

### A3. Checkpoint OCR chỉ ghi sau mỗi 1000 frame thực sự được OCR

Ghi nhớ mốc `submitted` của lần lưu trước; chỉ lưu khi `submitted - mốc >= 1000`.

### A4. Ghi JSON an toàn

Thêm `write_json_atomic(path, obj)` vào `utils.py`: ghi ra file tạm cùng thư mục rồi `os.replace`. Dùng cho `ocr_data.json` (checkpoint + bản cuối), `translation_map.json`, `translation_cache.json`, `character_profiles.json`.

### Kiểm chứng

- **Test đơn vị** (`tests/test_core.py`, cùng phong cách hiện có):
  - hai câu khác nhau cùng vị trí → hai segment;
  - câu đọc sai 1 chữ → vẫn một segment;
  - `write_json_atomic` giữ nguyên file cũ khi `json.dump` lỗi giữa chừng;
  - `run_translation` trả giá trị sai khi OCR rỗng.
- **Dữ liệu thật:** chạy lại thước đo với `ocr_pass.build_text_segments` đã sửa → phải cho 4427 segment / 132 frame sai câu.
- **Chạy thật, không tốn API:**
  - `python script.py --input test.mp4 --test` (video 720×1280 không khớp dữ liệu) phải dừng ngay với thông báo rõ, thay vì render ra rác;
  - `python ocr_pass.py --input khong_co.mp4` phải trả exit code 1.

### Ngoài phạm vi A

Mọi việc thuộc B–E. Cụ thể: **không** đổi prompt, glossary hay luật lọc chữ. Những thứ đó thuộc B, vì cần cấu hình theo từng video.

### Xong khi

Test cũ và mới đều qua, các kiểm chứng trên đúng như mô tả, mỗi mục A1–A4 là một commit trên `nang-cap`.

## 6. Nguyên tắc chung

- Làm trên `nang-cap`; `main` giữ bản gốc. Vẫn backup file sắp sửa vào `_backup_<mô-tả>_<timestamp>/` theo thói quen của user.
- Test chạy offline. Không gọi API thật (tốn quota) khi chưa được user cho phép.
- Không xóa dữ liệu hay media của user; chỉ liệt kê để user tự quyết.
- Không thêm: giao diện web, database, Docker, hệ plugin.
