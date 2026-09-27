# experiments/

Script thử nghiệm một lần, đã hết vai trò. Pipeline không dùng; giữ lại để tham khảo.
Hầu hết đọc/ghi file ở **thư mục hiện tại** (vd `input.mp4`), nên nếu cần thì chạy từ thư mục gốc: `python experiments/<file>.py`.

| File | Việc đã làm |
|---|---|
| `wm_remove.py` | Xóa watermark chạy bằng OCR dày mọi frame + nội suy (chậm, còn sót). Đã được `wm_track.py` thay. |
| `wm_ocr_probe.py` | Dò vị trí watermark bằng OCR (tạo `wm_anchors.json`, `_wm_probe_out/`) cho `wm_remove.py` |
| `wm_template.png` | Mẫu watermark của lần thử template-matching (bỏ dở) |
| `raw_map.py` | Gộp trùng `translation_map.json` cũ (định dạng trước khi có `lines.json`) |
| `debug_segs.py` | In segment quanh giây 90–92 để soi lỗi nhấp nháy phụ đề |
| `create_intro.py` | Tạo `intro.mp4` 1 giây từ `cat_intro.jpg` |
| `test_concat.py` | Thử ghép intro + 10 giây `input.mp4` |
| `test_edge.py` | Thử lồng tiếng edge-tts lên `input.mp4` |
| `demo_gen_scenes.py` | Vẽ cảnh mèo bằng Pillow (đường dẫn Linux, `/tmp/reel`); `story_engine/` đã thay thế |
