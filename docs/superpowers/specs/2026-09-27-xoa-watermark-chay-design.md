# Xóa watermark chạy (bám mọi frame, xóa theo nét chữ)

**Ngày:** 2026-09-27 · **Nhánh:** `nang-cap` · **Tiền đề:** pipeline job (`jobs/<video>/`), `ocr_data.json`, render `script.py`

## 1. Mục tiêu và yêu cầu (user chốt 2026-09-27)

- **Lỗi cần hết:** watermark chạy (vd "原创@哈加密") **còn sót, nhấp nháy** trong video ra: lúc bị xóa lúc không.
- **Loại watermark:** mỗi video/kênh một chữ ID khác nhau, nhưng cùng kiểu: chữ nhỏ trắng bán trong suốt, viền tối, trôi chậm khắp khung hình. Phải **tự nhận ra**, không bắt khai báo chữ.
- **Cách bật:** tự động trong `pipeline.py`. Có watermark chạy thì xóa, không có thì bỏ qua.
- **Cách làm được chọn:** mốc từ OCR sẵn có + khớp mẫu mọi frame. Không OCR thêm, không cài thêm thư viện.

## 2. Hiện trạng và số liệu đo (dữ liệu thật)

- Render chỉ inpaint chữ ở frame OCR bắt được. Trên video hihi (44.181 frame), OCR bắt watermark ở **8,5%** số frame.
  - Có 258 khoảng sót dài hơn 1 s, cộng lại **65%** thời lượng; 53 khoảng dài hơn 5 s.
  - Đó là nguyên nhân gây nhấp nháy.
- Watermark **không nhảy chỗ**: trôi đều khoảng 8–50 px/s theo từng đoạn thẳng, đổi hướng khi chạm biên, cỡ box không đổi (190×46, dao động 4%).
  - Những lần tưởng "nhảy" thực ra là OCR chỉ đọc một mảnh chữ.
- `wm_remove.py` (thử nghiệm cũ): OCR full-res mọi frame (≈0,25 s/frame) rồi nội suy thẳng. Chậm, vẫn sót (chính ghi chú trong file: OCR bắt ~20% frame), và xóa cả hộp.
- **Thử khả thi trên `newvideo` (3.308 frame, 1440×1080)** bằng script nháp. So với 19 mốc OCR full-res đọc đủ chữ:

| Cách | Kết quả |
|---|---|
| Bám cục bộ frame nối frame | **Hỏng**: mất dấu sau 1 s ở đoạn mờ, trôi dạt |
| Quỹ đạo tối ưu (Viterbi) toàn khung | Đúng ngoài đoạn mờ, **sai** trong đoạn mờ 10 s |
| **Dò toàn khung + mốc chắc + nội suy có dò lại** (chọn) | Phủ **100%** khoảng có watermark; lệch trung vị **3,5 px**, p90 5,3 px, lớn nhất **8,3 px**; khoảng 14 ms/frame |

- Ở frame watermark mờ tới mức không dò được (điểm khớp tại chỗ thật chỉ 0,05–0,13, bằng nhiễu), mắt người cũng gần như không thấy nó.
- Ở frame nội suy mà mắt còn thấy watermark, khung đều nằm đúng lên chữ.
- Khớp nhầm duy nhất là trên mảng nền **trơn**: NCC tự khuếch đại nhiễu ở vùng phẳng. **Cổng năng lượng** (mục 4.3) đã loại được nó.
- **Bản nháp chạy đúng luồng sản xuất** (tự nhận từ `ocr_data` của pipeline, dựng mẫu từ box OCR của pipeline):
  - `newvideo`: lệch trung vị 3,6 px, lớn nhất 7,8 px so với 19 mốc; phủ frame 152–3174;
  - mask chiếm 63% hộp (đúng hình nét chữ);
  - xóa thử trên 5 frame thật thì hết cả chữ lẫn viền. Nền nhiều vân còn vệt nhòe nhẹ của inpaint.

## 3. Kiến trúc

```
ocr_pass → wm_track → translate_pass → script.py (render)
```

- **`wm_track.py`** (mới, một việc):
  - vào: video làm việc của job, `ocr_data.json`;
  - ra: `wm_track.json`, `wm_mask.png`, `wm_preview.png` trong `jobs/<tên>/`.
- **`jobs.py`**:
  - `open_job()` thêm `wm_track` (đường dẫn `wm_track.json`);
  - tách hàm `working_video(job, ocr_data, input_path)`, trả về video mà chỉ số frame trong `ocr_data` tham chiếu tới (bản convert fps trong job nếu có). `script.py` và `wm_track.py` cùng dùng hàm này thay vì mỗi nơi tự suy.
- **`pipeline.py`**:
  - chạy `wm_track.py --input …` sau OCR;
  - `--only wm` để chạy riêng bước này;
  - `--skip-existing` bỏ qua khi đã có `wm_track.json` và OCR không chạy lại;
  - tiêu đề các bước thành "x/4".
- **`script.py`**: mỗi frame có vị trí thì đặt `wm_mask` vào đó và inpaint bằng `_inpaint_roi()` có sẵn (`INPAINT_METHOD`). Vẫn encode 1 lần; phần xóa và vẽ phụ đề giữ nguyên.
- **`config.py`**: `WM_TRACK = True`. Tắt cho từng video qua `MONGGU_CONFIG` như các hằng khác.

## 4. Thuật toán (`wm_track.py`)

Hằng số đặt ở đầu file, không đưa vào config (không cần chỉnh theo video).

### 4.1 Tự nhận watermark chạy (từ `ocr_data`)

Gom các lần OCR theo chữ (bỏ khoảng trắng). Chữ là **ứng viên** khi thỏa tất cả:
- dài ≥ 4 ký tự và xuất hiện ≥ 10 lần;
- **có mặt ≥ 40% số đoạn 10 giây** của video;
- tâm box tản ≥ 5% đường chéo khung (độ lệch chuẩn 2 chiều; ≈ 90 px ở 1440×1080);
- bề rộng box ổn định (hệ số biến thiên ≤ 0,15).

Chọn ứng viên xuất hiện nhiều nhất; không có thì `found: false`.

Số liệu chốt ngưỡng:

| Chữ | Có mặt | Tản | Kết quả |
|---|---|---|---|
| "原创@哈加密" (hihi) | 78% | 466 px | watermark |
| "原创@哈加密" (newvideo) | 57% | 471 px | watermark |
| "执事弟子" (nhãn) | 12% | — | loại |
| "道悟长老" (nhãn) | 21% | — | loại |
| watermark tĩnh | — | 4–6 px | loại |
| thoại | < 10% | — | loại |

### 4.2 Mẫu và mask

- Trong các lần đọc của chữ đã chọn, lấy những lần **đọc đủ** (bề rộng lệch ≤ 8% trung vị). Chọn tối đa 30 lần, ưu tiên độ tin cậy cao và rải đều theo thời gian.
- Có dưới 5 lần đọc đủ thì bỏ qua, kèm cảnh báo.
- Cắt vùng đó từ video, tính độ lớn gradient (Sobel) trên ảnh xám, căn thẳng hàng theo NCC (±8 px), lấy **trung vị**. Kết quả là mẫu full-res, kèm một bản thu nửa độ phân giải để dò toàn khung. Nếu bản nửa độ phân giải hẹp hơn 48 px (watermark nhỏ) thì dò toàn khung ở độ phân giải gốc.
- **`wm_mask`** = pixel mẫu có gradient mạnh (≥ 25% cực đại), đóng lỗ (morphology close) rồi nới 3 px. Đây là hình nét chữ + viền, không phải cả hộp.

### 4.3 Bám

1. **Dò toàn khung** mọi frame, ở nửa độ phân giải: `matchTemplate(TM_CCOEFF_NORMED)` trên gradient.
   - **Cổng năng lượng:** vị trí có gradient trung bình trong cửa sổ < 35% của mẫu thì bị loại, vì vùng trơn làm NCC khuếch đại nhiễu.
   - Ghi lại đỉnh tốt nhất và điểm.
2. **Mốc** (sửa sau kiểm thật, xem mục 10):
   - **nối đoạn quỹ đạo:** mọi đỉnh ≥ 0,30 được nối vào đoạn đang mở gần nhất mà nó với tới được (cách ≤ 1 s, dời ≤ 100 px/s + 6 px). Nhiều đoạn sống song song được;
   - **chỉ giữ đoạn** có ≥ 3 đỉnh, ít nhất 1 đỉnh chắc (≥ 0,45), và **không bị OCR phủ nhận**. Bị phủ nhận nghĩa là trong khoảng thời gian của đoạn có một lần OCR đọc đúng chữ watermark nằm cách đoạn > 30 px (nội suy vị trí đoạn tại frame đó). Đoạn như vậy là chữ tĩnh / nhãn khớp nhầm, kể cả khi nó với tới được quỹ đạo. Trên `newvideo`: cụm khóa nhầm cách lần đọc 97 px; mọi đoạn thật ≤ 14,3 px;
   - **chuỗi dài nhất:** chọn chuỗi đoạn nối được với nhau bằng chuyển động khả thi, có tổng số đỉnh lớn nhất. Mốc = các đỉnh của chuỗi. Watermark dừng thật (đoạn cuối `newvideo`, điểm 0,81; lần đọc cách 4,5 px) vẫn được giữ;
   - mốc được chỉnh lại ở full-res (±6 px). Có dưới 20 mốc thì bỏ qua, kèm cảnh báo.
3. **Khoảng giữa 2 mốc liên tiếp:**
   - dự đoán bằng **nội suy Hermite**: khớp vị trí và vận tốc 2 đầu, vận tốc = hệ số góc các mốc trong 1 s quanh mỗi đầu (các mốc đó phải trải ≥ 0,5 s, không thì dùng dây cung → thành nội suy thẳng);
   - rồi dò lại quanh điểm dự đoán ở full-res (±24 px). Điểm ≥ 0,30 thì lấy vị trí dò được (có bằng chứng);
   - nếu vị trí đó lệch khỏi đường dự đoán > 8 px thì đó là chỗ đổi hướng: nó thành mốc mới. Các frame **nội suy** giữa mốc trước và mốc mới được nội suy lại thẳng (không cần đọc lại ảnh), phần sau đi thẳng tới mốc kế; frame có bằng chứng giữ nguyên;
   - còn lại giữ vị trí dự đoán (không có bằng chứng).
4. Chỉ bám trong `[mốc đầu, mốc cuối]`, không đoán ra ngoài.
5. **Bộ nhớ:** đọc video tuần tự 2 lượt (dò toàn khung, rồi lấp khoảng). Không giữ ảnh trong RAM, chỉ giữ vị trí và điểm của mỗi frame.
6. **Thời gian đo trên bản nháp** (cùng code với plan): 71 s cho `newvideo` 132 s, tức khoảng 0,55× thời lượng video. Video 30 phút mất khoảng 16 phút.

### 4.4 Dữ liệu ra

```jsonc
// jobs/<tên>/wm_track.json
{ "found": true, "text": "原创@哈加密", "size": [191, 43],
  "total_frames": 3308, "fps": 25.0,
  "start": 150, "xy": [[599, 153], ...],   // tâm watermark, từ frame start trở đi, liên tục tới hết khoảng bám
  "evidence": "1101…",                      // 1 = có bằng chứng tại chỗ, 0 = nội suy
  "anchors": 704, "seconds": 45.1 }
// không có watermark chạy / bỏ qua: { "found": false, "reason": "..." }
```

- `wm_mask.png`: mask cùng kích thước mẫu; tâm mask trùng tâm trong `xy`.
- `wm_preview.png`: 6 frame rải đều, vẽ khung vị trí (xanh = có bằng chứng, vàng = nội suy), để user liếc kiểm.
- Log: `[WM] Thấy watermark chạy '原创@哈加密' (191×43) — bám 3.025 frame (41% có bằng chứng), 45s`.

### 4.5 Xóa khi render

- `main()` đọc `wm_track.json`. Dùng khi `found` và `total_frames` khớp video làm việc; lệch thì cảnh báo và bỏ qua, giống kiểm `lines.json`.
- Chia vị trí theo chunk rồi truyền cho worker cùng `wm_mask`.
- Trong `process_chunk`, ngay sau khi đọc frame: nếu frame có vị trí thì cắt ROI bằng kích thước mask quanh tâm, xén theo mép khung, rồi `_inpaint_roi(roi, mask_roi)`.
- Chữ watermark mà OCR bắt được vẫn được xóa thêm bởi luồng `SUPPRESS_TEXTS` hiện có. Hai lần xóa trùng vùng, không hại.

## 5. Lỗi và an toàn

- Không bao giờ xóa bừa:
  - không có ứng viên, dưới 5 lần đọc đủ, hoặc dưới 20 mốc thì ghi `found: false` kèm `reason`;
  - log in rõ con số.
- `WM_TRACK = False` thì ghi `found: false` và `reason: "tắt trong config"`.
- **Bước này không chặn pipeline:** `wm_track.py` lỗi (kể cả crash) thì `pipeline.py` in cảnh báo rồi dịch, render tiếp mà không xóa watermark. `run_step()` thêm tham số `required=False` cho riêng bước này; các bước khác vẫn dừng ngay khi lỗi như cũ.
- Không mở được video, hoặc thiếu `ocr_data.json`: thoát mã 1, pipeline cảnh báo như trên.

## 6. Kiểm thử (offline, không gọi API)

`tests/test_wm.py`, cùng phong cách các test hiện có:
1. **Tự nhận:** `ocr_data` giả gồm:
   - watermark chạy có mặt 90% đoạn 10 s;
   - watermark tĩnh;
   - các câu thoại;
   - nhãn di chuyển có mặt 15%.

   Chỉ được chọn watermark chạy. Bỏ watermark chạy thì phải ra `None`.
2. **Bám:** video tổng hợp nhỏ (khoảng 240 frame, 480×270; ghi ra thư mục tạm):
   - nền gồm vân ngẫu nhiên, dải màu và một mảng trơn;
   - chữ bán trong suốt, viền tối, trôi 2 px/frame và đổi hướng ở giữa;
   - có một đoạn độ mờ 0,1 (gần như vô hình);
   - `ocr_data` giả có vài lần "đọc đủ".

   Yêu cầu lệch ≤ 3 px ở ≥ 95% frame có bằng chứng, ≤ 8 px ở mọi frame, và không khóa vào mảng trơn.
3. **Render:** hàm xóa watermark đặt mask đúng tâm; mask sát mép khung không lỗi; pixel ngoài mask không đổi.

Dữ liệu thật:
- `newvideo`: chạy `ocr_pass` rồi `wm_track`. Lệch ≤ 10 px so với 19 mốc OCR full-res đọc đủ (`wm_anchors.json`), phủ ≈ 100% khoảng có watermark, báo thời gian.
- hihi (`ocr_data.json` gốc): tự nhận đúng "原创@哈加密".
- Render khoảng 20 s của `newvideo` có xóa watermark (bản dịch rỗng, không gọi API). Gửi user **ảnh trước/sau** để xem bằng mắt.

## 7. Dọn dẹp và tài liệu

- `wm_remove.py` chuyển vào `experiments/` và thêm một dòng vào `experiments/README.md`, vì `wm_track.py` thay nó.
- `wm_anchors.json` và `_wm_probe_out/` là dữ liệu cũ của user: để nguyên.
- README: mục "Xóa watermark chạy" gồm cách hoạt động, đọc log và `wm_preview.png`, tắt cho từng video; cập nhật sơ đồ các bước (4 bước).

## 8. Ngoài phạm vi

- Watermark tĩnh: đã có `SUPPRESS_TEXTS`.
- Logo dạng hình ảnh, watermark nhảy chỗ đột ngột, nhiều watermark chạy cùng lúc trong một video.
- Xóa đẹp hơn inpaint: tách lớp alpha nhiều frame, hoặc ProPainter/torch CUDA. Chỉ làm khi user thấy vết xóa theo nét chữ chưa đủ đẹp.
- Đoán vị trí ngoài `[mốc đầu, mốc cuối]`.

## 9. Xong khi

- Test cũ và mới đều qua.
- Các kiểm chứng dữ liệu thật ở mục 6 đạt, và user đã xem ảnh trước/sau.
- Mỗi task trong plan là một commit trên `nang-cap`.

## 10. Kiểm thật lần 1 → sửa (Task 3b) → giới hạn còn lại

**Kiểm thật lần 1.** Code đúng plan đạt tiêu chí số: 19 mốc OCR, lệch ≤ 7,8 px. Nhưng soi ảnh thì thấy 2 lỗi mà các mốc đó không lộ ra, vì chúng đều nằm từ giây 21 trở đi:
1. **Khóa nhầm vật đứng yên.** Ở frame 472–575, một cụm đỉnh đứng yên (điểm 0,46, cạnh nhãn "*女主内心") được nhận làm mốc vì "tự nhất quán". Hậu quả là xóa sai chỗ và để sót watermark thật.
2. **Nội suy thẳng qua đoạn cong.** Frame 194–396 là 8 giây không có bằng chứng; ở frame 300 lệch khoảng 60 px.

**Sửa (Task 3b):** chuỗi đoạn quỹ đạo và nội suy Hermite (mục 4.3). Test mới dùng video tổng hợp có vật giống chữ đứng yên: code cũ lệch 297 px ở 40 frame, code mới lệch ≤ 2,2 px.

**Review cuối → Task 3c.** Chỉ chọn chuỗi dài nhất chưa đủ. Vật đứng yên xuất hiện giữa một đoạn mờ dài thì *với tới được* quỹ đạo (4 px/frame cho vài giây là tới mọi chỗ), nên vẫn được nối vào nếu làm chuỗi thêm đỉnh. Ca dựng thử bám lệch 113 px; trên `newvideo` cụm 472–575 chỉ thua 26 đỉnh.
- Thêm phép **loại theo OCR** (mục 4.3): ca dựng thử giờ tránh xa vật đứng yên ≥ 30 px (bản nháp: 75 px); `newvideo` giữ nguyên 909 mốc.
- Thêm test cho nhánh dò lại / đổi hướng của `fill_gaps` (trước đó chưa test nào chạy qua).
- `--only wm` lỗi thì báo lỗi thật; `load_wm_track` chịu được JSON sai dạng; in tiến độ mỗi 10%.
- Còn hở: vật khớp nhầm xuất hiện lúc OCR không đọc được watermark lần nào mà vẫn nối được vào quỹ đạo thì vẫn có thể được chọn.

**Giới hạn còn lại.** Trong khoảng hụt rất dài mà watermark mờ tới mức khớp mẫu không phân biệt được với nhiễu, vị trí chỉ là đoán. Ví dụ `newvideo` frame 606–1019 (16,5 s, chỉ 4 frame có bằng chứng): ở frame 835 chữ còn thấy lờ mờ, nhưng điểm khớp tại chỗ thật chỉ 0,25, là đỉnh thứ 5, thấp hơn nhiễu 0,29–0,34. Ở những đoạn đó có thể còn thấy watermark mờ, kèm vết nhòe gần đó. Muốn hơn nữa phải đổi tín hiệu dò (ví dụ tách lớp alpha nhiều frame) — việc riêng, chỉ làm khi user thấy cần.
