# Tách lớp watermark + xóa bằng đảo pha trộn

**Ngày:** 2026-10-16 · **Nhánh:** `nang-cap` · **Tiền đề:** bước dò watermark chạy (`wm_track.py`, spec 2026-09-27), render `script.py`

## 1. Mục tiêu và yêu cầu (user chốt 2026-10-16)

- **Lỗi cần hết:** chỗ watermark bị xóa để lại **mảng nhòe** (inpaint Telea bôi mất chi tiết nền: lan can, cửa sổ, đồ vật), và trên nền có vân thì lộ rõ.
- **Hướng được chọn:** tách lớp watermark (độ đậm α + màu) từ nhiều frame rồi xóa bằng **đảo pha trộn**, khôi phục đúng nền gốc thay vì vẽ đoán. Không thêm thư viện, không mô hình AI.
- **Không làm thư viện watermark theo kênh:** mỗi video tự tách lớp từ chính nó.
- **Không bao giờ tệ hơn hiện nay:** cách mới chỉ được dùng khi tự chấm điểm thắng inpaint rõ ràng; không thì giữ inpaint.
- **Giai đoạn 2 (ngoài phạm vi lần này):** dùng lớp watermark tách được làm mẫu dò, để bắt watermark mờ / in nghiêng (video 奇异之修).

## 2. Hiện trạng và số liệu đo (thử nghiệm 2026-10-16, video Douyin 36 phút, watermark 粥咪咪)

- File video chỉ có 1 luồng hình + 1 luồng tiếng; metadata chỉ có mã video Douyin. Watermark của người đăng **in thẳng vào điểm ảnh**, không có lớp riêng nào để lấy.
- Watermark là ảnh chữ trắng viền tối, phủ với độ đậm đều α ≈ 0,25 (cao nhất 0,29) → nền dưới chữ còn lộ ~75%, khôi phục được.
- **Nền thật** lấy từ frame khác: watermark đã dời đi (3–12 s sau / trước) và cảnh đứng yên quanh đó. Từ 900 frame ứng viên lấy được 309 cặp (34%); 540 bị loại vì cảnh động, 51 vì điểm khớp thấp.
- **Ước lượng lớp watermark** từ nền thật tốt hơn hẳn dùng nền inpaint làm nền ước lượng (nền inpaint quá thô → hồi quy lệch, để lại bóng ma chữ).
- Chấm trên 61 cặp giữ riêng (PSNR vùng watermark so với nền thật):

| Cách xóa | PSNR | Bóng ma (tương quan mép ảnh với mép nét chữ) |
|---|---|---|
| Không xóa | 18,2 dB | — |
| Inpaint hiện tại (Telea) | 25,9 dB | 0,00 |
| Đảo pha trộn (lớp từ nền inpaint) | 26,6 dB | — |
| Đảo pha trộn (lớp từ nền thật) | 29,6 dB | 0,09 |
| Đảo, α riêng từng kênh / kênh màu làm mờ (YCrCb) | 29,6 dB | 0,09 |
| Lai theo độ chi tiết nền | 31,2 dB | 0,06 |
| **Đảo + làm sạch dải mép nét chữ (trung vị 5×5)** | **32,8 dB** | **−0,04** |

- Căn dưới 1 px + chỉnh độ đậm từng frame chỉ giảm bóng ma một nửa → không đưa vào.
- Còn lại: dải mép sau khi lọc trung vị hơi lấm tấm trên nền phẳng — tinh chỉnh bộ lọc dải mép bằng chính công cụ chấm điểm (mục 4.4).

## 3. Kiến trúc

| Thành phần | Việc | Ghi chú |
|---|---|---|
| `wm_matte.py` (mới) | Thu cặp nền thật → ước lượng lớp watermark → chấm điểm → chọn cách xóa → ghi `wm_matte.npz` | Thuần numpy / OpenCV |
| `wm_track.py` | `run()` gọi tách lớp ngay sau khi ghi `wm_track.json` (cùng bước 2/4) | Lỗi tách lớp không làm hỏng bước dò |
| `pipeline.py` | `--skip-existing`: có `wm_track.json` mà thiếu `wm_matte.npz` → chỉ tách lớp, không dò lại | Job cũ dùng được ngay |
| `jobs.py` | thêm đường dẫn `job.wm_matte` (`jobs/<video>/wm_matte.npz`) | |
| `script.py` | `load_wm_matte()`; `remove_wm()` xóa theo cách đã chọn | Không có / không dùng được → inpaint như cũ |

Luồng: `wm_track.json` + video → (1) chọn cặp (k, j) chỉ từ vị trí đã bám → (2) đọc video **một lượt tuần tự**, cắt các ô cần → (3) lọc cặp cảnh tĩnh → (4) ước lượng trên 80% cặp → (5) chấm trên 20% còn lại → `wm_matte.npz` → render.

## 4. Thuật toán (`wm_matte.py`)

### 4.1 Ô watermark

- Ô = mask watermark (`wm_mask.png`) nới `PAD = 10` px mỗi phía; `inside` = vùng mask; `ring` = vành 12 px quanh `inside`.
- Vị trí ô ở frame f: tâm từ `wm_track.json`, **căn lại ±3 px** bằng khớp mẫu tại chỗ (`_local_match`, như lúc dò); chỉ nhận khi điểm khớp ≥ 0,5.
- Ô sát / lòi ra mép khung: phần ngoài khung đánh dấu không hợp lệ (mặt nạ `valid`), không tính khi ước lượng và chấm.

### 4.2 Thu cặp nền thật

- Ứng viên k: frame có bằng chứng (`evidence = 1`), rải đều theo thời gian, tối đa `MAX_CAND = 600` ứng viên.
- Với mỗi k, xét j = k + {75, −75, 125, −125, 200, −200, 300, −300} frame (3–12 s ở 25 fps) theo thứ tự đó, giữ tối đa
  `TRIES = 2` frame j mà watermark đã rời chỗ (đọc 1 lượt phải giữ sẵn các ô cần trong RAM: 600 × (1 + 2) ô ≈ 100 MB;
  thử hết 8 j cho 1200 ứng viên thì ≈ 570 MB). Nhận j đầu tiên thỏa:
  - **watermark đã rời chỗ:** |x_j − x_k| ≥ rộng mask hoặc |y_j − y_k| ≥ cao mask (biết trước từ track, không cần đọc video);
  - **cảnh đứng yên:** sai khác trung bình |ô_k − ô_j| trên `ring` (điểm hợp lệ) < `STATIC_DIFF = 3.0` mức xám.
- Đọc video 1 lượt: lập trước danh sách (frame → các ô cần cắt), đọc tuần tự, chỉ giữ các ô (uint8).
- Giữ tối đa `MAX_PAIRS = 400` cặp (rải đều theo thời gian). Dưới `MIN_PAIRS = 60` cặp → `method = "inpaint"`, lý do "thiếu cảnh tĩnh".

### 4.3 Ước lượng lớp watermark

Mô hình pha trộn: `ảnh = (1 − α)·nền + α·màu` cho từng điểm, từng kênh. Với các cặp (ảnh, nền thật) ở cùng điểm, hồi quy tuyến tính bình phương tối thiểu có trọng số `valid`:

- `hệ số góc = cov(nền, ảnh) / var(nền)` → `α = 1 − hệ số góc` (lấy trung bình 3 kênh, kẹp [0, 0,95], = 0 ngoài `inside`);
- `aW = trung bình(ảnh) − (1 − α)·trung bình(nền)` (= α·màu, giữ riêng 3 kênh).
- Dải mép nét chữ `edge`: gradient (Sobel) của ảnh màu watermark (xám) ≥ `EDGE_FRAC = 0.5` × cực đại, trong `inside`.

### 4.4 Xóa (đảo pha trộn)

- `nền = (ô − aW) / (1 − α)` trong `inside`, kẹp [0, 255].
- Làm sạch dải mép `edge`: lọc trung vị 5×5 (mặc định đã đo, 32,8 dB). Khi làm có thể thử thêm inpaint nhỏ / lọc có hướng dẫn trên dải mép; **chỉ đổi mặc định khi thắng trên phần giữ riêng của video 36 phút**.
- Điểm gần đục `α ≥ OPAQUE = 0.9` → inpaint (Telea, bán kính 3) sau khi đảo.
- Chỉ ghi lại phần ô nằm trong khung.

### 4.5 Chấm điểm và chọn cách xóa

- Chia cặp cố định (seed 0): 80% ước lượng, 20% chấm (ít nhất 12 cặp chấm).
- PSNR trên `inside ∧ valid` của: inpaint hiện tại (`script._inpaint_roi` với `wm_mask`) và đảo pha trộn (4.4).
- `method = "invert"` nếu PSNR đảo ≥ PSNR inpaint + `MIN_GAIN_DB = 1.0`; ngược lại `"inpaint"`.
- In 1 dòng: `[WM] Lớp watermark: 309 cặp nền thật; đảo pha trộn 32.8 dB, inpaint 25.9 dB → dùng đảo pha trộn`.

### 4.6 `wm_matte.npz`

`alpha` (h, w, float16), `aW` (h, w, 3, float16), `edge` (h, w, bool), `pad`, `method`, `psnr_invert`, `psnr_inpaint`, `pairs`, và dấu của track: `track_start`, `track_len`, `mask_shape` (để nhận ra file cũ khi dò lại).

## 5. Render (`script.py`)

- `load_wm_matte(job, track)` → dict hoặc None. None khi: chưa có file, `method == "inpaint"`, dấu track / cỡ mask không khớp (dò lại sau khi tách lớp), hoặc file hỏng (WARN, không làm sập render — như `load_wm_track`).
- `remove_wm(frame, center, mask, matte=None)`: có `matte` → căn tâm ±3 px bằng hình watermark (gradient của `aW`), cắt ô, đảo pha trộn + làm sạch mép (4.4), ghi phần trong khung; không có → inpaint như hiện tại.
- Việc xóa vẫn diễn ra ở cỡ nguồn, trước khi phóng Full HD và vẽ phụ đề.

## 6. Lỗi và an toàn

- Mọi lỗi trong tách lớp (đọc video, thiếu cặp, số học) → bắt lại, ghi `method = "inpaint"` + lý do, bước 2/4 vẫn thành công.
- `--only wm` dò lại → tách lớp lại (file cũ bị ghi đè).
- Không thấy watermark / `WM_TRACK = False` → vẫn ghi `wm_matte.npz` với `method = "inpaint"` + lý do, để `--skip-existing`
  biết bước 2/4 đã xong và không chạy lại.
- Tách lớp không đổi vị trí watermark, không ảnh hưởng bộ lọc dòng OCR của bước dịch.

## 7. Kiểm thử (offline, không gọi API)

1. **Công thức:** ô tổng hợp với α, màu biết trước → đảo pha trộn khôi phục nền, sai số ≤ 1 mức xám (ngoài dải mép).
2. **Ước lượng:** video giả nền tĩnh từng đoạn (vài giây), watermark bán trong suốt α = 0,3, màu biết trước, trôi chậm → α ước lượng sai trung bình < 0,03 trong `inside`; `method = "invert"`; PSNR đảo > PSNR inpaint.
3. **Thiếu cảnh tĩnh:** video giả nền trôi liên tục → < 60 cặp → `method = "inpaint"`.
4. **Quy tắc chọn:** hơn < 1 dB → `"inpaint"`.
5. **File cũ / hỏng:** `load_wm_matte` trả None khi dấu track lệch hoặc file hỏng; render vẫn chạy (inpaint).
6. **Ô sát mép:** watermark lòi ra mép khung → không lỗi, phần ngoài khung không bị ghi.

## 8. Xong khi

- Toàn bộ test qua (cũ + mới).
- **Video 36 phút (粥咪咪):** `method = "invert"`, PSNR phần giữ riêng ≥ 30 dB và hơn inpaint ≥ 5 dB; ảnh so sánh trước/sau (≥ 6 thời điểm) không còn mảng nhòe hay bóng ma rõ.
- **Video thiếu cảnh tĩnh:** tự về inpaint, kết quả như hiện nay.
- Bước tách lớp ≤ 1/4 thời lượng video; render không chậm hơn quá 10%.
- Ghi số đo thật vào mục 9 của spec này.

## 9. Kết quả kiểm thật (2026-10-17, video Douyin 36 phút, watermark 粥咪咪)

Video gốc không còn trên máy → chạy `wm_matte.build` trực tiếp trên bản 25 fps của job (không render cả video).

- **Số đo:** 163 cặp nền thật (600 ứng viên × 2 lần thử); PSNR phần giữ riêng: đảo pha trộn **32,6 dB**, inpaint **25,8 dB** (+6,8 dB) → `method = "invert"`. Tách lớp **37 s** cho 36,4 phút (0,02× thời lượng). Xóa / frame: **2,8 ms** (inpaint 7,1 ms).
- **Ảnh trước / sau ở 8 thời điểm (đều là frame có bằng chứng, mẫu khớp đúng chỗ, lệch 0 px kể cả khi dò ±12 px):**
  - nền có chi tiết (mép ghế, lan can, nền tối): tốt hơn inpaint rõ — giữ nét nền thay vì mảng nhòe;
  - còn **bóng ma thấy được** ở 3/8 (nền tím, nền kem phẳng có viền chữ nổi, nền cam);
  - **hỏng** ở 1/8: frame 31 (giây 1,2) watermark đang mờ dần vào, độ đậm thật ~0,47 lần mức ước lượng → trừ quá tay, còn chữ tối, tệ hơn inpaint.
- **Thử nhanh (nháp):** ước lượng độ đậm riêng từng frame (hệ số k = 0,47–1,26, bình phương tối thiểu so với nền inpaint) chưa đáng tin — có frame tệ đi (k = 1,26); chốt an toàn từng frame theo chỉ số bóng ma không thấy việc inpaint phá nét nền nên nghiêng về inpaint, và chỉ số có dấu còn chọn nhầm ở nền phẳng.
- **Kết luận:** tiêu chí "không còn mảng nhòe hay bóng ma rõ" **chưa đạt**. Chấm PSNR trên cặp cảnh tĩnh không bắt được: (1) độ đậm watermark thay đổi theo thời gian (mờ dần vào / ra, mỗi đoạn hơi khác), (2) bóng ma ở mép nét chữ trên nền phẳng.

## 10. Bổ sung 2026-10-24: trung bình theo thời gian + bắt độ đậm (sửa bóng ma, chữ in ngược)

### 10.1 Nghiên cứu (nháp trên 213 cặp nền thật + frame thật của video 36 phút)

- **Phần dư sau đảo pha trộn** nằm ở kênh sáng (Y: lòng 6,1, mép 12,8 RMS; Cr/Cb ≈ 2) và **gần như ngẫu nhiên theo frame**: phần hệ thống (TB theo điểm qua các cặp) chỉ 1,3 / 8,7 RMS → nhiễu nén H.264 quanh nét chữ, không phải lớp ước lượng sai. Làm trung bình nhiều frame sẽ khử được.
- **Trung bình theo thời gian** (frame t ± 3 đã khôi phục, cắt tại ô của t, trọng số exp(−(Δ/6)²) trên ảnh xám mờ σ 2): nền tĩnh sạch hẳn (5/7 frame mẫu), nền chuyển động (tím, cam) còn vệt nhẹ như trước.
- **Đã thử, không dùng:** bù chuyển động nền trước khi lấy TB (tương quan pha / luồng quang học Farneback, lề 24 rồi 48 px) — nền tím đổi hình (đèn, hoạt hình) chứ không chỉ trượt, trọng số 0,46 → 0,50, ảnh không khá hơn; nhân lớp theo k ước lượng khi k ∈ [0,7; 1] (cho đoạn watermark đậm ~85% lúc mới hiện) — trên 42 cặp nền thật k ước lệch k thật 0,07 (tương quan 0,20) nên frame thường tệ đi 0,5 dB; căn lớp lẻ pixel (đỉnh NCC nội suy parabol) — +0,1 dB, chỉ số bóng ma không phân biệt được; tách kênh màu (lỗi không nằm ở Cr/Cb); mô hình học sâu (ProPainter / LaMa) — máy chỉ có torch CPU + GTX 1650 4 GB, quá chậm cho 54 nghìn frame, và inpaint vẫn thua đảo pha trộn 6–8 dB ở watermark bán trong suốt.
- **Độ đậm k theo frame** (bình phương tối thiểu (ảnh − nền inpaint) theo α·(màu − nền inpaint)) trên 201 cửa sổ 7 frame rải khắp video: trung vị 0,99, p5 0,86, độ lệch chuẩn trong cửa sổ 0,04. Lệch > 0,3 ở 7% cửa sổ:
  - k ≈ 0–0,04: **không có watermark** (vị trí track còn giữ) → bản hiện tại trừ lớp vào nền sạch, **in ngược chữ tối** (lỗi thấy rõ);
  - k ≈ 0,4–0,7: watermark **hiện dần từng chữ** (粥 rõ, 咪咪 nhạt) → nhân hệ số k cho cả lớp không được, inpaint là an toàn;
  - k > 1,3: ước lượng k sai (nền inpaint kém, có chữ khác đè) nhưng đảo pha trộn vẫn xóa tốt → giữ đảo.
- Nền inpaint cho k: Telea trên ô thu nhỏ 2 lần (~2 ms) lệch ≤ 0,05 so với Telea cỡ thật (6,1 ms) kể cả ở frame chữ hiện dần / vắng watermark; thu 4 lần (1,1 ms) khớp nhóm 99,3% frame ngẫu nhiên nhưng lệch thấp 0,1–0,2 đúng ở frame chữ hiện dần (k 0,39 → 0,20, rơi nhầm sang "không có watermark").

### 10.2 Thiết kế (`wm_matte.py`)

- `opacity(patch, valid, matte)` → k (nền Telea cố định trên ô thu nhỏ 2 lần — ngưỡng chấm trên Telea, không theo `INPAINT_METHOD`; chỉ điểm α > 0,1 trong khung; ô gần như ngoài khung → 1).
- `stream(frames, pos, matte, mask_shape)`: nhận lặp (số frame, ảnh gốc) liên tiếp, trả lặp (số frame, ảnh đã xóa) cùng thứ tự, trễ `TEMPORAL_K` frame. Mỗi frame có vị trí: căn ±3 px, đảo pha trộn, đo k. Frame t ra theo trung vị k của t ± `TEMPORAL_K` (frame có vị trí):
  - k < `K_ABSENT` = 0,25 → trả ảnh gốc;
  - k < `K_FADE` = 0,7 → inpaint ô (trên ảnh gốc);
  - còn lại → ô của t = TB có trọng số exp(−(Δ / `TEMPORAL_DIFF`)²) các frame t ± `TEMPORAL_K` đã khôi phục cắt tại ô của t (Δ: ảnh xám mờ σ 2; luôn gồm chính t; bỏ láng giềng ngoài track — có thể còn watermark, và là frame người nhận sửa tại chỗ — hoặc có k < `K_FADE`; điểm ngoài khung trọng số 0). `TEMPORAL_K` = 3, `TEMPORAL_DIFF` = 6,0.
- `apply` giữ nguyên (xóa 1 frame, không TB) — dùng chung hàm căn ô với `stream`.

### 10.3 Render (`script.py`)

- `wm_frames(cap, start, end, wm_pos, wm_mask, matte)` → lặp (số frame, ảnh) cho [start, end) đã xóa watermark. Có lớp → `wm_matte.stream`, đọc thêm `TEMPORAL_K` frame mỗi phía của chunk (không ghi ra); không có lớp → inpaint từng frame như cũ. `process_chunk` lặp qua hàm này.
- `remove_wm` bỏ tham số `matte` (không còn ai gọi nhánh đó).

### 10.4 Xong khi

- Toàn bộ test qua; test mới trên video giả không nén: TB theo thời gian giảm sai số so với nền thật khi có nhiễu theo frame; đoạn không có watermark giữ nguyên ảnh; đoạn watermark nhạt (k ≈ 0,5) → inpaint; số frame ra đúng thứ tự, đủ, kể cả chunk sát đầu / cuối video.
- **Kết quả (2026-10-24):** đạt cả 3 ý dưới. Nhấp nháy trong ô (frame liền nhau, cùng ô) cũ → mới: nền kem 5,8 → 2,5; nền cam 4,4 → 0,9; nền tím 13,9 → 11,2; đoạn vắng watermark 1,8 → 1,5; mở đầu hiện dần 8,2 → 7,2. Xóa 9,1 ms / frame. Còn lại: ~0,5 s lúc watermark mới hiện (đậm ~85%) còn bóng nhẹ trên nền phẳng; nền chuyển động mạnh còn vệt nhẹ.
- Video 36 phút: ảnh trước / sau ở 11 frame (8 frame mục 9 + 2 frame không có watermark + 1 frame hiện dần) không còn chữ in ngược, nền tĩnh không còn bóng ma rõ.
- Xóa watermark ≤ 10 ms / frame (cũ 2,2 ms; render cả frame vài chục ms).
