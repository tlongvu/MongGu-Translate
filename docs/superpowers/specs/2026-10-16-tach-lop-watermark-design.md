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

- Ứng viên k: frame có bằng chứng (`evidence = 1`), rải đều theo thời gian, tối đa `MAX_PAIRS·3` ứng viên.
- Với mỗi k, thử j = k ± {75, 125, 200, 300} frame (3–12 s ở 25 fps), lần lượt; nhận j đầu tiên thỏa:
  - **watermark đã rời chỗ:** |x_j − x_k| ≥ rộng mask hoặc |y_j − y_k| ≥ cao mask (biết trước từ track, không cần đọc video);
  - **cảnh đứng yên:** sai khác trung bình |ô_k − ô_j| trên `ring` (điểm hợp lệ) < `STATIC_DIFF = 3.0` mức xám.
- Đọc video 1 lượt: lập trước danh sách (frame → các ô cần cắt), đọc tuần tự, chỉ giữ các ô (uint8).
- Dừng khi đủ `MAX_PAIRS = 400` cặp. Dưới `MIN_PAIRS = 60` cặp → `method = "inpaint"`, lý do "thiếu cảnh tĩnh".

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

## 9. Kết quả kiểm thật

(Điền khi làm xong.)
