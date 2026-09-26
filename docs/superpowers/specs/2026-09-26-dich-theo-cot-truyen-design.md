# Dự án con B — Dịch theo cốt truyện, đúng nhân vật, có cache

**Ngày:** 2026-09-26 · **Nhánh:** `nang-cap` · **Tiền đề:** dự án con A đã xong (`docs/superpowers/specs/2026-09-26-nang-cap-design.md`)

## 1. Mục tiêu và yêu cầu (user chốt 2026-09-26)

- **Lỗi cần hết:** sai xưng hô, nhầm người nói, sai ý theo cốt truyện. Tên riêng không nhất quán không phải vấn đề chính.
- **Loại video:** có cả series nhiều tập lẫn video độc lập. Phải chọn được "video này thuộc bộ nào".
- **Backend thực dùng:** Gemini (5 key xoay vòng) và DeepSeek. Freemodel, NVIDIA và Google Translate không dùng.
- **Cách làm được chọn:** tách 3 bước — hồ sơ truyện → người nói/người nghe → dịch theo từng lần xuất hiện, xưng hô tra từ bảng theo cặp.
- **Bắt buộc có cache** để giảm tải cho hệ thống dịch.

## 2. Nguyên nhân gốc (đã kiểm chứng trong code)

1. Một câu tiếng Trung chỉ có một bản dịch cho cả video (`translation_map` khóa theo câu, render tra theo câu).
2. Hồ sơ nhân vật chỉ suy từ 90 dòng đầu (`translate_pass.py:1121`).
3. Hầu hết các câu không có thông tin người nói/người nghe; gợi ý chỉ đến từ nhãn tên trên màn hình.
4. Ngữ cảnh lẫn giữa các phim: prompt viết cứng một video cũ, glossary ép `我→Tôi`, hồ sơ và cache dùng chung toàn cục.
5. Đường "dịch rời rạc" dự phòng không có ngữ cảnh, lại gán theo vị trí khi model trả thiếu câu.

## 3. Chia việc

| | Phần | Nội dung |
|---|---|---|
| **B1** | Lõi LLM + cache | Mọi lời gọi qua `llm.ask_json()`; cache SQLite; xóa 3 backend thừa, code copy và đường dịch rời rạc |
| **B2** | Thư mục job + series | `jobs/<video>/`, `series/<bộ>/story.json`, `--series`; bỏ prompt/glossary/hồ sơ toàn cục |
| **B3** | Dịch theo cốt truyện | 5 bước ở mục 5; render tra bản dịch theo segment |

Làm theo thứ tự B1 → B2 → B3: B3 cần cache (để không tốn quota) và chỗ lưu riêng theo video/series. Dự án con C cũ (lõi LLM) nay là B1. D (reel gốc) và E (dọn dẹp) giữ nguyên.

## 4. Dữ liệu (B2, B3)

```
jobs/<tên-video>/          mỗi video một thư mục
  ocr_data.json
  story.json               hồ sơ truyện (video độc lập); video thuộc series dùng series/<bộ>/story.json
  lines.json               từng lần xuất hiện của từng câu
  output.mp4
series/<tên-bộ>/story.json dùng chung cho mọi tập; mỗi tập chỉ bổ sung
cache/llm.sqlite           cache lời gọi LLM
```

**`story.json`:**
```json
{ "tom_tat": "…",
  "nhan_vat": { "tieu_thu": {"ten": "Bản tiểu thư", "ten_goc": ["本小姐"], "gioi_tinh": "nữ", "vai": "nữ chính"} },
  "xung_ho":  { "tieu_thu>yiyi": {"xung": "bản tiểu thư", "goi": "Y Y"},
                "tieu_thu>*":    {"xung": "ta", "goi": "ngươi"} },
  "ten_rieng": {"香蕉长生": "Chuối Trường Sinh"} }
```
- `A>B`: khi A nói với B thì A xưng gì và gọi B là gì.
- `A>*`: mặc định khi không rõ người nghe, hoặc nói với nhiều người.
- ID nhân vật là chuỗi ASCII không dấu, ổn định; `lines.json` tham chiếu tới chúng.

**`lines.json`** — mỗi phần tử là một segment của `ocr_data`:
```json
{"id": 812, "zh": "我要去修仙！", "noi": "tieu_thu", "nghe": "yiyi", "chac": 0.9, "vi": "Bản tiểu thư muốn đi tu tiên!"}
```

**File là nguồn sự thật:**
- Mỗi bước chỉ điền phần còn thiếu, không ghi đè thứ đã có, nên sửa tay luôn được giữ.
- Series: mỗi tập chỉ **thêm** nhân vật và cặp xưng hô mới vào hồ sơ chung.
- Sửa tay rồi chạy lại thì cache bảo đảm chỉ phần liên quan phải gọi lại LLM.

## 5. Luồng dịch (B3)

| Bước | Vào LLM | Ra | Lời gọi (video 3.000 câu) |
|---|---|---|---|
| 0. Chuẩn hóa | Chữ OCR thô | Chữ sạch, hoặc `""` nếu là rác | ~21 |
| 1. Hồ sơ truyện | **Toàn bộ** lời thoại theo thứ tự, kèm hồ sơ series nếu có | `story.json` | 1 |
| 2. Người nói | Cửa sổ 200 câu, kèm hồ sơ và nhãn tên đang hiện trên màn hình | `noi`, `nghe`, `chac` từng câu | ~15 |
| 3. Dịch | Cửa sổ 120 câu gắn `(A→B)`, kèm tóm tắt, **chỉ các cặp xưng hô có trong cửa sổ**, và 15 câu đã dịch ngay trước | `vi` theo `id` | ~25 |
| 4. Rà soát | Câu đã dịch, kèm người nói và bảng xưng hô | Chỉ các câu cần sửa; chỉ nhận `id` thuộc cửa sổ | ~20 |

- **Mỗi lời gọi sở hữu riêng các câu của nó.** Câu ngữ cảnh gối đầu chỉ để đọc, không được gán hay dịch lại.
- **Render** tra bản dịch theo segment. Chữ OCR thô ở frame f lấy bản dịch của segment đang phủ f có chữ giống nhất (≥ 0.7). Bỏ `translation_map.json`.
- **Chi phí:** khoảng 82 lời gọi (hiện nay khoảng 71), thêm khoảng 30–40% token, chủ yếu do bước 1. Chạy lại gần như bằng 0 nhờ cache.

## 6. Cache và xử lý lỗi

### Cache — `cache/llm.sqlite`, dùng `sqlite3` có sẵn trong Python

- **Khóa:** `sha256([CACHE_VERSION, temperature, system, user])`. Không đưa tên model vào khóa, nên câu trả lời DeepSeek đã có thì lần sau Gemini không phải trả lời lại. Tăng `CACHE_VERSION` để cố ý làm mới toàn bộ.
- **Chỉ lưu JSON hợp lệ**, không bao giờ lưu lỗi.
- `translation_cache.json` cũ (khóa theo câu) thôi dùng. File vẫn để nguyên trên đĩa.

### Xử lý lỗi

- **Gemini:**
  - Xoay key khi gặp lỗi quota (429 / `RESOURCE_EXHAUSTED` / `quota`).
  - Cả vòng key đều bị 429 thì đợi 30 giây rồi thử thêm một vòng (giới hạn theo phút thường hết sau chừng đó).
  - Vẫn 429 thì coi Gemini đã cạn cho cả phiên và dùng DeepSeek.
  - JSON hỏng thì thử lại một lần, **không** tắt Gemini.
- **Lỗi khác** (mạng, request): thử lại một lần sau 2 giây, rồi chuyển sang provider tiếp theo **cho riêng lời gọi đó**.
- **Mọi provider đều hỏng:** `LLMError` → pass thoát với exit 1. Phần đã xong nằm trong cache, chạy lại sẽ làm tiếp.
- **Thiếu câu trong kết quả:**
  - hỏi lại riêng các câu còn thiếu (prompt khác nên khóa cache khác);
  - vẫn thiếu thì exit 1 và liệt kê các câu thiếu;
  - không bao giờ âm thầm để trống.
- **`noi`/`nghe` không có trong hồ sơ:** gán `?` với `chac` = 0, rồi dùng cặp xưng hô `A>*`.

## 7. B1 chi tiết — Lõi LLM + cache

### `llm.py` (mới)

- `ask_json(system: str, user: str, temperature: float = 0.1, providers=None) -> dict | list`
  - Tra cache → gọi lần lượt các provider → parse JSON → lưu cache → trả về.
  - Hỏng hết thì raise `LLMError`.
  - `providers` là danh sách `(tên, hàm(system, user, temperature) -> str)`, dùng làm chỗ cắm cho test. Mặc định: Gemini (nếu có key và chưa cạn), rồi DeepSeek (nếu có key).
- `parse_json_loose(txt)` — chuyển từ `translate_pass._parse_json_loose`.
- `LLMError(RuntimeError)`.
- **Gemini:** SDK `google-genai`, model `GEMINI_MODEL`, JSON mode, xoay key như mục 6. Không in bất kỳ phần nào của key ra log.
- **DeepSeek:** SDK `openai`, `base_url="https://api.deepseek.com"`, model `DEEPSEEK_MODEL`, `response_format=json_object`. System prompt được nối thêm "(Trả về đúng định dạng JSON.)" vì JSON mode của DeepSeek yêu cầu chữ "json" xuất hiện trong prompt.

### `translate_pass.py`

**Xóa:**
- `load_translation_cache`, `save_translation_cache`
- `GeminiClientRotator`, `get_gemini_rotator`
- `normalize_texts_with_*` (4), `translate_with_*` (5)
- `_gemini_json`, `_freemodel_json`, `_deepseek_json`, `_nvidia_json`, `_llm_json`, `_translate_isolated`
- các biến toàn cục `_active_backend`, `_gemini_dead`, `_*_client`

**Thêm:** `normalize_texts(unique_texts, batch_size=150) -> dict`, dùng `ask_json`. Câu model bỏ sót, hoặc giá trị không phải chuỗi → giữ nguyên chữ gốc (như cũ).

**Giữ các phase 1–4 hiện có** (B3 sẽ thay), chỉ đổi sang `ask_json`:
- Phase 1 thiếu câu → gọi lại một lần cho riêng các câu thiếu; vẫn thiếu thì raise `LLMError`.
- Phase 4 chỉ nhận `id` thuộc cửa sổ.
- Câu sạch không nằm trong transcript (703/3.145 biến thể OCR trên dữ liệu thật) → mượn bản dịch của dòng transcript giống nhất (≥ 0.7), không có thì `""`.

**`run_translation()`:**
- Bỏ tham số `backend` và phần cập nhật cache cũ.
- `LLMError` → in lỗi rồi `return False`.
- CLI bỏ `--backend`.

### Các file khác

- **`config.py`:**
  - bỏ `OPENAI_API_KEY`, `FREEMODEL_*`, `NVIDIA_*`, `GEMINI_API_KEY` (đơn), `TRANSLATION_CACHE_PATH`;
  - thêm `GEMINI_MODEL = "gemini-2.5-flash"`, `DEEPSEEK_MODEL = "deepseek-chat"`, `LLM_CACHE_PATH = <thư mục script>/cache/llm.sqlite`.
- **`pipeline.py`:** bỏ `--backend`.
- **`.gitignore`:** thêm `cache/`.

### Kiểm chứng B1 (không tốn API)

`tests/test_llm.py`, dùng provider giả và cache trong thư mục tạm:
- hỏi y hệt → provider chỉ bị gọi một lần; đổi prompt → gọi lại;
- JSON hỏng không được cache;
- provider đầu lỗi → dùng provider sau;
- mọi provider lỗi → `LLMError`;
- `parse_json_loose` bóc được JSON bọc trong ```json.

`tests/test_core.py`:
- bỏ test cache cũ;
- thêm test `translate_contextual` với `ask_json` giả: thiếu câu → hỏi lại đúng các câu thiếu; vẫn thiếu → `LLMError`; biến thể mượn bản dịch dòng giống nhất.

Không chạy `translate_pass.py` thật (sẽ gọi API). Trường hợp thiếu `ocr_data` đã có test `run_translation` trả `False`; chỉ cần sửa theo chữ ký mới.
