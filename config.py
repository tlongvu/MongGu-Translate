"""
Monggu Translate - Centralized Configuration
Tất cả config tập trung tại đây, các module khác import từ file này.
"""

import os
from dotenv import load_dotenv

# Load .env từ thư mục cha (d:\monggutranslate\.env)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# ──────────────────────────────────────────────
# VIDEO
# ──────────────────────────────────────────────
VIDEO_PATH = "input.mp4"
TEST_MODE = False
TEST_FRAMES = 500

# ──────────────────────────────────────────────
# OCR (PaddleOCR)
# ──────────────────────────────────────────────
OCR_LANG = "ch"
OCR_INTERVAL_SEC = 0.0          # Quét mỗi frame (0.0s) để không bỏ sót
TARGET_FPS = 25                 # Convert video về fps này trước OCR (0 = không convert)
OCR_CONFIDENCE_THRESHOLD = 0.70  # Lọc text có confidence thấp
OCR_RESIZE_FACTOR = 0.5         # Downscale trước khi OCR (tăng tốc CPU)
CROP_TOP_RATIO = 0.0            # 0.0 = scan toàn bộ frame

# ── Scene-diff skip: bỏ qua OCR các frame gần như giống hệt frame trước ──
# Phụ đề thường đứng yên 1-3s → không cần OCR lại 25-75 lần cho cùng 1 câu.
OCR_SCENE_DIFF = True            # Bật/tắt cơ chế bỏ qua frame trùng lặp
OCR_SCENE_DIFF_THRESHOLD = 5.0   # Mean abs diff (0-255) tối thiểu để coi là "frame khác"
OCR_SCENE_DIFF_WIDTH = 64        # Bề rộng ảnh xám downscale dùng để so sánh nhanh
OCR_MAX_SKIP_SEC = 0.35          # Không bao giờ skip quá ngần này (giữ keyframe đủ dày cho segment)

# Watermark patterns (bỏ qua nếu text chứa các chuỗi này)
WATERMARK_PATTERNS = [
    '@', '抖音', 'id:', 'ID:', '关注', '点赞', '收藏', '快手', 'tiktok', 'douyin',
    # Disclaimer phổ biến trong video TQ
    '内容仅供参考',   # "Nội dung chỉ để tham khảo"
    '不良引导',        # "không hướng dẫn xấu"
    '仅供参考',        # rút gọn
    '小鸡快跑',        # tên kênh
    '小鸡',             # prefix kênh
]

# ──────────────────────────────────────────────
# TRANSLATION & NORMALIZATION
# ──────────────────────────────────────────────
GEMINI_API_KEYS = []
_primary_gemini = os.getenv("GEMINI_API_KEY", "")
if _primary_gemini:
    GEMINI_API_KEYS.append(_primary_gemini)

for i in range(1, 20):
    _key_val = os.getenv(f"GEMINI_API_KEY{i}", "")
    if _key_val and _key_val not in GEMINI_API_KEYS:
        GEMINI_API_KEYS.append(_key_val)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
DEEPSEEK_MODEL = "deepseek-chat"

NORMALIZATION_SYSTEM_PROMPT = """Bạn là hệ thống tiền xử lý phụ đề video Trung Quốc.

Nhiệm vụ: Tôi sẽ cung cấp một JSON array chứa các đoạn text OCR. Hãy xử lý CLEAN + NORMALIZE.

HƯỚNG DẪN:
1. LỌC RÁC:
- Định dạng Latin/Ký tự lặp vô nghĩa (vd "meme", "neme", "ADIATN").
- Text quá ngắn (<=1 ký tự) ko có nghĩa (Ngoại trừ: "我", "你", "他"...).
- Watermark.
- ĐỐI VỚI TEXT RÁC: Gán value = "" (chuỗi rỗng).

2. NORMALIZE TEXT (Sửa lỗi OCR):
- Sửa lặp chữ vô lý: "飞飞行" -> "飞行", "楷悄" -> "悄悄".

YÊU CẦU OUTPUT MỚI (BẮT BUỘC):
Trả về MỘT JSON Object (Dictionary).
Key: Text GỐC. Value: Text ĐÃ CHUẨN HÓA (hoặc "" nếu bỏ đi).
Ví dụ:
INPUT: ["mem", "我", "飞飞行", "正常句子"]
OUTPUT: {
  "mem": "",
  "我": "我",
  "飞飞行": "飞行",
  "正常句子": "正常句子"
}
KHÔNG giải thích. CHỈ trả về duy nhất JSON Object."""

# Prompt dịch (dùng cho mọi LLM qua llm.ask_json). Định dạng đầu ra do Phase 1 tự nối thêm.
TRANSLATION_GENRE = "phim ngắn Trung Quốc"
TRANSLATION_SYSTEM_PROMPT = """Bạn là dịch giả chuyên nghiệp phụ đề phim Trung Quốc sang tiếng Việt.
Thể loại phim: {genre}

Bảng tên riêng (BẮT BUỘC sử dụng đúng):
{glossary}

Quy tắc dịch:
1. Dịch tự nhiên, đúng ngữ cảnh phim, KHÔNG dịch word-by-word
2. Giữ nguyên tên riêng theo bảng glossary ở trên
3. Giữ giọng văn phù hợp thể loại phim (kịch tính, trang trọng khi cần)
4. Nếu là nhãn tên nhân vật (1-3 từ, ví dụ "姜姐", "王建成"), chỉ cần dịch tên
5. Câu ngắn 1-2 từ → dịch ngắn gọn, không thêm thắt"""

# ──────────────────────────────────────────────
# RENDERING
# ──────────────────────────────────────────────
FONT_PATH = os.path.join(_SCRIPT_DIR, "..", "assets", "fonts", "BeVietnamPro-Bold.ttf")
TEXT_COLOR = (255, 255, 0)      # Vàng (RGB cho Pillow)
OUTLINE_COLOR = (0, 0, 0)       # Đen
OUTLINE_WIDTH = 2
INPAINT_RADIUS = 4              # Giảm từ 7 → 4 (nhanh hơn, vẫn clean tốt)
INPAINT_METHOD = "telea"        # "telea" | "ns" (chất lượng) | "fill" (siêu nhanh: blur fill)

# ── Nền mờ bán trong suốt sau chữ (plate) — tăng độ đọc trên nền sáng/nhiễu ──
SUBTITLE_BG_ENABLE = True        # Vẽ hộp nền mờ phía sau mỗi cụm phụ đề
SUBTITLE_BG_COLOR = (0, 0, 0)    # Màu nền (RGB)
SUBTITLE_BG_ALPHA = 120          # Độ mờ 0-255 (0 = trong suốt, 255 = đặc)
SUBTITLE_BG_PAD = 8              # Đệm quanh chữ (px)
SUBTITLE_BG_RADIUS = 10          # Bo góc hộp nền (px)

# Text bị ẩn: inpaint xóa chữ gốc nhưng KHÔNG vẽ lại tiếng Việt
# (dùng cho tiêu đề bay quanh, nhãn gây nhiễu...)
SUPPRESS_TEXTS = [
    "命定新郎",   # "Chú rể định mệnh" — tiêu đề bay quanh
    "命定新",     # OCR đọc thiếu
    "命定新命",   # OCR đọc sai
    "meme",       # Watermark/label meme
    "邮猫 memle",  # OCR lỗi chữ meme
    "的猫 meme",   # meme mèo
    "蛋子的猫 meme",  # meme mèo của Danzi
    "的猫 neme",   # OCR đọc sai meme
    "ADIATN",     # OCR đọc nhầm logo
    "《柿子道符》",  # Watermark tiêu đề
    "柿子道符",      # Watermark tiêu đề (không dấu ngoặc)
    "明月猫 meme",    # Watermark kênh
    "明月猫",        # Watermark kênh (không meme)
    "《明月之钩》",    # Watermark tiêu đề mới
    "明月之钩",        # Watermark tiêu đề mới (không dấu ngoặc)
    # ── Các biến thể OCR của Watermark "明月之钩" ──
    "明月", "明目", "明县", "明洱", "明有", "明厨", "明之", "阴之",
    "之钧", "之钓", "之钩", "之铃", "手之钓", "月之钓",
    "《空空无心》",  # Watermark mới
    "空空无心",      # Watermark mới (không dấu)
    "美乐猫 meme",    # Watermark mới
    "美乐猫",        # Watermark mới (không meme)
    # ── Thêm mới ──
    "《魔丸天师》",  # Watermark tiêu đề kênh hiện tại
    "魔丸天师",      # Tiêu đề (không dấu ngoặc)
    "《魔丸跑腿》",  # Watermark tiêu đề kênh
    "魔丸跑腿",      # Tiêu đề (không dấu ngoặc)
    "猫猫meme小剧场",  # Nhãn kênh "猫猫meme小剧场" — xóa nhưng không dịch
    "猫猫 meme小剧场",  # Biến thể OCR có khoảng trắng
    "猫猫meme",     # Rút gọn nhãn
    "猫猫",         # Prefix kênh
    # Disclaimer dọc cạnh trái (an toàn hơn lọc ở đây)
    "内容仅供参考",
    "内容仅供参考无任何不良引导",
    "仅供参考",
    "小鸡快跑",
    # ── Video mới ──
    "《翻天养女》",    # Watermark tiêu đề
    "翻天养女",        # Tiêu đề (không dấu ngoặc)
    "猫meme的猫me",    # Nhãn kênh meme
    "猫meme的猫",      # OCR đọc thiếu
    "猫meme的",        # OCR đọc thiếu
    "《言之高手》",    # Watermark tiêu đề "Ngôn chi cao thủ"
    "言之高手",        # Tiêu đề (không dấu ngoặc)
    # ── Video mới ──
    "原创◎加更",      # Nhãn "nguyên sang ◎ tập thêm"
    "原创加更",        # Biến thể OCR mất ký tự ◎
    "原创",            # OCR đọc thiếu
    "加更",            # OCR đọc thiếu
    "原文《桃桃师妹》",  # Watermark tiêu đề gốc
    "《桃桃师妹》",    # Tiêu đề (không có chữ 原文)
    "桃桃师妹",        # Tiêu đề (không dấu ngoặc)
    # ── Video mới (thêm theo yêu cầu người dùng) ──
    "虚构演绎",        # Disclaimer: diễn xuất hư cấu
    "仅供娱乐",        # Disclaimer: chỉ để giải trí
    "请勿模仿",        # Disclaimer: vui lòng không bắt chước
    "无任何不良引导",  # Disclaimer: không có hướng dẫn xấu
    "《楚楚暴富》",    # Watermark tiêu đề
    "楚楚暴富",        # Tiêu đề (không dấu ngoặc)
    "初与猫meme",      # Watermark kênh/nhãn meme
    "初与猫",          # Biến thể nhãn
    "住院部",          # Biển hiệu tòa nhà (tránh dịch lẫn vào phụ đề)
    # ── Video mới (thêm theo yêu cầu người dùng) ──
    "原创@哈加密",      # Watermark/Nhãn kênh
    "原创哈加密",        # Biến thể OCR mất ký tự @
    "哈加密",            # Tên kênh rút gọn
    # ── Các biến thể OCR của Watermark "原文《香蕉长生》" ──
    "原文《香蕉",
    "《香蕉长生",
    "香蕉长生》",
    "香蕉长生）",
    "文《香蕉长生",
    "原文（香蕉",
    "原文6香蕉",
    "原文《香薄",
    "安《香蕉",
    "云《香蕉",
]

# Auto-suppress: text chỉ chứa chữ Latin (không có chữ Trung) và ngắn → thường là logo/watermark bị OCR đọc nhầm
AUTO_SUPPRESS_LATIN_ONLY = True

# ──────────────────────────────────────────────
# TIKTOK PASS MODE (Bypass detection)
# ──────────────────────────────────────────────
TIKTOK_PASS_MODE = False

# Biến đổi khung hình để đổi "vân tay" (Hash) của video
PASS_ZOOM_RATIO = 1.05          # Zoom 5% (cắt bớt viền cũ, đổi hash)
PASS_SHARPEN = True              # Tăng độ sắc nét (giảm cảm giác low quality)
PASS_COLOR_ADJUST = True         # Chỉnh nhẹ độ sáng/tương phản

# Chuyển đổi định dạng dọc 9:16 (Bỏ vạch đen, thêm nền mở rộng mờ)
PASS_PORTRAIT_916 = True        # Nếu True, output sẽ là 1080x1920

# Tự động phát hiện và xóa mã QR (Tránh vi phạm)
SUPPRESS_QR_CODES = True

# ──────────────────────────────────────────────
# PERFORMANCE
# ──────────────────────────────────────────────
NUM_WORKERS = 4                  # Giảm từ 6 → 4 (bớt tải CPU, tránh đơ máy)

# ──────────────────────────────────────────────
# GPU ACCELERATION (GTX 1650)
# ──────────────────────────────────────────────
USE_GPU = True                   # Bật/tắt GPU acceleration
USE_NVENC = True                 # FFmpeg NVENC encode (chip riêng, ~5-10% GPU)
USE_NVDEC = True                 # FFmpeg NVDEC decode (chip riêng, ~5% GPU)
USE_TORCH_CUDA = True            # PyTorch CUDA cho resize/blur/cvtColor (~15% GPU)
GPU_MAX_LOAD = 0.30              # Giới hạn GPU load tối đa (30% → không đơ máy)

# ──────────────────────────────────────────────
# ĐƯỜNG DẪN DỮ LIỆU
# ──────────────────────────────────────────────
# Cache lời gọi LLM (llm.ask_json): khóa = hash prompt đầy đủ → an toàn giữa các video
LLM_CACHE_PATH = os.path.join(_SCRIPT_DIR, "cache", "llm.sqlite")
# Dữ liệu theo video (jobs.py): jobs/<tên-video>/…; series dùng chung hồ sơ: series/<tên-bộ>/…
JOBS_DIR = os.path.join(_SCRIPT_DIR, "jobs")
SERIES_DIR = os.path.join(_SCRIPT_DIR, "series")

# ──────────────────────────────────────────────
# OVERRIDE PER-VIDEO (tùy chọn)
# ──────────────────────────────────────────────
# Đặt biến môi trường MONGGU_CONFIG=path/to/override.json để ghi đè bất kỳ hằng số
# IN HOA nào ở trên cho từng video (vd: SUPPRESS_TEXTS, TEXT_COLOR, INPAINT_METHOD...)
# mà không phải sửa file này. Ví dụ override.json:
#   { "SUPPRESS_TEXTS": ["..."], "TIKTOK_PASS_MODE": true }
_override_path = os.getenv("MONGGU_CONFIG", "")
if _override_path and os.path.exists(_override_path):
    import json as _json
    try:
        with open(_override_path, "r", encoding="utf-8") as _f:
            _ov = _json.load(_f)
        _applied = [k for k in _ov if k.isupper()]
        for _k in _applied:
            globals()[_k] = _ov[_k]
        # ASCII-only để config import được từ mọi tiến trình (kể cả chưa set UTF-8 stdout)
        print(f"[config] Override applied from {_override_path}: {_applied}")
    except Exception as _e:
        print(f"[config][WARN] Cannot read override {_override_path}: {_e}")
