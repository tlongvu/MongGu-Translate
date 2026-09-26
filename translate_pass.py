"""
Monggu Translate - Pass 2: Dịch thuật bằng Gemini
Load ocr_data.json → gửi Gemini 1 lần → xuất translation_map.json

Usage:
    python translate_pass.py                    # Dịch bằng Gemini (mặc định)
    python translate_pass.py --backend google   # Fallback dùng Google Translate
"""

import sys

# Đảm bảo console in được tiếng Việt Unicode (kể cả khi stdout là pipe/subprocess)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import json
import os
import re
import time
import argparse

from config import (
    GEMINI_API_KEY, GEMINI_API_KEYS, DEEPSEEK_API_KEY, GLOSSARY_PATH,
    FREEMODEL_API_KEY, FREEMODEL_BASE_URL, FREEMODEL_MODEL,
    NVIDIA_API_KEY, NVIDIA_MODEL,
    TRANSLATION_GENRE, TRANSLATION_SYSTEM_PROMPT,
    NORMALIZATION_SYSTEM_PROMPT,
    OCR_DATA_PATH, TRANSLATION_MAP_PATH, TRANSLATION_CACHE_PATH,
)


def load_translation_cache() -> dict:
    """Load cache bản dịch {cleaned_zh: vi} từ các lần chạy trước (nếu có)."""
    if os.path.exists(TRANSLATION_CACHE_PATH):
        try:
            with open(TRANSLATION_CACHE_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"  [CACHE] Đã nạp {len(cache)} bản dịch cũ từ {os.path.basename(TRANSLATION_CACHE_PATH)}")
            return cache
        except Exception as e:
            print(f"  [CACHE][WARN] Không đọc được cache, bỏ qua: {e}")
    return {}


def save_translation_cache(cache: dict) -> None:
    """Lưu cache bản dịch để lần sau chỉ dịch câu mới."""
    try:
        with open(TRANSLATION_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  [CACHE][WARN] Không lưu được cache: {e}")


def load_glossary() -> dict:
    """Load bảng glossary tên riêng."""
    if os.path.exists(GLOSSARY_PATH):
        with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
            glossary = json.load(f)
        print(f"  Loaded glossary: {len(glossary)} entries")
        return glossary
    print("  [WARN] Không tìm thấy glossary.json, bỏ qua")
    return {}


def load_ocr_data() -> dict:
    """Load kết quả OCR từ Pass 1."""
    if not os.path.exists(OCR_DATA_PATH):
        print(f"  [ERROR] Không tìm thấy {OCR_DATA_PATH}")
        print(f"  Hãy chạy ocr_pass.py trước!")
        return None

    with open(OCR_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded OCR data: {len(data.get('unique_texts', []))} unique texts")
    return data


def format_glossary_for_prompt(glossary: dict) -> str:
    """Format glossary thành chuỗi đẹp cho prompt."""
    if not glossary:
        return "(Không có)"
    lines = [f"  {zh} → {vi}" for zh, vi in glossary.items()]
    return "\n".join(lines)


class GeminiClientRotator:
    """Quản lý khởi tạo và xoay tua tự động giữa các API Key của Gemini."""
    def __init__(self):
        self.keys = GEMINI_API_KEYS
        self.current_idx = 0
        self.client = None
        self.model_name = "gemini-2.5-flash"
        self._init_current_client()

    def _init_current_client(self) -> bool:
        if not self.keys:
            self.client = None
            return False
        try:
            from google import genai
            key = self.keys[self.current_idx]
            masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else "..."
            print(f"  [Gemini Rotator] Đang dùng Key index {self.current_idx} ({masked_key})")
            self.client = genai.Client(api_key=key)
            return True
        except Exception as e:
            print(f"  [Gemini Rotator] Lỗi khởi tạo client với key {self.current_idx}: {e}")
            self.client = None
            return False

    def rotate_key(self) -> bool:
        if len(self.keys) <= 1:
            print("  [Gemini Rotator] Chỉ có 1 key, giữ nguyên.")
            return False
        self.current_idx = (self.current_idx + 1) % len(self.keys)
        return self._init_current_client()

    def generate_content(self, contents, config, max_attempts=5):
        """Gọi API của Gemini, nếu bị giới hạn tần suất (429/Quota) sẽ tự động đổi key và thử lại."""
        if not self.client:
            if not self._init_current_client():
                raise RuntimeError("Không có Gemini key hợp lệ.")

        attempt = 0
        max_total_retries = max(max_attempts, len(self.keys) * 2)

        while attempt < max_total_retries:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config
                )
                return response
            except Exception as e:
                attempt += 1
                error_str = str(e)
                # Phát hiện lỗi rate limit hoặc hết hạn ngạch
                is_rate_limit = any(term in error_str.lower() for term in ['429', 'quota', 'resourceexhausted', 'limit', 'exhausted'])
                
                print(f"  [Gemini Error] Lỗi khi gọi API: {error_str[:150]}")
                
                if is_rate_limit:
                    if len(self.keys) > 1:
                        self.rotate_key()
                        time.sleep(1)
                    else:
                        wait_time = attempt * 5
                        print(f"  [Gemini] Bị rate limit (chỉ có 1 key). Đợi {wait_time}s rồi thử lại...")
                        time.sleep(wait_time)
                else:
                    # Lỗi mạng hoặc lỗi khác, xoay key hoặc đợi nhẹ rồi thử lại
                    if len(self.keys) > 1:
                        self.rotate_key()
                    time.sleep(2)
                    
        raise RuntimeError(f"Gọi Gemini thất bại sau {max_total_retries} lần thử.")


# Rotator toàn cục để duy trì key đang hoạt động tốt qua các lượt gọi khác nhau
_gemini_rotator = None

def get_gemini_rotator() -> GeminiClientRotator:
    global _gemini_rotator
    if _gemini_rotator is None:
        _gemini_rotator = GeminiClientRotator()
    return _gemini_rotator


def normalize_texts_with_gemini(unique_texts: list, batch_size: int = 150) -> dict:
    """
    Tiền xử lý (Clean + Normalize) OCR text bằng Gemini.
    Trả về mapping: {raw_text: cleaned_text}
    """
    print("\n  [NORMALIZE] Bắt đầu gọi Gemini để làm sạch text OCR...")
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("  [ERROR] Chưa cài gói google-genai! Chạy: pip install google-genai")
        return None

    if not GEMINI_API_KEYS:
        print("  [ERROR] Chưa có GEMINI_API_KEY trong .env!")
        return None

    rotator = get_gemini_rotator()
    
    normalization_map = {}
    total_batches = (len(unique_texts) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(unique_texts))
        batch = unique_texts[start:end]

        print(f"    Batch {batch_idx + 1}/{total_batches} ({len(batch)} câu)...")
        user_prompt = json.dumps(batch, ensure_ascii=False)

        try:
            config = types.GenerateContentConfig(
                system_instruction=NORMALIZATION_SYSTEM_PROMPT,
                temperature=0.0
            )
            response = rotator.generate_content(
                contents=user_prompt,
                config=config
            )
            response_text = response.text.strip()
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    clean_dict = json.loads(json_match.group())
                    for zh in batch:
                        normalization_map[zh] = clean_dict.get(zh, zh)
                except json.JSONDecodeError:
                    print("  [WARN] Không parse được JSON. Fallback giữ nguyên gốc.")
                    for zh in batch: normalization_map[zh] = zh
            else:
                for zh in batch: normalization_map[zh] = zh

        except Exception as e:
            print(f"    [ERROR] Gemini API lỗi: {e}")
            return None
        
        if batch_idx < total_batches - 1:
            time.sleep(2)

    return normalization_map


def normalize_texts_with_deepseek(unique_texts: list, batch_size: int = 150) -> dict:
    """
    Tiền xử lý (Clean + Normalize) OCR text bằng DeepSeek API.
    """
    if not DEEPSEEK_API_KEY:
        print("  [ERROR] Chưa có DEEPSEEK_API_KEY trong .env!")
        return {t: t for t in unique_texts}

    try:
        from openai import OpenAI
    except ImportError:
        print("  [ERROR] Chưa cài gói openai! Chạy: pip install openai")
        return {t: t for t in unique_texts}

    print("\n  [NORMALIZE] Bắt đầu gọi DeepSeek để làm sạch text OCR...")
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    except Exception as e:
        print(f"  [ERROR] Lỗi khởi tạo client DeepSeek: {e}")
        return {t: t for t in unique_texts}
    
    normalization_map = {}
    total_batches = (len(unique_texts) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(unique_texts))
        batch = unique_texts[start:end]

        print(f"    Batch {batch_idx + 1}/{total_batches} ({len(batch)} câu)...")
        user_prompt = json.dumps(batch, ensure_ascii=False)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": NORMALIZATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0
                )
                response_text = response.choices[0].message.content.strip()
                
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        clean_dict = json.loads(json_match.group())
                        for zh in batch:
                            normalization_map[zh] = clean_dict.get(zh, zh)
                    except json.JSONDecodeError:
                        print("  [WARN] Không parse được JSON. Fallback giữ nguyên gốc.")
                        for zh in batch: normalization_map[zh] = zh
                else:
                    for zh in batch: normalization_map[zh] = zh
                break

            except Exception as e:
                print(f"    [ERROR] DeepSeek API lỗi (lần {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    for zh in batch: normalization_map[zh] = zh
                time.sleep(2 * (attempt + 1))
        
        if batch_idx < total_batches - 1:
            time.sleep(1)

    return normalization_map


def normalize_texts_with_freemodel(unique_texts: list, batch_size: int = 150) -> dict:
    """
    Tiền xử lý (Clean + Normalize) OCR text bằng Freemodel (OpenAI endpoint).
    """
    if not FREEMODEL_API_KEY:
        print("  [ERROR] Chưa có FREEMODEL_API_KEY trong .env!")
        return {t: t for t in unique_texts}

    try:
        from openai import OpenAI
    except ImportError:
        print("  [ERROR] Chưa cài gói openai! Chạy: pip install openai")
        return {t: t for t in unique_texts}

    print("\n  [NORMALIZE] Bắt đầu gọi Freemodel để làm sạch text OCR...")
    try:
        client = OpenAI(api_key=FREEMODEL_API_KEY, base_url=FREEMODEL_BASE_URL)
    except Exception as e:
        print(f"  [ERROR] Lỗi khởi tạo client Freemodel: {e}")
        return {t: t for t in unique_texts}
    
    normalization_map = {}
    total_batches = (len(unique_texts) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(unique_texts))
        batch = unique_texts[start:end]

        print(f"    Batch {batch_idx + 1}/{total_batches} ({len(batch)} câu)...")
        user_prompt = json.dumps(batch, ensure_ascii=False)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=FREEMODEL_MODEL,
                    messages=[
                        {"role": "system", "content": NORMALIZATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0
                )
                response_text = response.choices[0].message.content.strip()
                
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        clean_dict = json.loads(json_match.group())
                        for zh in batch:
                            normalization_map[zh] = clean_dict.get(zh, zh)
                    except json.JSONDecodeError:
                        print("  [WARN] Không parse được JSON. Fallback giữ nguyên gốc.")
                        for zh in batch: normalization_map[zh] = zh
                else:
                    for zh in batch: normalization_map[zh] = zh
                break

            except Exception as e:
                print(f"    [ERROR] Freemodel API lỗi (lần {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    for zh in batch: normalization_map[zh] = zh
                time.sleep(2 * (attempt + 1))
        
        if batch_idx < total_batches - 1:
            time.sleep(1)

    return normalization_map


def normalize_texts_with_nvidia(unique_texts: list, batch_size: int = 150) -> dict:
    """
    Tiền xử lý (Clean + Normalize) OCR text bằng Nvidia API.
    """
    if not NVIDIA_API_KEY:
        print("  [ERROR] Chưa có NVIDIA_API_KEY trong .env!")
        return {t: t for t in unique_texts}

    try:
        from openai import OpenAI
    except ImportError:
        print("  [ERROR] Chưa cài gói openai! Chạy: pip install openai")
        return {t: t for t in unique_texts}

    print("\n  [NORMALIZE] Bắt đầu gọi Nvidia API để làm sạch text OCR...")
    try:
        client = OpenAI(api_key=NVIDIA_API_KEY, base_url="https://integrate.api.nvidia.com/v1")
    except Exception as e:
        print(f"  [ERROR] Lỗi khởi tạo client Nvidia: {e}")
        return {t: t for t in unique_texts}
    
    normalization_map = {}
    total_batches = (len(unique_texts) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(unique_texts))
        batch = unique_texts[start:end]

        print(f"    Batch {batch_idx + 1}/{total_batches} ({len(batch)} câu)...")
        user_prompt = json.dumps(batch, ensure_ascii=False)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=NVIDIA_MODEL,
                    messages=[
                        {"role": "system", "content": NORMALIZATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0
                )
                response_text = response.choices[0].message.content.strip()
                
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        clean_dict = json.loads(json_match.group())
                        for zh in batch:
                            normalization_map[zh] = clean_dict.get(zh, zh)
                    except json.JSONDecodeError:
                        print("  [WARN] Không parse được JSON. Fallback giữ nguyên gốc.")
                        for zh in batch: normalization_map[zh] = zh
                else:
                    for zh in batch: normalization_map[zh] = zh
                break

            except Exception as e:
                print(f"    [ERROR] Nvidia API lỗi (lần {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    for zh in batch: normalization_map[zh] = zh
                time.sleep(2 * (attempt + 1))
        
        if batch_idx < total_batches - 1:
            time.sleep(1)

    return normalization_map


def translate_with_gemini(unique_texts: list, glossary: dict, batch_size: int = 50) -> dict:
    """
    Dịch bằng Gemini API.
    Gửi theo batch_size để tránh quá dài.
    Trả về: {text_zh: text_vi}
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("  [ERROR] Chưa cài gói google-genai! Chạy: pip install google-genai")
        return None

    if not GEMINI_API_KEYS:
        print("  [ERROR] Chưa có GEMINI_API_KEY trong .env!")
        return None

    rotator = get_gemini_rotator()

    glossary_str = format_glossary_for_prompt(glossary)
    system_prompt = TRANSLATION_SYSTEM_PROMPT.format(
        genre=TRANSLATION_GENRE,
        glossary=glossary_str,
    )

    translation_map = {}
    total_batches = (len(unique_texts) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(unique_texts))
        batch = unique_texts[start:end]

        print(f"  Batch {batch_idx + 1}/{total_batches} ({len(batch)} câu)...")

        user_prompt = json.dumps(batch, ensure_ascii=False)

        try:
            # Structured output: ép Gemini trả về đúng JSON array of string
            # → hết lỗi parse khi model thêm lời dẫn / bọc ```json.
            try:
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=list[str],
                )
            except Exception:
                # SDK cũ không hỗ trợ response_schema → fallback cấu hình cơ bản
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1,
                )
            response = rotator.generate_content(
                contents=user_prompt,
                config=config
            )
            response_text = response.text.strip()

            # Với structured output, response_text đã là JSON array thuần.
            try:
                translated_list = json.loads(response_text)
            except json.JSONDecodeError:
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    translated_list = json.loads(json_match.group())
                else:
                    print(f"  [WARN] Không parse được JSON từ Gemini response")
                    print(f"  Response: {response_text[:200]}...")
                    translated_list = batch

            # Map từng câu
            if len(translated_list) == len(batch):
                for zh, vi in zip(batch, translated_list):
                    translation_map[zh] = vi
            else:
                print(f"  [WARN] Số lượng không khớp: gửi {len(batch)}, nhận {len(translated_list)}")
                for i, zh in enumerate(batch):
                    if i < len(translated_list):
                        translation_map[zh] = translated_list[i]
                    else:
                        translation_map[zh] = zh

        except Exception as e:
            print(f"  [ERROR] Gemini API dịch lỗi: {e}")
            return None

        # Delay giữa các batch để tránh rate limit
        if batch_idx < total_batches - 1:
            time.sleep(3)

    return translation_map


def translate_with_deepseek(unique_texts: list, glossary: dict, batch_size: int = 50) -> dict:
    """
    Dịch bằng DeepSeek API sử dụng thư viện openai.
    """
    if not DEEPSEEK_API_KEY:
        print("  [ERROR] Chưa có DEEPSEEK_API_KEY trong .env!")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        print("  [ERROR] Chưa cài gói openai! Chạy: pip install openai")
        return None

    print("  Khởi tạo kết nối DeepSeek...")
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    except Exception as e:
        print(f"  [ERROR] Lỗi khởi tạo client DeepSeek: {e}")
        return None

    glossary_str = format_glossary_for_prompt(glossary)
    system_prompt = TRANSLATION_SYSTEM_PROMPT.format(
        genre=TRANSLATION_GENRE,
        glossary=glossary_str,
    )

    translation_map = {}
    total_batches = (len(unique_texts) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(unique_texts))
        batch = unique_texts[start:end]

        print(f"  [DeepSeek] Batch {batch_idx + 1}/{total_batches} ({len(batch)} câu)...")
        user_prompt = json.dumps(batch, ensure_ascii=False)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1
                )
                response_text = response.choices[0].message.content.strip()

                # Trích xuất JSON array
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    translated_list = json.loads(json_match.group())
                else:
                    print(f"  [WARN] Không parse được JSON từ DeepSeek response")
                    translated_list = batch

                if len(translated_list) == len(batch):
                    for zh, vi in zip(batch, translated_list):
                        translation_map[zh] = vi
                else:
                    print(f"  [WARN] Số lượng không khớp: gửi {len(batch)}, nhận {len(translated_list)}")
                    for i, zh in enumerate(batch):
                        if i < len(translated_list):
                            translation_map[zh] = translated_list[i]
                        else:
                            translation_map[zh] = zh
                break

            except Exception as e:
                print(f"  [ERROR] DeepSeek API lỗi (lần {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 * (attempt + 1))

        if batch_idx < total_batches - 1:
            time.sleep(1)

    return translation_map


def translate_with_freemodel(unique_texts: list, glossary: dict, batch_size: int = 50) -> dict:
    """
    Dịch bằng Freemodel API sử dụng thư viện openai.
    """
    if not FREEMODEL_API_KEY:
        print("  [ERROR] Chưa có FREEMODEL_API_KEY trong .env!")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        print("  [ERROR] Chưa cài gói openai! Chạy: pip install openai")
        return None

    print("  Khởi tạo kết nối Freemodel...")
    try:
        client = OpenAI(api_key=FREEMODEL_API_KEY, base_url=FREEMODEL_BASE_URL)
    except Exception as e:
        print(f"  [ERROR] Lỗi khởi tạo client Freemodel: {e}")
        return None

    glossary_str = format_glossary_for_prompt(glossary)
    system_prompt = TRANSLATION_SYSTEM_PROMPT.format(
        genre=TRANSLATION_GENRE,
        glossary=glossary_str,
    )

    translation_map = {}
    total_batches = (len(unique_texts) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(unique_texts))
        batch = unique_texts[start:end]

        print(f"  [Freemodel] Batch {batch_idx + 1}/{total_batches} ({len(batch)} câu)...")
        user_prompt = json.dumps(batch, ensure_ascii=False)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=FREEMODEL_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1
                )
                response_text = response.choices[0].message.content.strip()

                # Trích xuất JSON array
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    translated_list = json.loads(json_match.group())
                else:
                    print(f"  [WARN] Không parse được JSON từ Freemodel response")
                    translated_list = batch

                if len(translated_list) == len(batch):
                    for zh, vi in zip(batch, translated_list):
                        translation_map[zh] = vi
                else:
                    print(f"  [WARN] Số lượng không khớp: gửi {len(batch)}, nhận {len(translated_list)}")
                    for i, zh in enumerate(batch):
                        if i < len(translated_list):
                            translation_map[zh] = translated_list[i]
                        else:
                            translation_map[zh] = zh
                break

            except Exception as e:
                print(f"  [ERROR] Freemodel API lỗi (lần {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 * (attempt + 1))

        if batch_idx < total_batches - 1:
            time.sleep(1)

    return translation_map


def translate_with_nvidia(unique_texts: list, glossary: dict, batch_size: int = 50) -> dict:
    """
    Dịch bằng Nvidia API sử dụng thư viện openai.
    """
    if not NVIDIA_API_KEY:
        print("  [ERROR] Chưa có NVIDIA_API_KEY trong .env!")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        print("  [ERROR] Chưa cài gói openai! Chạy: pip install openai")
        return None

    print("  Khởi tạo kết nối Nvidia...")
    try:
        client = OpenAI(api_key=NVIDIA_API_KEY, base_url="https://integrate.api.nvidia.com/v1")
    except Exception as e:
        print(f"  [ERROR] Lỗi khởi tạo client Nvidia: {e}")
        return None

    glossary_str = format_glossary_for_prompt(glossary)
    system_prompt = TRANSLATION_SYSTEM_PROMPT.format(
        genre=TRANSLATION_GENRE,
        glossary=glossary_str,
    )

    translation_map = {}
    total_batches = (len(unique_texts) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(unique_texts))
        batch = unique_texts[start:end]

        print(f"  [Nvidia] Batch {batch_idx + 1}/{total_batches} ({len(batch)} câu)...")
        user_prompt = json.dumps(batch, ensure_ascii=False)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=NVIDIA_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1
                )
                response_text = response.choices[0].message.content.strip()

                # Trích xuất JSON array
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    translated_list = json.loads(json_match.group())
                else:
                    print(f"  [WARN] Không parse được JSON từ Nvidia response")
                    translated_list = batch

                if len(translated_list) == len(batch):
                    for zh, vi in zip(batch, translated_list):
                        translation_map[zh] = vi
                else:
                    print(f"  [WARN] Số lượng không khớp: gửi {len(batch)}, nhận {len(translated_list)}")
                    for i, zh in enumerate(batch):
                        if i < len(translated_list):
                            translation_map[zh] = translated_list[i]
                        else:
                            translation_map[zh] = zh
                break

            except Exception as e:
                print(f"  [ERROR] Nvidia API lỗi (lần {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 * (attempt + 1))

        if batch_idx < total_batches - 1:
            time.sleep(1)

    return translation_map


def translate_with_google(unique_texts: list, glossary: dict) -> dict:
    """
    Fallback: dịch bằng Google Translate + glossary pre/post-replace.
    Pre-replace tên riêng bằng placeholder trước khi gửi Google → tránh dịch sai tên.
    """
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source='zh-CN', target='vi')
    translation_map = {}

    # Sắp xếp glossary theo độ dài giảm dần (tránh replace từ ngắn trước)
    sorted_glossary = sorted(glossary.items(), key=lambda x: len(x[0]), reverse=True)

    print(f"  Đang dịch {len(unique_texts)} câu bằng Google Translate + glossary...")

    for i, text in enumerate(unique_texts):
        try:
            # Pre-replace: thay tên riêng bằng placeholder
            modified_text = text
            replacements = []  # (placeholder, vi_name)
            for idx, (zh_name, vi_name) in enumerate(sorted_glossary):
                if zh_name in modified_text:
                    # Dùng placeholder có delimiter để tránh trùng (XMGX01X vs XMGX10X)
                    placeholder = f"XMGX{idx:02d}X"
                    modified_text = modified_text.replace(zh_name, placeholder)
                    replacements.append((placeholder, vi_name))

            # Dịch text đã thay placeholder
            vi_text = translator.translate(modified_text)

            if vi_text:
                # Post-replace: khôi phục tên riêng từ placeholder (dài trước)
                replacements.sort(key=lambda x: len(x[0]), reverse=True)
                for placeholder, vi_name in replacements:
                    # Google Translate có thể viết hoa/thường placeholder
                    vi_text = re.sub(re.escape(placeholder), vi_name, vi_text, flags=re.IGNORECASE)

                translation_map[text] = vi_text
            else:
                translation_map[text] = text

        except Exception as e:
            print(f"  [ERROR] Google Translate lỗi [{text}]: {e}")
            translation_map[text] = text

        # Rate limit protection
        if i % 10 == 0 and i > 0:
            time.sleep(1)

        if (i + 1) % 20 == 0 or i == len(unique_texts) - 1:
            print(f"  Progress: {i + 1}/{len(unique_texts)}")

    return translation_map


# ──────────────────────────────────────────────
# CONTEXTUAL TRANSLATION (Phase 1-4): dịch theo ngữ cảnh kịch bản
#   Phase 1: dịch cả transcript theo thứ tự thời gian, có ngữ cảnh
#   Phase 2: gán người nói (speaker attribution) từ nhãn tên nhân vật
#   Phase 3: hồ sơ nhân vật (giới tính/vai vế/xưng hô) tự suy luận
#   Phase 4: pass QA rà soát nhất quán chủ ngữ/xưng hô
# ──────────────────────────────────────────────

CHAR_PROFILES_PATH = os.path.join(os.path.dirname(GLOSSARY_PATH), "character_profiles.json")

_PROFILE_SYS = (
    "Bạn là trợ lý phân tích nhân vật phim Trung Quốc. "
    "Tôi cung cấp các câu thoại theo thứ tự thời gian (kèm vài tên đã biết để tham khảo). "
    "Hãy TỰ NHẬN DIỆN các nhân vật chính và suy luận hồ sơ từng người từ ngữ cảnh hội thoại. "
    "Trả về DUY NHẤT một JSON object: key = tên (tiếng Việt nếu suy ra được, không thì để tiếng Trung), "
    'value = {"gender":"nam|nữ|?","role":"vai trò/quan hệ ngắn gọn","xung_ho":"cách tự xưng và gọi người khác"}. '
    "Chỉ liệt kê nhân vật thực sự (bỏ qua rác OCR). Nếu không chắc dùng '?'. KHÔNG giải thích, CHỈ JSON."
)

_QA_SYS = (
    "Bạn là biên tập viên phụ đề tiếng Việt cho phim Trung Quốc. "
    "Tôi đưa kịch bản đã dịch (mỗi dòng: số | người nói | tiếng Trung | tiếng Việt) kèm hồ sơ nhân vật. "
    "Hãy SOÁT các dòng bị SAI chủ ngữ / xưng hô / nhầm người nói so với ngữ cảnh hội thoại và hồ sơ nhân vật "
    "(tiếng Trung hay lược bỏ chủ ngữ nên dễ dịch nhầm 'tôi/anh/nó'). "
    "Trả về DUY NHẤT một JSON object CHỈ gồm các dòng CẦN SỬA: key = số dòng, value = câu tiếng Việt đã sửa. "
    "Nếu không dòng nào sai, trả về {}. KHÔNG giải thích."
)


def _gemini_json(rotator, types, system_instruction, user_prompt, temperature=0.1):
    """Gọi Gemini yêu cầu JSON, parse object/array. Trả về python obj hoặc None."""
    try:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json",
        )
    except Exception:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction, temperature=temperature)
    resp = rotator.generate_content(contents=user_prompt, config=config)
    txt = (resp.text or "").strip()
    return _parse_json_loose(txt)


def _parse_json_loose(txt: str):
    """Parse JSON object/array, chịu được khi model bọc thêm text."""
    if not txt:
        return None
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r'(\{.*\}|\[.*\])', txt, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                return None
    return None


_deepseek_client = None
_freemodel_client = None
_active_backend = "gemini"


def _freemodel_json(system_instruction, user_prompt, temperature=0.1):
    """Gọi Freemodel (OpenAI SDK) yêu cầu JSON object. Trả về python obj hoặc None."""
    global _freemodel_client
    if not FREEMODEL_API_KEY:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    if _freemodel_client is None:
        _freemodel_client = OpenAI(api_key=FREEMODEL_API_KEY, base_url=FREEMODEL_BASE_URL)
    sys_msg = system_instruction + "\n(Trả về đúng định dạng JSON.)"
    for attempt in range(3):
        try:
            resp = _freemodel_client.chat.completions.create(
                model=FREEMODEL_MODEL,
                messages=[{"role": "system", "content": sys_msg},
                          {"role": "user", "content": user_prompt}],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            return _parse_json_loose((resp.choices[0].message.content or "").strip())
        except Exception as e:
            if attempt == 2:
                print(f"  [Freemodel][WARN] {str(e)[:90]}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


_deepseek_client = None


def _deepseek_json(system_instruction, user_prompt, temperature=0.1):
    """Gọi DeepSeek (OpenAI SDK) yêu cầu JSON object. Trả về python obj hoặc None."""
    global _deepseek_client
    if not DEEPSEEK_API_KEY:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    if _deepseek_client is None:
        _deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    # DeepSeek json_object mode yêu cầu chữ "json" xuất hiện trong prompt
    sys_msg = system_instruction + "\n(Trả về đúng định dạng JSON.)"
    for attempt in range(3):
        try:
            resp = _deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": sys_msg},
                          {"role": "user", "content": user_prompt}],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            return _parse_json_loose((resp.choices[0].message.content or "").strip())
        except Exception as e:
            if attempt == 2:
                print(f"  [DeepSeek][WARN] {str(e)[:90]}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


_nvidia_client = None


def _nvidia_json(system_instruction, user_prompt, temperature=0.1):
    """Gọi Nvidia (OpenAI SDK) yêu cầu JSON object. Trả về python obj hoặc None."""
    global _nvidia_client
    if not NVIDIA_API_KEY:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    if _nvidia_client is None:
        _nvidia_client = OpenAI(api_key=NVIDIA_API_KEY, base_url="https://integrate.api.nvidia.com/v1")
    sys_msg = system_instruction + "\n(Trả về đúng định dạng JSON.)"
    for attempt in range(3):
        try:
            resp = _nvidia_client.chat.completions.create(
                model=NVIDIA_MODEL,
                messages=[{"role": "system", "content": sys_msg},
                          {"role": "user", "content": user_prompt}],
                temperature=temperature,
            )
            return _parse_json_loose((resp.choices[0].message.content or "").strip())
        except Exception as e:
            if attempt == 2:
                print(f"  [Nvidia][WARN] {str(e)[:90]}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


_gemini_dead = False  # circuit-breaker: tắt Gemini cho cả phiên sau lần lỗi đầu (tránh phí retry quota)


def _llm_json(system_instruction, user_prompt, temperature=0.1):
    """LLM-as-JSON provider-agnostic: hỗ trợ Nvidia, Freemodel, Gemini, DeepSeek."""
    global _gemini_dead, _active_backend
    
    if _active_backend == "nvidia":
        try:
            res = _nvidia_json(system_instruction, user_prompt, temperature)
            if res is not None:
                return res
        except Exception as e:
            print(f"  [LLM] Nvidia lỗi ({str(e)[:60]})")
        if DEEPSEEK_API_KEY:
            return _deepseek_json(system_instruction, user_prompt, temperature)
        elif GEMINI_API_KEYS and not _gemini_dead:
            try:
                from google.genai import types
                rot = get_gemini_rotator()
                return _gemini_json(rot, types, system_instruction, user_prompt, temperature)
            except Exception:
                pass
        return None

    if _active_backend == "freemodel":
        try:
            res = _freemodel_json(system_instruction, user_prompt, temperature)
            if res is not None:
                return res
        except Exception as e:
            print(f"  [LLM] Freemodel lỗi ({str(e)[:60]})")
        if DEEPSEEK_API_KEY:
            return _deepseek_json(system_instruction, user_prompt, temperature)
        elif GEMINI_API_KEYS and not _gemini_dead:
            try:
                from google.genai import types
                rot = get_gemini_rotator()
                return _gemini_json(rot, types, system_instruction, user_prompt, temperature)
            except Exception:
                pass
        return None

    if GEMINI_API_KEYS and not _gemini_dead:
        try:
            from google.genai import types
            rot = get_gemini_rotator()
            res = _gemini_json(rot, types, system_instruction, user_prompt, temperature)
            if res is not None:
                return res
        except Exception as e:
            print(f"  [LLM] Gemini lỗi ({str(e)[:60]})")
        # Gemini đã exhaust retry nội bộ mà vẫn fail → tắt hẳn cho phiên này
        _gemini_dead = True
        print("  [LLM] Tắt Gemini cho phiên này → chuyển hẳn sang DeepSeek.")
    return _deepseek_json(system_instruction, user_prompt, temperature)


def _is_name_label(cleaned: str, glossary: dict) -> bool:
    """NHÃN TÊN nhân vật — chỉ tin glossary (chính xác cao, tránh nhận nhầm verb/rác OCR)."""
    c = (cleaned or "").strip()
    if not c:
        return False
    return c in glossary or c in glossary.values()


def build_ordered_transcript(text_segments: list, normalized_mapping: dict, glossary: dict) -> list:
    """
    Phase 1+2: dựng transcript theo thứ tự thời gian, DEDUP theo lần xuất hiện đầu
    (mỗi câu unique 1 dòng) để giữ ngữ cảnh mà không phí API cho text lặp lại.
    Mỗi item: {idx, raw, cleaned, start, end, is_label, onscreen, speaker}.
    """
    raw_items = []
    for seg in text_segments:
        raw = seg.get("text", "")
        cleaned = normalized_mapping.get(raw, raw)
        if not cleaned:  # bị lọc thành rác
            continue
        raw_items.append({
            "raw": raw,
            "cleaned": cleaned,
            "start": int(seg.get("start_frame", 0)),
            "end": int(seg.get("end_frame", seg.get("start_frame", 0))),
            "is_label": _is_name_label(cleaned, glossary),
        })
    raw_items.sort(key=lambda it: (it["start"], it["end"]))

    # Cửa sổ thời gian của các nhãn tên (dùng cho speaker attribution, lấy từ TOÀN BỘ segment)
    labels = [it for it in raw_items if it["is_label"]]

    # Dedup theo cleaned, giữ lần xuất hiện ĐẦU (thứ tự thời gian)
    seen, transcript = set(), []
    for it in raw_items:
        if it["cleaned"] in seen:
            continue
        seen.add(it["cleaned"])
        transcript.append(it)

    for i, it in enumerate(transcript):
        it["idx"] = i + 1
        if it["is_label"]:
            it["onscreen"], it["speaker"] = [], None
            continue
        onscreen = []
        for lb in labels:
            if lb["start"] <= it["end"] and it["start"] <= lb["end"]:  # giao thời gian
                name = glossary.get(lb["cleaned"], lb["cleaned"])
                if name not in onscreen:
                    onscreen.append(name)
        it["onscreen"] = onscreen
        it["speaker"] = onscreen[0] if len(onscreen) == 1 else None  # chỉ chắc khi đúng 1
    return transcript


def infer_character_profiles(transcript: list, glossary: dict) -> dict:
    """Phase 3: suy luận hồ sơ nhân vật từ THOẠI (không phụ thuộc nhãn). Lưu file tái dùng."""
    sample = [f"[{it['idx']}] {it['cleaned']}" for it in transcript[:90]]
    known = sorted(set(list(glossary.values())))
    user = ("TÊN ĐÃ BIẾT (tham khảo, có thể không đủ):\n" + json.dumps(known, ensure_ascii=False)
            + "\n\nCÂU THOẠI THEO THỨ TỰ (hãy tự nhận diện nhân vật chính từ đây):\n"
            + "\n".join(sample))
    try:
        profiles = _llm_json(_PROFILE_SYS, user, temperature=0.0)
    except Exception as e:
        print(f"  [Phase 3][WARN] Suy luận hồ sơ nhân vật lỗi: {e}")
        return {}
    if not isinstance(profiles, dict):
        return {}

    # Trộn với hồ sơ thủ công nếu đã có (ưu tiên file cũ do người dùng chỉnh)
    existing = {}
    if os.path.exists(CHAR_PROFILES_PATH):
        try:
            existing = json.load(open(CHAR_PROFILES_PATH, encoding="utf-8"))
        except Exception:
            existing = {}
    merged = {**profiles, **existing}
    try:
        json.dump(merged, open(CHAR_PROFILES_PATH, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"  [Phase 3] Hồ sơ nhân vật: {len(merged)} người -> {os.path.basename(CHAR_PROFILES_PATH)}")
    except Exception as e:
        print(f"  [Phase 3][WARN] Không lưu được hồ sơ: {e}")
    return merged


def _format_profiles(profiles: dict) -> str:
    if not profiles:
        return "(Không có)"
    lines = []
    for name, info in profiles.items():
        if isinstance(info, dict):
            g = info.get("gender", "?"); r = info.get("role", "?"); x = info.get("xung_ho", "")
            lines.append(f"  - {name}: giới tính={g}, vai={r}" + (f", xưng hô: {x}" if x else ""))
        else:
            lines.append(f"  - {name}: {info}")
    return "\n".join(lines)


def _line_tag(it: dict) -> str:
    if it["is_label"]:
        return "(Nhãn tên) "
    if it.get("speaker"):
        return f"(Người nói: {it['speaker']}) "
    if it.get("onscreen"):
        return f"(Trên màn hình: {', '.join(it['onscreen'])}) "
    return ""


def translate_script(transcript: list, profiles: dict, glossary: dict,
                     window: int = 120, overlap: int = 15) -> dict:
    """
    Phase 1: dịch cả kịch bản theo NGỮ CẢNH. Chia cửa sổ có gối đầu để giữ mạch
    hội thoại cho video dài. Trả về {idx: vi}.
    """
    glossary_str = format_glossary_for_prompt(glossary)
    base_sys = TRANSLATION_SYSTEM_PROMPT.format(genre=TRANSLATION_GENRE, glossary=glossary_str)
    sys_prompt = (
        base_sys
        + "\n\nHỒ SƠ NHÂN VẬT (dùng để xưng hô đúng & nhất quán):\n" + _format_profiles(profiles)
        + "\n\nQUAN TRỌNG:\n"
        "- Mỗi dòng được đánh số [n] và có thể kèm gợi ý '(Người nói: X)' hoặc '(Trên màn hình: ...)'.\n"
        "- Tiếng Trung hay LƯỢC BỎ chủ ngữ. Hãy dựa vào MẠCH hội thoại liên tiếp + gợi ý người nói "
        "để xác định ĐÚNG chủ ngữ và xưng hô tiếng Việt.\n"
        "- Giữ xưng hô NHẤT QUÁN xuyên suốt theo hồ sơ nhân vật.\n"
        "- Trả về DUY NHẤT một JSON object: key = số dòng (trong []), value = câu tiếng Việt. KHÔNG giải thích."
    )

    vi_by_idx = {}
    n = len(transcript)
    step = max(1, window - overlap)
    for start in range(0, n, step):
        chunk = transcript[start:start + window]
        if not chunk:
            break
        user = "\n".join(f"[{it['idx']}] {_line_tag(it)}{it['cleaned']}" for it in chunk)
        try:
            data = _llm_json(sys_prompt, user, temperature=0.1)
        except Exception as e:
            print(f"  [Phase 1][WARN] Dịch cửa sổ {start} lỗi: {e}")
            data = None
        if isinstance(data, dict):
            for it in chunk:
                vi = data.get(str(it["idx"]))
                if vi:
                    vi_by_idx[it["idx"]] = vi.strip()
        if start + window >= n:
            break
    return vi_by_idx


def qa_consistency_pass(transcript: list, vi_by_idx: dict, profiles: dict,
                        window: int = 150) -> dict:
    """Phase 4: rà soát & sửa các dòng sai chủ ngữ/xưng hô. Cập nhật vi_by_idx tại chỗ."""
    prof_str = _format_profiles(profiles)
    n = len(transcript)
    fixed_total = 0
    for start in range(0, n, window):
        chunk = transcript[start:start + window]
        rows = []
        for it in chunk:
            vi = vi_by_idx.get(it["idx"], "")
            if not vi:
                continue
            sp = it.get("speaker") or (", ".join(it.get("onscreen", [])) or "?")
            rows.append(f"[{it['idx']}] | {sp} | {it['cleaned']} | {vi}")
        if not rows:
            continue
        user = "HỒ SƠ NHÂN VẬT:\n" + prof_str + "\n\nKỊCH BẢN:\n" + "\n".join(rows)
        try:
            fixes = _llm_json(_QA_SYS, user, temperature=0.0)
        except Exception as e:
            print(f"  [Phase 4][WARN] QA cửa sổ {start} lỗi: {e}")
            fixes = None
        if isinstance(fixes, dict):
            for k, v in fixes.items():
                try:
                    idx = int(k)
                except (ValueError, TypeError):
                    continue
                if v and isinstance(v, str):
                    vi_by_idx[idx] = v.strip()
                    fixed_total += 1
    if fixed_total:
        print(f"  [Phase 4] QA đã sửa {fixed_total} dòng cho nhất quán chủ ngữ/xưng hô")
    return vi_by_idx


def translate_contextual(text_segments: list, normalized_mapping: dict,
                         texts_to_translate: list, glossary: dict) -> dict:
    """
    Orchestrator Phase 1-4. Trả về {cleaned_zh: vi} phủ tối đa texts_to_translate.
    Ném exception nếu hỏng nặng → caller fallback sang dịch rời rạc.
    """
    if not (GEMINI_API_KEYS or DEEPSEEK_API_KEY or FREEMODEL_API_KEY):
        raise RuntimeError("Không có LLM provider (Gemini/DeepSeek/Freemodel) cho dịch ngữ cảnh")

    print("  [Phase 1] Dựng transcript theo thứ tự thời gian...")
    transcript = build_ordered_transcript(text_segments, normalized_mapping, glossary)
    if not transcript:
        raise RuntimeError("Transcript rỗng (không có segment hợp lệ)")
    n_label = sum(1 for it in transcript if it["is_label"])
    print(f"    {len(transcript)} dòng ({n_label} nhãn tên, {len(transcript)-n_label} thoại)")

    print("  [Phase 3] Suy luận hồ sơ nhân vật...")
    profiles = infer_character_profiles(transcript, glossary)

    print("  [Phase 1+2] Dịch kịch bản theo ngữ cảnh + người nói...")
    vi_by_idx = translate_script(transcript, profiles, glossary)
    if not vi_by_idx:
        raise RuntimeError("Phase 1 không trả về bản dịch nào")

    print("  [Phase 4] Rà soát nhất quán...")
    qa_consistency_pass(transcript, vi_by_idx, profiles)

    # Dựng cleaned -> vi
    cleaned_to_vi = {}
    for it in transcript:
        vi = vi_by_idx.get(it["idx"])
        if vi:
            cleaned_to_vi[it["cleaned"]] = vi

    # Phủ nốt các câu nằm trong unique_texts nhưng không có trong transcript (segment)
    leftover = [t for t in texts_to_translate if t not in cleaned_to_vi]
    if leftover:
        print(f"  [Bổ sung] {len(leftover)} câu ngoài transcript -> dịch rời rạc...")
        if _active_backend == "freemodel":
            iso_backend = "freemodel"
        else:
            iso_backend = "deepseek" if _gemini_dead else "gemini"
        iso = _translate_isolated(leftover, glossary, iso_backend)
        cleaned_to_vi.update(iso or {})

    return cleaned_to_vi


def _translate_isolated(texts_to_translate: list, glossary: dict, backend: str) -> dict:
    """Dịch RỜI RẠC (cách cũ): cache + batch. Trả về {cleaned_zh: vi}. Dùng làm fallback."""
    translation_cache = load_translation_cache()
    cache_hits = {zh: translation_cache[zh] for zh in texts_to_translate if zh in translation_cache}
    texts_new = [zh for zh in texts_to_translate if zh not in translation_cache]
    if cache_hits:
        print(f"  [CACHE] {len(cache_hits)} câu lấy từ cache, {len(texts_new)} câu cần gọi API")

    cleaned_translation_map = {}
    if not texts_new:
        print("  Tất cả câu đã có trong cache → bỏ qua gọi API dịch.")
    else:
        print(f"  Dịch rời rạc (backend: {backend}) cho {len(texts_new)} câu...")
        api_map = None
        if backend == "nvidia":
            api_map = translate_with_nvidia(texts_new, glossary)
            if not api_map:
                print("\n  [FALLBACK] Nvidia không khả dụng, chuyển sang DeepSeek...")
                api_map = translate_with_deepseek(texts_new, glossary)
            if not api_map:
                print("\n  [FALLBACK] DeepSeek không khả dụng, chuyển sang Gemini...")
                api_map = translate_with_gemini(texts_new, glossary)
            if not api_map:
                api_map = translate_with_google(texts_new, glossary)
        elif backend == "freemodel":
            api_map = translate_with_freemodel(texts_new, glossary)
            if not api_map:
                print("\n  [FALLBACK] Freemodel không khả dụng, chuyển sang DeepSeek...")
                api_map = translate_with_deepseek(texts_new, glossary)
            if not api_map:
                print("\n  [FALLBACK] DeepSeek không khả dụng, chuyển sang Gemini...")
                api_map = translate_with_gemini(texts_new, glossary)
            if not api_map:
                api_map = translate_with_google(texts_new, glossary)
        elif backend == "gemini":
            api_map = translate_with_gemini(texts_new, glossary)
            if not api_map:
                print("\n  [FALLBACK] Gemini không khả dụng, chuyển sang DeepSeek...")
                api_map = translate_with_deepseek(texts_new, glossary)
            if not api_map:
                print("\n  [FALLBACK] DeepSeek không khả dụng, chuyển sang Google Translate...")
                api_map = translate_with_google(texts_new, glossary)
            else:
                untranslated = sum(1 for zh, vi in api_map.items() if zh == vi)
                if untranslated == len(api_map):
                    print(f"\n  [FALLBACK] Gemini trả nguyên bản, bổ sung DeepSeek...")
                    deepseek_map = translate_with_deepseek(texts_new, glossary)
                    if deepseek_map:
                        api_map.update(deepseek_map)
                    still = [zh for zh, vi in api_map.items() if zh == vi]
                    if still:
                        google_map = translate_with_google(still, glossary)
                        api_map.update(google_map)
        elif backend == "deepseek":
            api_map = translate_with_deepseek(texts_new, glossary)
            if not api_map:
                api_map = translate_with_gemini(texts_new, glossary)
            if not api_map:
                api_map = translate_with_google(texts_new, glossary)
        else:
            api_map = translate_with_google(texts_new, glossary)

        cleaned_translation_map = api_map or {}
        new_cached = 0
        for zh, vi in cleaned_translation_map.items():
            if vi and vi != zh:
                translation_cache[zh] = vi
                new_cached += 1
        if new_cached:
            save_translation_cache(translation_cache)
            print(f"  [CACHE] Đã lưu thêm {new_cached} bản dịch mới vào cache")

    return {**cache_hits, **cleaned_translation_map}


def run_translation(backend: str = "deepseek"):
    """Chạy dịch thuật, xuất translation_map.json."""
    print("=" * 55)
    print("  Monggu Translate - Pass 2: Translation")
    print("=" * 55)

    # Load data
    print("\n[1/3] Đọc dữ liệu...")
    ocr_data = load_ocr_data()
    if not ocr_data:
        return

    glossary = load_glossary()
    unique_texts = ocr_data.get("unique_texts", [])

    if not unique_texts:
        print("  [WARN] Không có text nào để dịch!")
        return

    print(f"  Tổng cộng {len(unique_texts)} câu ban đầu")

    global _gemini_dead, _active_backend
    _active_backend = backend
    if backend in ("deepseek", "nvidia") or not GEMINI_API_KEYS:
        _gemini_dead = True

    # Normalize Pass — chọn provider theo backend
    print("\n[1.5/3] Normalize Texts...")
    t_start_norm = time.time()
    
    if backend == "nvidia":
        normalized_mapping = normalize_texts_with_nvidia(unique_texts)
        if not normalized_mapping:
            print("\n  [FALLBACK] Làm sạch bằng Nvidia thất bại, chuyển sang DeepSeek...")
            normalized_mapping = normalize_texts_with_deepseek(unique_texts)
        if not normalized_mapping:
            normalized_mapping = normalize_texts_with_gemini(unique_texts)
    elif backend == "freemodel":
        normalized_mapping = normalize_texts_with_freemodel(unique_texts)
        if not normalized_mapping:
            print("\n  [FALLBACK] Làm sạch bằng Freemodel thất bại, chuyển sang DeepSeek...")
            normalized_mapping = normalize_texts_with_deepseek(unique_texts)
        if not normalized_mapping:
            normalized_mapping = normalize_texts_with_gemini(unique_texts)
    elif backend == "gemini" and GEMINI_API_KEYS and not _gemini_dead:
        normalized_mapping = normalize_texts_with_gemini(unique_texts)
        if not normalized_mapping:
            print("\n  [FALLBACK] Làm sạch bằng Gemini thất bại, chuyển sang DeepSeek...")
            normalized_mapping = normalize_texts_with_deepseek(unique_texts)
    else:
        normalized_mapping = normalize_texts_with_deepseek(unique_texts)
        if not normalized_mapping:
            normalized_mapping = normalize_texts_with_gemini(unique_texts)
    
    # Lọc ra những text đã clean và có ý nghĩa
    texts_to_translate_dict = {}
    for raw, cleaned in normalized_mapping.items():
        if cleaned: # Bỏ qua những chuỗi rỗng do lọc rác
            texts_to_translate_dict[cleaned] = raw
            
    texts_to_translate = list(texts_to_translate_dict.keys())
    print(f"  => Sau khi clean, còn lại {len(texts_to_translate)} câu cần dịch (thời gian: {time.time() - t_start_norm:.1f}s)")

    if not texts_to_translate:
        print("  [WARN] Tất cả text đều bị lọc thành rác. Không có gì để dịch!")
        return

    # ── Dịch: ưu tiên CONTEXTUAL (Phase 1-4) cho backend LLM (gemini/deepseek) ──
    text_segments = ocr_data.get("text_segments", [])
    t0 = time.time()
    cleaned_translation_map = None
    has_llm = bool(GEMINI_API_KEYS or DEEPSEEK_API_KEY or FREEMODEL_API_KEY or NVIDIA_API_KEY)

    if backend in ("gemini", "deepseek", "freemodel", "nvidia") and text_segments and has_llm:
        print(f"\n[2/3] Dịch theo NGỮ CẢNH kịch bản (Phase 1-4) — {len(text_segments)} segment...")
        try:
            cleaned_translation_map = translate_contextual(
                text_segments, normalized_mapping, texts_to_translate, glossary)
        except Exception as e:
            print(f"  [FALLBACK] Dịch ngữ cảnh lỗi ({e}); quay về dịch rời rạc...")
            cleaned_translation_map = None
    elif backend in ("gemini", "deepseek", "freemodel", "nvidia") and not text_segments:
        print("\n[2/3] [WARN] Không có text_segments → không dịch ngữ cảnh được, dùng dịch rời rạc.")

    if not cleaned_translation_map:
        print(f"\n[2/3] Dịch rời rạc (backend: {backend})...")
        cleaned_translation_map = _translate_isolated(texts_to_translate, glossary, backend)

    # Lưu kết quả ngữ cảnh vào cache để lần sau / tool khác tái dùng
    if cleaned_translation_map:
        _cache = load_translation_cache()
        _n = 0
        for zh, vi in cleaned_translation_map.items():
            if vi and vi != zh and _cache.get(zh) != vi:
                _cache[zh] = vi
                _n += 1
        if _n:
            save_translation_cache(_cache)
            print(f"  [CACHE] Cập nhật {_n} bản dịch vào cache")

    # Reconstruct final translation_map matching RAW texts
    translation_map = {}
    for raw, cleaned in normalized_mapping.items():
        if cleaned and cleaned_translation_map and cleaned in cleaned_translation_map:
            translation_map[raw] = cleaned_translation_map[cleaned]
        else:
            # Gán text rỗng để Pass 3 inpaint mà không vẽ chữ lên (giấu text rác)
            translation_map[raw] = ""

    elapsed = time.time() - t0

    # Xuất JSON
    print(f"\n[3/3] Xuất translation_map.json...")
    with open(TRANSLATION_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(translation_map, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 55}")
    print(f"  Dịch thuật hoàn tất!")
    print(f"  Thời gian  : {elapsed:.1f}s")
    print(f"  Số câu dịch: {len(translation_map)}")
    print(f"  Output     : {TRANSLATION_MAP_PATH}")
    print(f"{'=' * 55}")

    # Preview 5 câu đầu
    print("\n  Preview bản dịch:")
    for i, (zh, vi) in enumerate(translation_map.items()):
        if i >= 5:
            print(f"  ... và {len(translation_map) - 5} câu nữa")
            break
        print(f"  [{zh}]")
        print(f"  → {vi}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monggu Translate - Translation Pass")
    parser.add_argument("--backend", "-b", type=str, default="gemini",
                        choices=["gemini", "deepseek", "freemodel", "nvidia", "google"],
                        help="Backend dịch thuật (gemini, deepseek, freemodel, nvidia hoặc google)")
    args = parser.parse_args()

    run_translation(backend=args.backend)

