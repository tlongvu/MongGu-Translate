"""
Monggu Translate - Lõi LLM dùng chung: ask_json() + cache.

- Gemini (xoay vòng GEMINI_API_KEYS) trước, DeepSeek dự phòng.
- Cache SQLite theo hash(prompt đầy đủ): hỏi y hệt thì trả ngay, không gọi API.
  Ngữ cảnh nằm trong khóa nên không bao giờ trả nhầm bản dịch của phim khác.
- Chỉ cache câu trả lời JSON hợp lệ. Mọi provider đều hỏng → LLMError.
"""

import hashlib
import json
import os
import re
import sqlite3
import time

from config import GEMINI_API_KEYS, DEEPSEEK_API_KEY, GEMINI_MODEL, DEEPSEEK_MODEL, LLM_CACHE_PATH

CACHE_VERSION = 1   # tăng lên để cố ý bỏ toàn bộ cache cũ (vd đổi model, muốn dịch lại hết)
RETRY_DELAY = 2     # giây chờ trước khi thử lại sau lỗi mạng


class LLMError(RuntimeError):
    """Không provider nào trả được JSON hợp lệ."""


def parse_json_loose(txt: str):
    """Parse JSON object/array, chịu được khi model bọc thêm chữ hoặc ```json."""
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


# ── Cache ──

_db = None


def _cache():
    global _db
    if _db is None:
        os.makedirs(os.path.dirname(LLM_CACHE_PATH), exist_ok=True)
        _db = sqlite3.connect(LLM_CACHE_PATH)
        _db.execute("CREATE TABLE IF NOT EXISTS llm_cache "
                    "(key TEXT PRIMARY KEY, response TEXT, provider TEXT, created REAL)")
    return _db


def _cache_key(system: str, user: str, temperature: float) -> str:
    raw = json.dumps([CACHE_VERSION, temperature, system, user], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ── Providers: hàm(system, user, temperature) -> text ──

_gemini_clients = {}
_gemini_idx = 0             # key đang dùng tốt, giữ qua các lời gọi
_gemini_exhausted = False   # mọi key hết quota trong phiên này → bỏ qua Gemini


def _is_quota_error(e: Exception) -> bool:
    s = str(e).lower()
    return "429" in s or "resource_exhausted" in s or "resourceexhausted" in s or "quota" in s


def _ask_gemini(system: str, user: str, temperature: float) -> str:
    global _gemini_idx, _gemini_exhausted
    from google import genai
    from google.genai import types
    config = types.GenerateContentConfig(system_instruction=system, temperature=temperature,
                                         response_mime_type="application/json")
    for round_ in range(2):  # vòng 2 sau 30s: giới hạn theo phút thường đã hồi
        for _ in range(len(GEMINI_API_KEYS)):
            key = GEMINI_API_KEYS[_gemini_idx]
            client = _gemini_clients.get(key)
            if client is None:
                client = _gemini_clients[key] = genai.Client(api_key=key)
            try:
                resp = client.models.generate_content(model=GEMINI_MODEL, contents=user, config=config)
                return resp.text or ""
            except Exception as e:
                if not _is_quota_error(e):
                    raise
                print(f"  [Gemini] key #{_gemini_idx + 1} bị giới hạn quota → đổi key")
                _gemini_idx = (_gemini_idx + 1) % len(GEMINI_API_KEYS)
        if round_ == 0:
            print("  [Gemini] Mọi key đều bị giới hạn, đợi 30s rồi thử thêm một vòng...")
            time.sleep(30)
    _gemini_exhausted = True
    raise LLMError("mọi key Gemini đều hết quota — chuyển hẳn sang DeepSeek cho phiên này")


_deepseek_client = None


def _ask_deepseek(system: str, user: str, temperature: float) -> str:
    global _deepseek_client
    if _deepseek_client is None:
        from openai import OpenAI
        _deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    resp = _deepseek_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        # JSON mode của DeepSeek yêu cầu chữ "json" xuất hiện trong prompt
        messages=[{"role": "system", "content": system + "\n(Trả về đúng định dạng JSON.)"},
                  {"role": "user", "content": user}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""


def _default_providers() -> list:
    out = []
    if GEMINI_API_KEYS and not _gemini_exhausted:
        out.append(("gemini", _ask_gemini))
    if DEEPSEEK_API_KEY:
        out.append(("deepseek", _ask_deepseek))
    return out


def ask_json(system: str, user: str, temperature: float = 0.1, providers=None):
    """Hỏi LLM, trả về JSON đã parse (dict/list). Cache theo (system, user, temperature).
    providers: [(tên, hàm)] — mặc định Gemini rồi DeepSeek; test truyền provider giả."""
    key = _cache_key(system, user, temperature)
    row = _cache().execute("SELECT response FROM llm_cache WHERE key = ?", (key,)).fetchone()
    if row:
        return json.loads(row[0])

    errors = []
    for name, ask in (providers if providers is not None else _default_providers()):
        for _ in range(2):  # mỗi provider thử 2 lần (lỗi mạng / JSON hỏng)
            try:
                obj = parse_json_loose(ask(system, user, temperature))
            except LLMError as e:  # vd Gemini cạn quota: thử lại vô ích
                errors.append(f"{name}: {e}")
                break
            except Exception as e:
                errors.append(f"{name}: {str(e)[:150]}")
                time.sleep(RETRY_DELAY)
                continue
            if obj is None:
                errors.append(f"{name}: JSON hỏng")
                continue
            _cache().execute("INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?, ?)",
                             (key, json.dumps(obj, ensure_ascii=False), name, time.time()))
            _cache().commit()
            return obj
    raise LLMError("; ".join(errors) if errors
                   else "Không có LLM nào dùng được (thiếu GEMINI_API_KEY / DEEPSEEK_API_KEY trong .env)")
