"""
Monggu Translate - Lõi LLM dùng chung: ask_json() + cache.

- Gemini (xoay vòng GEMINI_API_KEYS) trước, DeepSeek dự phòng.
- Cache SQLite theo hash(prompt đầy đủ): hỏi y hệt thì trả ngay, không gọi API.
  Ngữ cảnh nằm trong khóa nên không bao giờ trả nhầm bản dịch của phim khác.
- Chỉ cache câu trả lời JSON hợp lệ VÀ đạt check (nếu có). Mọi provider đều hỏng → LLMError.
- Không in bất kỳ phần nào của API key: mọi thông báo lỗi đi qua _redact().
"""

import hashlib
import json
import os
import re
import sqlite3
import time

from config import GEMINI_API_KEYS, DEEPSEEK_API_KEY, GEMINI_MODEL, DEEPSEEK_MODEL, LLM_CACHE_PATH

CACHE_VERSION = 1           # tăng lên để cố ý bỏ toàn bộ cache cũ (vd đổi model, muốn dịch lại hết)
RETRY_DELAY = 2             # giây chờ trước khi thử lại sau lỗi mạng
GEMINI_TIMEOUT_MS = 300_000  # 1 lời gọi tối đa 5 phút (hồ sơ truyện gửi cả video)
DEEPSEEK_MAX_TOKENS = 8192   # mặc định 4K dễ cắt cụt JSON dài (chuẩn hóa 150 câu, hồ sơ truyện)


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


def _redact(text: str) -> str:
    """Xóa dấu vết API key khỏi thông báo lỗi (SDK hay chép nguyên body lỗi của server).
    Làm TRƯỚC khi cắt ngắn chuỗi — cắt giữa key thì .replace không bắt được."""
    for k in list(GEMINI_API_KEYS) + [DEEPSEEK_API_KEY]:
        if k:
            text = text.replace(k, "***")
    return re.sub(r"\*{2,}\w+|api_key:[^'\"\s]+|AIza[0-9A-Za-z_\-]{10,}|sk-[0-9A-Za-z]{10,}", "***", text)


# ── Cache (tốt nhất có thể: cache lỗi thì cảnh báo và chạy tiếp, không làm mất câu trả lời) ──

_db = None


def _cache():
    global _db
    if _db is None:
        os.makedirs(os.path.dirname(LLM_CACHE_PATH), exist_ok=True)
        _db = sqlite3.connect(LLM_CACHE_PATH, timeout=30)
        _db.execute("CREATE TABLE IF NOT EXISTS llm_cache "
                    "(key TEXT PRIMARY KEY, response TEXT, provider TEXT, created REAL)")
    return _db


def _cache_key(system: str, user: str, temperature: float) -> str:
    raw = json.dumps([CACHE_VERSION, temperature, system, user], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str):
    try:
        row = _cache().execute("SELECT response FROM llm_cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None
    except sqlite3.Error as e:
        print(f"  [LLM][WARN] Cache lỗi ({e}) — bỏ qua cache. Lặp lại thì xóa {LLM_CACHE_PATH}")
        return None


def _cache_put(key: str, obj, provider: str) -> None:
    try:
        _cache().execute("INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?, ?)",
                         (key, json.dumps(obj, ensure_ascii=False), provider, time.time()))
        _cache().commit()
    except sqlite3.Error as e:
        print(f"  [LLM][WARN] Không ghi được cache ({e})")


# ── Providers: hàm(system, user, temperature) -> text ──

_gemini_clients = {}
_gemini_idx = 0             # key đang dùng tốt, giữ qua các lời gọi
_gemini_exhausted = False   # mọi key hết quota / không dùng được trong phiên này → bỏ qua Gemini


def _error_code(e: Exception):
    return getattr(e, "code", None) or getattr(e, "status_code", None)


def _is_quota_error(e: Exception) -> bool:
    return _error_code(e) == 429 or "resource_exhausted" in str(e).lower()


def _is_dead_key_error(e: Exception) -> bool:
    """Key bị thu hồi / sai / tài khoản bị khóa — đổi key, KHÔNG phải lỗi của request."""
    code, s = _error_code(e), str(e).lower()
    return code in (401, 403) or (code == 400 and "api key" in s)


def _gemini_client(key: str):
    from google import genai
    from google.genai import types
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS))


def _ask_gemini(system: str, user: str, temperature: float) -> str:
    global _gemini_idx, _gemini_exhausted
    if not GEMINI_API_KEYS:
        raise LLMError("không có key Gemini")
    from google.genai import types
    config = types.GenerateContentConfig(system_instruction=system, temperature=temperature,
                                         response_mime_type="application/json")
    for round_ in range(2):  # vòng 2 sau 30s: giới hạn theo phút thường đã hồi
        for _ in range(len(GEMINI_API_KEYS)):
            key = GEMINI_API_KEYS[_gemini_idx]
            client = _gemini_clients.get(key)
            if client is None:
                client = _gemini_clients[key] = _gemini_client(key)
            try:
                resp = client.models.generate_content(model=GEMINI_MODEL, contents=user, config=config)
                return resp.text or ""
            except Exception as e:
                if _is_quota_error(e):
                    print(f"  [Gemini] key #{_gemini_idx + 1} bị giới hạn quota → đổi key")
                elif _is_dead_key_error(e):
                    print(f"  [Gemini] key #{_gemini_idx + 1} không dùng được ({_error_code(e)}) → đổi key")
                else:
                    raise  # lỗi mạng / request: để ask_json thử lại hoặc chuyển provider
                _gemini_idx = (_gemini_idx + 1) % len(GEMINI_API_KEYS)
        if round_ == 0:
            print("  [Gemini] Mọi key đều bị giới hạn, đợi 30s rồi thử thêm một vòng...")
            time.sleep(30)
    _gemini_exhausted = True
    raise LLMError("mọi key Gemini đều hết quota / không dùng được — chuyển hẳn sang DeepSeek cho phiên này")


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
        max_tokens=DEEPSEEK_MAX_TOKENS,
    )
    return resp.choices[0].message.content or ""


def _default_providers() -> list:
    out = []
    if GEMINI_API_KEYS and not _gemini_exhausted:
        out.append(("gemini", _ask_gemini))
    if DEEPSEEK_API_KEY:
        out.append(("deepseek", _ask_deepseek))
    return out


def _reject_reason(obj, check):
    """None = dùng được; ngược lại là lý do bỏ câu trả lời (không cache; hỏi lại / đổi provider)."""
    if obj is None:
        return "JSON hỏng"
    try:
        return None if check is None or check(obj) else "trả lời thiếu"
    except Exception as e:  # LLM trả sai kiểu làm check vấp → trả lời sai, không làm sập cả chuỗi provider
        return f"trả lời sai ({type(e).__name__}: {str(e)[:80]})"


def ask_json(system: str, user: str, temperature: float = 0.1, providers=None, check=None):
    """Hỏi LLM, trả về JSON đã parse (dict/list). Cache theo (system, user, temperature).
    check: hàm(obj) -> bool; False = trả lời thiếu/sai (vd bỏ sót dòng) → KHÔNG cache, thử lại,
    rồi sang provider sau — tránh cache mãi một câu trả lời thiếu.
    providers: [(tên, hàm)] — mặc định Gemini rồi DeepSeek; test truyền provider giả."""
    key = _cache_key(system, user, temperature)
    cached = _cache_get(key)
    if cached is not None and _reject_reason(cached, check) is None:  # luật check chặt hơn → hỏi lại
        return cached

    plist = providers if providers is not None else _default_providers()
    errors = []
    for n, (name, ask) in enumerate(plist):
        for attempt in range(2):  # mỗi provider thử 2 lần (lỗi mạng / JSON hỏng / trả lời thiếu)
            try:
                obj = parse_json_loose(ask(system, user, temperature))
            except LLMError as e:  # vd Gemini cạn quota: thử lại vô ích
                errors.append(f"{name}: {_redact(str(e))}")
                break
            except Exception as e:
                errors.append(f"{name}: {_redact(str(e))[:150]}")
                if attempt == 0:
                    time.sleep(RETRY_DELAY)
                continue
            why = _reject_reason(obj, check)
            if why:
                errors.append(f"{name}: {why}")
                continue
            _cache_put(key, obj, name)
            return obj
        if n + 1 < len(plist):
            print(f"  [LLM] {name} không trả lời được ({errors[-1]}) → dùng {plist[n + 1][0]}")
    raise LLMError("; ".join(errors) if errors
                   else "Không có LLM nào dùng được (thiếu GEMINI_API_KEY / DEEPSEEK_API_KEY trong .env)")
