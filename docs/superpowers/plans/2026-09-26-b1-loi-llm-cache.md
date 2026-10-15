# B1 — Lõi LLM + cache — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mọi lời gọi LLM của pass dịch đi qua một hàm `llm.ask_json()`, có cache SQLite theo prompt đầy đủ. Xóa code backend thừa (Freemodel, NVIDIA, Google) và đường dịch rời rạc không ngữ cảnh.

**Architecture:**
- `llm.py` mới: tra cache → Gemini (xoay key) → DeepSeek → parse JSON → lưu cache.
- `translate_pass.py` giữ nguyên các phase 1–4 hiện có (B3 sẽ thay), chỉ đổi đường gọi LLM.
- Dòng bị model bỏ sót được hỏi lại; biến thể OCR mượn bản dịch của dòng giống nhất.

**Tech Stack:**
- `sqlite3` và `difflib` (thư viện chuẩn);
- `google-genai`, `openai` (đã cài).

**Spec:** `docs/superpowers/specs/2026-09-26-dich-theo-cot-truyen-design.md`, mục 6–7.

## Global Constraints

- Nhánh `nang-cap`. Backup file sắp sửa vào `_backup_nang-cap-B1_<timestamp>/`.
- **Không gọi API thật.**
  - Không chạy `translate_pass.py` hay `pipeline.py`.
  - Test chỉ dùng provider giả, hoặc `ask_json` giả.
  - Test `llm` phải xóa key thật khỏi module (`llm.GEMINI_API_KEYS = []`, `llm.DEEPSEEK_API_KEY = ""`), để lỡ quên truyền provider thì cũng không gọi mạng.
- Không in bất kỳ phần nào của API key ra log.
- Giữ phong cách code: comment tiếng Việt, `print`, test là hàm `test_*` thuần với `assert` và có runner chạy độc lập.
- Windows + PowerShell, `$env:PYTHONIOENCODING="utf-8"`.
- `SCRATCH` = `C:\Users\20010\AppData\Local\Temp\claude\D--monggutranslate-test-gpu-copy\b143f228-8f05-4cf8-8116-0fdb84454354\scratchpad`.

## Chuẩn bị

- [ ] Backup:

```powershell
$b = "_backup_nang-cap-B1_" + (Get-Date -Format "yyyyMMdd_HHmmss"); New-Item -ItemType Directory $b | Out-Null
Copy-Item translate_pass.py, config.py, pipeline.py $b; Copy-Item tests\test_core.py "$b\test_core.py"; $b
```

- [ ] `python tests/test_core.py` → `14/14 tests passed`

---

### Task 1: `llm.py` — `ask_json` + cache

**Files:**
- Create: `llm.py`, `tests/test_llm.py`
- Modify: `config.py` (thêm `GEMINI_MODEL`, `DEEPSEEK_MODEL`, `LLM_CACHE_PATH`), `.gitignore` (thêm `cache/`)

**Interfaces — Produces:**
- `llm.ask_json(system: str, user: str, temperature: float = 0.1, providers=None) -> dict | list` — raise `llm.LLMError` nếu mọi provider hỏng.
- `llm.parse_json_loose(txt: str) -> dict | list | None`
- `llm.LLMError(RuntimeError)`
- Hằng số module: `llm.CACHE_VERSION`, `llm.RETRY_DELAY`, `llm.LLM_CACHE_PATH`.

- [ ] **Step 1: Config.** Trong `config.py`:
  - ngay sau dòng `DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")`, thêm:

```python
GEMINI_MODEL = "gemini-2.5-flash"
DEEPSEEK_MODEL = "deepseek-chat"
```

  - ngay sau dòng `TRANSLATION_CACHE_PATH = ...`, thêm:

```python
# Cache lời gọi LLM (llm.ask_json): khóa = hash prompt đầy đủ → an toàn giữa các video
LLM_CACHE_PATH = os.path.join(_SCRIPT_DIR, "cache", "llm.sqlite")
```

`.gitignore`: thêm dòng `cache/` sau dòng `.claude/*.lock`.

- [ ] **Step 2: Viết test (fail)** — tạo `tests/test_llm.py`:

```python
"""
Test lõi LLM (llm.py) với provider GIẢ — không gọi API thật.
Chạy:  python -m pytest tests/   hoặc   python tests/test_llm.py
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm  # noqa: E402

# An toàn: lỡ quên truyền providers cũng không gọi mạng thật
llm.GEMINI_API_KEYS = []
llm.DEEPSEEK_API_KEY = ""
llm.RETRY_DELAY = 0


def _fresh_cache():
    """Trỏ cache sang thư mục tạm mới; trả về thư mục để dọn."""
    if llm._db is not None:
        llm._db.close()
        llm._db = None
    d = tempfile.mkdtemp()
    llm.LLM_CACHE_PATH = os.path.join(d, "cache", "llm.sqlite")
    return d


def _cleanup(d):
    if llm._db is not None:
        llm._db.close()
        llm._db = None
    shutil.rmtree(d, ignore_errors=True)


def _broken(system, user, temperature):
    raise ConnectionError("mất mạng")


def test_ask_json_caches_identical_prompt():
    d = _fresh_cache()
    try:
        calls = []
        fake = [("fake", lambda s, u, t: calls.append(u) or '{"a": 1}')]
        assert llm.ask_json("sys", "hỏi 1", providers=fake) == {"a": 1}
        assert llm.ask_json("sys", "hỏi 1", providers=fake) == {"a": 1}
        assert len(calls) == 1, "Hỏi y hệt lần 2 phải lấy từ cache"
        llm.ask_json("sys", "hỏi 2", providers=fake)
        llm.ask_json("sys khác", "hỏi 1", providers=fake)
        assert len(calls) == 3, "Đổi system hoặc user phải gọi lại"
    finally:
        _cleanup(d)


def test_ask_json_does_not_cache_broken_json():
    d = _fresh_cache()
    try:
        replies = iter(["không phải json", '{"ok": true}'])
        calls = []

        def fake(system, user, temperature):
            calls.append(1)
            return next(replies)

        assert llm.ask_json("sys", "q", providers=[("fake", fake)]) == {"ok": True}
        assert len(calls) == 2, "JSON hỏng → thử lại lần 2"
        assert llm.ask_json("sys", "q", providers=[("fake", fake)]) == {"ok": True}
        assert len(calls) == 2, "Chỉ câu trả lời hợp lệ được cache"
    finally:
        _cleanup(d)


def test_ask_json_falls_back_to_next_provider():
    d = _fresh_cache()
    try:
        got = llm.ask_json("sys", "q", providers=[("gemini", _broken), ("deepseek", lambda s, u, t: "[1, 2]")])
        assert got == [1, 2]
    finally:
        _cleanup(d)


def test_ask_json_raises_when_all_providers_fail():
    d = _fresh_cache()
    try:
        try:
            llm.ask_json("sys", "q", providers=[("a", _broken), ("b", lambda s, u, t: "rác")])
            raise AssertionError("Phải ném LLMError")
        except llm.LLMError as e:
            assert "a:" in str(e) and "b:" in str(e), "Thông báo lỗi phải nêu từng provider"
        try:
            llm.ask_json("sys", "q2", providers=[])
            raise AssertionError("Không có provider phải ném LLMError")
        except llm.LLMError:
            pass
    finally:
        _cleanup(d)


def test_parse_json_loose():
    assert llm.parse_json_loose('```json\n{"x": "Tôi"}\n```') == {"x": "Tôi"}
    assert llm.parse_json_loose("Kết quả: [1, 2]") == [1, 2]
    assert llm.parse_json_loose("không có json") is None
    assert llm.parse_json_loose("") is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
```

Run: `python tests/test_llm.py` → lỗi `ModuleNotFoundError: No module named 'llm'`

- [ ] **Step 3: Viết `llm.py`**

```python
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
```

- [ ] **Step 4: Pass**

Run: `python tests/test_llm.py` → `5/5 tests passed`; `python -m pytest tests -q` → `19 passed`

- [ ] **Step 5: Commit**

```powershell
git add llm.py tests/test_llm.py config.py .gitignore
git commit -m "feat(llm): lõi ask_json + cache SQLite theo prompt đầy đủ" -m "Gemini xoay key (cạn quota cả vòng thì đợi 30s thử thêm, rồi chuyển hẳn DeepSeek); JSON hỏng thử lại, không tắt Gemini; chỉ cache JSON hợp lệ; không in key ra log." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Pass dịch dùng `ask_json`; bỏ dịch rời rạc

**Files:**
- Modify: `translate_pass.py` (import; `infer_character_profiles`; `translate_script`; `qa_consistency_pass`; `translate_contextual`; `run_translation`; `__main__`; hàm mới `normalize_texts`, `_take_translations`)
- Modify: `pipeline.py` (bỏ `--backend`)
- Test: `tests/test_core.py`

**Interfaces:**
- Consumes: `llm.ask_json`, `llm.LLMError` (Task 1)
- Produces:
  - `translate_pass.normalize_texts(unique_texts: list, batch_size: int = 150) -> dict`
  - `translate_pass.run_translation() -> bool` (bỏ tham số `backend`)
  - `translate_pass.translate_contextual(...)` raise `LLMError` khi model vẫn bỏ sót dòng sau khi hỏi lại

- [ ] **Step 1: Viết test (fail)** trong `tests/test_core.py`:
  - xóa `test_translation_cache_roundtrip`;
  - trong `test_run_translation_fails_without_input`, đổi 2 chỗ `tp.run_translation("google")` thành `tp.run_translation()`;
  - thêm:

```python
def _fake_llm(first_window_reply):
    """ask_json giả cho Phase 1-4: hồ sơ {}, QA {}, hỏi lại dòng thiếu → dịch dòng 3."""
    calls = []

    def fake(system, user, temperature=0.1):
        calls.append(user)
        if 'CÂU THOẠI THEO THỨ TỰ' in user or 'KỊCH BẢN:' in user:  # Phase 3 hồ sơ / Phase 4 QA
            return {}
        if 'CHỈ dịch các dòng' in user:
            return {"3": "Đi thôi"}
        return first_window_reply
    return fake, calls


def _run_contextual(fake, segs_texts, norm, texts):
    segs = [{'text': t, 'start_frame': i * 10, 'end_frame': i * 10 + 5} for i, t in enumerate(segs_texts)]
    orig_ask, orig_prof = tp.ask_json, tp.CHAR_PROFILES_PATH
    tmp = tempfile.mkdtemp()
    try:
        tp.ask_json = fake
        tp.CHAR_PROFILES_PATH = os.path.join(tmp, "profiles.json")
        return tp.translate_contextual(segs, norm, texts, {})
    finally:
        tp.ask_json, tp.CHAR_PROFILES_PATH = orig_ask, orig_prof
        shutil.rmtree(tmp, ignore_errors=True)


def test_translate_contextual_retries_missing_lines_and_borrows_variants():
    fake, calls = _fake_llm({"1": "Xin chào", "2": "Ta muốn tu tiên!"})  # lần đầu bỏ sót dòng 3
    norm = {'你好': '你好', '我要去修仙！': '我要去修仙！', '走吧': '走吧', '我要去修仙': '我要去修仙'}
    out = _run_contextual(fake, ['你好', '我要去修仙！', '走吧'], norm, ['你好', '我要去修仙！', '走吧', '我要去修仙'])
    assert out['走吧'] == 'Đi thôi', "Dòng bị bỏ sót phải được hỏi lại"
    assert sum('CHỈ dịch các dòng: 3' in u for u in calls) == 1
    assert out['我要去修仙'] == 'Ta muốn tu tiên!', "Biến thể OCR mượn bản dịch dòng giống nhất"


def test_translate_contextual_raises_when_lines_still_missing():
    fake, _ = _fake_llm({"1": "Xin chào"})  # hỏi lại cũng chỉ ra dòng 3, dòng 2 vẫn thiếu
    try:
        _run_contextual(fake, ['你好', '走吧', '好的'], {'你好': '你好', '走吧': '走吧', '好的': '好的'},
                        ['你好', '走吧', '好的'])
        raise AssertionError("Phải ném LLMError khi vẫn thiếu dòng")
    except tp.LLMError:
        pass
```

Run: `python -m pytest tests/test_core.py -q -k "contextual or run_translation"` → FAIL (`tp` chưa có `ask_json` / `LLMError`; `run_translation()` thiếu đối số)

- [ ] **Step 2: Import.** Trong `translate_pass.py`:
  - thêm `from difflib import get_close_matches` sau `import argparse`;
  - thêm `from llm import ask_json, LLMError` ngay trước dòng `from utils import write_json_atomic`;
  - thay docstring đầu file bằng:

```python
"""
Monggu Translate - Pass 2: Dịch theo ngữ cảnh kịch bản
Load ocr_data.json → chuẩn hóa chữ OCR → dịch theo kịch bản (Phase 1-4) → xuất translation_map.json
Mọi lời gọi LLM qua llm.ask_json (Gemini, dự phòng DeepSeek; có cache → chạy lại không tốn lại).

Usage:
    python translate_pass.py
"""
```

- [ ] **Step 3: `normalize_texts`.** Chèn ngay TRƯỚC khối comment `# ──...` / `# CONTEXTUAL TRANSLATION (Phase 1-4)`:

```python
def normalize_texts(unique_texts: list, batch_size: int = 150) -> dict:
    """Làm sạch chữ OCR bằng LLM: {raw: cleaned, hoặc "" nếu là rác}.
    Câu model bỏ sót hoặc trả giá trị không phải chuỗi → giữ nguyên chữ gốc."""
    mapping = {}
    total = (len(unique_texts) + batch_size - 1) // batch_size
    for b, start in enumerate(range(0, len(unique_texts), batch_size), 1):
        batch = unique_texts[start:start + batch_size]
        print(f"    Batch {b}/{total} ({len(batch)} câu)...")
        got = ask_json(NORMALIZATION_SYSTEM_PROMPT, json.dumps(batch, ensure_ascii=False), temperature=0.0)
        got = got if isinstance(got, dict) else {}
        for zh in batch:
            v = got.get(zh, zh)
            mapping[zh] = v if isinstance(v, str) else zh
    return mapping

```

- [ ] **Step 4: Phase 3.** Trong `infer_character_profiles`, thay:

```python
        profiles = _llm_json(_PROFILE_SYS, user, temperature=0.0)
    except Exception as e:
```

bằng:

```python
        profiles = ask_json(_PROFILE_SYS, user, temperature=0.0)
    except LLMError as e:
```

- [ ] **Step 5: Phase 1 — hỏi lại dòng thiếu.** Trong `translate_script`, thay:

```python
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
```

bằng:

```python
        user = "\n".join(f"[{it['idx']}] {_line_tag(it)}{it['cleaned']}" for it in chunk)
        _take_translations(ask_json(sys_prompt, user, temperature=0.1), chunk, vi_by_idx)
        missing = [it for it in chunk if it["idx"] not in vi_by_idx]
        if missing:
            # Hỏi lại riêng các dòng thiếu (prompt khác → khóa cache khác), vẫn kèm cả cửa sổ làm ngữ cảnh
            ids = ", ".join(str(it["idx"]) for it in missing)
            _take_translations(ask_json(sys_prompt, user + f"\n\nCHỈ dịch các dòng: {ids}", temperature=0.1),
                               missing, vi_by_idx)
        if start + window >= n:
            break
    missing = [it["idx"] for it in transcript if it["idx"] not in vi_by_idx]
    if missing:
        raise LLMError(f"Model bỏ sót {len(missing)} dòng kể cả khi hỏi lại: {missing[:20]}")
    return vi_by_idx


def _take_translations(data, items: list, vi_by_idx: dict) -> None:
    """Lấy bản dịch {"idx": vi} cho đúng các dòng của cửa sổ.
    Chuỗi rỗng = model cố ý để trống (rác) → vẫn tính là đã trả lời; thiếu key mới là bỏ sót."""
    if not isinstance(data, dict):
        return
    for it in items:
        vi = data.get(str(it["idx"]))
        if isinstance(vi, str):
            vi_by_idx[it["idx"]] = vi.strip()
```

- [ ] **Step 6: Phase 4 — chỉ nhận dòng của cửa sổ.** Trong `qa_consistency_pass`:
  - thêm dòng `        window_ids = {it["idx"] for it in chunk}` ngay sau `chunk = transcript[start:start + window]`;
  - thay:

```python
            fixes = _llm_json(_QA_SYS, user, temperature=0.0)
        except Exception as e:
```

bằng:

```python
            fixes = ask_json(_QA_SYS, user, temperature=0.0)
        except LLMError as e:
```

  - thay `if v and isinstance(v, str):` bằng `if idx in window_ids and v and isinstance(v, str):`.

- [ ] **Step 7: Orchestrator.** Trong `translate_contextual`:
  - docstring:

```python
    """
    Orchestrator Phase 1-4. Trả về {cleaned_zh: vi} phủ tối đa texts_to_translate.
    Ném LLMError nếu LLM hỏng / vẫn bỏ sót dòng → caller dừng (exit 1); phần đã xong nằm trong cache.
    """
```

  - xóa 2 dòng kiểm tra provider (`if not (GEMINI_API_KEYS or DEEPSEEK_API_KEY or FREEMODEL_API_KEY):` + `raise RuntimeError(...)`);
  - xóa 2 dòng `if not vi_by_idx:` + `raise RuntimeError("Phase 1 không trả về bản dịch nào")`;
  - thay toàn bộ khối từ `# Phủ nốt các câu nằm trong unique_texts...` tới `cleaned_to_vi.update(iso or {})` bằng:

```python
    # Câu sạch không có trong transcript = biến thể OCR của một dòng đã dịch → mượn bản dịch dòng
    # giống nhất (giữ đúng xưng hô theo ngữ cảnh), không dịch rời rạc.
    leftover = [t for t in texts_to_translate if t not in cleaned_to_vi]
    if leftover:
        known = list(cleaned_to_vi)
        borrowed = 0
        for t in leftover:
            match = get_close_matches(t, known, n=1, cutoff=0.7)
            if match:
                cleaned_to_vi[t] = cleaned_to_vi[match[0]]
                borrowed += 1
        print(f"  [Bổ sung] {len(leftover)} biến thể ngoài transcript: {borrowed} mượn bản dịch dòng "
              f"giống nhất, {len(leftover) - borrowed} để trống")
```

- [ ] **Step 8: `run_translation` + `__main__`.**
  - Chữ ký thành `def run_translation():`.
  - Thay toàn bộ phần từ dòng `print(f"  Tổng cộng {len(unique_texts)} câu ban đầu")` tới hết khối `# Reconstruct final translation_map matching RAW texts` (tức ngay trước `elapsed = time.time() - t0`) bằng:

```python
    text_segments = ocr_data.get("text_segments", [])
    if not text_segments:
        print("  [ERROR] ocr_data.json thiếu text_segments — hãy chạy lại OCR.")
        return False

    print(f"  Tổng cộng {len(unique_texts)} câu ban đầu")

    try:
        print("\n[1.5/3] Normalize Texts...")
        t_start_norm = time.time()
        normalized_mapping = normalize_texts(unique_texts)

        # Câu đã làm sạch, có nghĩa, bỏ trùng (giữ thứ tự)
        texts_to_translate = list(dict.fromkeys(c for c in normalized_mapping.values() if c))
        print(f"  => Sau khi clean, còn lại {len(texts_to_translate)} câu cần dịch "
              f"(thời gian: {time.time() - t_start_norm:.1f}s)")
        if not texts_to_translate:
            print("  [ERROR] Tất cả text đều bị lọc thành rác. Không có gì để dịch!")
            return False

        print(f"\n[2/3] Dịch theo NGỮ CẢNH kịch bản (Phase 1-4) — {len(text_segments)} segment...")
        t0 = time.time()
        cleaned_translation_map = translate_contextual(
            text_segments, normalized_mapping, texts_to_translate, glossary)
    except (LLMError, RuntimeError) as e:
        print(f"  [ERROR] Dịch thất bại: {e}")
        print("  Phần đã dịch xong nằm trong cache/llm.sqlite — chạy lại sẽ làm tiếp, không tốn lại.")
        return False

    # Dựng translation_map theo chữ OCR THÔ; rỗng = Pass 3 xóa chữ mà không vẽ (rác)
    translation_map = {raw: (cleaned_translation_map.get(cleaned, "") if cleaned else "")
                       for raw, cleaned in normalized_mapping.items()}

```

  - `__main__` thành:

```python
if __name__ == "__main__":
    argparse.ArgumentParser(description="Monggu Translate - Translation Pass").parse_args()
    sys.exit(0 if run_translation() else 1)
```

`pipeline.py`:
  - xóa dòng `python pipeline.py --input video.mp4 --backend deepseek --workers 4` trong docstring, thay bằng `python pipeline.py --input video.mp4 --workers 4`;
  - xóa 2 dòng `parser.add_argument("--backend", ...)` (dòng khai báo + `choices=...`);
  - xóa dòng `print(f"  Backend : {args.backend}")`;
  - đổi `run_step("Pass 2/3 — Translation", "translate_pass.py", ["--backend", args.backend])` thành `run_step("Pass 2/3 — Translation", "translate_pass.py", [])`.

- [ ] **Step 9: Pass**

Run: `python tests/test_core.py` → `15/15 tests passed`; `python tests/test_llm.py` → `5/5`
Run: `Select-String -Path translate_pass.py -Pattern '_llm_json|_translate_isolated|load_translation_cache|save_translation_cache' | Where-Object { $_.Line -notmatch '^\s*def ' }` → không còn chỗ GỌI nào (định nghĩa cũ còn, Task 3 xóa)

- [ ] **Step 10: Commit**

```powershell
git add translate_pass.py pipeline.py tests/test_core.py
git commit -m "refactor(dịch): mọi lời gọi LLM qua llm.ask_json; bỏ dịch rời rạc" -m "Phase 1 thiếu dòng -> hỏi lại riêng dòng thiếu, vẫn thiếu -> LLMError (exit 1, không để trống âm thầm). Biến thể OCR ngoài transcript mượn bản dịch dòng giống nhất thay vì dịch không ngữ cảnh. Phase 4 chỉ nhận id của cửa sổ. Bỏ --backend (Gemini trước, DeepSeek dự phòng)." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Xóa code chết (backend thừa, rotator cũ, cache cũ, config chết)

**Files:**
- Modify: `translate_pass.py` (xóa theo mốc), `config.py`, import trong `translate_pass.py`

- [ ] **Step 1: Xóa theo mốc.** Tạo `$SCRATCH\b1_cut.py`, chạy một lần từ thư mục project. Script xóa từ dưới lên, theo tên hàm, không theo số dòng:

```python
"""Xóa code chết trong translate_pass.py theo MỐC tên hàm (xóa từ dưới lên)."""
p = "translate_pass.py"
L = open(p, encoding="utf-8").read().split("\n")


def idx(prefix):
    hits = [i for i, s in enumerate(L) if s.startswith(prefix)]
    assert len(hits) == 1, f"mốc '{prefix}': {len(hits)} kết quả"
    return hits[0]


cuts = [  # [bắt đầu, kết thúc) — kết thúc là mốc được GIỮ lại
    ("def _translate_isolated", "def run_translation"),
    ("def _gemini_json", "def _is_name_label"),
    ("class GeminiClientRotator", "def normalize_texts("),  # có "(" để không khớp normalize_texts_with_*
    ("def load_translation_cache", "def load_glossary"),
]
for a, b in cuts:  # thứ tự trên là từ dưới lên
    i, j = idx(a), idx(b)
    assert i < j, (a, b)
    print(f"xóa {a} .. trước {b}: {j - i} dòng")
    del L[i:j]
open(p, "w", encoding="utf-8", newline="").write("\n".join(L))
```

Run: `python "$SCRATCH\b1_cut.py"` → 4 dòng `xóa ...`, tổng khoảng 950 dòng.

- [ ] **Step 2: Dọn import và config.**

Khối `from config import (...)` trong `translate_pass.py` thành:

```python
from config import (
    GLOSSARY_PATH,
    TRANSLATION_GENRE, TRANSLATION_SYSTEM_PROMPT,
    NORMALIZATION_SYSTEM_PROMPT,
    OCR_DATA_PATH, TRANSLATION_MAP_PATH,
)
```

`config.py`:
- xóa các dòng `GEMINI_API_KEY = GEMINI_API_KEYS[0] if ...`, `OPENAI_API_KEY = ...`, `FREEMODEL_API_KEY = ...`, `FREEMODEL_BASE_URL = ...`, `FREEMODEL_MODEL = ...`, `NVIDIA_API_KEY = ...`, `NVIDIA_MODEL = ...`;
- xóa dòng comment `# Cache bản dịch giữa các lần chạy: ...` cùng dòng `TRANSLATION_CACHE_PATH = ...`;
- đổi comment `# Prompt cho Deepseek` (trên `TRANSLATION_GENRE`) thành `# Prompt dịch (dùng cho mọi LLM qua llm.ask_json)`.

- [ ] **Step 3: Kiểm chứng**

- Chạy `python -c "import ast,sys; ast.parse(open('translate_pass.py',encoding='utf-8').read())"` → không lỗi.
- Chạy `Select-String -Path *.py,tests\*.py -Pattern 'FREEMODEL|NVIDIA_|OPENAI_API_KEY|TRANSLATION_CACHE_PATH|GEMINI_API_KEY\b|_gemini_dead|_active_backend|GeminiClientRotator|normalize_texts_with_|translate_with_'` → không còn kết quả (ngoài `wm_remove.py` không dính).
- Chạy `python -m pytest tests -q` → `20 passed`.
- Chạy `python -c "import translate_pass"` → import được.
- Kiểm tra độ dài: `(Get-Content translate_pass.py).Count` giảm khoảng 950 dòng so với bản backup.

- [ ] **Step 4: Commit**

```powershell
git add translate_pass.py config.py
git commit -m "chore(dịch): xóa backend Freemodel/NVIDIA/Google, rotator + cache cũ, config chết" -m "User chỉ dùng Gemini + DeepSeek (nay qua llm.py). translation_cache.json cũ (khóa theo câu, gây lẫn giữa các phim) thôi dùng; file vẫn nằm trên đĩa." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Kiểm chứng tổng, không tốn API

- [ ] **Step 1: Chạy trọn pass dịch trên dữ liệu THẬT với LLM giả.** Tạo `$SCRATCH\b1_dryrun.py`:

```python
"""Chạy run_translation() trên ocr_data.json THẬT (hihi, ~3.100 câu) với ask_json GIẢ — không gọi mạng.
Kiểm tra: luồng chạy trọn, mọi chữ thô có mặt trong map, biến thể được mượn bản dịch, tốc độ."""
import json, os, re, sys, time
PROJ = r"D:\monggutranslate\test-gpu copy"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "b1_dry")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, PROJ)
os.chdir(PROJ)
import llm
llm.GEMINI_API_KEYS, llm.DEEPSEEK_API_KEY = [], ""          # chặn mạng tuyệt đối
import translate_pass as tp

calls = {"norm": 0, "prof": 0, "tr": 0, "qa": 0}


def fake(system, user, temperature=0.1):
    if system == tp.NORMALIZATION_SYSTEM_PROMPT:
        calls["norm"] += 1
        return {t: t for t in json.loads(user)}
    if 'CÂU THOẠI THEO THỨ TỰ' in user:
        calls["prof"] += 1
        return {}
    if 'KỊCH BẢN:' in user:
        calls["qa"] += 1
        return {}
    calls["tr"] += 1
    return {m.group(1): f"VI{m.group(1)}" for m in re.finditer(r'^\[(\d+)\]', user, re.M)}


tp.ask_json = fake
tp.TRANSLATION_MAP_PATH = os.path.join(OUT, "translation_map.json")
tp.CHAR_PROFILES_PATH = os.path.join(OUT, "profiles.json")
t = time.time()
ok = tp.run_translation()
tm = json.load(open(tp.TRANSLATION_MAP_PATH, encoding="utf-8"))
raw = json.load(open("ocr_data.json", encoding="utf-8"))["unique_texts"]
print("OK", ok, "| thời gian", round(time.time() - t, 1), "s | lời gọi", calls)
print("chữ thô có trong map:", sum(r in tm for r in raw), "/", len(raw), "| có bản dịch:", sum(bool(v) for v in tm.values()))
```

Run: `python "$SCRATCH\b1_dryrun.py"`
Expected:
- `OK True`;
- mọi chữ thô đều có trong map;
- số lời gọi khoảng: norm 21, prof 1, tr ~24 (2.442 dòng transcript, cửa sổ 120 gối 15), qa ~17;
- dòng log `[Bổ sung] … mượn bản dịch dòng giống nhất` xuất hiện, với số mượn lớn hơn 0;
- thời gian dưới 60 giây (nếu `get_close_matches` quá chậm thì dừng và báo);
- không có file mới trong project.

- [ ] **Step 2: Bộ test đầy đủ**

Run: `python -m pytest tests -q` → `20 passed`; `python tests/test_core.py` → `15/15`; `python tests/test_llm.py` → `5/5`

- [ ] **Step 3: Review độc lập.** Dispatch agent review (read-only) cho range B1, theo mẫu `requesting-code-review`. Sửa các điểm Critical/Important trước khi coi B1 là xong.
