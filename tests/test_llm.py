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
