"""
Test story_engine (reel gốc) — không gọi API, không dựng video.
Chạy:  python -m pytest tests/   hoặc   python tests/test_story.py
"""

import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "story_engine"))
sys.path.insert(0, ROOT)

import llm  # noqa: E402
import make_story_reel as msr  # noqa: E402

# An toàn: không bao giờ gọi mạng thật
llm.GEMINI_API_KEYS = []
llm.DEEPSEEK_API_KEY = ""
llm.RETRY_DELAY = 0

CFG = {"characters": {"mc": {"name": "Mèo Chuối", "emotions": {"cry": "c.mp4"}}, "rival": {"name": "Hắc Miêu"}},
       "emotions": {"shock": "a.mp4", "angry": "b.mp4"},
       "backgrounds": {"mountain": "m.png", "hall": "h.png"}}
GOOD = {"title": "T", "scenes": [{"char": "mc", "emotion": "idle", "bg": "mountain", "say": "Ta là Mèo Chuối."}]}
BAD = {"title": "T", "scenes": [{"char": "boss", "emotion": "idle", "bg": "mountain", "say": "Ta là trùm!"}]}


class _FakeLLM:
    """Provider giả + cache tạm: trả lần lượt các kịch bản trong replies, đếm số lời gọi."""

    def __init__(self, replies):
        self.replies, self.calls = iter(replies), 0

    def __enter__(self):
        self.orig = (llm._default_providers, llm.LLM_CACHE_PATH, llm._db)
        self.dir = tempfile.mkdtemp()
        llm._db, llm.LLM_CACHE_PATH = None, os.path.join(self.dir, "llm.sqlite")
        llm._default_providers = lambda: [("fake", self._ask)]
        return self

    def _ask(self, system, user, temperature):
        self.calls += 1
        return json.dumps(next(self.replies), ensure_ascii=False)

    def __exit__(self, *exc):
        if llm._db is not None:
            llm._db.close()
        llm._default_providers, llm.LLM_CACHE_PATH, llm._db = self.orig
        shutil.rmtree(self.dir, ignore_errors=True)


def test_script_problems_accepts_valid_script():
    scr = {"title": "T", "scenes": [{"char": "mc", "emotion": "idle", "bg": "mountain", "say": "Ta là Mèo Chuối."},
                                    {"char": "rival", "emotion": "angry", "bg": "hall", "say": "Hừ!"}]}
    assert msr.script_problems(scr, CFG) == []


def test_script_problems_rejects_ids_outside_library():
    scr = {"scenes": [{"char": "boss", "emotion": "idle", "bg": "mountain", "say": "A"},      # nhân vật lạ
                      {"char": "mc", "emotion": "dance", "bg": "mountain", "say": "B"},       # cảm xúc lạ
                      {"char": "mc", "emotion": "idle", "bg": "space", "say": "C"},           # bối cảnh lạ
                      {"char": "mc", "emotion": "idle", "bg": "hall", "say": "  "}]}          # thiếu lời
    probs = msr.script_problems(scr, CFG)
    assert len(probs) == 4, probs
    assert msr.script_problems({"scenes": []}, CFG) and msr.script_problems("rác", CFG)


def test_script_problems_reports_bad_types_instead_of_crashing():
    scr = {"resolution": "1080x1920",
           "scenes": [{"char": ["mc", "rival"], "bg": "hall", "say": "A"},            # LLM gộp 2 nhân vật
                      {"char": "mc", "emotion": ["shock"], "bg": {"id": "hall"}, "say": "B"},
                      {"char": "mc", "bg": "hall", "say": "C", "dur": "3 giây"}]}
    probs = msr.script_problems(scr, CFG)
    assert len(probs) == 5, probs  # resolution, nhân vật, cảm xúc, bối cảnh, dur


def test_script_problems_accepts_per_character_emotion_and_null_emotion():
    scr = {"scenes": [{"char": "mc", "emotion": "cry", "bg": "hall", "say": "Hu hu", "dur": 2.5},
                      {"char": "mc", "emotion": None, "bg": "hall", "say": "Thôi"}]}
    assert msr.script_problems(scr, CFG) == []
    # cảm xúc riêng của mc không dùng được cho rival
    assert msr.script_problems({"scenes": [{"char": "rival", "emotion": "cry", "bg": "hall", "say": "A"}]}, CFG)


def test_script_from_idea_never_returns_or_caches_script_outside_library():
    with _FakeLLM([BAD, GOOD]) as fake:
        assert msr.script_from_idea("ý tưởng", CFG) == GOOD
        assert fake.calls == 2, "kịch bản dùng nhân vật lạ phải bị từ chối và hỏi lại"
        assert msr.script_from_idea("ý tưởng", CFG) == GOOD
        assert fake.calls == 2, "lần 2 phải lấy kịch bản hợp lệ từ cache"


def _run_main(*argv):
    orig = sys.argv
    sys.argv = ["make_story_reel.py", *argv]
    try:
        msr.main()
    finally:
        sys.argv = orig


def test_idea_refuses_to_overwrite_existing_script():
    d = tempfile.mkdtemp()
    try:
        chars = os.path.join(d, "characters.json")
        json.dump(CFG, open(chars, "w", encoding="utf-8"))
        kb = os.path.join(d, "kb.json")
        open(kb, "w", encoding="utf-8").write("kịch bản đã sửa tay")
        with _FakeLLM([GOOD]) as fake:
            for argv in (["--idea", "x", "--script", kb], ["--idea", "  "]):
                try:
                    _run_main("--chars", chars, *argv)
                    raise AssertionError(f"phải dừng: {argv}")
                except SystemExit:
                    pass
            assert fake.calls == 0, "dừng trước khi tốn lời gọi LLM"
        assert open(kb, encoding="utf-8").read() == "kịch bản đã sửa tay"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_hand_written_script_with_bom_or_syntax_error_gives_clear_message():
    d = tempfile.mkdtemp()
    try:
        chars = os.path.join(d, "characters.json")
        json.dump(CFG, open(chars, "w", encoding="utf-8"))
        kb = os.path.join(d, "kb.json")
        open(kb, "w", encoding="utf-8").write('{"scenes": [{"char": "mc", "bg": "hall", "say": "A",}]}')
        try:
            _run_main("--chars", chars, "--script", kb)
            raise AssertionError("JSON lỗi phải dừng")
        except SystemExit as e:
            assert "dòng 1" in str(e), e
        # BOM (Notepad / PowerShell 5 lưu "UTF-8") vẫn đọc được: lỗi phía sau phải là lỗi kịch bản, không phải JSON
        open(kb, "w", encoding="utf-8-sig").write(json.dumps(BAD))
        try:
            _run_main("--chars", chars, "--script", kb)
            raise AssertionError("kịch bản sai phải dừng")
        except SystemExit as e:
            assert "boss" in str(e), e
    finally:
        shutil.rmtree(d, ignore_errors=True)


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
