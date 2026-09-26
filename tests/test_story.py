"""
Test story_engine (reel gốc) — không gọi API, không dựng video.
Chạy:  python -m pytest tests/   hoặc   python tests/test_story.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "story_engine"))
sys.path.insert(0, ROOT)

import make_story_reel as msr  # noqa: E402

CFG = {"characters": {"mc": {"name": "Mèo Chuối"}, "rival": {"name": "Hắc Miêu"}},
       "emotions": {"shock": "a.mp4", "angry": "b.mp4"},
       "backgrounds": {"mountain": "m.png", "hall": "h.png"}}


def test_script_problems_accepts_valid_script():
    scr = {"title": "T", "scenes": [{"char": "mc", "emotion": "idle", "bg": "mountain", "say": "Ta là Mèo Chuối."},
                                    {"char": "rival", "emotion": "angry", "bg": "hall", "say": "Hừ!"}]}
    assert msr.script_problems(scr, CFG) == []


def test_script_problems_rejects_ids_outside_library():
    scr = {"scenes": [{"char": "boss", "emotion": "idle", "bg": "mountain", "say": "A"},      # nhân vật lạ
                      {"char": "mc", "emotion": "cry", "bg": "mountain", "say": "B"},         # cảm xúc lạ
                      {"char": "mc", "emotion": "idle", "bg": "space", "say": "C"},           # bối cảnh lạ
                      {"char": "mc", "emotion": "idle", "bg": "hall", "say": "  "}]}          # thiếu lời
    probs = msr.script_problems(scr, CFG)
    assert len(probs) == 4, probs
    assert msr.script_problems({"scenes": []}, CFG) and msr.script_problems("rác", CFG)


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
