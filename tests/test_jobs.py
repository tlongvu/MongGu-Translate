"""
Test thư mục job (jobs.py) — chỉ dùng file giả trong thư mục tạm.
Chạy:  python -m pytest tests/   hoặc   python tests/test_jobs.py
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jobs  # noqa: E402


def _setup():
    d = tempfile.mkdtemp()
    jobs.JOBS_DIR = os.path.join(d, "jobs")
    jobs.SERIES_DIR = os.path.join(d, "series")
    return d


def _video(d, sub, name, content):
    os.makedirs(os.path.join(d, sub), exist_ok=True)
    p = os.path.join(d, sub, name)
    with open(p, "wb") as f:
        f.write(content)
    return p


def test_open_job_create_and_reopen():
    d = _setup()
    try:
        v = _video(d, "a", "tap1.mp4", b"video-1")
        try:
            jobs.open_job(v)
            raise AssertionError("Chưa có job mà không create phải lỗi")
        except jobs.JobError:
            pass
        j = jobs.open_job(v, create=True)
        assert j.dir == os.path.join(jobs.JOBS_DIR, "tap1")
        assert j.ocr_data == os.path.join(j.dir, "ocr_data.json")
        assert j.output == os.path.join(j.dir, "tap1_vi.mp4")
        assert j.story == os.path.join(j.dir, "story.json"), "Video lẻ: hồ sơ truyện nằm trong job"
        assert j.lines == os.path.join(j.dir, "lines.json")
        assert jobs.open_job(v).dir == j.dir, "Mở lại cùng video → cùng job"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_open_job_rejects_other_video_with_same_name():
    d = _setup()
    try:
        jobs.open_job(_video(d, "a", "tap1.mp4", b"video-1"), create=True)
        try:
            jobs.open_job(_video(d, "b", "tap1.mp4", b"video khac hoan toan"), create=True)
            raise AssertionError("Video khác trùng tên phải bị chặn")
        except jobs.JobError:
            pass
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_series_shares_profiles_between_episodes():
    d = _setup()
    try:
        j1 = jobs.open_job(_video(d, "a", "daosiep1.mp4", b"ep1"), series="daosi", create=True)
        j2 = jobs.open_job(_video(d, "a", "daosiep2.mp4", b"ep2"), series="daosi", create=True)
        assert j1.story == j2.story == os.path.join(jobs.SERIES_DIR, "daosi", "story.json")
        assert jobs.open_job(os.path.join(d, "a", "daosiep1.mp4")).series == "daosi", "Series được nhớ trong job.json"
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
