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


_ORIG_DIRS = (jobs.JOBS_DIR, jobs.SERIES_DIR)


def _setup():
    d = tempfile.mkdtemp()
    jobs.JOBS_DIR = os.path.join(d, "jobs")
    jobs.SERIES_DIR = os.path.join(d, "series")
    return d


def _cleanup(d):
    jobs.JOBS_DIR, jobs.SERIES_DIR = _ORIG_DIRS
    shutil.rmtree(d, ignore_errors=True)


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
        _cleanup(d)


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
        _cleanup(d)


def test_series_shares_profiles_between_episodes():
    d = _setup()
    try:
        j1 = jobs.open_job(_video(d, "a", "daosiep1.mp4", b"ep1"), series="daosi", create=True)
        j2 = jobs.open_job(_video(d, "a", "daosiep2.mp4", b"ep2"), series="daosi", create=True)
        assert j1.story == j2.story == os.path.join(jobs.SERIES_DIR, "daosi", "story.json")
        assert jobs.open_job(os.path.join(d, "a", "daosiep1.mp4")).series == "daosi", "Series được nhớ trong job.json"
    finally:
        _cleanup(d)


def test_open_job_errors_are_clean_and_side_effect_free():
    d = _setup()
    try:
        v = _video(d, "a", "tap1.mp4", b"video-1")
        try:
            jobs.open_job(v)                                    # chưa có job, không create
        except jobs.JobError:
            pass
        assert not os.path.exists(jobs.JOBS_DIR), "create=False không được tạo thư mục nào"
        for bad in ("..\\x", "a:b", "a?b", "  "):
            try:
                jobs.open_job(v, series=bad, create=True)
                raise AssertionError(f"Tên series {bad!r} phải bị từ chối")
            except jobs.JobError:
                pass
        j = jobs.open_job(v, create=True)
        with open(os.path.join(j.dir, "job.json"), "w", encoding="utf-8") as f:
            f.write("{hỏng")
        try:
            jobs.open_job(v)
            raise AssertionError("job.json hỏng phải báo JobError, không phải traceback")
        except jobs.JobError:
            pass
    finally:
        _cleanup(d)


def test_open_job_has_watermark_paths_and_working_video():
    d = _setup()
    try:
        v = _video(d, "a", "tap1.mp4", b"video-1")
        j = jobs.open_job(v, create=True)
        assert j.wm_track == os.path.join(j.dir, "wm_track.json")
        assert j.wm_mask == os.path.join(j.dir, "wm_mask.png")
        assert j.wm_preview == os.path.join(j.dir, "wm_preview.png")
        # OCR không convert fps → đọc thẳng video gốc; có convert → bản trong job
        assert jobs.working_video(j, {"video": "tap1.mp4", "original_video": "tap1.mp4"}, v) == v
        assert jobs.working_video(j, {"video": "tap1_25fps.mp4", "original_video": "tap1.mp4"}, v) == \
            os.path.join(j.dir, "tap1_25fps.mp4")
        assert jobs.working_video(j, {}, v) == v
    finally:
        _cleanup(d)


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
