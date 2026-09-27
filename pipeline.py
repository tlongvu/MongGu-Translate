"""
Monggu Translate - Pipeline Orchestrator
Chạy toàn bộ quy trình bằng 1 lệnh: OCR → dò watermark chạy → Dịch → Render.

Usage:
    python pipeline.py --input video.mp4
    python pipeline.py --input video.mp4 --workers 4
    python pipeline.py --input video.mp4 --only ocr          # chỉ chạy 1 pass
    python pipeline.py --input video.mp4 --only wm           # chỉ dò lại watermark chạy
    python pipeline.py --input video.mp4 --skip-existing      # bỏ qua pass đã có output
    python pipeline.py --input video.mp4 --test              # test mode (giới hạn frames)
    python pipeline.py --input daosiep4.mp4 --series daosi   # tập của một bộ: dùng chung hồ sơ nhân vật

Dữ liệu và video ra của mỗi video nằm trong jobs/<tên-video>/ (vd jobs/daosiep4/daosiep4_vi.mp4).

Mỗi bước chạy như một tiến trình con; bước nào lỗi (returncode != 0) thì pipeline dừng ngay (fail-fast)
để tránh render bằng dữ liệu hỏng — trừ bước dò watermark chạy: lỗi thì chỉ cảnh báo, vẫn dịch và render.
"""

import sys

# Đảm bảo console in được tiếng Việt / ký tự Unicode trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import argparse
import json
import os
import subprocess
import time

from config import VIDEO_PATH
from jobs import open_job, JobError

_HERE = os.path.dirname(os.path.abspath(__file__))


def run_step(title: str, script: str, extra_args: list, required: bool = True) -> float:
    """Chạy 1 bước bằng tiến trình con. Trả về thời gian chạy (s).
    Bước bắt buộc lỗi → dừng pipeline; bước không bắt buộc (dò watermark) lỗi → cảnh báo rồi chạy tiếp."""
    cmd = [sys.executable, os.path.join(_HERE, script)] + extra_args
    print("\n" + "═" * 60)
    print(f"  ▶ {title}")
    print(f"    {' '.join(cmd)}")
    print("═" * 60)
    # Ép subprocess dùng UTF-8 stdout để in tiếng Việt qua pipe không lỗi cp1252
    child_env = dict(os.environ, PYTHONIOENCODING="utf-8")
    t0 = time.time()
    result = subprocess.run(cmd, env=child_env)
    elapsed = time.time() - t0
    if result.returncode != 0:
        if not required:
            print(f"\n  ⚠ '{title}' lỗi (exit code {result.returncode}) — bỏ qua bước này, chạy tiếp.")
            return elapsed
        print(f"\n  ✗ '{title}' THẤT BẠI (exit code {result.returncode}). Dừng pipeline.")
        sys.exit(result.returncode)
    print(f"  ✓ '{title}' xong sau {elapsed:.1f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Monggu Translate - Pipeline (OCR → watermark → Dịch → Render)")
    parser.add_argument("--input", "-i", type=str, default=VIDEO_PATH, help="Đường dẫn video gốc")
    parser.add_argument("--series", "-s", type=str, default=None, help="Tên bộ (series) nếu video là một tập")
    parser.add_argument("--workers", "-w", type=int, default=None, help="Số workers render")
    parser.add_argument("--test", action="store_true", help="Test mode (giới hạn frames)")
    parser.add_argument("--no-audio", action="store_true", help="Render không ghép audio")
    parser.add_argument("--only", type=str, default="all",
                        choices=["all", "ocr", "wm", "translate", "render"],
                        help="Chỉ chạy 1 pass cụ thể")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Bỏ qua pass đã xong trong thư mục job của video")
    args = parser.parse_args()

    print("=" * 60)
    print("  Monggu Translate - PIPELINE")
    print("=" * 60)
    print(f"  Input   : {args.input}")
    print(f"  Only    : {args.only}")

    run_ocr = args.only in ("all", "ocr")
    run_wm = args.only in ("all", "wm")
    run_tr = args.only in ("all", "translate")
    run_rd = args.only in ("all", "render")

    try:  # chỉ OCR mới được tạo job mới; --only translate/render cần job đã có
        job = open_job(args.input, args.series, create=run_ocr)
    except JobError as e:
        print(f"  [ERROR] {e}")
        sys.exit(1)
    print(f"  Job     : {job.dir}" + (f"  (series: {job.series})" if job.series else ""))

    if args.skip_existing:
        if run_ocr and os.path.exists(job.ocr_data):
            try:
                with open(job.ocr_data, encoding="utf-8") as f:
                    done = json.load(f)
            except ValueError:
                done = {"complete": False}
            # Dữ liệu của lần chạy --test chỉ phủ ~20s đầu → không coi là xong khi chạy thật
            if done.get("complete", True) and (args.test or not done.get("test_mode")):
                print("  [SKIP] OCR đã xong cho video này → bỏ qua OCR")
                run_ocr = False
        if run_wm and not run_ocr and os.path.exists(job.wm_track):   # OCR vừa chạy lại → phải dò lại
            print("  [SKIP] Đã dò watermark chạy cho video này → bỏ qua")
            run_wm = False
        if run_tr and not run_ocr and os.path.exists(job.lines):  # OCR vừa chạy lại → bản dịch cũ không còn khớp
            print("  [SKIP] Đã có bản dịch cho video này → bỏ qua dịch")
            run_tr = False

    wall0 = time.time()

    if run_ocr:
        ocr_args = ["--input", args.input]
        if args.series:
            ocr_args += ["--series", args.series]
        if args.test:
            ocr_args.append("--test")
        run_step("Bước 1/4 — OCR", "ocr_pass.py", ocr_args)

    if run_wm:
        # chạy riêng (--only wm) mà lỗi thì phải báo lỗi thật; trong cả pipeline thì chỉ cảnh báo
        run_step("Bước 2/4 — Dò watermark chạy", "wm_track.py", ["--input", args.input], required=(args.only == "wm"))

    if run_tr:
        run_step("Bước 3/4 — Dịch", "translate_pass.py", ["--input", args.input])

    if run_rd:
        rd_args = ["--input", args.input]
        if args.workers is not None:
            rd_args += ["--workers", str(args.workers)]
        if args.test:
            rd_args.append("--test")
        if args.no_audio:
            rd_args.append("--no-audio")
        run_step("Bước 4/4 — Render", "script.py", rd_args)

    print("\n" + "=" * 60)
    print(f"  PIPELINE HOÀN TẤT trong {time.time() - wall0:.1f}s")
    if run_rd:
        print(f"  Video ra: {job.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
