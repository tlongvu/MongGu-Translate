"""
Monggu Translate - Pipeline Orchestrator
Chạy toàn bộ quy trình bằng 1 lệnh: Pass 1 (OCR) → Pass 2 (Dịch) → Pass 3 (Render).

Usage:
    python pipeline.py --input video.mp4
    python pipeline.py --input video.mp4 --workers 4
    python pipeline.py --input video.mp4 --only ocr          # chỉ chạy 1 pass
    python pipeline.py --input video.mp4 --skip-existing      # bỏ qua pass đã có output
    python pipeline.py --input video.mp4 --test              # test mode (giới hạn frames)

Mỗi pass chạy như một tiến trình con; nếu một pass lỗi (returncode != 0),
pipeline dừng ngay (fail-fast) để tránh render bằng dữ liệu hỏng.
"""

import sys

# Đảm bảo console in được tiếng Việt / ký tự Unicode trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import argparse
import os
import subprocess
import time

from config import OCR_DATA_PATH, TRANSLATION_MAP_PATH, VIDEO_PATH

_HERE = os.path.dirname(os.path.abspath(__file__))


def run_step(title: str, script: str, extra_args: list) -> float:
    """Chạy 1 pass bằng tiến trình con. Trả về thời gian chạy (s). Thoát nếu lỗi."""
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
        print(f"\n  ✗ '{title}' THẤT BẠI (exit code {result.returncode}). Dừng pipeline.")
        sys.exit(result.returncode)
    print(f"  ✓ '{title}' xong sau {elapsed:.1f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Monggu Translate - Pipeline (OCR → Dịch → Render)")
    parser.add_argument("--input", "-i", type=str, default=VIDEO_PATH, help="Đường dẫn video gốc")
    parser.add_argument("--workers", "-w", type=int, default=None, help="Số workers render")
    parser.add_argument("--test", action="store_true", help="Test mode (giới hạn frames)")
    parser.add_argument("--no-audio", action="store_true", help="Render không ghép audio")
    parser.add_argument("--only", type=str, default="all",
                        choices=["all", "ocr", "translate", "render"],
                        help="Chỉ chạy 1 pass cụ thể")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Bỏ qua pass đã có file output (ocr_data.json / translation_map.json)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Monggu Translate - PIPELINE")
    print("=" * 60)
    print(f"  Input   : {args.input}")
    print(f"  Only    : {args.only}")

    run_ocr = args.only in ("all", "ocr")
    run_tr = args.only in ("all", "translate")
    run_rd = args.only in ("all", "render")

    if args.skip_existing:
        if run_ocr and os.path.exists(OCR_DATA_PATH):
            print(f"  [SKIP] Đã có {os.path.basename(OCR_DATA_PATH)} → bỏ qua OCR")
            run_ocr = False
        if run_tr and os.path.exists(TRANSLATION_MAP_PATH):
            print(f"  [SKIP] Đã có {os.path.basename(TRANSLATION_MAP_PATH)} → bỏ qua dịch")
            run_tr = False

    wall0 = time.time()

    if run_ocr:
        ocr_args = ["--input", args.input]
        if args.test:
            ocr_args.append("--test")
        run_step("Pass 1/3 — OCR Scan", "ocr_pass.py", ocr_args)

    if run_tr:
        run_step("Pass 2/3 — Translation", "translate_pass.py", [])

    if run_rd:
        rd_args = ["--input", args.input]
        if args.workers is not None:
            rd_args += ["--workers", str(args.workers)]
        if args.test:
            rd_args.append("--test")
        if args.no_audio:
            rd_args.append("--no-audio")
        run_step("Pass 3/3 — Render Video", "script.py", rd_args)

    print("\n" + "=" * 60)
    print(f"  PIPELINE HOÀN TẤT trong {time.time() - wall0:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
