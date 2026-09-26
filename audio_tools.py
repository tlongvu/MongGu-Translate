"""
Monggu Translate - Audio Tools
Tách nhạc nền bản quyền ra khỏi giọng nhân vật / meme sound bằng AI (Demucs).

Workflow:
  1. Tách audio từ video gốc
  2. Dùng Demucs AI tách vocals (giọng nói/meme) ra khỏi nhạc nền
  3. Ghép lại vocals vào video đã render

Chức năng:
  separate  - Tách vocals (giọng/meme) ra khỏi nhạc nền bằng AI
  extract   - Tách audio gốc ra file riêng
  mux       - Ghép audio file vào video
  analyze   - Xem thông tin audio của video
  auto      - Tự động: tách vocals + ghép vào video đã render (1 lệnh duy nhất)

Usage:
    # === WORKFLOW CHÍNH (Loại bỏ nhạc bản quyền) ===
    # Cách 1: Tự động - 1 lệnh duy nhất
    python audio_tools.py auto input.mp4 output_final_45.mp4

    # Cách 2: Từng bước
    python audio_tools.py separate input.mp4           # Tách vocals → input_vocals.wav
    python audio_tools.py mux output_final_45.mp4 input_vocals.wav  # Ghép vào video

    # === Tiện ích khác ===
    python audio_tools.py extract input.mp4            # Tách audio gốc
    python audio_tools.py analyze input.mp4            # Xem info audio

Yêu cầu:
    pip install demucs
    (ffmpeg đã cài sẵn)
"""

import subprocess
import os
import sys
import json
import argparse
import shutil
import time


def run_ffmpeg(cmd, description=""):
    """Chạy ffmpeg command và xử lý lỗi."""
    print(f"  → {description}")
    print(f"    CMD: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"  ✓ Thành công!")
            return True
        else:
            print(f"  ✗ Lỗi: {result.stderr[:500]}")
            return False
    except FileNotFoundError:
        print("  ✗ ffmpeg không tìm thấy! Hãy cài ffmpeg và thêm vào PATH.")
        return False
    except subprocess.TimeoutExpired:
        print("  ✗ Quá thời gian (timeout 600s)!")
        return False


def run_ffprobe(video_path):
    """Lấy thông tin audio stream từ video."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return json.loads(result.stdout)
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def format_time(seconds):
    """Format giây thành MM:SS."""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:05.2f}"


def check_demucs_installed():
    """Kiểm tra demucs đã cài chưa."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "demucs", "--help"],
            capture_output=True, text=True, timeout=15
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def install_demucs():
    """Cài đặt demucs."""
    print("\n  ⏳ Đang cài đặt demucs (lần đầu, có thể mất vài phút)...")
    cmd = [sys.executable, "-m", "pip", "install", "-U", "demucs"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode == 0:
        print("  ✓ Cài đặt demucs thành công!")
        return True
    else:
        print(f"  ✗ Cài đặt thất bại: {result.stderr[:300]}")
        return False


# ──────────────────────────────────────────────
# COMMANDS
# ──────────────────────────────────────────────

def cmd_separate(args):
    """
    Tách vocals (giọng nhân vật / meme sound) ra khỏi nhạc nền bằng Demucs AI.
    Dùng Python API trực tiếp (bypass lỗi torchcodec).
    
    Output:
      - {base}_vocals.wav   → Giọng nói + meme sound (GIỮA LẠI)
      - {base}_no_vocals.wav → Nhạc nền (BỎ ĐI hoặc giữ tùy ý)
    """
    video = args.input
    if not os.path.exists(video):
        print(f"  ✗ File không tồn tại: {video}")
        return None

    base = os.path.splitext(os.path.basename(video))[0]
    work_dir = os.path.dirname(os.path.abspath(video)) or "."

    # Bước 1: Tách audio từ video thành WAV
    print(f"\n{'─' * 50}")
    print(f"  Bước 1/3: Tách audio từ video...")
    print(f"{'─' * 50}")
    
    temp_audio = os.path.join(work_dir, f"{base}_temp_audio.wav")
    if not run_ffmpeg(
        ["ffmpeg", "-y", "-i", video, "-vn", "-acodec", "pcm_s16le",
         "-ar", "44100", "-ac", "2", temp_audio],
        f"Tách audio → {temp_audio}"
    ):
        return None

    # Bước 2: Chạy Demucs bằng Python API (bypass lỗi torchcodec)
    print(f"\n{'─' * 50}")
    print(f"  Bước 2/3: Demucs AI đang tách vocals khỏi nhạc nền...")
    print(f"  (Lần đầu chạy sẽ tải model ~80MB)")
    print(f"{'─' * 50}")

    t0 = time.time()
    model_name = getattr(args, 'model', 'htdemucs')
    use_cpu = getattr(args, 'cpu', False)

    try:
        import torch
        import numpy as np
        from scipy.io import wavfile
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        # Load model
        print(f"  → Loading model: {model_name}...")
        model = get_model(model_name)
        
        if use_cpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"  → Device: {device}")
        model.to(device)

        # Load audio bằng scipy (bypass torchcodec)
        print(f"  → Loading audio: {temp_audio}...")
        sr, audio_np = wavfile.read(temp_audio)
        # wavfile trả về int16 → chuyển sang float32 [-1, 1]
        audio_np = audio_np.astype(np.float32) / 32768.0
        # (samples, channels) → (channels, samples)
        if audio_np.ndim == 1:
            audio_np = np.stack([audio_np, audio_np])  # mono → stereo
        else:
            audio_np = audio_np.T
        wav = torch.from_numpy(audio_np)
        
        # Resample nếu cần (demucs yêu cầu sample rate của model)
        if sr != model.samplerate:
            print(f"  → Resampling {sr}Hz → {model.samplerate}Hz...")
            import torchaudio
            wav = torchaudio.functional.resample(wav, sr, model.samplerate)
            sr = model.samplerate

        # Chuẩn bị input: (batch=1, channels, samples)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        wav = wav.unsqueeze(0).to(device)

        # Chạy model
        print(f"  → Đang tách vocals... (có thể mất vài phút)")
        with torch.no_grad():
            sources = apply_model(model, wav, progress=True)
        
        # sources shape: (batch, num_sources, channels, samples)
        # Tìm index của "vocals" trong model.sources
        source_names = model.sources
        print(f"  → Sources: {source_names}")
        
        vocals_idx = source_names.index("vocals") if "vocals" in source_names else 0
        
        # Lấy vocals và no_vocals
        vocals = sources[0, vocals_idx]  # (channels, samples)
        
        # no_vocals = tổng tất cả sources trừ vocals
        no_vocals = sources[0].sum(dim=0) - vocals  # (channels, samples)
        
        # Denormalize
        vocals = vocals * ref.std() + ref.mean()
        no_vocals = no_vocals * ref.std() + ref.mean()
        
        elapsed = time.time() - t0
        print(f"\n  ✓ Demucs hoàn thành trong {elapsed:.0f}s")

        # Bước 3: Lưu output bằng scipy (bypass torchcodec)
        print(f"\n{'─' * 50}")
        print(f"  Bước 3/3: Lưu output...")
        print(f"{'─' * 50}")

        vocals_dst = os.path.join(work_dir, f"{base}_vocals.wav")
        no_vocals_dst = os.path.join(work_dir, f"{base}_no_vocals.wav")

        def save_wav(tensor, path, sample_rate):
            """Lưu tensor thành WAV bằng scipy."""
            audio_np = tensor.cpu().numpy()
            # (channels, samples) → (samples, channels)
            if audio_np.ndim == 2:
                audio_np = audio_np.T
            # Clip và chuyển sang int16
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_int16 = (audio_np * 32767).astype(np.int16)
            wavfile.write(path, sample_rate, audio_int16)

        save_wav(vocals, vocals_dst, sr)
        save_wav(no_vocals, no_vocals_dst, sr)

    except ImportError as e:
        print(f"\n  ✗ Thiếu thư viện: {e}")
        print(f"  → Hãy cài: pip install demucs torchaudio scipy")
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        return None
    except Exception as e:
        print(f"\n  ✗ Lỗi khi chạy Demucs: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        return None

    # Cleanup temp
    if os.path.exists(temp_audio):
        os.remove(temp_audio)



    # Report
    vocals_size = os.path.getsize(vocals_dst) / (1024 * 1024)
    print(f"\n{'=' * 50}")
    print(f"  ✓ HOÀN THÀNH!")
    print(f"{'=' * 50}")
    print(f"  🎤 Vocals (giọng/meme): {vocals_dst} ({vocals_size:.1f} MB)")
    if os.path.exists(no_vocals_dst):
        bg_size = os.path.getsize(no_vocals_dst) / (1024 * 1024)
        print(f"  🎵 Nhạc nền (bản quyền): {no_vocals_dst} ({bg_size:.1f} MB)")
    print(f"\n  → Ghép vocals vào video đã render:")
    print(f"    python audio_tools.py mux <video_rendered.mp4> {vocals_dst}")
    print(f"{'=' * 50}")

    return vocals_dst


def cmd_extract(args):
    """Tách audio gốc ra khỏi video (không xử lý AI)."""
    video = args.input
    if not os.path.exists(video):
        print(f"  ✗ File không tồn tại: {video}")
        return

    # Detect audio codec
    info = run_ffprobe(video)
    audio_codec = "unknown"
    if info:
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "audio":
                audio_codec = stream.get("codec_name", "unknown")
                break

    if args.output:
        output = args.output
    else:
        base = os.path.splitext(video)[0]
        ext_map = {"aac": ".aac", "mp3": ".mp3", "opus": ".opus", "vorbis": ".ogg"}
        ext = ext_map.get(audio_codec, ".aac")
        output = f"{base}_audio{ext}"

    out_ext = os.path.splitext(output)[1].lower()
    if out_ext in [".aac", ".m4a"] and audio_codec == "aac":
        cmd = ["ffmpeg", "-y", "-i", video, "-vn", "-acodec", "copy", output]
        desc = f"Tách audio (copy {audio_codec})"
    elif out_ext == ".mp3":
        cmd = ["ffmpeg", "-y", "-i", video, "-vn", "-acodec", "libmp3lame", "-q:a", "2", output]
        desc = "Tách audio (MP3)"
    elif out_ext == ".wav":
        cmd = ["ffmpeg", "-y", "-i", video, "-vn", "-acodec", "pcm_s16le", output]
        desc = "Tách audio (WAV)"
    else:
        cmd = ["ffmpeg", "-y", "-i", video, "-vn", "-acodec", "aac", "-b:a", "192k", output]
        desc = "Tách audio (AAC)"

    if run_ffmpeg(cmd, f"{desc} → {output}"):
        size_mb = os.path.getsize(output) / (1024 * 1024)
        print(f"  📁 Output: {output} ({size_mb:.1f} MB)")


def cmd_mux(args):
    """Ghép audio file vào video."""
    video = args.input
    audio = args.audio

    if not os.path.exists(video):
        print(f"  ✗ Video không tồn tại: {video}")
        return
    if not os.path.exists(audio):
        print(f"  ✗ Audio không tồn tại: {audio}")
        return

    if args.output:
        output = args.output
    else:
        base = os.path.splitext(video)[0]
        output = f"{base}_final.mp4"

    cmd = [
        "ffmpeg", "-y",
        "-i", video,
        "-i", audio,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output
    ]

    if run_ffmpeg(cmd, f"Ghép audio → {output}"):
        size_mb = os.path.getsize(output) / (1024 * 1024)
        print(f"  📁 Output: {output} ({size_mb:.1f} MB)")


def cmd_analyze(args):
    """Phân tích audio stream của video."""
    video = args.input
    if not os.path.exists(video):
        print(f"  ✗ File không tồn tại: {video}")
        return

    info = run_ffprobe(video)
    if not info:
        print("  ✗ Không thể phân tích file (ffprobe lỗi)")
        return

    fmt = info.get("format", {})
    duration = float(fmt.get("duration", 0))
    total_bitrate = int(fmt.get("bit_rate", 0)) // 1000

    print(f"\n{'=' * 50}")
    print(f"  📹 File: {video}")
    print(f"{'=' * 50}")
    print(f"  Duration : {format_time(duration)}")
    print(f"  Bitrate  : {total_bitrate} kbps")
    print(f"  Format   : {fmt.get('format_long_name', 'N/A')}")

    for stream in info.get("streams", []):
        codec_type = stream.get("codec_type", "")
        if codec_type == "audio":
            print(f"\n  🔊 Audio Stream:")
            print(f"    Codec     : {stream.get('codec_name', 'N/A')}")
            print(f"    Sample Rate: {stream.get('sample_rate', 'N/A')} Hz")
            print(f"    Channels  : {stream.get('channels', 'N/A')}")
            br = int(stream.get("bit_rate", 0))
            print(f"    Bitrate   : {br // 1000 if br else '?'} kbps")
        elif codec_type == "video":
            print(f"\n  🎬 Video Stream:")
            print(f"    Codec     : {stream.get('codec_name', 'N/A')}")
            print(f"    Resolution: {stream.get('width', '?')}x{stream.get('height', '?')}")

    print(f"{'=' * 50}")


def cmd_auto(args):
    """
    Tự động: Tách vocals từ video gốc → ghép vào video đã render.
    1 lệnh duy nhất thay vì phải chạy separate + mux.
    
    Usage:
        python audio_tools.py auto input.mp4 output_final_45.mp4
    """
    source_video = args.input         # Video gốc (có audio bản quyền)
    rendered_video = args.rendered     # Video đã render (không audio hoặc audio cũ)

    if not os.path.exists(source_video):
        print(f"  ✗ Video gốc không tồn tại: {source_video}")
        return
    if not os.path.exists(rendered_video):
        print(f"  ✗ Video đã render không tồn tại: {rendered_video}")
        return

    print(f"\n{'=' * 55}")
    print(f"  🔄 AUTO MODE: Loại bỏ nhạc bản quyền")
    print(f"{'=' * 55}")
    print(f"  Video gốc (nguồn audio)  : {source_video}")
    print(f"  Video đã render (ghép vào): {rendered_video}")
    print()

    # Bước 1: Tách vocals bằng AI
    sep_args = argparse.Namespace(
        input=source_video,
        model=getattr(args, 'model', 'htdemucs'),
        cpu=getattr(args, 'cpu', False)
    )
    vocals_path = cmd_separate(sep_args)

    if not vocals_path or not os.path.exists(vocals_path):
        print("\n  ✗ Tách vocals thất bại!")
        return

    # Bước 2: Ghép vocals vào video đã render
    print(f"\n{'─' * 50}")
    print(f"  Ghép vocals vào video đã render...")
    print(f"{'─' * 50}")

    if args.output:
        output = args.output
    else:
        base = os.path.splitext(rendered_video)[0]
        output = f"{base}_clean.mp4"

    mux_args = argparse.Namespace(
        input=rendered_video,
        audio=vocals_path,
        output=output
    )
    cmd_mux(mux_args)

    if os.path.exists(output):
        size_mb = os.path.getsize(output) / (1024 * 1024)
        print(f"\n{'=' * 55}")
        print(f"  🎉 HOÀN THÀNH!")
        print(f"  📁 Output (không nhạc bản quyền): {output} ({size_mb:.1f} MB)")
        print(f"{'=' * 55}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monggu Translate - Audio Tools (Tách nhạc bản quyền bằng AI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
═══════════════════════════════════════════════════
  WORKFLOW CHÍNH: Loại bỏ nhạc nền bản quyền
═══════════════════════════════════════════════════

  Cách 1 - Tự động (1 lệnh):
    python audio_tools.py auto input.mp4 output_final_45.mp4
    → Tạo: output_final_45_clean.mp4 (chỉ có giọng/meme, không nhạc nền)

  Cách 2 - Thủ công (2 bước):
    python audio_tools.py separate input.mp4
    python audio_tools.py mux output_final_45.mp4 input_vocals.wav

  Tiện ích:
    python audio_tools.py analyze input.mp4    # Xem info audio
    python audio_tools.py extract input.mp4    # Tách audio gốc ra file
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Lệnh cần thực hiện")

    # auto - Workflow tự động
    p_auto = subparsers.add_parser("auto",
        help="Tự động: tách vocals + ghép vào video (1 lệnh)")
    p_auto.add_argument("input", help="Video GỐC (có audio bản quyền)")
    p_auto.add_argument("rendered", help="Video ĐÃ RENDER (cần ghép vocals)")
    p_auto.add_argument("-o", "--output", help="Đường dẫn output")
    p_auto.add_argument("--model", default="htdemucs",
        help="Demucs model (mặc định: htdemucs, chất lượng cao: htdemucs_ft)")
    p_auto.add_argument("--cpu", action="store_true", help="Dùng CPU thay vì GPU")

    # separate - Tách vocals bằng AI
    p_sep = subparsers.add_parser("separate",
        help="Tách vocals (giọng/meme) ra khỏi nhạc nền bằng AI")
    p_sep.add_argument("input", help="Video hoặc audio file")
    p_sep.add_argument("--model", default="htdemucs",
        help="Demucs model (htdemucs / htdemucs_ft)")
    p_sep.add_argument("--cpu", action="store_true", help="Dùng CPU thay vì GPU")

    # extract - Tách audio thô
    p_extract = subparsers.add_parser("extract", help="Tách audio gốc ra file")
    p_extract.add_argument("input", help="Đường dẫn video")
    p_extract.add_argument("-o", "--output", help="Đường dẫn output")

    # mux - Ghép audio vào video
    p_mux = subparsers.add_parser("mux", help="Ghép audio vào video")
    p_mux.add_argument("input", help="Video (không có audio)")
    p_mux.add_argument("audio", help="File audio cần ghép")
    p_mux.add_argument("-o", "--output", help="Đường dẫn output")

    # analyze - Phân tích audio
    p_analyze = subparsers.add_parser("analyze", help="Phân tích audio stream")
    p_analyze.add_argument("input", help="Đường dẫn video")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    print("=" * 55)
    print("  Monggu Translate - Audio Tools")
    print("  (AI-powered vocal/music separation)")
    print("=" * 55)

    if args.command == "auto":
        cmd_auto(args)
    elif args.command == "separate":
        cmd_separate(args)
    elif args.command == "extract":
        cmd_extract(args)
    elif args.command == "mux":
        cmd_mux(args)
    elif args.command == "analyze":
        cmd_analyze(args)
