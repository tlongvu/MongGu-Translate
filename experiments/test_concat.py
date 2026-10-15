import subprocess
import os

INTRO_VIDEO = "intro.mp4"
MAIN_VIDEO = "input.mp4"
OUTPUT_VIDEO = "final_output.mp4"

def concat_videos():
    print(f"Đang chuẩn hóa và ghép nối 2 video...\n - Intro: {INTRO_VIDEO}\n - Gốc: {MAIN_VIDEO}")
    
    # Giải thích lệnh FFmpeg filter_complex:
    # 1. Ép video 1 (Intro) về kích thước chuẩn (1406x1080), 30fps, và chuẩn hóa âm thanh (44.1kHz, stereo).
    # 2. Ép video 2 (Gốc) về cùng kích thước, cùng fps, cùng chuẩn âm thanh.
    #    (Ghi chú: Ở đây mình thêm -t 10 cho video gốc để test cho nhanh, chỉ lấy 10s đầu).
    # 3. Dùng lệnh 'concat' để nối mượt mà 2 dòng hình ảnh và âm thanh này lại.
    
    cmd = (
        f'ffmpeg -y -i {INTRO_VIDEO} -t 10 -i {MAIN_VIDEO} '
        f'-filter_complex "'
        f'[0:v]scale=1406:1080:force_original_aspect_ratio=decrease,pad=1406:1080:(ow-iw)/2:(oh-ih)/2,fps=30,setpts=PTS-STARTPTS[v0];'
        f'[0:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[a0];'
        f'[1:v]scale=1406:1080:force_original_aspect_ratio=decrease,pad=1406:1080:(ow-iw)/2:(oh-ih)/2,fps=30,setpts=PTS-STARTPTS[v1];'
        f'[1:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[a1];'
        f'[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]" '
        f'-map "[v]" -map "[a]" '
        f'-c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k {OUTPUT_VIDEO}'
    )
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"\n✅ Thành công! Video ghép nối đã được lưu tại: {OUTPUT_VIDEO}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi ghép video: {e}")

if __name__ == "__main__":
    if os.path.exists(INTRO_VIDEO) and os.path.exists(MAIN_VIDEO):
        concat_videos()
    else:
        print("Thiếu file intro.mp4 hoặc input.mp4")
