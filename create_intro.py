import subprocess
import os

IMAGE_PATH = "cat_intro.jpg"
OUTPUT_INTRO = "intro.mp4"
DURATION = 1 # Thời lượng intro (giây)

def create_intro_from_image():
    print(f"Đang tạo video Intro {DURATION} giây từ ảnh {IMAGE_PATH}...")
    
    # Lệnh FFmpeg:
    # -loop 1 -i ... : Lặp lại bức ảnh liên tục
    # -f lavfi -i anullsrc... : Tạo một track âm thanh trống (im lặng) chuẩn stereo 44.1kHz. 
    # Bắt buộc phải có âm thanh thì lát nữa ghép với video gốc mới không bị lỗi mất tiếng.
    # -t 2 : Cắt đúng 2 giây
    # -pix_fmt yuv420p : Chuẩn màu tương thích với mọi nền tảng (quan trọng)
    
    cmd = (
        f'ffmpeg -y -loop 1 -i {IMAGE_PATH} '
        f'-f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 '
        f'-c:v libx264 -t {DURATION} -pix_fmt yuv420p -c:a aac -shortest {OUTPUT_INTRO}'
    )
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ Đã tạo thành công Video Intro: {OUTPUT_INTRO}")
        print("Bây giờ bạn có thể dùng file này để ghép nối với video chính!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi tạo intro: {e}")

if __name__ == "__main__":
    if os.path.exists(IMAGE_PATH):
        create_intro_from_image()
    else:
        print(f"Không tìm thấy ảnh {IMAGE_PATH}. Bạn nhớ lưu ảnh vào thư mục này nhé!")
