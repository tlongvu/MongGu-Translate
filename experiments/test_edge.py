import asyncio
import edge_tts
import subprocess
import os

TEXT = "Ngày thiên kim thật trở về. Phu nhân, đại tiểu thư về rồi ạ."
VOICE = "vi-VN-NamMinhNeural" # Giọng nam
OUTPUT_AUDIO = "test_audio_nam.mp3"
INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "test_voiceover_nam.mp4"

async def generate_audio():
    print(f"Đang sinh giọng đọc AI (Giọng: {VOICE})...")
    # Tăng tốc độ đọc lên một chút (+10%) để nghe tự nhiên và phù hợp với video hơn
    communicate = edge_tts.Communicate(TEXT, VOICE, rate="+10%")
    await communicate.save(OUTPUT_AUDIO)
    print(f"Đã lưu file âm thanh: {OUTPUT_AUDIO}")

def mix_video():
    print(f"Đang mix âm thanh AI vào video gốc ({INPUT_VIDEO})...")
    # Cắt 7 giây đầu của video, giảm âm lượng video gốc xuống 20% và chèn giọng đọc AI vào
    cmd = (
        f'ffmpeg -y -t 7 -i {INPUT_VIDEO} -i {OUTPUT_AUDIO} '
        f'-filter_complex "[0:a]volume=0.2[a0];[1:a]volume=1.5[a1];[a0][a1]amix=inputs=2:duration=first[a]" '
        f'-map 0:v -map "[a]" -c:v copy -c:a aac {OUTPUT_VIDEO}'
    )
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        print(f"Thành công! Video output đã được lưu tại: {OUTPUT_VIDEO}")
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi mix video: {e.stderr.decode('utf-8', errors='ignore')}")

async def main():
    await generate_audio()
    if os.path.exists(INPUT_VIDEO):
        mix_video()
    else:
        print(f"Không tìm thấy video gốc {INPUT_VIDEO} để mix.")

if __name__ == "__main__":
    asyncio.run(main())
