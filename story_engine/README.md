# Story Engine — Dựng reel meme theo cốt truyện, KHÓA NHÂN VẬT

Bộ khung này biến **cốt truyện (JSON)** + **thư viện nhân vật (JSON)** thành một reel dọc 1080×1920, đảm bảo **không nhầm nhân vật** xuyên suốt phim.

## 1. Cơ chế "khóa nhân vật" (điểm quan trọng nhất)

- Mọi nhân vật **dùng chung một thân "mèo chuối"** (`base_idle` — clip idle lặp).
- Mỗi nhân vật phân biệt bằng **một bộ phụ kiện riêng** (tóc + vũ khí), tra **theo `character_id`**.
- Khi đổi cảm xúc/thao tác → tạm thay thân bằng **clip meme** tương ứng (giận, sốc, buồn…), **nhưng vẫn đắp đúng phụ kiện của nhân vật đó** → người xem vẫn nhận ra ai.
- Hết cảm xúc → tự quay lại `base_idle` (mèo chuối ban đầu).

Vì phụ kiện luôn được lấy theo ID ở **mọi cảnh**, không thể gắn nhầm phụ kiện của nhân vật A sang B. Đó là cách khóa danh tính. Bonus: vì thân giống nhau nên toạ độ tóc/vũ khí cố định → luôn rơi đúng chỗ.

## 2. Cấu trúc thư mục

```
story_engine/
  make_story_reel.py        # engine
  characters.json           # thư viện nhân vật + thân chung + pool cảm xúc
  script.json               # cốt truyện (danh sách cảnh)
  assets/
    clips/base.mp4          # THÂN mèo chuối (idle, phông xanh)
    clips/emo_*.mp4         # các clip cảm xúc dùng chung (phông xanh)
    char/<id>/hair.png      # tóc của nhân vật <id> (nền trong suốt)
    char/<id>/weapon.png    # vũ khí của nhân vật <id> (nền trong suốt)
    backgrounds/<id>.png    # nền cảnh
    music/bg.wav            # nhạc nền (tự sinh nếu thiếu)
```

## 3. `characters.json`

```jsonc
{
  "chroma": "0x00FF00", "chroma_sim": 0.30, "chroma_blend": 0.12,  // tách phông xanh
  "voice": null,                         // giọng edge-tts mặc định (vd "vi-VN-NamMinhNeural")
  "base_idle": "assets/clips/base.mp4",  // THÂN dùng chung
  "emotions": {                          // pool cảm xúc dùng chung
    "shock": "assets/clips/emo_shock.mp4",
    "angry": "assets/clips/emo_angry.mp4",
    "sad":   "assets/clips/emo_sad.mp4"
  },
  "backgrounds": { "mountain": "...", "hall": "..." },
  "characters": {
    "mc": {
      "name": "Mèo Chuối",
      "hair": "assets/char/mc/hair.png",
      "weapon": "assets/char/mc/weapon.png",
      "cat_h": 1050, "cat_y": 600,                  // kích thước & vị trí thân
      "hair_w": 500, "hair_dx": 0, "hair_y": 540,   // tóc: rộng, lệch ngang, cao
      "weap_w": 200, "weap_x": 770, "weap_y": 640,  // vũ khí: rộng, x, y
      "base_idle": "assets/clips/mc_idle.mp4",       // (tùy chọn) thân riêng nếu muốn
      "emotions": { "angry": "assets/char/mc/angry.mp4" } // (tùy chọn) override cảm xúc riêng
    },
    "rival": { "name": "Hắc Miêu", "hair": "...", "weapon": "...", ... }
  }
}
```

Quy tắc chọn clip mỗi cảnh: `idle` → `base_idle`; cảm xúc khác → tìm trong `characters[id].emotions` trước, không có thì lấy `emotions` chung.

## 4. `script.json`

```jsonc
{
  "title": "TU TIÊN KÝ",
  "resolution": [1080, 1920],
  "scenes": [
    { "char": "mc", "emotion": "idle",  "bg": "mountain", "say": "Ta là Mèo Chuối..." },
    { "char": "mc", "emotion": "shock", "bg": "mountain", "say": "Cái gì?!", "dur": 3.0 },
    { "char": "rival", "emotion": "idle", "bg": "hall", "say": "Hừ...", "voice": "vi-VN-NamMinhNeural" }
  ]
}
```

Mỗi cảnh: `char` (ID nhân vật), `emotion` (`idle`/`shock`/`angry`/…), `bg` (ID nền), `say` (caption + lời lồng tiếng), tùy chọn `dur` (giây) và `voice`.

## 5. Cắm ASSET THẬT của bạn vào

1. **Thân mèo chuối**: thay `assets/clips/base.mp4` bằng clip idle mèo chuối của bạn (nền xanh lá).
2. **Clip cảm xúc**: bỏ các meme cảm xúc (nền xanh) vào `assets/clips/` và khai báo trong `emotions`.
3. **Tóc & vũ khí**: bỏ PNG nền trong suốt vào `assets/char/<id>/` rồi trỏ trong `characters.json`. Tinh chỉnh `hair_w/hair_y/weap_*` cho khớp đầu mèo (chỉ cần chỉnh 1 lần vì thân cố định).
4. **Nền**: bỏ ảnh 1080×1920 vào `assets/backgrounds/`.
5. **Cốt truyện**: viết `script.json` theo mạch bạn muốn.

## 6. Lồng giọng (tùy chọn)

Đặt `"voice": "vi-VN-NamMinhNeural"` (hoặc `vi-VN-HoaiMyNeural`) ở cấp cảnh hoặc toàn cục. Engine gọi **edge-tts** (cần mạng) để sinh giọng theo `say`, tự khớp thời lượng cảnh và trộn cùng tiếng meme. Không có mạng/edge-tts → bỏ qua, dùng tiếng clip.

## 7. Chạy

```bash
pip install pillow numpy edge-tts          # numpy & edge-tts là tùy chọn
python make_story_reel.py --chars characters.json --script script.json -o reel.mp4

# Ý tưởng → kịch bản bằng LLM (Gemini → DeepSeek, dùng chung lõi llm.py + cache với pipeline dịch):
python make_story_reel.py --idea "Mèo Chuối đi thi tuyển đệ tử nhưng quên mang kiếm" -o reel.mp4
```

Với `--idea`:
- Cần cài thêm `pip install -r ../requirements.txt` và API key trong `D:\monggutranslate\.env`, giống pipeline dịch.
- LLM **chỉ được dùng** nhân vật, cảm xúc và bối cảnh có trong `characters.json`. Kịch bản dùng id lạ sẽ bị từ chối, không cache, và hỏi lại.
- Kịch bản được lưu vào `kich_ban/kb_<thời gian>.json`, hoặc vào đường dẫn `--script` nếu có đưa. File đã có thì **không bao giờ bị ghi đè**: lệnh sẽ dừng và báo.
- Muốn sửa lời thoại, sửa file đó rồi dựng lại bằng `--script kich_ban/<file>.json`, **bỏ `--idea`**.
- Có thể thêm `"mo_ta"` (tính cách, quan hệ) cho từng nhân vật trong `characters.json` để kịch bản hợp vai hơn.
- Cùng một ý tưởng thì cache trả về cùng một kịch bản. Muốn một bản khác thì đổi câu ý tưởng.
- Kịch bản viết tay (`--script`) cũng được kiểm tra với thư viện trước khi dựng.

## 8. Giới hạn hiện tại & hướng nâng

- Phụ kiện đặt theo toạ độ cố định → khớp tốt khi thân là **cùng một con mèo chuối**; nếu clip cảm xúc là con mèo khác (như demo) thì tóc có thể lệch. Khắc phục: dùng cùng nhân vật cho mọi cảm xúc, hoặc override `hair_y/weap_y` riêng cho từng cảm xúc.
- Muốn tóc bám theo đầu khi mèo cử động → cần head-tracking (ví dụ MediaPipe) — có thể bổ sung sau.
- Nhân vật/nền nên thay bằng ảnh tự sinh (ComfyUI/Forge) để vừa đẹp vừa nguyên bản (tránh lỗi unoriginal của Facebook).
