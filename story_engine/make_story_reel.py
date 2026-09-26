#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_story_reel.py — Engine dựng reel meme theo cốt truyện, KHÓA NHÂN VẬT.

Cơ chế khóa nhân vật:
  - Mọi nhân vật dùng chung một "thân" mèo chuối (base_idle).
  - Phân biệt bằng bộ phụ kiện (tóc/vũ khí) + bảng tên, tra theo character_id.
  - Đổi cảm xúc -> thay thân bằng clip meme, nhưng vẫn đắp phụ kiện + tên của
    nhân vật đó -> không nhầm. Hết cảm xúc -> quay lại base_idle.

Chạy:
  python make_story_reel.py --chars characters.json --script script.json -o reel.mp4
  python make_story_reel.py --idea "Mèo Chuối đi thi tuyển đệ tử nhưng quên mang kiếm" -o reel.mp4
    (--idea: LLM viết kịch bản CHỈ dùng nhân vật / cảm xúc / bối cảnh trong characters.json,
     lưu ra kich_ban/*.json để sửa tay rồi dựng lại bằng --script)
Phụ thuộc: ffmpeg, Pillow. (Tùy chọn: edge-tts để lồng giọng; numpy để sinh nhạc.)
"""
import os, sys, json, argparse, subprocess, tempfile, shutil, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))  # lõi dùng chung với pipeline dịch (llm.py: Gemini → DeepSeek, có cache)

SCRIPT_SYS = """Bạn là biên kịch reel hài ngắn (30-60 giây) cho mạng xã hội, giọng văn tiếng Việt tự nhiên, hài hước.
Tôi đưa THƯ VIỆN (nhân vật, cảm xúc, bối cảnh có sẵn) và một Ý TƯỞNG. Hãy viết kịch bản 5-10 cảnh.
Trả về DUY NHẤT một JSON object:
{"title": "tên reel", "scenes": [{"char": "<id nhân vật>", "emotion": "<cảm xúc>", "bg": "<id bối cảnh>", "say": "<lời thoại>"}]}
Quy tắc:
- CHỈ dùng id nhân vật, cảm xúc, bối cảnh có trong THƯ VIỆN (viết đúng id).
- Mỗi cảnh một nhân vật nói một câu ngắn (≤ 90 ký tự), xưng hô đúng quan hệ giữa các nhân vật.
- Có mở đầu, cao trào, và câu chốt gây cười ở cảnh cuối.
KHÔNG giải thích."""


def script_problems(scr, cfg) -> list:
    """Các lỗi của kịch bản so với thư viện characters.json (rỗng = dựng được)."""
    if not isinstance(scr, dict) or not isinstance(scr.get("scenes"), list) or not scr["scenes"]:
        return ["kịch bản không có 'scenes'"]
    emotions = {"idle", "base", *cfg.get("emotions", {})}
    probs = []
    for i, sc in enumerate(scr["scenes"], 1):
        if not isinstance(sc, dict):
            probs.append(f"cảnh {i}: không phải object")
            continue
        if sc.get("char") not in cfg["characters"]:
            probs.append(f"cảnh {i}: nhân vật lạ {sc.get('char')!r}")
        if sc.get("emotion", "idle") not in emotions:
            probs.append(f"cảnh {i}: cảm xúc lạ {sc.get('emotion')!r}")
        if sc.get("bg") not in cfg["backgrounds"]:
            probs.append(f"cảnh {i}: bối cảnh lạ {sc.get('bg')!r}")
        if not isinstance(sc.get("say"), str) or not sc["say"].strip():
            probs.append(f"cảnh {i}: thiếu lời thoại")
    return probs


def script_from_idea(idea, cfg) -> dict:
    """Ý tưởng → kịch bản (cùng dạng script.json). Kịch bản dùng id ngoài thư viện → KHÔNG cache,
    hỏi lại / đổi provider (check=); hỏng hết → LLMError."""
    from llm import ask_json
    lib = {"nhan_vat": {cid: {"ten": c.get("name", cid), **({"mo_ta": c["mo_ta"]} if c.get("mo_ta") else {})}
                        for cid, c in cfg["characters"].items()},
           "cam_xuc": ["idle", *cfg.get("emotions", {})],
           "boi_canh": list(cfg["backgrounds"])}
    user = "THƯ VIỆN:\n" + json.dumps(lib, ensure_ascii=False) + "\n\nÝ TƯỞNG:\n" + idea
    return ask_json(SCRIPT_SYS, user, temperature=0.7, check=lambda o: not script_problems(o, cfg))
FONT = os.environ.get("STORY_FONT",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

def rel(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)

def probe_dur(path):
    out = subprocess.run(["ffprobe","-v","quiet","-show_entries",
        "format=duration","-of","csv=p=0", path], capture_output=True, text=True)
    try: return float(out.stdout.strip())
    except: return 0.0

def render_caption(text, size, out_png, name="", name_color=(255,210,110)):
    """Caption chữ vàng viền đen + hộp nền mờ; có name -> vẽ BẢNG TÊN phía trên."""
    from PIL import Image, ImageDraw, ImageFont
    W, H = size
    img = Image.new("RGBA", (W, H), (0,0,0,0)); d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, int(W*0.054))
    lines, cur = [], ""
    for w in text.split():
        t = (cur+" "+w).strip()
        if d.textlength(t, font=f) > W-140: lines.append(cur); cur = w
        else: cur = t
    lines.append(cur)
    lh = int(W*0.074); total = lh*len(lines); y0 = H - int(H*0.16) - total; pad = 26
    if name:
        nf = ImageFont.truetype(FONT, int(W*0.046))
        nw = d.textlength(name, font=nf); nh = int(W*0.062)
        nx0, ny0 = 55-pad, y0-pad-nh-14
        d.rounded_rectangle([nx0, ny0, nx0+nw+44, ny0+nh], radius=16,
                            fill=(20,18,30,235), outline=name_color, width=4)
        d.text((nx0+22, ny0+(nh-int(W*0.046))//2-2), name, font=nf, fill=name_color)
    d.rounded_rectangle([55-pad, y0-pad, W-55+pad, y0+total+pad],
                        radius=26, fill=(0,0,0,165))
    for i, ln in enumerate(lines):
        tw = d.textlength(ln, font=f); x = (W-tw)/2; y = y0+i*lh
        for ox in (-3,0,3):
            for oy in (-3,0,3): d.text((x+ox,y+oy), ln, font=f, fill=(0,0,0,255))
        d.text((x,y), ln, font=f, fill=(255,255,0,255))
    img.save(out_png)

def try_tts(text, voice, out_mp3):
    try:
        import asyncio, edge_tts
        async def _g(): await edge_tts.Communicate(text, voice, rate="+6%").save(out_mp3)
        asyncio.run(_g())
        return os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 0
    except Exception as e:
        print(f"  [tts] bỏ qua ({e})"); return False

def ensure_music(path, seconds):
    if os.path.exists(path): return path
    try:
        import numpy as np, wave
        sr = 44100
        P = {'C':261.63,'Eb':311.13,'F':349.23,'G':392.0,'Bb':466.16}
        seq = ['C','Eb','G','F','Eb','C','Bb','C','Eb','F','G','Bb']
        def tone(fr,d):
            t = np.linspace(0,d,int(sr*d),False)
            return (0.5*np.sin(2*np.pi*fr*t)+0.15*np.sin(2*np.pi*2*fr*t))*np.exp(-3*t)
        mel = np.concatenate([tone(P[n],0.5) for n in seq])
        need = int(sr*seconds)
        while len(mel) < need: mel = np.concatenate([mel, mel])
        mel = mel[:need]; mel = (mel/max(1e-6,np.max(np.abs(mel)))*0.5*32767).astype(np.int16)
        st = np.stack([mel,mel],1).flatten()
        with wave.open(path,'w') as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr); w.writeframes(st.tobytes())
        return path
    except Exception as e:
        print(f"  [music] bỏ qua nhạc nền ({e})"); return None

def pick_clip(cfg, char, emotion):
    if emotion in (None, "idle", "base"):
        return rel(char.get("base_idle", cfg["base_idle"]))
    src = char.get("emotions", {}).get(emotion) or cfg.get("emotions", {}).get(emotion)
    if not src:
        print(f"  [warn] không có clip cảm xúc '{emotion}', dùng base idle")
        return rel(char.get("base_idle", cfg["base_idle"]))
    return rel(src)

def render_scene(cfg, scene, idx, tmp, W, H):
    chars = cfg["characters"]
    cid = scene["char"]
    if cid not in chars:
        raise SystemExit(f"Cảnh {idx}: nhân vật '{cid}' không có trong characters.json")
    char = chars[cid]
    emotion = scene.get("emotion", "idle")
    clip = pick_clip(cfg, char, emotion)
    bg = rel(cfg["backgrounds"][scene["bg"]])
    hair = rel(char["hair"]) if char.get("hair") else None
    weapon = rel(char["weapon"]) if char.get("weapon") else None

    cat_h = int(char.get("cat_h", 1050)); cat_y = int(char.get("cat_y", 600))
    hair_w = int(char.get("hair_w", 500)); hair_dx = int(char.get("hair_dx", 0)); hair_y = int(char.get("hair_y", cat_y-40))
    weap_w = int(char.get("weap_w", 200)); weap_x = int(char.get("weap_x", W-300)); weap_y = int(char.get("weap_y", cat_y-40))
    chroma = cfg.get("chroma", "0x00FF00"); sim = cfg.get("chroma_sim", 0.30); bl = cfg.get("chroma_blend", 0.12)

    say = scene.get("say", "")
    dur = float(scene.get("dur", 3.6))
    voice = scene.get("voice") or cfg.get("voice")
    tts_mp3 = None
    if say and voice:
        cand = os.path.join(tmp, f"tts{idx}.mp3")
        if try_tts(say, voice, cand):
            tts_mp3 = cand; dur = max(dur, probe_dur(cand) + 0.5)

    cap_png = os.path.join(tmp, f"cap{idx}.png")
    nm = char.get("name", "")
    nc = char.get("name_color", [255,210,110])
    render_caption(say, (W,H), cap_png, name=nm, name_color=tuple(nc))

    parts = [f"[0:v]chromakey={chroma}:{sim}:{bl},scale=-1:{cat_h}[cat]"]
    ov_idx = 2; labels = {}
    if hair:   labels["hair"] = ov_idx; ov_idx += 1
    if weapon: labels["weap"] = ov_idx; ov_idx += 1
    cap_i = ov_idx
    if hair:   parts.append(f"[{labels['hair']}:v]scale={hair_w}:-1[hair]")
    if weapon: parts.append(f"[{labels['weap']}:v]scale={weap_w}:-1[weap]")
    parts.append(f"[1:v][cat]overlay=(W-w)/2:{cat_y}[a]"); last = "a"
    if hair:
        parts.append(f"[{last}][hair]overlay=(W-w)/2+{hair_dx}:{hair_y}[b]"); last = "b"
    if weapon:
        parts.append(f"[{last}][weap]overlay={weap_x}:{weap_y}[c]"); last = "c"
    parts.append(f"[{last}][{cap_i}:v]overlay=0:0,format=yuv420p[v]")
    parts.append(f"[0:a]aresample=44100,volume=1.0[ca]")
    fc = ";".join(parts)

    inputs = ["-t", f"{dur}", "-i", clip, "-i", bg]
    if hair:   inputs += ["-i", hair]
    if weapon: inputs += ["-i", weapon]
    inputs += ["-i", cap_png]

    out = os.path.join(tmp, f"scene{idx}.mp4")
    amap = "[ca]"
    if tts_mp3:
        inputs += ["-i", tts_mp3]
        fc += f";[ca][{cap_i+1}:a]amix=inputs=2:duration=first:normalize=0[mix]"
        amap = "[mix]"
    cmd = ["ffmpeg","-y", *inputs, "-filter_complex", fc,
           "-map","[v]","-map",amap, "-t", f"{dur}", "-r","25",
           "-c:v","libx264","-preset","veryfast","-pix_fmt","yuv420p",
           "-c:a","aac","-b:a","160k","-ar","44100", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1200:]); raise SystemExit(f"Lỗi render cảnh {idx}")
    print(f"  ✓ cảnh {idx}: {cid}/{emotion}  ({dur:.1f}s)")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", default="characters.json")
    ap.add_argument("--script", default=None, help="kịch bản JSON (mặc định script.json; với --idea: nơi lưu kịch bản)")
    ap.add_argument("--idea", default=None, help="ý tưởng → LLM viết kịch bản từ thư viện nhân vật rồi dựng")
    ap.add_argument("-o","--out", default="reel.mp4")
    ap.add_argument("--music", default="assets/music/bg.wav")
    a = ap.parse_args()

    cfg = json.load(open(rel(a.chars), encoding="utf-8"))
    if a.idea:
        from llm import LLMError
        try:
            scr = script_from_idea(a.idea, cfg)
        except LLMError as e:
            raise SystemExit(f"Không viết được kịch bản: {e}")
        scr.setdefault("resolution", [1080, 1920])
        path = rel(a.script or os.path.join("kich_ban", time.strftime("kb_%Y%m%d_%H%M%S.json")))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(scr, f, ensure_ascii=False, indent=2)
        print(f"Kịch bản: {path}  (sửa tay rồi dựng lại: --script \"{path}\")")
    else:
        scr = json.load(open(rel(a.script or "script.json"), encoding="utf-8"))
        probs = script_problems(scr, cfg)
        if probs:
            raise SystemExit("Kịch bản không khớp characters.json: " + "; ".join(probs[:5]))
    W, H = scr.get("resolution", [1080,1920])

    tmp = tempfile.mkdtemp(prefix="story_")
    print(f"== Dựng '{scr.get('title','reel')}' — {len(scr['scenes'])} cảnh ==")
    clips = []
    for i, sc in enumerate(scr["scenes"], 1):
        clips.append(render_scene(cfg, sc, i, tmp, W, H))

    lst = os.path.join(tmp,"list.txt")
    with open(lst,"w") as f:
        for c in clips: f.write(f"file '{c}'\n")
    base = os.path.join(tmp,"base.mp4")
    subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",lst,
        "-c:v","copy","-c:a","aac","-b:a","160k", base], check=True, capture_output=True)
    total = probe_dur(base)

    out = rel(a.out)
    if a.music:
        os.makedirs(os.path.dirname(rel(a.music)), exist_ok=True)
    mus = ensure_music(rel(a.music), total+1) if a.music else None
    if mus:
        subprocess.run(["ffmpeg","-y","-i",base,"-i",mus,"-filter_complex",
            f"[1:a]volume=0.13,afade=t=out:st={max(0,total-1):.1f}:d=1[m];"
            f"[0:a][m]amix=inputs=2:duration=first:normalize=0[a]",
            "-map","0:v","-map","[a]","-c:v","copy","-c:a","aac","-b:a","192k",
            "-shortest", out], check=True, capture_output=True)
    else:
        shutil.copy(base, out)

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nXONG: {out}  ({total:.1f}s, {W}x{H})")

if __name__ == "__main__":
    main()
