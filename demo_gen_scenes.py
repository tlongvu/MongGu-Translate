from PIL import Image, ImageDraw, ImageFont
import math

W, H = 1080, 1920
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
big = ImageFont.truetype(FONT, 64)
cap = ImageFont.truetype(FONT, 58)

def lerp(a,b,t): return tuple(int(a[i]+(b[i]-a[i])*t) for i in range(3))

def bg_gradient(img, top, bot):
    d = ImageDraw.Draw(img)
    for y in range(H):
        d.line([(0,y),(W,y)], fill=lerp(top,bot,y/H))

def draw_cat(d, cx, cy, s, mood="happy"):
    # đầu mèo tròn vàng cam
    fur=(245,190,90); dark=(60,40,20)
    d.ellipse([cx-s,cy-s,cx+s,cy+s], fill=fur, outline=dark, width=6)
    # tai
    for sx in (-1,1):
        d.polygon([(cx+sx*s*0.7,cy-s*0.55),(cx+sx*s*1.05,cy-s*1.15),(cx+sx*s*0.3,cy-s*0.95)], fill=fur, outline=dark)
    # mắt
    ey=cy-s*0.15; ex=s*0.42
    for sx in (-1,1):
        d.ellipse([cx+sx*ex-26,ey-30,cx+sx*ex+26,ey+30], fill=(255,255,255), outline=dark, width=4)
        d.ellipse([cx+sx*ex-12,ey-12,cx+sx*ex+12,ey+18], fill=(30,30,30))
    # mũi
    d.polygon([(cx-14,cy+s*0.18),(cx+14,cy+s*0.18),(cx,cy+s*0.32)], fill=(220,120,120))
    # miệng
    if mood=="happy":
        d.arc([cx-50,cy+s*0.2,cx+50,cy+s*0.5], 20,160, fill=dark, width=6)
    elif mood=="shock":
        d.ellipse([cx-22,cy+s*0.30,cx+22,cy+s*0.55], fill=(120,40,40), outline=dark, width=4)
    else:
        d.line([(cx-45,cy+s*0.4),(cx+45,cy+s*0.4)], fill=dark, width=6)
    # ria
    for sx in (-1,1):
        for k in (-1,0,1):
            d.line([(cx+sx*22,cy+s*0.2+k*14),(cx+sx*s*0.9,cy+s*0.15+k*22)], fill=dark, width=3)

def caption(img, text):
    d = ImageDraw.Draw(img, "RGBA")
    # wrap
    words=text.split(); lines=[]; cur=""
    for w in words:
        t=(cur+" "+w).strip()
        if d.textlength(t, font=cap) > W-120: lines.append(cur); cur=w
        else: cur=t
    lines.append(cur)
    lh=78; total=lh*len(lines); y0=H-360
    # plate
    pad=24
    d.rounded_rectangle([60-pad, y0-pad, W-60+pad, y0+total+pad], radius=24, fill=(0,0,0,150))
    for i,ln in enumerate(lines):
        tw=d.textlength(ln, font=cap); x=(W-tw)/2; y=y0+i*lh
        # outline đen
        for ox in (-3,0,3):
            for oy in (-3,0,3):
                d.text((x+ox,y+oy), ln, font=cap, fill=(0,0,0))
        d.text((x,y), ln, font=cap, fill=(255,255,0))  # vàng

scenes = [
    (((120,180,255),(40,80,180)), "happy",  "Hôm nay tôi tự tin đi xin việc!"),
    (((255,200,140),(200,120,60)), "happy",  "Sếp hỏi: Em biết làm những gì?"),
    (((255,170,170),(190,70,70)),  "shock",  "Tôi đáp: Dạ... em biết nằm ạ."),
    (((180,255,200),(60,170,110)), "flat",   "Và thế là tôi lại thất nghiệp."),
]
for i,(grad,mood,text) in enumerate(scenes):
    img=Image.new("RGB",(W,H))
    bg_gradient(img, grad[0], grad[1])
    d=ImageDraw.Draw(img)
    # sàn
    d.rectangle([0,int(H*0.72),W,H], fill=lerp(grad[1],(20,20,20),0.3))
    draw_cat(d, W//2, int(H*0.42), 250, mood)
    caption(img, text)
    img.save(f"/tmp/reel/scene{i+1}.png")
    print("saved scene", i+1)
