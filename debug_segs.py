import json, bisect

with open('ocr_data.json', encoding='utf-8') as f:
    d = json.load(f)

fps = d['fps']
fd = d['frame_detections']
segs = d.get('text_segments', [])

# 1:30 = 90s * 60fps = 5400, 1:32 = 92s * 60fps = 5520
target_start = int(90 * fps)
target_end = int(92 * fps)

out = []
out.append(f'FPS={fps}, checking frames {target_start}-{target_end} (1:30-1:32)')
out.append('')

# Check segments covering this range
out.append('=== SEGMENTS covering 1:30-1:32 ===')
for s in segs:
    if s['end_frame'] >= target_start and s['start_frame'] <= target_end:
        bbox = [int(v) for v in s['bbox']]
        out.append(f'  [{s["start_frame"]}->{s["end_frame"]}] bbox={bbox} "{s["text"]}"')

out.append('')
out.append('=== FRAME DETECTIONS at 1:30-1:32 (every 5th frame) ===')
consecutive_miss = 0
max_miss = 0
prev_had_target = False

for fn in range(target_start, target_end):
    fn_str = str(fn)
    if fn_str in fd:
        texts = [det['text'] for det in fd[fn_str]]
        if fn % 5 == 0:  # sample every 5 frames
            out.append(f'  Frame {fn}: {texts}')
    else:
        if fn % 5 == 0:
            out.append(f'  Frame {fn}: [NO DETECTION]')

with open('debug_flicker.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('Done -> debug_flicker.txt')
