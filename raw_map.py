import json
import difflib

def clean_and_dedup_translation_map(input_path, output_path, similarity_threshold=0.75):
    # Đọc file JSON rác
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_map = json.load(f)

    # Sắp xếp các key theo độ dài giảm dần (ưu tiên câu dài, đầy đủ làm bản chuẩn)
    sorted_keys = sorted(raw_map.keys(), key=len, reverse=True)
    
    canonical_texts = {}  # Lưu {text_chuẩn: bản_dịch_chuẩn}
    cleaned_map = {}      # Lưu mapping cuối cùng
    
    # Những ký tự rác phổ biến do OCR sinh ra
    trash_chars = ['+', 'n', '2', ' ', '*', '『', '「', ']', '」', '【', '】', '“', '”', '.', '？', '！', '!', '?']

    for text in sorted_keys:
        # Bỏ qua rác quá ngắn (<= 1 ký tự mà không phải chữ Hán)
        clean_text = ''.join(c for c in text if c not in trash_chars)
        if len(clean_text) <= 1 and not any('\u4e00' <= c <= '\u9fff' for c in clean_text):
            cleaned_map[text
] = ""
            continue
            
        matched_canonical = None
        
        for canonical in canonical_texts:
            # 1. Kiểm tra Mảnh vỡ (Substring Match)
            # Dùng clean_text để check tránh bị nhiễu bởi dấu câu
            canonical_clean = ''.join(c for c in canonical if c not in trash_chars)
            if clean_text in canonical_clean and len(clean_text) > 0:
                matched_canonical = canonical
                break
                
            # 2. Kiểm tra Lỗi Typo (Fuzzy Match)
            # Chỉ check fuzzy nếu độ dài xấp xỉ nhau (tránh câu ngắn match bừa câu dài)
            if len(text) >= 3 and abs(len(text) - len(canonical)) <= 3:
                ratio = difflib.SequenceMatcher(None, text, canonical).ratio()
                if ratio > similarity_threshold:
                    matched_canonical = canonical
                    break
        
        if matched_canonical:
            # Nếu là mảnh vỡ/typo -> Lấy bản dịch của câu chuẩn gán cho nó
            # Ví dụ: "三师克" -> sẽ được gán là "Tam sư huynh"
            
            # Ưu tiên lấy bản dịch dài/có ý nghĩa hơn nếu bản thân nó đã được dịch tốt
            if len(raw_map[text
]) > len(canonical_texts[matched_canonical
]):
                 canonical_texts[matched_canonical
] = raw_map[text
]
            
            cleaned_map[text
] = canonical_texts[matched_canonical
]
        else:
            # Nếu là text mới hoàn toàn -> Đăng ký làm bản chuẩn
            canonical_texts[text
] = raw_map[text
]
            cleaned_map[text
] = raw_map[text
]

    # Ghi ra file mới
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_map, f, ensure_ascii=False, indent=2)
        
    print(f"Hoàn tất! Đã gom {len(raw_map)} text rác về {len(canonical_texts)} text chuẩn.")
    print(f"Đã lưu kết quả tại: {output_path}")

# Chạy thử
if __name__ == "__main__":
    # Thay tên file tùy ý bạn
    clean_and_dedup_translation_map('translation_map.json', 'translation_map_cleaned.json')