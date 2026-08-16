from pypdf import PdfReader

reader = PdfReader('data/965984.pdf')

# Ek-1 starts on page 145 (index 144) to page 176 (index 175)
ek1_words = []
for p_idx in range(144, 176):
    text = reader.pages[p_idx].extract_text()
    for line in text.split('\n'):
        line = line.strip()
        if not line or line.isdigit() or '8. EKLER' in line or 'Ek-1:' in line or 'Bilinirlik' in line or 'Esas Alınması' in line:
            continue
        if len(line) == 1 and line.isalpha():
            continue
        for tok in line.split():
            tok = tok.strip(" .,;:()[]{}!?\"'0123456789-*–—/\\")
            if tok and not tok.isdigit():
                lower = tok.replace('I', 'ı').replace('İ', 'i').lower()
                ek1_words.append(lower)

print("Ek-1 unique words:", len(set(ek1_words)))

# Ek-3 starts on page 179 (index 178)
ek3_words = []
for p_idx in [178]:
    text = reader.pages[p_idx].extract_text()
    for line in text.split('\n'):
        line = line.strip()
        if not line or line.isdigit() or 'Ek-3:' in line or 'Eklenen' in line:
            continue
        for tok in line.split():
            tok = tok.strip(" .,;:()[]{}!?\"'0123456789-*–—/\\")
            if tok and not tok.isdigit():
                lower = tok.replace('I', 'ı').replace('İ', 'i').lower()
                ek3_words.append(lower)

print("Ek-3 unique words:", len(set(ek3_words)))

all_words = sorted(list(set(ek1_words).union(set(ek3_words))))
print(f"Total combined unique words: {len(all_words)}")
print("First 20:", all_words[:20])
print("Last 20:", all_words[-20:])

with open('data/kalyoncu_words_4600.txt', 'w', encoding='utf-8') as f:
    for w in all_words:
        f.write(w + '\n')

print("Saved to data/kalyoncu_words_4600.txt")
