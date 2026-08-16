"""
Comparative Benchmark: Akana vs Zeyrek
Testing morphological analysis speed and vocabulary coverage across thousands of words.
"""

import time
import akana
import zeyrek

# Initialize analyzers
print("Initializing Akana...")
t0 = time.perf_counter()
akana_morph = akana.Morphology()
akana_init = time.perf_counter() - t0
print(f"Akana initialized in {akana_init*1000:.2f} ms")

print("Initializing Zeyrek (Zemberek Python Port)...")
t0 = time.perf_counter()
zeyrek_morph = zeyrek.MorphAnalyzer()
zeyrek_init = time.perf_counter() - t0
print(f"Zeyrek initialized in {zeyrek_init*1000:.2f} ms")

# Test Words (Everyday words, technical words, loanwords, compound words, inflected words)
test_words = [
    "kitap", "kitabım", "kitaplarımızda", "ev", "evdeki", "evlerimizden",
    "banyo", "banyoya", "piknik", "piknikte", "kahvaltı", "kahvaltıda",
    "oynamak", "oynadılar", "oynuyorlardı", "öğretmen", "öğretmenimizden",
    "pencere", "pencerelerden", "bilgisayar", "bilgisayarlaşmak", "uçak",
    "havaalanı", "denizaltı", "akbaba", "koşuyordu", "biliyordu",
    "yapacaklardı", "seviyordu", "temizliğini", "fırçaladı", "arabasıyla",
    "çocuklarımızın", "gözlükçülük", "çalışkanlık", "akşamki", "dünkü",
    "yapamayacaklar", "gidemem", "küçücük", "affetmek", "hakkımızda"
] * 250 # 10,500 word queries

print(f"\nBenchmarking on {len(test_words)} morphological analysis queries...")

# Benchmark Akana
t0 = time.perf_counter()
akana_parsed_count = 0
for w in test_words:
    res = akana_morph.analyze(w)
    if res:
        akana_parsed_count += 1
akana_time = time.perf_counter() - t0
akana_qps = len(test_words) / akana_time

# Benchmark Zeyrek
t0 = time.perf_counter()
zeyrek_parsed_count = 0
for w in test_words:
    res = zeyrek_morph.analyze(w)
    if res:
        zeyrek_parsed_count += 1
zeyrek_time = time.perf_counter() - t0
zeyrek_qps = len(test_words) / zeyrek_time

print("\n" + "=" * 70)
print(f"{'METRIC':<30} | {'AKANA (Rust)':<17} | {'ZEYREK (Python)':<17}")
print("=" * 70)
print(f"{'Total Queries':<30} | {len(test_words):<17} | {len(test_words):<17}")
print(f"{'Successfully Parsed':<30} | {akana_parsed_count:<17} | {zeyrek_parsed_count:<17}")
print(f"{'Total Execution Time':<30} | {akana_time*1000:>14.2f} ms | {zeyrek_time*1000:>14.2f} ms")
print(f"{'Throughput (Words/sec)':<30} | {akana_qps:>14.1f}   | {zeyrek_qps:>14.1f}  ")
print(f"{'Speed Advantage':<30} | {akana_qps/zeyrek_qps:>14.1f}x  | {'1.0x':<17}")
print("=" * 70)
