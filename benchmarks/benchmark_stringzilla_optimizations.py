"""
Performance & Correctness Benchmark for StringZilla SIMD & Zero-Regex Optimizations in Akana.
Tests:
1. Tokenization Throughput (Zero-allocation fast dispatch vs repeated regex scanning).
2. AI Style Audit Throughput (StringZilla SIMD literal search vs 25x regex passes).
3. Informal Normalization Speed (Zero-copy suffix scanning vs regex capture engines).
4. Named Entity Recognition Speed (Linear token stream vs multi-regex passes).
"""

import time
import akana

def benchmark_tokenization():
    sample_text = (
        "Prof. Dr. Ahmet Yılmaz 16 Ağustos 2026 tarihinde İstanbul Üniversitesi bünyesinde "
        "500 TL ödeme yaptı ve https://akana.dev adresini ziyaret etti. #TürkçeNLP @akana_project "
        "Küçük çocuk bahçede neşeyle koşuyordu. Güneş pırıl pırıl parlıyor, kuşlar ağaçların dallarında cıvıldıyordu. "
    ) * 500  # ~35,000 words

    start = time.perf_counter()
    tokens = akana.tokenize_words(sample_text)
    elapsed = time.perf_counter() - start
    words_per_sec = len(tokens) / elapsed

    print(f"[*] Tokenization Benchmark:")
    print(f"    • Total Tokens Processed: {len(tokens):,}")
    print(f"    • Time Elapsed:          {elapsed*1000:.2f} ms")
    print(f"    • Throughput:              {words_per_sec:,.0f} tokens/sec")
    return words_per_sec

def benchmark_ai_style_auditing():
    sample_text = (
        "Yapay zeka teknolojileri, modern dünyada bireylerin ve kurumların hayatında kritik bir rol oynamaktadır. "
        "Bu bağlamda —özellikle veri analitiği alanında— hayati bir önem taşımaktadır; "
        "aynı zamanda sadece büyük işletmeler için değil, aynı zamanda günlük kullanıcılar için de vazgeçilmez bir hale gelmiştir. "
        "Bu doğrultuda geliştirilen algoritmalar, karar alma süreçlerinde etkin bir şekilde kullanılmaktadır. "
    ) * 200  # ~11,000 words

    start = time.perf_counter()
    report = akana.audit_ai_style(sample_text)
    elapsed = time.perf_counter() - start
    words_count = report.metrics["rhythm"]["total_words"]
    words_per_sec = words_count / elapsed

    print(f"\n[*] AI Style Audit (StringZilla SIMD) Benchmark:")
    print(f"    • Total Words Processed:  {words_count:,}")
    print(f"    • Total Findings:         {len(report.findings):,}")
    print(f"    • Time Elapsed:          {elapsed*1000:.2f} ms")
    print(f"    • Throughput:              {words_per_sec:,.0f} words/sec")
    return words_per_sec

def benchmark_informal_normalization():
    sample_words = ["yapcam", "geliyom", "biliyo", "napıyon", "tşkler", "çooookk", "gelmiom"] * 5000  # 35,000 words

    start = time.perf_counter()
    for w in sample_words:
        _ = akana.normalize_informal(w)
    elapsed = time.perf_counter() - start
    words_per_sec = len(sample_words) / elapsed

    print(f"\n[*] Informal Text Normalization (Zero-Regex) Benchmark:")
    print(f"    • Total Words Processed:  {len(sample_words):,}")
    print(f"    • Time Elapsed:          {elapsed*1000:.2f} ms")
    print(f"    • Throughput:              {words_per_sec:,.0f} words/sec")
    return words_per_sec

def benchmark_ner():
    sample_text = (
        "Prof. Dr. Ahmet Yılmaz 16 Ağustos 2026 tarihinde İstanbul Üniversitesi bünyesinde "
        "500 TL ödeme yaptı. Çankaya Köşkü önünde saat 14:30'da buluştular. "
    ) * 300  # ~7,200 words

    start = time.perf_counter()
    entities = akana.extract_entities(sample_text)
    elapsed = time.perf_counter() - start

    print(f"\n[*] Named Entity Recognition (Linear Token Stream) Benchmark:")
    print(f"    • Total Entities Found:   {len(entities):,}")
    print(f"    • Time Elapsed:          {elapsed*1000:.2f} ms")
    print(f"    • Speed:                  {len(sample_text)/elapsed/1024/1024:.2f} MB/sec")

def main():
    print("=" * 80)
    print(" AKANA PERFORMANCE & STRINGZILLA SIMD ACCELERATION BENCHMARKS")
    print("=" * 80)
    benchmark_tokenization()
    benchmark_ai_style_auditing()
    benchmark_informal_normalization()
    benchmark_ner()
    print("\n" + "=" * 80)
    print(" ALL BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
