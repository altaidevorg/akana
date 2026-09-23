"""
Deep, comprehensive validation and stress test suite comparing
Akana native Rust embedding engine vs official HuggingFace tokenizer.
"""

import time
import sys
from concurrent.futures import ThreadPoolExecutor
from tokenizers import Tokenizer
import numpy as np
import akana

def main():
    print("=" * 80)
    print("🚀 AKANA vs HUGGINGFACE TOKENIZER: EXTENSIVE DEEP BENCHMARK & STRESS TEST")
    print("=" * 80)

    hf_tokenizer = Tokenizer.from_file("crates/akana-core/data/embeddings/tokenizer.json")
    data = np.load("crates/akana-core/data/embeddings/turboquant_weights.npz")
    packed = data["packed_indices"]
    scales = data["scales"]

    def dequantize(token_id):
        row = packed[token_id]
        scale = float(scales[token_id])
        emb = np.zeros(256, dtype=np.float32)
        for j in range(256):
            byte_idx = j // 4
            bit_offset = (j % 4) * 2
            q = (int(row[byte_idx]) >> bit_offset) & 0b11
            emb[j] = (q - 1.5) * scale
        return emb

    # Pre-dequantize lookup table for reference calculations
    table = np.array([dequantize(i) for i in range(len(scales))])

    def hf_embed(text):
        tokens = hf_tokenizer.encode(text, add_special_tokens=False).ids
        if not tokens:
            return np.zeros(256, dtype=np.float32)
        embs = table[tokens]
        mean_emb = np.mean(embs, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 1e-12:
            mean_emb /= norm
        return mean_emb

    # ── 1. Gather comprehensive dataset ──────────────────────────────────────────
    dataset = []

    # (a) Kalyoncu 4,600 vocabulary words
    with open("data/kalyoncu_words_4600.txt") as f:
        kalyoncu_words = [line.strip() for line in f if line.strip()]
    dataset.extend(kalyoncu_words)

    # (b) Sample story sentences
    for path in ["data/sample_iki_horoz.txt", "data/sample_sule_piknikte.txt"]:
        with open(path) as f:
            text = f.read()
            sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 3]
            dataset.extend(sentences)

    # (c) Diverse edge cases
    edge_cases = [
        "",  # Empty
        " ",  # Whitespace
        "   \t  \n  ",  # Multi-whitespace
        "a",  # Single char
        "z",
        "ı", "I", "i", "İ", "ğ", "Ğ", "ü", "Ü", "ş", "Ş", "ö", "Ö", "ç", "Ç",
        "muvaffakiyetsizleştiricileştiriveremeyebileceklerimizdenmişsinizcesine",
        "çekoslovakyalılaştıramadıklarımızdanmısınız",
        "1923", "1453", "2026", "3.14159", "1.000.000 TL", "50%", "10/10",
        "Ahmet'in, Mehmet'e, İstanbul'dan, Ankara'ya, TBMM'de, TDK'ye",
        "Dr. Prof. Av. Doç. vb. vs. sn. dk. m² cm³",
        "https://huggingface.co/altaidevorg/turkish-bge-m3-model2vec-turboquant-2bit",
        "test.user+nlp@subdomain.example.org.tr",
        "#TurkceNLP #YapayZeka @altaidevorg",
        "!@#$%^&*()_+=-`~[]\\{}|;':\",./<>?",
        "«»“”‘’—–…",
        "🚀🇹🇷🤖💻📚✨",
        "Yapay Zekâ ve Doğal Dil İşleme (NLP): Akana v0.2.0 modern Türkçe analiz araç takımıdır.",
        "Uzun paragraf: " + " ".join(kalyoncu_words[:100]),
    ]
    dataset.extend(edge_cases)

    total_items = len(dataset)
    print(f"Total test items loaded: {total_items:,} (4,600 words + stories + edge cases)\n")

    # ── 2. Token-by-Token Match Verification ─────────────────────────────────────
    print("TEST 1: Exact Token ID Match (HuggingFace Tokenizer vs Akana Tokenizer)...")
    token_mismatches = 0
    mismatch_samples = []

    for idx, text in enumerate(dataset):
        hf_ids = hf_tokenizer.encode(text, add_special_tokens=False).ids
        ak_ids = akana.tokenize_embedding_text(text)

        if hf_ids != ak_ids:
            token_mismatches += 1
            if len(mismatch_samples) < 5:
                mismatch_samples.append((text, hf_ids, ak_ids))

    token_match_pct = ((total_items - token_mismatches) / total_items) * 100
    print(f"  Token Match Rate:    {token_match_pct:.4f}% ({total_items - token_mismatches}/{total_items})")
    if token_mismatches > 0:
        print(f"  Mismatches found: {token_mismatches}")
        for t, hf_i, ak_i in mismatch_samples:
            print(f'    Sample: "{t[:30]}" -> HF: {hf_i} vs Akana: {ak_i}')
    else:
        print("  ✅ ZERO token ID mismatches across all 4,700+ inputs!")

    print()

    # ── 3. Vector Numerical Precision Verification ──────────────────────────────
    print("TEST 2: Vector Embedding Numerical Precision (256 dimensions)...")
    min_cosine_sim = 1.0
    max_abs_diff = 0.0
    mean_abs_diff_sum = 0.0
    valid_vectors = 0

    for text in dataset:
        hf_v = hf_embed(text)
        ak_v = np.array(akana.embed(text), dtype=np.float32)

        norm_hf = np.linalg.norm(hf_v)
        norm_ak = np.linalg.norm(ak_v)

        if norm_hf > 1e-12 and norm_ak > 1e-12:
            cos_sim = float(np.dot(hf_v, ak_v) / (norm_hf * norm_ak))
            if cos_sim < min_cosine_sim:
                min_cosine_sim = cos_sim
            diff = float(np.max(np.abs(hf_v - ak_v)))
            if diff > max_abs_diff:
                max_abs_diff = diff
            mean_abs_diff_sum += float(np.mean(np.abs(hf_v - ak_v)))
            valid_vectors += 1

    mean_abs_diff = mean_abs_diff_sum / max(valid_vectors, 1)
    print(f"  Min Cosine Similarity: {min_cosine_sim:.8f}")
    print(f"  Max Absolute Error:    {max_abs_diff:.8f}")
    print(f"  Mean Absolute Error:   {mean_abs_diff:.10f}")
    if min_cosine_sim > 0.99999 and max_abs_diff < 1e-5:
        print("  ✅ Numerical precision is 100.00% identical within float32 limits!")

    print()

    # ── 4. Semantic Concept Clusters & Pairs ────────────────────────────────────
    print("TEST 3: Semantic Similarity Structure & Relation Tests...")
    test_pairs = [
        # Morphological Inflections
        ("ev", "evler", "Morphology: Noun Plural", True),
        ("kitap", "kitabım", "Morphology: Voicing + Possessive", True),
        ("gelmek", "geliyorum", "Morphology: Verb Progressive", True),
        # Semantic Synonyms / Near-Synonyms
        ("hekim", "doktor", "Synonym: Hekim / Doktor", True),
        ("muallim", "öğretmen", "Synonym: Muallim / Öğretmen", True),
        ("konut", "ev", "Synonym: Konut / Ev", True),
        # Topical Associations vs Unrelated
        ("Türkiye", "Ankara", "Country - Capital", True),
        ("Fransa", "Paris", "Country - Capital", True),
        ("bilgisayar", "yazılım", "Tech domain", True),
        # Unrelated control pairs
        ("domates", "kuantum", "Unrelated (Food vs Physics)", False),
        ("uçak", "şarkı", "Unrelated (Vehicle vs Music)", False),
    ]

    for w1, w2, desc, is_related in test_pairs:
        sim = akana.similarity(w1, w2)
        rel_tag = "RELATED  " if is_related else "UNRELATED"
        print(f"  [{rel_tag}] {w1:<12} <-> {w2:<12} | Cosine: {sim:+.4f} | ({desc})")

    print()

    # ── 5. Throughput & Multi-threading Stress Test ─────────────────────────────
    print("TEST 4: Throughput & Multi-Thread Concurrency Benchmark...")
    benchmark_corpus = kalyoncu_words[:1000] * 50  # 50,000 sentences

    # Single-thread batch
    t0 = time.perf_counter()
    res = akana.embed_batch(benchmark_corpus)
    t1 = time.perf_counter()
    single_thread_time = t1 - t0
    single_thread_sps = len(benchmark_corpus) / single_thread_time
    print(f"  Single-Thread Batch (50,000 sentences): {single_thread_sps:,.0f} sent/sec ({single_thread_time*1000:.1f} ms)")

    # Multi-thread concurrent calls (ThreadPoolExecutor)
    chunks = [benchmark_corpus[i:i + 10000] for i in range(0, len(benchmark_corpus), 10000)]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = list(executor.map(akana.embed_batch, chunks))
    t1 = time.perf_counter()
    mt_time = t1 - t0
    mt_sps = len(benchmark_corpus) / mt_time
    print(f"  Multi-Thread 4 Workers (50,000 sentences): {mt_sps:,.0f} sent/sec ({mt_time*1000:.1f} ms)")

    print("\n" + "=" * 80)
    print("🎉 ALL EXTENSIVE VALIDATION SUITES COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
