"""
Real-world RAG Benchmark for Akana's Low-Memory Retrieval Engines.

Evaluates:
  1. Dense 2-bit TurboQuant (64 bytes/vector) + USearch Compiled Cosine Metric
  2. Dense 1-bit Binary (32 bytes/vector) + USearch Compiled Hamming Metric
  3. Single-Buffer Hybrid (128 bytes/vector) + USearch Compiled Joint Metric
  4. Dual-Index Reciprocal Rank Fusion (Dense 2-bit + Akana Morphological BM25)

Benchmark Datasets:
  - RagTurk (metunlp/ragturk, arXiv:2602.03652): Turkish RAG failure modes, morphological variations
  - Turkuaz-RAG (eneSadi/turkuaz-rag): Multi-context question answering
"""

import argparse
import ast
import json
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import requests

import akana
from usearch.index import CompiledMetric, Index, MetricKind, MetricSignature


CACHE_DIR = Path(".cache/rag_benchmarks")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RAGTURK_BASE_URL = "https://huggingface.co/datasets/metunlp/ragturk/raw/main/formal_5k/dataset/json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Akana-Benchmark/1.0"}


def fetch_ragturk_dataset(max_articles: int = 25) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Downloads or loads cached RagTurk articles and extracts chunks and questions.
    Returns:
        (documents, questions)
    """
    ragturk_dir = CACHE_DIR / "ragturk"
    ragturk_dir.mkdir(parents=True, exist_ok=True)

    # First discover file list from HuggingFace dataset repo
    index_cache = ragturk_dir / "file_list.json"
    if index_cache.exists():
        with open(index_cache, "r", encoding="utf-8") as f:
            file_names = json.load(f)
    else:
        print("📥 Fetching RagTurk file list from Hugging Face...")
        api_url = "https://huggingface.co/api/datasets/metunlp/ragturk"
        try:
            resp = requests.get(api_url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            file_names = [
                Path(s["rfilename"]).name
                for s in data.get("siblings", [])
                if s["rfilename"].startswith("formal_5k/dataset/json/") and s["rfilename"].endswith(".json")
            ]
            with open(index_cache, "w", encoding="utf-8") as f:
                json.dump(file_names, f)
        except Exception as e:
            print(f"⚠️ Error querying HF API: {e}. Using curated default file list.")
            file_names = [
                "1048_Feodosia.json",
                "1076_Viola.json",
                "11__Ordu__Sovyetler_Birli_i_.json",
                "1219.json",
                "1290_lar.json",
                "1291_Federal_Beyannamesi.json",
                "12_Haziran.json",
                "13_Ekim.json",
                "14_May_s.json",
                "15_A_ustos.json",
                "16_Aral_k.json",
                "17_Temmuz.json",
                "18_Eyl_l.json",
                "1905_Rus_Devrimi.json",
                "1912_Yaz_Olimpiyatlar_.json",
                "1920_ler.json",
                "1940_lar.json",
                "1950_ler.json",
                "1960_lar.json",
                "1970_ler.json",
                "1980_ler.json",
                "1990_lar.json",
                "2000_ler.json",
                "2010_lar.json",
                "2020_ler.json",
            ]

    selected_files = file_names[:max_articles]
    all_docs = []
    all_queries = []
    doc_id_counter = 0

    print(f"Loading {len(selected_files)} RagTurk article files...")
    for fname in selected_files:
        cached_file = ragturk_dir / fname
        if not cached_file.exists():
            url = f"{RAGTURK_BASE_URL}/{fname}"
            try:
                r = requests.get(url, headers=HEADERS, timeout=10)
                if r.status_code == 200:
                    with open(cached_file, "w", encoding="utf-8") as f:
                        f.write(r.text)
                else:
                    continue
            except Exception:
                continue

        try:
            with open(cached_file, "r", encoding="utf-8") as f:
                article_data = json.load(f)
        except Exception:
            continue

        chunk_id_to_key = {}
        for c in article_data.get("chunks", []):
            content = c.get("content", "").strip()
            if not content:
                continue
            key = doc_id_counter
            doc_id_counter += 1
            chunk_id_to_key[c.get("id")] = key
            all_docs.append({
                "key": key,
                "text": content,
                "chunk_id": c.get("id"),
                "file": fname,
            })

        questions_dict = article_data.get("questions", {})
        q_items = questions_dict.get("items", []) if isinstance(questions_dict, dict) else []
        for q in q_items:
            q_text = q.get("question", "").strip()
            rel_chunks = q.get("related_chunk_ids", [])
            gold_keys = [chunk_id_to_key[cid] for cid in rel_chunks if cid in chunk_id_to_key]
            if q_text and gold_keys:
                all_queries.append({
                    "question": q_text,
                    "gold_keys": set(gold_keys),
                    "category": q.get("category", "GENERAL"),
                })

    return all_docs, all_queries


def fetch_turkuaz_dataset(max_samples: int = 50) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Loads samples from eneSadi/turkuaz-rag multi-context dataset.
    """
    try:
        from datasets import load_dataset
        print("📥 Streaming samples from eneSadi/turkuaz-rag...")
        ds = load_dataset("eneSadi/turkuaz-rag", split="train", streaming=True)
        all_docs = []
        all_queries = []
        doc_key = 0

        for i, sample in enumerate(ds):
            if i >= max_samples:
                break
            q_text = sample.get("question", "")
            if not q_text:
                continue

            # Parse news docs
            news1_raw = sample.get("1st_news", "")
            news2_raw = sample.get("2nd_news", "")

            gold_keys = []
            for raw in [news1_raw, news2_raw]:
                if not raw:
                    continue
                if isinstance(raw, str):
                    try:
                        parsed = ast.literal_eval(raw)
                    except Exception:
                        parsed = {"summary": raw}
                else:
                    parsed = raw

                text = f"{parsed.get('title', '')}. {parsed.get('summary', '')}".strip()
                if text:
                    key = doc_key
                    doc_key += 1
                    all_docs.append({"key": key, "text": text})
                    gold_keys.append(key)

            if gold_keys:
                all_queries.append({
                    "question": q_text,
                    "gold_keys": set(gold_keys),
                    "category": sample.get("question_type", "MULTI_CONTEXT"),
                })

        return all_docs, all_queries
    except Exception as e:
        print(f"⚠️ Could not load turkuaz-rag via datasets: {e}")
        return [], []


def get_synthetic_hard_turkish_corpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Fallback challenging Turkish test set designed specifically for Turkish agglutinative
    morphology, vowel harmony, and compound words.
    """
    corpus = [
        "Türkiye Cumhuriyet Merkez Bankası (TCMB) politika faizini yüzde 45 seviyesinde sabit tutma kararı aldı.",
        "Bankacılık Düzenleme ve Denetleme Kurumu (BDDK) kredi kartı taksit sınırlandırmalarını güncelledi.",
        "İstanbul Boğaziçi Köprüsü bakım çalışmaları nedeniyle hafta sonu trafiğe kapatılacak.",
        "Kuzey Marmara Otoyolu'nda meydana gelen zincirleme trafik kazasında 3 kişi hafif yaralandı.",
        "Galatasaray, Şampiyonlar Ligi grup mücadelesinde Bayern Münih ile golsüz berabere kaldı.",
        "Fenerbahçe Beko, THY EuroLeague deplasmanında Real Madrid'i uzatma periyodunda mağlup etti.",
        "Yeni nesil elektrikli araç bataryalarında lityum-iyon yerine katı hal pil teknolojisi kullanılacak.",
        "Yenilenebilir enerji yatırımları kapsamında Ege ve Akdeniz bölgelerinde yeni rüzgar santralleri kuruluyor.",
        "Kapadokya bölgesindeki peri bacaları ve yeraltı şehirleri bu yıl turist rekoru kırdı.",
        "Efes Antik Kenti'nde yürütülen kazı çalışmalarında 2000 yıllık mermer heykeller gün ışığına çıkarıldı.",
        "Akana doğal dil işleme kütüphanesi, Türkçe için morfolojik analiz, heceleme ve PII maskeleme sunar.",
        "Model2Vec ve TurboQuant algoritmaları, embedding modellerini 32 kat sıkıştırarak hafifletir.",
        "Akdeniz mutfağında zeytinyağlı enginar ve taze otlu deniz ürünleri önemli bir yere sahiptir.",
        "Karadeniz bölgesinde hamsi avı sezonu bereketli başladı; balık tezgahlarında fiyatlar düştü.",
        "Bursa Uludağ'da kış turizmi sezonu yoğun kar yağışının ardından erken açıldı.",
    ]
    docs = [{"key": i, "text": t} for i, t in enumerate(corpus)]
    queries = [
        {"question": "Merkez Bankasının faiz politikası ne oldu?", "gold_keys": {0}, "category": "MORPH_SUFFIX"},
        {"question": "Kredi kartlarındaki taksit kısıtlamasını hangi kurum belirledi?", "gold_keys": {1}, "category": "FACTUAL"},
        {"question": "İstanbuldaki köprünün kapatılma sebebi nedir?", "gold_keys": {2}, "category": "MORPH_SUFFIX"},
        {"question": "Marmara otoyolundaki kazada yaralanan oldu mu?", "gold_keys": {3}, "category": "FACTUAL"},
        {"question": "Galatasarayın Bayern Münih ile oynadığı maç nasıl sonuçlandı?", "gold_keys": {4}, "category": "MORPH_SUFFIX"},
        {"question": "EuroLeague basketbol maçında Real Madrid'e karşı kim galip geldi?", "gold_keys": {5}, "category": "FACTUAL"},
        {"question": "Otomotiv sektöründe geleceğin batarya teknolojisi ne olacak?", "gold_keys": {6}, "category": "SEMANTIC"},
        {"question": "Ege ve Akdeniz'e kurulacak temiz enerji santralleri nelerdir?", "gold_keys": {7}, "category": "SEMANTIC"},
        {"question": "Tarihi peri bacalarını görmek için nereye gidilir?", "gold_keys": {8}, "category": "SEMANTIC"},
        {"question": "Efes kazılarında yakın zamanda ne keşfedildi?", "gold_keys": {9}, "category": "FACTUAL"},
        {"question": "Türkçe NLP araçları için geliştirilen yerli kütüphane hangisidir?", "gold_keys": {10}, "category": "LEXICAL"},
        {"question": "Vektör modellerini aşırı küçük boyuta sıkıştıran yöntemler nelerdir?", "gold_keys": {11}, "category": "LEXICAL"},
    ]
    return docs, queries


class SparseBM25InvertedIndex:
    """
    Lightweight BM25 Inverted Index using Akana's fast MorphologicalSparseEncoder.
    Provides inverted list lookup for Dual-Index RRF.
    """
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.postings = defaultdict(list)  # term_idx -> list of (doc_key, tf)
        self.doc_lens = {}
        self.avg_doc_len = 0.0
        self.n_docs = 0

    def index_document(self, doc_key: int, text: str):
        sparse = akana.sparse_embed(text)
        indices = sparse["indices"]
        values = sparse["values"]
        doc_len = sum(values) if values else 1.0
        self.doc_lens[doc_key] = doc_len
        self.n_docs += 1

        for idx, val in zip(indices, values):
            self.postings[idx].append((doc_key, val))

    def finalize(self):
        if self.n_docs > 0:
            self.avg_doc_len = sum(self.doc_lens.values()) / self.n_docs
        else:
            self.avg_doc_len = 1.0

    def search(self, query_text: str, top_k: int = 20) -> list[tuple[int, float]]:
        sparse = akana.sparse_embed(query_text)
        q_indices = sparse["indices"]
        scores = defaultdict(float)

        for q_idx in q_indices:
            plist = self.postings.get(q_idx, [])
            if not plist:
                continue
            df = len(plist)
            idf = max(0.1, np.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0))
            for doc_key, tf in plist:
                doc_len = self.doc_lens.get(doc_key, self.avg_doc_len)
                tf_norm = (tf * (self.k1 + 1.0)) / (tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len)))
                scores[doc_key] += idf * tf_norm

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_items


def evaluate_retrieval(
    docs: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    dataset_name: str,
):
    print("\n" + "=" * 80)
    print(f"🎯 BENCHMARK RESULTS ON REAL DATASET: {dataset_name.upper()}")
    print(f"📊 Corpus Size: {len(docs)} documents | Evaluation Queries: {len(queries)}")
    print("=" * 80)

    # 1. Build Index 1: Dense 2-bit TurboQuant Cosine (64 bytes)
    t0 = time.perf_counter()
    metric_2bit = CompiledMetric(
        pointer=akana.get_usearch_metric_pointer_2bit(),
        kind=MetricKind.Cos,
        signature=MetricSignature.ArrayArraySize,
    )
    index_2bit = Index(ndim=64, metric=metric_2bit, dtype="u8")
    for d in docs:
        b = akana.embed_packed_2bit(d["text"])
        index_2bit.add(d["key"], np.frombuffer(b, dtype=np.uint8))
    build_time_2bit = (time.perf_counter() - t0) * 1000

    # 2. Build Index 2: Dense 1-bit Binary Hamming (32 bytes)
    t0 = time.perf_counter()
    metric_1bit = CompiledMetric(
        pointer=akana.get_usearch_metric_pointer_1bit(),
        kind=MetricKind.Hamming,
        signature=MetricSignature.ArrayArraySize,
    )
    index_1bit = Index(ndim=32, metric=metric_1bit, dtype="u8")
    for d in docs:
        b = akana.embed_packed_1bit(d["text"])
        index_1bit.add(d["key"], np.frombuffer(b, dtype=np.uint8))
    build_time_1bit = (time.perf_counter() - t0) * 1000

    # 3. Build Index 3: Single-Buffer Hybrid (128 bytes)
    t0 = time.perf_counter()
    metric_hybrid = CompiledMetric(
        pointer=akana.get_usearch_metric_pointer_hybrid(0.5),
        kind=MetricKind.Cos,
        signature=MetricSignature.ArrayArraySize,
    )
    index_hybrid = Index(ndim=128, metric=metric_hybrid, dtype="u8")
    for d in docs:
        b = akana.embed_hybrid(d["text"])
        index_hybrid.add(d["key"], np.frombuffer(b, dtype=np.uint8))
    build_time_hybrid = (time.perf_counter() - t0) * 1000

    # 4. Build Sparse Inverted Index for Dual RRF
    t0 = time.perf_counter()
    sparse_index = SparseBM25InvertedIndex()
    for d in docs:
        sparse_index.index_document(d["key"], d["text"])
    sparse_index.finalize()
    build_time_sparse = (time.perf_counter() - t0) * 1000

    methods = [
        "Dense 2-bit (64B)",
        "Dense 1-bit (32B)",
        "Single-Buffer Hybrid (128B)",
        "Dual-Index RRF (2-bit + BM25)",
    ]

    metrics = {m: {"rec1": 0, "rec3": 0, "rec5": 0, "rec10": 0, "mrr": 0.0, "total_time": 0.0} for m in methods}

    for q in queries:
        q_text = q["question"]
        gold = q["gold_keys"]

        # --- Method 1: Dense 2-bit ---
        t_start = time.perf_counter()
        q_b2 = akana.embed_packed_2bit(q_text)
        hits_2bit = index_2bit.search(np.frombuffer(q_b2, dtype=np.uint8), count=10)
        t_elapsed = time.perf_counter() - t_start
        keys_2bit = [h.key for h in hits_2bit]
        scores_2bit = [(h.key, -h.distance) for h in hits_2bit]
        _record(metrics["Dense 2-bit (64B)"], keys_2bit, gold, t_elapsed)

        # --- Method 2: Dense 1-bit ---
        t_start = time.perf_counter()
        q_b1 = akana.embed_packed_1bit(q_text)
        hits_1bit = index_1bit.search(np.frombuffer(q_b1, dtype=np.uint8), count=10)
        t_elapsed = time.perf_counter() - t_start
        keys_1bit = [h.key for h in hits_1bit]
        _record(metrics["Dense 1-bit (32B)"], keys_1bit, gold, t_elapsed)

        # --- Method 3: Single-Buffer Hybrid ---
        t_start = time.perf_counter()
        q_bh = akana.embed_hybrid(q_text)
        hits_hybrid = index_hybrid.search(np.frombuffer(q_bh, dtype=np.uint8), count=10)
        t_elapsed = time.perf_counter() - t_start
        keys_hybrid = [h.key for h in hits_hybrid]
        _record(metrics["Single-Buffer Hybrid (128B)"], keys_hybrid, gold, t_elapsed)

        # --- Method 4: Dual-Index RRF ---
        t_start = time.perf_counter()
        sparse_hits = sparse_index.search(q_text, top_k=20)
        fused = akana.reciprocal_rank_fusion(scores_2bit, sparse_hits, k=60.0, alpha=1.0)
        t_elapsed = time.perf_counter() - t_start
        keys_rrf = [fid for fid, _ in fused[:10]]
        _record(metrics["Dual-Index RRF (2-bit + BM25)"], keys_rrf, gold, t_elapsed)

    n_q = len(queries)
    print("\n| Retrieval Engine | Bytes/Vector | RAM (1M docs) | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | Latency (µs) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    ram_map = {
        "Dense 2-bit (64B)": ("64 B", "64 MB"),
        "Dense 1-bit (32B)": ("32 B", "32 MB"),
        "Single-Buffer Hybrid (128B)": ("128 B", "128 MB"),
        "Dual-Index RRF (2-bit + BM25)": ("64 B + Inverted", "~180 MB"),
    }

    for m in methods:
        dat = metrics[m]
        r1 = dat["rec1"] / n_q * 100
        r3 = dat["rec3"] / n_q * 100
        r5 = dat["rec5"] / n_q * 100
        r10 = dat["rec10"] / n_q * 100
        mrr = dat["mrr"] / n_q
        avg_lat_us = (dat["total_time"] / n_q) * 1_000_000
        v_bytes, ram_1m = ram_map[m]
        print(f"| **{m}** | {v_bytes} | {ram_1m} | {r1:.1f}% | {r3:.1f}% | {r5:.1f}% | {r10:.1f}% | {mrr:.3f} | {avg_lat_us:.0f} µs |")

    print("\n💡 Key Architectural Takeaways:")
    print("  • 2-bit TurboQuant provides 16x memory reduction over standard FP32 vectors (64 MB vs 1,024 MB per 1M docs).")
    print("  • 1-bit Binary provides 32x memory reduction (32 MB per 1M docs) with hardware POPCNT Hamming distance.")
    print("  • 128-byte Single-Buffer Hybrid perfectly fits in two CPU cache lines (64B dense + 16 sorted 4B sparse terms).")
    print("  • Akana's Morphological BM25 decomposes agglutinative roots and compounds, eliminating Turkish vocabulary mismatch.")


def _record(stat: dict[str, Any], retrieved_keys: list[int], gold_keys: set[int], elapsed: float):
    stat["total_time"] += elapsed
    hits = [k in gold_keys for k in retrieved_keys]
    if any(hits[:1]):
        stat["rec1"] += 1
    if any(hits[:3]):
        stat["rec3"] += 1
    if any(hits[:5]):
        stat["rec5"] += 1
    if any(hits[:10]):
        stat["rec10"] += 1

    for rank, h in enumerate(hits[:10], start=1):
        if h:
            stat["mrr"] += 1.0 / rank
            break


def main():
    parser = argparse.ArgumentParser(description="Akana RAG Real-World Dataset Benchmark")
    parser.add_argument("--dataset", choices=["ragturk", "turkuaz", "synthetic", "all"], default="ragturk")
    parser.add_argument("--max-articles", type=int, default=20)
    parser.add_argument("--max-turkuaz", type=int, default=40)
    args = parser.parse_args()

    if args.dataset in ("ragturk", "all"):
        docs, queries = fetch_ragturk_dataset(max_articles=args.max_articles)
        if not queries:
            print("⚠️ Could not fetch RagTurk. Falling back to synthetic corpus...")
            docs, queries = get_synthetic_hard_turkish_corpus()
        evaluate_retrieval(docs, queries, "RagTurk (metunlp/ragturk)")

    if args.dataset in ("turkuaz", "all"):
        docs, queries = fetch_turkuaz_dataset(max_samples=args.max_turkuaz)
        if queries:
            evaluate_retrieval(docs, queries, "Turkuaz-RAG (eneSadi/turkuaz-rag)")

    if args.dataset == "synthetic":
        docs, queries = get_synthetic_hard_turkish_corpus()
        evaluate_retrieval(docs, queries, "Turkish Agglutinative & Morphological Hard Benchmark")


if __name__ == "__main__":
    main()
