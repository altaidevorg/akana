"""
Experiment: 1-Bit Binary Dense + Morphological BM25 Hybrid Retrieval vs 2-Bit TurboQuant Hybrid.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import time
import numpy as np
from usearch.index import CompiledMetric, Index, MetricKind, MetricSignature

import akana
from scripts.benchmark_rag_real_data import (
    fetch_ragturk_dataset,
    fetch_turkuaz_dataset,
    SparseBM25InvertedIndex,
    _record,
)


def run_experiment():
    print("=" * 90)
    print("🧪 EXPERIMENT: 1-BIT BINARY DENSE + BM25 RRF vs 2-BIT TURBOQUANT + BM25 RRF")
    print("=" * 90)

    datasets = [
        ("RagTurk (metunlp/ragturk)", fetch_ragturk_dataset, 15),
        ("Turkuaz-RAG (eneSadi/turkuaz-rag)", fetch_turkuaz_dataset, 40),
    ]

    for ds_name, fetch_fn, arg in datasets:
        docs, queries = fetch_fn(arg)
        if not queries:
            print(f"Skipping {ds_name}: no queries.")
            continue

        n_docs = len(docs)
        n_queries = len(queries)

        # 1. 2-Bit USearch Index
        m2 = CompiledMetric(
            pointer=akana.get_usearch_metric_pointer_2bit(),
            kind=MetricKind.Cos,
            signature=MetricSignature.ArrayArraySize,
        )
        idx2 = Index(ndim=64, metric=m2, dtype="u8")
        for d in docs:
            idx2.add(d["key"], np.frombuffer(akana.embed_packed_2bit(d["text"]), dtype=np.uint8))

        # 2. 1-Bit USearch Index
        m1 = CompiledMetric(
            pointer=akana.get_usearch_metric_pointer_1bit(),
            kind=MetricKind.Hamming,
            signature=MetricSignature.ArrayArraySize,
        )
        idx1 = Index(ndim=32, metric=m1, dtype="u8")
        for d in docs:
            idx1.add(d["key"], np.frombuffer(akana.embed_packed_1bit(d["text"]), dtype=np.uint8))

        # 3. Sparse Inverted Index
        s_idx = SparseBM25InvertedIndex()
        for d in docs:
            s_idx.index_document(d["key"], d["text"])
        s_idx.finalize()

        methods = [
            "Dense 2-bit (64B)",
            "Dense 1-bit (32B)",
            "Sparse BM25 (Only)",
            "RRF (2-bit + BM25)",
            "RRF (1-bit + BM25)",
        ]
        metrics = {
            m: {"rec1": 0, "rec3": 0, "rec5": 0, "rec10": 0, "mrr": 0.0, "total_time": 0.0}
            for m in methods
        }

        for q in queries:
            gold = q["gold_keys"]
            qt = q["question"]

            # Dense 2bit search
            t0 = time.perf_counter()
            h2 = idx2.search(np.frombuffer(akana.embed_packed_2bit(qt), dtype=np.uint8), count=10)
            t_2b = time.perf_counter() - t0
            k2 = [h.key for h in h2]
            s2 = [(h.key, -h.distance) for h in h2]
            _record(metrics["Dense 2-bit (64B)"], k2, gold, t_2b)

            # Dense 1bit search
            t0 = time.perf_counter()
            h1 = idx1.search(np.frombuffer(akana.embed_packed_1bit(qt), dtype=np.uint8), count=10)
            t_1b = time.perf_counter() - t0
            k1 = [h.key for h in h1]
            s1 = [(h.key, -h.distance) for h in h1]
            _record(metrics["Dense 1-bit (32B)"], k1, gold, t_1b)

            # Sparse BM25 search
            t0 = time.perf_counter()
            sp_hits = s_idx.search(qt, top_k=20)
            t_sp = time.perf_counter() - t0
            k_sp = [doc_id for doc_id, _ in sp_hits[:10]]
            _record(metrics["Sparse BM25 (Only)"], k_sp, gold, t_sp)

            # RRF 2-bit + BM25
            t0 = time.perf_counter()
            fused2 = akana.reciprocal_rank_fusion(s2, sp_hits, k=60.0, alpha=1.0)
            t_rrf2 = (time.perf_counter() - t0) + t_2b + t_sp
            _record(metrics["RRF (2-bit + BM25)"], [fid for fid, _ in fused2[:10]], gold, t_rrf2)

            # RRF 1-bit + BM25
            t0 = time.perf_counter()
            fused1 = akana.reciprocal_rank_fusion(s1, sp_hits, k=60.0, alpha=1.0)
            t_rrf1 = (time.perf_counter() - t0) + t_1b + t_sp
            _record(metrics["RRF (1-bit + BM25)"], [fid for fid, _ in fused1[:10]], gold, t_rrf1)

        print(f"\n--- {ds_name} (Corpus: {n_docs} docs | Queries: {n_queries}) ---")
        print(f"{'Retrieval Engine':<22} | {'Bytes':<6} | {'Recall@1':<8} | {'Recall@5':<8} | {'Recall@10':<9} | {'MRR@10':<7} | {'Latency'}")
        print("-" * 85)

        bytes_map = {
            "Dense 2-bit (64B)": "64 B",
            "Dense 1-bit (32B)": "32 B",
            "Sparse BM25 (Only)": "Inv.",
            "RRF (2-bit + BM25)": "64B+Inv",
            "RRF (1-bit + BM25)": "32B+Inv",
        }

        for m in methods:
            dat = metrics[m]
            r1 = dat["rec1"] / n_queries * 100
            r5 = dat["rec5"] / n_queries * 100
            r10 = dat["rec10"] / n_queries * 100
            mrr = dat["mrr"] / n_queries
            lat = (dat["total_time"] / n_queries) * 1e6
            b_str = bytes_map[m]
            print(f"{m:<22} | {b_str:<6} | {r1:7.1f}% | {r5:7.1f}% | {r10:8.1f}% | {mrr:7.3f} | {lat:5.0f} µs")


if __name__ == "__main__":
    run_experiment()
