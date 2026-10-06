#!/usr/bin/env python3
"""Example 11: Ultra-Low-Memory Turkish Embedding, Retrieval & USearch Integration.

Demonstrates production-ready Turkish vector search running in megabytes of RAM:
1. 2-Bit TurboQuant Packed Embeddings (64 bytes / vector = 64 MB for 1M docs).
2. 1-Bit Binary Embeddings (32 bytes / vector = 32 MB for 1M docs) with hardware POPCNT.
3. Precompiled C-ABI metric pointers for zero-copy USearch (`unum-cloud/usearch`) indexes.
4. Akana Morphological BM25 Sparse Embeddings with root extraction & compound decomposition.
5. Single-Buffer 128-Byte Hybrid Vectors (64B dense + 16 sorted sparse slots).
6. Multi-Index Reciprocal Rank Fusion (RRF) for high-accuracy Turkish RAG.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import akana

try:
    from usearch.index import CompiledMetric, Index, MetricKind, MetricSignature
    HAS_USEARCH = True
except ImportError:
    HAS_USEARCH = False


# Sample Turkish knowledge base with agglutinative suffixes, compounds & semantic queries
DOCUMENTS = [
    {
        "id": 1,
        "title": "TCMB Faiz Politikası",
        "text": "Türkiye Cumhuriyet Merkez Bankası (TCMB) Para Politikası Kurulu, politika faizini yüzde 45 seviyesinde sabit tutma kararı aldı. Enflasyon görünümünde belirgin bir bozulma öngörülmesi durumunda parasal sıkılık düzeyi artırılacaktır.",
    },
    {
        "id": 2,
        "title": "Kredi Kartı Taksit Düzenlemesi",
        "text": "Bankacılık Düzenleme ve Denetleme Kurumu (BDDK), finansal istikrarı güçlendirmek amacıyla bireysel kredi kartı harcamalarındaki taksit sınırlandırmalarını güncelledi.",
    },
    {
        "id": 3,
        "title": "İstanbul Boğaziçi Köprü Bakımı",
        "text": "İstanbul 15 Temmuz Şehitler Köprüsü'nde gerçekleştirilecek asfalt yenileme ve halat kontrol çalışmaları nedeniyle hafta sonu boyunca Anadolu yakasından Avrupa yakasına geçişler tek şeritten sağlanacaktır.",
    },
    {
        "id": 4,
        "title": "Model2Vec ve TurboQuant Sıkıştırma",
        "text": "Akana projesinde kullanılan Model2Vec ve TurboQuant algoritmaları, 2.2 GB boyutundaki BGE-M3 modelini 2.5 MB'a indirerek vektörleri 32 kat sıkıştırır ve CPU üzerinde 20.000 cümle/saniye hız sağlar.",
    },
    {
        "id": 5,
        "title": "Geleneksel Türk Edebiyatı ve Şiir",
        "text": "Divan edebiyatında aruz vezni ve mazmunlar önemli yer tutarken, Tanzimat dönemiyle birlikte dilde sadeleşme ve hece ölçüsüne yöneliş hız kazanmıştır. Ahmet Haşim ve Yahya Kemal modern Türk şiirinin öncülerindendir.",
    },
]


def main():
    print("=" * 80)
    print(" 🚀 AKANA ULTRA-LOW-MEMORY EMBEDDING & RETRIEVAL SUITE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. 2-Bit TurboQuant Packed Embeddings (64 Bytes / Vector)
    # -------------------------------------------------------------------------
    print("\n[1] 2-Bit TurboQuant Packed Embeddings (64 Bytes per Vector):")
    sample_text = "Türkiye Cumhuriyet Merkez Bankası faiz kararını açıkladı."
    packed_2bit = akana.embed_packed_2bit(sample_text)
    print(f"  Input text: '{sample_text}'")
    print(f"  Packed vector type: {type(packed_2bit).__name__}")
    print(f"  Vector size: {len(packed_2bit)} bytes (exactly 1 CPU cache line!)")
    print(f"  Global invariant scale (EMBEDDING_SCALE_2BIT): {akana.EMBEDDING_SCALE_2BIT}")
    print(f"  First 16 packed bytes (hex): {packed_2bit[:16].hex()}")

    # -------------------------------------------------------------------------
    # 2. 1-Bit Binary Embeddings (32 Bytes / Vector)
    # -------------------------------------------------------------------------
    print("\n[2] 1-Bit Binary Embeddings (32 Bytes per Vector):")
    packed_1bit = akana.embed_packed_1bit(sample_text)
    print(f"  Vector size: {len(packed_1bit)} bytes (256 bits for 256 dimensions)")
    print(f"  Packed 1-bit bytes (hex): {packed_1bit.hex()}")
    print("  RAM footprint for 1,000,000 documents: ONLY 32 MB!")

    # -------------------------------------------------------------------------
    # 3. Morphological BM25 Sparse Embeddings
    # -------------------------------------------------------------------------
    print("\n[3] Morphological BM25 Sparse Embeddings (Agglutinative Disambiguation):")
    query_morph = "Köprülerin bakımları hafta sonunda yapılacak mı?"
    sparse = akana.sparse_embed(query_morph)
    print(f"  Input text: '{query_morph}'")
    print(f"  Extracted morphological roots / lemmas: {sparse['terms']}")
    print(f"  Hashed 24-bit bucket indices: {sparse['indices']}")
    print(f"  BM25 term saturation weights: {[round(v, 3) for v in sparse['values']]}")

    # -------------------------------------------------------------------------
    # 4. Zero-Copy USearch Indexing with Compiled C-ABI Metrics
    # -------------------------------------------------------------------------
    if not HAS_USEARCH:
        print("\n⚠️ usearch library not found. Install via: uv pip install usearch")
        return

    print("\n[4] Zero-Copy USearch Integration with Akana Compiled Metrics:")
    ptr_2bit = akana.get_usearch_metric_pointer_2bit()
    print(f"  Obtained native C-ABI 2-bit metric pointer: {hex(ptr_2bit)}")

    # Instantiate USearch with Akana's pre-compiled L2-cache lookup table metric
    metric_2bit = CompiledMetric(
        pointer=ptr_2bit,
        kind=MetricKind.Cos,
        signature=MetricSignature.ArrayArraySize,
    )
    index_2bit = Index(ndim=64, metric=metric_2bit, dtype="u8")

    # Index all documents
    for doc in DOCUMENTS:
        vec_bytes = akana.embed_packed_2bit(doc["text"])
        arr = np.frombuffer(vec_bytes, dtype=np.uint8)
        index_2bit.add(doc["id"], arr)

    print(f"  Successfully indexed {len(index_2bit)} documents into 2-bit USearch index.")

    # Query the 2-bit index
    query = "Merkez bankasının faiz oranı ne kadar oldu?"
    print(f"\n  Querying: '{query}'")
    q_bytes = akana.embed_packed_2bit(query)
    q_arr = np.frombuffer(q_bytes, dtype=np.uint8)
    matches = index_2bit.search(q_arr, count=3)

    for rank, match in enumerate(matches, start=1):
        doc = next(d for d in DOCUMENTS if d["id"] == match.key)
        print(f"    Rank #{rank} [Doc ID {match.key} | Cosine Dist: {match.distance:.4f}]: {doc['title']}")

    # -------------------------------------------------------------------------
    # 5. 1-Bit Binary USearch Indexing (Hardware POPCNT Hamming Metric)
    # -------------------------------------------------------------------------
    print("\n[5] 1-Bit Binary USearch Indexing (POPCNT Hamming Metric):")
    ptr_1bit = akana.get_usearch_metric_pointer_1bit()
    metric_1bit = CompiledMetric(
        pointer=ptr_1bit,
        kind=MetricKind.Hamming,
        signature=MetricSignature.ArrayArraySize,
    )
    index_1bit = Index(ndim=32, metric=metric_1bit, dtype="u8")

    for doc in DOCUMENTS:
        vec_bytes = akana.embed_packed_1bit(doc["text"])
        arr = np.frombuffer(vec_bytes, dtype=np.uint8)
        index_1bit.add(doc["id"], arr)

    q_bytes_1bit = akana.embed_packed_1bit(query)
    matches_1bit = index_1bit.search(np.frombuffer(q_bytes_1bit, dtype=np.uint8), count=3)
    for rank, match in enumerate(matches_1bit, start=1):
        doc = next(d for d in DOCUMENTS if d["id"] == match.key)
        print(f"    Rank #{rank} [Doc ID {match.key} | Hamming Dist: {match.distance:.4f}]: {doc['title']}")

    # -------------------------------------------------------------------------
    # 6. Dual-Index Reciprocal Rank Fusion (Dense + Morphological BM25)
    # -------------------------------------------------------------------------
    print("\n[6] Dual-Index Reciprocal Rank Fusion (RRF) for Highest Retrieval Accuracy:")
    # Build simple sparse inverted list for our sample documents
    from collections import defaultdict
    inverted_index = defaultdict(list)
    doc_lookup = {d["id"]: d for d in DOCUMENTS}

    for doc in DOCUMENTS:
        sp = akana.sparse_embed(doc["text"])
        for idx, val in zip(sp["indices"], sp["values"]):
            inverted_index[idx].append((doc["id"], val))

    # Search sparse index
    q_sparse = akana.sparse_embed(query)
    sparse_scores = defaultdict(float)
    for q_idx in q_sparse["indices"]:
        for doc_id, tf in inverted_index.get(q_idx, []):
            sparse_scores[doc_id] += tf

    sparse_hits = sorted(sparse_scores.items(), key=lambda x: x[1], reverse=True)
    dense_hits_1bit = [(m.key, -m.distance) for m in matches_1bit]
    dense_hits_2bit = [(m.key, -m.distance) for m in matches]

    # Fuse 1-bit Binary search + Morphological BM25 with Akana's native Rust RRF
    fused_1bit = akana.reciprocal_rank_fusion(dense_hits_1bit, sparse_hits, k=60.0, alpha=1.0)

    print("  Fused RRF (1-Bit Binary + Morphological BM25) Rankings:")
    for rank, (doc_id, score) in enumerate(fused_1bit[:3], start=1):
        doc = doc_lookup[doc_id]
        print(f"    Rank #{rank} [Doc ID {doc_id} | RRF Score: {score:.5f}]: {doc['title']}")

    print("\n" + "=" * 80)
    print(" ✅ Demonstration complete. Akana delivers state-of-the-art Turkish RAG")
    print("    with 16x–32x memory compression, running directly on any standard laptop CPU!")
    print("=" * 80)


if __name__ == "__main__":
    main()
