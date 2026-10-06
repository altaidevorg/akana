"""
Integration tests for Akana's low-memory packed embeddings and compiled USearch metrics.
"""

import numpy as np
import pytest
import akana

try:
    from usearch.index import CompiledMetric, Index, MetricKind, MetricSignature
    HAS_USEARCH = True
except ImportError:
    HAS_USEARCH = False


def test_packed_2bit_constants_and_types():
    assert akana.PACKED_2BIT_BYTES == 64
    assert akana.PACKED_1BIT_BYTES == 32
    assert akana.HYBRID_VECTOR_BYTES == 128
    assert akana.EMBEDDING_SCALE_2BIT == pytest.approx(0.0625, rel=1e-5)

    vec = akana.embed_packed_2bit("Türkiye'nin başkenti Ankara'dır.")
    assert isinstance(vec, bytes)
    assert len(vec) == 64

    batch = akana.embed_batch_packed_2bit(["Metin 1", "Metin 2", "Metin 3"])
    assert len(batch) == 3
    for b in batch:
        assert isinstance(b, bytes)
        assert len(b) == 64


def test_packed_1bit_embedding():
    vec = akana.embed_packed_1bit("Yapay zeka ve doğal dil işleme.")
    assert isinstance(vec, bytes)
    assert len(vec) == 32

    batch = akana.embed_batch_packed_1bit(["Cümle A", "Cümle B"])
    assert len(batch) == 2
    for b in batch:
        assert isinstance(b, bytes)
        assert len(b) == 32


def test_sparse_morphological_embedding():
    res = akana.sparse_embed("Kitaplarımı ve defterlerimi çalışma masasına bıraktım.")
    assert "indices" in res
    assert "values" in res
    assert "terms" in res

    indices = res["indices"]
    values = res["values"]
    terms = res["terms"]

    assert len(indices) == len(values) == len(terms)
    assert len(indices) > 0

    # Ensure terms include base roots / lemmas like 'kitap', 'defter', 'masa', 'bırak'
    terms_set = set(terms)
    assert "kitap" in terms_set
    assert "defter" in terms_set
    assert "bırak" in terms_set

    # Verify 24-bit bounds
    for idx in indices:
        assert 0 <= idx < (1 << 24)

    # Verify positive BM25 TF values
    for val in values:
        assert val > 0.0


def test_hybrid_embedding_and_alpha():
    vec = akana.embed_hybrid("İstanbul Boğazı'nda vapur seferleri aksadı.")
    assert isinstance(vec, bytes)
    assert len(vec) == 128

    batch = akana.embed_batch_hybrid(["Test 1", "Test 2"])
    assert len(batch) == 2
    for b in batch:
        assert isinstance(b, bytes)
        assert len(b) == 128

    # Test alpha setting and getting
    prev_alpha = akana.get_hybrid_metric_alpha()
    akana.set_hybrid_metric_alpha(0.65)
    assert akana.get_hybrid_metric_alpha() == pytest.approx(0.65, rel=1e-4)
    akana.set_hybrid_metric_alpha(prev_alpha)


def test_reciprocal_rank_fusion():
    dense_results = [(101, 0.95), (102, 0.85), (103, 0.70)]
    sparse_results = [(102, 12.5), (104, 11.0), (101, 5.0)]

    fused = akana.reciprocal_rank_fusion(dense_results, sparse_results, k=60.0, alpha=1.0)
    assert len(fused) == 4

    fused_dict = dict(fused)
    # 102 is ranked 2nd in dense (rank 1) and 1st in sparse (rank 0) -> should be top ranked
    # 101 is ranked 1st in dense (rank 0) and 3rd in sparse (rank 2)
    top_id, top_score = fused[0]
    assert top_id in (101, 102)
    assert fused_dict[102] > fused_dict[103]
    assert fused_dict[102] > fused_dict[104]


@pytest.mark.skipif(not HAS_USEARCH, reason="usearch not installed")
def test_usearch_2bit_compiled_metric():
    ptr = akana.get_usearch_metric_pointer_2bit()
    assert ptr > 0

    metric = CompiledMetric(
        pointer=ptr,
        kind=MetricKind.Cos,
        signature=MetricSignature.ArrayArraySize,
    )
    index = Index(ndim=64, metric=metric, dtype="u8")

    corpus = [
        "Türkiye'nin başkenti Ankara'dır ve İç Anadolu bölgesindedir.",
        "İstanbul, Türkiye'nin en kalabalık ve tarihi şehridir.",
        "Derin öğrenme modelleri doğal dil işlemede çığır açtı.",
        "Akana kütüphanesi çok hızlı ve hafif Türkçe dil araçları sunar.",
        "Güneşli bir günde ormanda yürüyüş yapmak ruha iyi gelir.",
    ]

    for idx, text in enumerate(corpus):
        raw_bytes = akana.embed_packed_2bit(text)
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        index.add(idx, arr)

    assert len(index) == len(corpus)

    # Query with exact first sentence
    q_bytes = akana.embed_packed_2bit(corpus[0])
    q_arr = np.frombuffer(q_bytes, dtype=np.uint8)
    matches = index.search(q_arr, count=3)

    assert len(matches) > 0
    top_match = matches[0]
    assert top_match.key == 0
    # Exact vector cosine distance should be ~0.0
    assert top_match.distance == pytest.approx(0.0, abs=1e-3)


@pytest.mark.skipif(not HAS_USEARCH, reason="usearch not installed")
def test_usearch_1bit_compiled_metric():
    ptr = akana.get_usearch_metric_pointer_1bit()
    assert ptr > 0

    metric = CompiledMetric(
        pointer=ptr,
        kind=MetricKind.Hamming,
        signature=MetricSignature.ArrayArraySize,
    )
    index = Index(ndim=32, metric=metric, dtype="u8")

    corpus = [
        "Kedi koltuğun üzerinde uyuyor.",
        "Köpek bahçede neşeyle koşuyor.",
        "Python programlama dili veri biliminde yaygındır.",
    ]

    for idx, text in enumerate(corpus):
        raw_bytes = akana.embed_packed_1bit(text)
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        index.add(idx, arr)

    q_bytes = akana.embed_packed_1bit(corpus[1])
    q_arr = np.frombuffer(q_bytes, dtype=np.uint8)
    matches = index.search(q_arr, count=2)

    assert len(matches) > 0
    assert matches[0].key == 1
    assert matches[0].distance == pytest.approx(0.0, abs=1e-3)


@pytest.mark.skipif(not HAS_USEARCH, reason="usearch not installed")
def test_usearch_hybrid_compiled_metric():
    ptr = akana.get_usearch_metric_pointer_hybrid(0.5)
    assert ptr > 0

    metric = CompiledMetric(
        pointer=ptr,
        kind=MetricKind.Cos,
        signature=MetricSignature.ArrayArraySize,
    )
    index = Index(ndim=128, metric=metric, dtype="u8")

    corpus = [
        "Galatasaray ile Fenerbahçe arasındaki derbi maçı nefes kesti.",
        "Merkez Bankası faiz kararını açıkladı ve piyasalar hareketlendi.",
        "Uzay araştırmaları için yeni nesil roket motorları geliştiriliyor.",
    ]

    for idx, text in enumerate(corpus):
        raw_bytes = akana.embed_hybrid(text)
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        index.add(idx, arr)

    # Query with morphological overlap and semantic match
    q_text = "Fenerbahçe Galatasaray derbisi ne zaman oynandı?"
    q_bytes = akana.embed_hybrid(q_text)
    q_arr = np.frombuffer(q_bytes, dtype=np.uint8)

    matches = index.search(q_arr, count=2)
    assert len(matches) > 0
    assert matches[0].key == 0
