"""
Integration tests for Akana's low-memory packed embeddings and compiled USearch metrics.
"""

import pytest
import akana

try:
    import numpy as np
    from usearch.index import CompiledMetric, Index, MetricKind, MetricSignature
    HAS_USEARCH = True
except ImportError:
    HAS_USEARCH = False
    np = None


def _byte_index(nbytes, metric):
    """USearch index that stores Akana's packed vectors byte for byte.

    USearch reads a numpy uint8 array as packed bits: into a `u8` or `i8` index it unpacks the
    first `ndim` bits into 0/1 bytes, so `Index(ndim=nbytes, dtype="u8")` keeps 1/8 of each vector.
    A `b1` index of `8 * nbytes` dimensions stores the bytes as given, and the compiled metric
    receives `nbytes` as its size argument.
    """
    return Index(ndim=8 * nbytes, metric=metric, dtype="b1")


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

    # One lemma per word in context: 'kitap', 'defter', 'masa', and the verb's lemma 'bırakmak'
    terms_set = set(terms)
    assert "kitap" in terms_set
    assert "defter" in terms_set
    assert "bırakmak" in terms_set
    assert "bırak" not in terms_set

    # 32-bit term ids, never 0 or 2**32 - 1, ascending
    for idx in indices:
        assert 0 < idx < (1 << 32) - 1
    assert indices == sorted(indices)

    # One term per word occurrence: "kitap" once, not once per reading of "Kitaplarımı"
    res = akana.sparse_embed("Kitaplarımı okudum. Kitap güzeldi.")
    once = akana.sparse_embed("Kitaplarımı okudum.")
    kitap = dict(zip(res["terms"], res["values"]))["kitap"]
    kitap_once = dict(zip(once["terms"], once["values"]))["kitap"]
    # BM25 saturation of tf 2 over tf 1: (2 * 2.2 / 3.2) / 1
    assert kitap / kitap_once == pytest.approx(1.375, rel=1e-4)

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


@pytest.mark.skipif(not HAS_USEARCH, reason="usearch or numpy not installed")
def test_usearch_2bit_compiled_metric():
    ptr = akana.get_usearch_metric_pointer_2bit()
    assert ptr > 0

    metric = CompiledMetric(
        pointer=ptr,
        kind=MetricKind.Cos,
        signature=MetricSignature.ArrayArraySize,
    )
    index = _byte_index(64, metric)

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


@pytest.mark.skipif(not HAS_USEARCH, reason="usearch or numpy not installed")
def test_usearch_1bit_compiled_metric():
    ptr = akana.get_usearch_metric_pointer_1bit()
    assert ptr > 0

    metric = CompiledMetric(
        pointer=ptr,
        kind=MetricKind.Hamming,
        signature=MetricSignature.ArrayArraySize,
    )
    index = _byte_index(32, metric)

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


@pytest.mark.skipif(not HAS_USEARCH, reason="usearch or numpy not installed")
def test_usearch_hybrid_compiled_metric():
    ptr = akana.get_usearch_metric_pointer_hybrid(0.5)
    assert ptr > 0

    metric = CompiledMetric(
        pointer=ptr,
        kind=MetricKind.Cos,
        signature=MetricSignature.ArrayArraySize,
    )
    index = _byte_index(128, metric)

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


@pytest.mark.skipif(not HAS_USEARCH, reason="usearch or numpy not installed")
@pytest.mark.parametrize(
    "embed, nbytes, pointer",
    [
        (akana.embed_packed_2bit, 64, akana.get_usearch_metric_pointer_2bit),
        (akana.embed_packed_1bit, 32, akana.get_usearch_metric_pointer_1bit),
        (akana.embed_hybrid, 128, lambda: akana.get_usearch_metric_pointer_hybrid(0.5)),
    ],
)
def test_usearch_stores_vectors_verbatim_and_uses_the_akana_metric(embed, nbytes, pointer):
    """The index must hold the exact bytes Akana produced, and USearch's distances must be the
    ones the compiled metric computes on those bytes (a self-match alone cannot show either)."""
    import ctypes

    ptr = pointer()
    rust_metric = ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)(ptr)
    index = _byte_index(nbytes, CompiledMetric(pointer=ptr, kind=MetricKind.Cos,
                                               signature=MetricSignature.ArrayArraySize))
    corpus = [
        "Davacının tazminat talebi kısmen kabul edildi.",
        "Ankara'da kar yağışı etkili oldu.",
        "Kira bedelinin uyarlanması davası açıldı.",
        "Yapay zeka modelleri hızla gelişiyor.",
    ]
    vectors = [np.frombuffer(embed(t), dtype=np.uint8) for t in corpus]
    for key, v in enumerate(vectors):
        index.add(key, v)
        assert np.array_equal(np.asarray(index.get(key)).ravel(), v)

    q = np.frombuffer(embed("tazminat davası"), dtype=np.uint8).copy()
    for m in index.search(q, count=len(corpus), exact=True):
        v = np.ascontiguousarray(vectors[int(m.key)])
        assert m.distance == pytest.approx(rust_metric(q.ctypes.data, v.ctypes.data, nbytes), abs=1e-6)


@pytest.mark.skipif(not HAS_USEARCH, reason="usearch or numpy not installed")
@pytest.mark.parametrize("view", [False, True])
def test_reloaded_index_needs_the_compiled_metric_again(tmp_path, view):
    """load()/view() restore the metric from the file header, which holds no function pointer;
    setting index.metric afterwards brings back Akana's metric (searching without it crashes)."""
    metric = CompiledMetric(pointer=akana.get_usearch_metric_pointer_hybrid(0.5), kind=MetricKind.Cos,
                            signature=MetricSignature.ArrayArraySize)
    built = _byte_index(128, metric)
    texts = ["Taşınmaz davacıya aittir.", "Vergi ziyaı cezası kaldırıldı.", "Makul sürede yargılanma hakkı ihlal edildi."]
    for key, t in enumerate(texts):
        built.add(key, np.frombuffer(akana.embed_hybrid(t), dtype=np.uint8))
    path = str(tmp_path / "hybrid.usearch")
    built.save(path)

    q = np.frombuffer(akana.embed_hybrid("vergi cezası"), dtype=np.uint8)
    expected = [(int(m.key), float(m.distance)) for m in built.search(q, count=3, exact=True)]

    reloaded = _byte_index(128, metric)
    (reloaded.view if view else reloaded.load)(path)
    reloaded.metric = metric
    got = [(int(m.key), float(m.distance)) for m in reloaded.search(q, count=3, exact=True)]
    assert [k for k, _ in got] == [k for k, _ in expected]
    assert [d for _, d in got] == pytest.approx([d for _, d in expected], abs=1e-6)
    del reloaded  # release the memory map before tmp_path is removed
