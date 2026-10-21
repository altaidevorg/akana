"""
Binding-level behaviour: character offsets, panics, import errors, GIL release and engine
reuse.
"""

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import akana

SAMPLE = open(
    os.path.join(os.path.dirname(__file__), "..", "data", "sample_iki_horoz.txt"),
    encoding="utf-8",
).read()


# --- Character offsets -------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Sayın Zxqy Wqerty konuştu.",
        "İğdır'da Şükrü Çelik çalışıyor.",
        "Şükrü Öztürk 15 Ocak 2024 tarihinde İstanbul'a gitti ve %20 indirim aldı, 500 TL ödedi.",
    ],
)
def test_entity_offsets_slice_the_text(text):
    entities = akana.extract_entities(text)
    assert entities
    for e in entities:
        assert text[e.start : e.end] == e.text


def test_entity_offsets_after_non_ascii():
    text = "Sayın Zxqy Wqerty konuştu."
    names = [e for e in akana.extract_entities(text) if e.text == "Zxqy Wqerty"]
    assert [(e.start, e.end) for e in names] == [(6, 17)]


def test_grammar_offsets_after_non_ascii():
    text = "Şükrü geldi, Ali de geldi, Veli te geldi."
    [finding] = akana.check_grammar(text).findings
    assert (finding.start_offset, finding.end_offset) == (32, 34)
    assert text[32:34] == "te"


@pytest.mark.parametrize(
    "text",
    [
        "Çiçekler'de güzel, Ünal'da geldi mi?",
        "Şükrü geldi, Ali de geldi, Veli te geldi.",
        "Öğretmen geldi. Evi terketmek zorunda kaldı ve 1923'de kurulan cumhuriyeti andık.",
    ],
)
def test_grammar_offsets_slice_the_text(text):
    results = [
        akana.check_grammar(text),
        akana.GrammarCheckResult(json.loads(akana.GrammarChecker().check_json(text))),
    ]
    for result in results:
        assert result.findings
        for f in result.findings:
            assert text[f.start_offset : f.end_offset] == f.original_text


def test_style_offsets_slice_the_text():
    text = "Şükrü çığlık attı. Bu bağlamda öğrenciler gelmiştir."
    findings = [f for f in akana.audit_ai_style(text).findings if f.end > f.start]
    assert findings
    spans = [text[f.start : f.end] for f in findings]
    # As byte offsets this span read 'mda öğrencil'.
    assert "Bu bağlamda" in spans, spans


def test_pii_and_chunk_offsets_slice_the_text():
    text = "Şükrü Öztürk'ün telefonu 0532 123 45 67, e-postası şükrü@örnek.com.tr değil sukru@gmail.com."
    entities = akana.pii_mask(text)["entities"]
    assert entities
    for e in entities:
        assert text[e["start"] : e["end"]] == e["text"]
    text = "Şükrü İğdır'da çalışıyor. Öğretmenler çok çalışkan. Çiçekler açtı."
    for chunk in akana.chunk_sentences(text, chunk_size=8):
        assert text[chunk.start_index : chunk.end_index] == chunk.text


# --- Errors --------------------------------------------------------------------------------


def test_panic_raises_runtime_error():
    with pytest.raises(RuntimeError, match="boom") as info:
        akana._core._panic_for_tests("boom")
    assert isinstance(info.value, Exception)
    # The module keeps working after a panic.
    assert akana.stem("kitaplar") == "kitap"


def test_failed_native_import_is_raised():
    code = "import sys; sys.modules['akana._core'] = None; import akana"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode != 0
    assert "ImportError" in proc.stderr
    assert "akana._core" in proc.stderr


# --- GIL release ----------------------------------------------------------------------------


def _long_call(make_call, min_seconds=0.15):
    """A call whose single native call takes at least `min_seconds`, by doubling its input."""
    scale = 1
    while True:
        call = make_call(scale)
        t0 = time.perf_counter()
        call()
        if time.perf_counter() - t0 >= min_seconds or scale >= 1024:
            return call
        scale *= 2


def _python_runs_during(call):
    """How many times the main thread ran Python code in the middle of `call`, running in
    another thread. A call that holds the GIL lets it run none."""
    window = {}

    def worker():
        window["start"] = time.perf_counter()
        call()
        window["end"] = time.perf_counter()

    ticks = []
    t = threading.Thread(target=worker)
    t.start()
    while t.is_alive():
        ticks.append(time.perf_counter())
    t.join()
    start, end = window["start"], window["end"]
    # Leave out the edges, where the worker runs Python code around the native call and
    # can hand the GIL over at the switch interval.
    margin = (end - start) * 0.2
    return sum(1 for x in ticks if start + margin < x < end - margin)


def test_embed_batch_releases_the_gil():
    # Under the threading threshold, so the call runs on the worker thread alone.
    call = _long_call(lambda k: (lambda: akana.embed_batch([SAMPLE * k] * 200)))
    assert _python_runs_during(call) > 0


def test_pii_mask_releases_the_gil():
    engine = akana.TurkishPiiEngine()
    call = _long_call(lambda k: (lambda: engine.mask(SAMPLE * k)))
    assert _python_runs_during(call) > 0


def test_threads_get_the_serial_results():
    texts = [SAMPLE[i : i + 300] for i in range(0, len(SAMPLE) - 300, 50)]
    engine = akana.TurkishPiiEngine(use_embeddings=True)
    checker = akana.GrammarChecker()
    work = [
        lambda: akana.embed_batch(texts),
        lambda: [engine.mask(t)["masked_text"] for t in texts],
        lambda: [json.loads(checker.check_json(t))["findings"] for t in texts[:20]],
        lambda: akana.audit_ai_style_json(SAMPLE),
        lambda: [c.to_dict() for c in akana.chunk_semantic(SAMPLE)],
    ]
    expected = [f() for f in work]
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda i: work[i % len(work)](), range(4 * len(work))))
    for i, r in enumerate(results):
        assert r == expected[i % len(work)]


def test_large_batches_match_single_embeddings():
    # Large enough to be split across threads.
    texts = [f"{w} {i}" for i, w in enumerate(SAMPLE.split())][:1500]
    assert akana.embed_batch(texts) == [akana.embed(t) for t in texts]
    assert akana.embed_batch_hybrid(texts) == [akana.embed_hybrid(t) for t in texts]
    assert akana.embed_batch_packed_2bit(texts) == [akana.embed_packed_2bit(t) for t in texts]


# --- Shared engines ---------------------------------------------------------------------------


def test_user_dictionary_stays_with_its_morphology():
    custom = akana.Morphology()
    custom.load_dictionary_str("zxqy zxqy Noun")
    assert any(p["root"] == "zxqy" for p in custom.analyze("zxqyler"))
    assert not any(p["root"] == "zxqy" for p in akana.Morphology().analyze("zxqyler"))


def test_module_functions_reuse_engines():
    # Rebuilding the engine cost 100-1400 ms per call; reusing it costs well under a
    # millisecond here. The bound is loose so that slow debug builds pass.
    calls = [
        lambda: akana.analyze("Ali eve gitti."),
        lambda: akana.stem("kitaplar"),
        lambda: akana.audit_ai_style("Ali eve gitti."),
        lambda: akana.analyze_readability("Ali eve gitti."),
        lambda: akana.syntactic_analyze("evlerde"),
    ]
    for call in calls:
        call()
        t0 = time.perf_counter()
        for _ in range(5):
            call()
        assert (time.perf_counter() - t0) / 5 < 0.05
