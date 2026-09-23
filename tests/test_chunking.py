"""Tests for Akana Semantic, Sentence, and SDPM Chunkers."""

import akana


def test_sentence_chunker():
    chunker = akana.SentenceChunker(chunk_size=20, chunk_overlap=0)
    text = (
        "Ankara Türkiye'nin başkentidir. "
        "İstanbul ise en kalabalık şehridir. "
        "İzmir Ege'nin incisidir. "
        "Antalya Akdeniz'in turizm başkentidir."
    )
    chunks = chunker(text)

    assert len(chunks) >= 2
    for c in chunks:
        assert isinstance(c, akana.Chunk)
        assert c.token_count > 0
        assert text[c.start_index : c.end_index] == c.text
        assert len(c.sentences) > 0

    assert chunks[0].start_index == 0
    assert chunks[-1].end_index == len(text)


def test_chunk_attributes_and_dict():
    chunker = akana.SentenceChunker(chunk_size=50)
    text = "Birinci test cümlesi. İkinci test cümlesi."
    chunks = chunker.chunk(text)

    assert len(chunks) == 1
    c = chunks[0]
    assert c.text == text
    assert c.start_index == 0
    assert c.end_index == len(text)
    assert c.token_count > 0
    assert len(c.sentences) == 2

    d = c.to_dict()
    assert d["text"] == text
    assert d["start_index"] == 0
    assert d["end_index"] == len(text)
    assert d["token_count"] == c.token_count
    assert d["sentences"] == ["Birinci test cümlesi.", "İkinci test cümlesi."]


def test_semantic_chunker_multi_topic():
    chunker = akana.SemanticChunker(
        chunk_size=512,
        threshold_mode="percentile",
        threshold_value=0.75,
        min_chunk_size=5,
    )
    text = (
        "Kuantum fiziği, atom ve atom altı parçacıkların davranışlarını inceler. "
        "Dalga-parçacık ikiliği kuantum kuramının merkezindedir. "
        "Fenerbahçe futbol takımı dün akşam şampiyonluk yarışında kritik bir galibiyet aldı. "
        "Teknik direktör oyuncuların sahadaki performansından memnun olduğunu belirtti. "
        "Geleneksel Türk mutfağında kuru fasulye ve pilav vazgeçilmez bir ikilidir. "
        "Yanında turşu ve ayran ile servis edilir."
    )
    chunks = chunker.chunk(text)

    assert len(chunks) >= 2
    for c in chunks:
        assert text[c.start_index : c.end_index] == c.text
        assert c.token_count > 0

    # Ensure topic isolation
    first_chunk = chunks[0].text
    assert "Kuantum" in first_chunk
    assert "mutfağında" not in first_chunk


def test_semantic_chunker_threshold_modes():
    text = (
        "Yapay zeka modelleri doğal dil işlemede çığır açtı. "
        "Büyük dil modelleri insan benzeri metinler üretebiliyor. "
        "Uzay araştırmalarında Mars yüzeyinde yeni mineral keşifleri yapıldı. "
        "Rover araçları kayaç örnekleri toplamaya devam ediyor."
    )

    for mode in ["percentile", "similarity", "stdev", "iqr", "auto"]:
        chunker = akana.SemanticChunker(
            chunk_size=256, threshold_mode=mode, min_chunk_size=5
        )
        chunks = chunker(text)
        assert len(chunks) >= 1
        for c in chunks:
            assert text[c.start_index : c.end_index] == c.text


def test_sdpm_chunker():
    chunker = akana.SDPMChunker(
        chunk_size=512,
        threshold_mode="similarity",
        threshold=0.85,
        merge_threshold=0.50,
    )
    text = (
        "Derin öğrenme yapay sinir ağlarına dayanır. "
        "Geriye yayılım algoritması ağırlıkları günceller. "
        "Güneş sisteminde sekiz gezegen bulunmaktadır. "
        "Merkür güneşe en yakın gezegendir."
    )
    chunks = chunker.chunk(text)

    assert len(chunks) >= 1
    for c in chunks:
        assert text[c.start_index : c.end_index] == c.text
        assert c.token_count <= 512


def test_standalone_chunk_functions():
    text = "Birinci cümle burada. İkinci cümle burada. Üçüncü cümle burada."

    chunks_sem = akana.chunk_semantic(text, chunk_size=256)
    assert len(chunks_sem) >= 1

    chunks_sent = akana.chunk_sentences(text, chunk_size=256)
    assert len(chunks_sent) >= 1

    chunks_sdpm = akana.chunk_sdpm(text, chunk_size=256)
    assert len(chunks_sdpm) >= 1


def test_chunk_batch():
    chunker = akana.SemanticChunker(chunk_size=256)
    texts = [
        "Birinci belgenin birinci cümlesi. Birinci belgenin ikinci cümlesi.",
        "İkinci belgenin birinci cümlesi. İkinci belgenin ikinci cümlesi.",
    ]
    batch_chunks = chunker.chunk_batch(texts)

    assert len(batch_chunks) == 2
    assert len(batch_chunks[0]) >= 1
    assert len(batch_chunks[1]) >= 1
