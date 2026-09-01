"""Comprehensive and detailed tests for Akana Chunking Suite."""

import pytest
import akana


def test_chunking_unicode_and_emojis():
    text = (
        "🇹🇷 Türkiye'nin yerli elektrikli otomobili Togg yollarda! 🚗⚡ "
        "Batarya teknolojisi ve menzil performansı kullanıcılar tarafından beğenildi. 🔋 "
        "İstanbul ve Ankara gibi büyükşehirlerde şarj istasyonları hızla yaygınlaşıyor. 🔌"
    )
    chunker = akana.SemanticChunker(chunk_size=256, threshold_mode="percentile", threshold_value=0.75)
    chunks = chunker(text)

    assert len(chunks) >= 1
    for c in chunks:
        assert text[c.start_index:c.end_index] == c.text
        assert c.token_count > 0


def test_chunking_numbers_abbreviations_and_quotes():
    text = (
        "Prof. Dr. Mehmet Özdemir ve Doç. Dr. Selin Vural saat 10.45'te 2. toplantı salonunda buluştu. "
        "Toplantıda \"Türkiye Yüzyılı\" vizyonu ve Ar-Ge projeleri ele alındı. "
        "Bütçe olarak 1.500.000 TL ayrılması kararlaştırıldı. "
        "Resmi duyuru www.resmigazete.gov.tr üzerinden ilan edilecek."
    )
    chunker = akana.SentenceChunker(chunk_size=100)
    chunks = chunker(text)

    assert len(chunks) >= 1
    for c in chunks:
        assert text[c.start_index:c.end_index] == c.text
        assert c.token_count > 0


def test_chunking_large_multi_domain_corpus():
    domains = [
        # Domain 1: Quantum Physics
        "Kuantum teorisi mikroskobik dünyadaki fiziksel olayları betimler. "
        "Heisenberg belirsizlik ilkesi bir parçacığın konum ve momentumunun aynı anda kesin olarak ölçülemeyeceğini belirtir. "
        "Schrödinger dalga denklemi kuantum durumlarının zamana bağlı evrimini tanımlar.",
        # Domain 2: Turkish History
        "Kurtuluş Savaşı 1919 yılında Mustafa Kemal Atatürk'ün Samsun'a çıkışıyla başlamıştır. "
        "Büyük Millet Meclisi 23 Nisan 1920'de Ankara'da açılarak milli iradeyi temsil etmiştir. "
        "29 Ekim 1923 tarihinde Türkiye Cumhuriyeti resmen ilan edilmiştir.",
        # Domain 3: Turkish Cuisine
        "Gaziantep mutfağı zengin etli yemekleri ve tatlılarıyla UNESCO tescillidir. "
        "Ali Nazik kebabı ve beyran çorbası kentin en bilinen simgelerindendir. "
        "Fıstıklı baklava çıtır yufkası ve doğal şerbetiyle eşsiz bir lezzet sunar.",
        # Domain 4: Macroeconomics
        "Merkez bankaları enflasyon oranını kontrol altında tutmak için politika faizini belirler. "
        "Cari açık ve dış ticaret dengesi döviz kuru hareketlerinde belirleyici rol oynar. "
        "Doğrudan yabancı yatırımlar ekonomik büyümeyi ve istihdamı destekler.",
    ]

    full_document = "\n\n".join(domains)

    # 1. Semantic Chunker
    semantic_chunker = akana.SemanticChunker(
        chunk_size=512,
        threshold_mode="percentile",
        threshold_value=0.75,
        min_chunk_size=10,
    )
    sem_chunks = semantic_chunker(full_document)

    assert len(sem_chunks) >= 3
    for c in sem_chunks:
        assert full_document[c.start_index:c.end_index] == c.text
        assert c.token_count > 0

    # Verify domain isolation in chunks
    assert "Kuantum" in sem_chunks[0].text
    assert "Mustafa Kemal" in full_document
    assert any("Gaziantep" in c.text or "kebap" in c.text for c in sem_chunks)
    assert any("enflasyon" in c.text or "Merkez bankaları" in c.text for c in sem_chunks)

    # 2. SDPM Chunker
    sdpm_chunker = akana.SDPMChunker(
        chunk_size=512,
        threshold_mode="percentile",
        threshold_value=0.75,
        merge_threshold=0.60,
    )
    sdpm_chunks = sdpm_chunker(full_document)
    assert len(sdpm_chunks) >= 2
    for c in sdpm_chunks:
        assert full_document[c.start_index:c.end_index] == c.text
        assert c.token_count <= 512


def test_chunker_parameter_boundaries():
    text = "Birinci cümle. İkinci cümle. Üçüncü cümle. Dördüncü cümle."

    # Very small chunk size
    chunker_tiny = akana.SentenceChunker(chunk_size=5)
    chunks_tiny = chunker_tiny(text)
    assert len(chunks_tiny) >= 3

    # Very large chunk size
    chunker_huge = akana.SentenceChunker(chunk_size=2048)
    chunks_huge = chunker_huge(text)
    assert len(chunks_huge) == 1
    assert chunks_huge[0].text == text


def test_threshold_modes_consistency():
    text = (
        "Yapay zeka sistemleri görüntü tanıma ve doğal dil işlemede kullanılır. "
        "Konvolüsyonel sinir ağları pikselleri analiz eder. "
        "Astronotlar uzay istasyonunda yerçekimsiz ortamda bilimsel araştırmalar yürütmektedir. "
        "Yörüngede yapılan biyoloji deneyleri tıp dünyasına katkı sağlamaktadır."
    )

    for mode in ["percentile", "similarity", "stdev", "iqr", "auto"]:
        chunker = akana.SemanticChunker(chunk_size=512, threshold_mode=mode, min_chunk_size=5)
        chunks = chunker(text)
        assert len(chunks) >= 1
        for c in chunks:
            assert text[c.start_index:c.end_index] == c.text
            assert c.token_count > 0
