#!/usr/bin/env python3
"""Benchmark script for Akana Chunking Suite (SemanticChunker, SentenceChunker, SDPMChunker)."""

import time
import akana

# Sample realistic multi-domain corpus
PARAGRAPHS = [
    (
        "Kuantum fiziği, evrenin en küçük yapı taşlarının davranışlarını anlamamıza olanak tanır. "
        "Dalga-parçacık ikiliği, süperpozisyon ve kuantum dolanıklığı gibi fenomenler klasik fiziğin sınırlarını aşar. "
        "Son yıllarda kuantum bilgisayarlar kriptografi ve optimizasyon alanlarında çığır açmaktadır. "
        "Kuantum algoritmaları geleneksel süper bilgisayarların binlerce yılda çözebileceği problemleri dakikalar içinde çözebilir."
    ),
    (
        "Türkiye Cumhuriyeti'nin kuruluş dönemi köklü siyasi, toplumsal ve ekonomik dönüşümlere sahne olmuştur. "
        "Mustafa Kemal Atatürk önderliğinde gerçekleştirilen inkılaplar, modern bir hukuk devleti inşa etmeyi hedeflemiştir. "
        "Eğitim, sanayi ve tarım alanlarında başlatılan kalkınma hamleleri genç cumhuriyetin temellerini sağlamlaştırmıştır. "
        "1923 yılından itibaren Ankara başkent olarak ülkenin yönetim merkezi haline gelmiştir."
    ),
    (
        "Geleneksel Türk mutfağı Orta Asya, Selçuklu ve Osmanlı miraslarının birleşimiyle zenginleşmiştir. "
        "Güneydoğu Anadolu'nun baharatlı kebapları, Karadeniz'in hamsili lezzetleri ve Ege'nin zeytinyağlıları farklı coğrafyaların zenginliğini sunar. "
        "Kayseri mantısı, Adana kebabı ve Antep baklavası dünyaca tanınan lezzetler arasındadır. "
        "Geleneksel Türk çayı ve Türk kahvesi ise misafirperverliğin ve sohbet kültürünün simgesidir."
    ),
    (
        "Yapay zeka ve makine öğrenmesi algoritmaları günümüzde birçok endüstride kritik roller üstlenmektedir. "
        "Derin öğrenme mimarileri, büyük veri kümelerinden anlamlı örüntüler çıkararak bilgisayarla görme ve doğal dil işleme alanlarını dönüştürmüştür. "
        "Büyük dil modelleri metin üretimi, kodlama ve dil çevirisinde olağanüstü başarılar elde etmektedir. "
        "Gelecekte otonom sistemler ve robotik teknolojilerin insan hayatını daha da kolaylaştırması beklenmektedir."
    ),
    (
        "Küresel finans piyasaları merkez bankalarının faiz kararları ve enflasyon verileriyle şekillenmektedir. "
        "Uluslararası ticaret hacmi, jeopolitik gelişmeler ve emtia fiyatları döviz kurlarında dalgalanmalara yol açabilir. "
        "Yatırımcılar portföy çeşitlendirmesi yaparak riskleri minimize etmeye çalışırlar. "
        "Sürdürülebilir finans ve yeşil tahviller son yıllarda piyasalarda giderek daha fazla ilgi görmektedir."
    ),
]

def run_benchmark():
    full_doc = "\n\n".join(PARAGRAPHS * 10)  # 50 paragraphs, ~200 sentences, ~2500 words
    word_count = len(full_doc.split())

    print("=" * 70)
    print(" 🚀 AKANA CHUNKING PERFORMANCE BENCHMARK")
    print(f" Document stats: {len(full_doc):,} chars | {word_count:,} words | 50 paragraphs")
    print("=" * 70)

    # 1. Benchmark SentenceChunker
    sentence_chunker = akana.SentenceChunker(chunk_size=512, chunk_overlap=30)
    # Warmup
    _ = sentence_chunker(full_doc)

    iters = 100
    t0 = time.perf_counter()
    for _ in range(iters):
        chunks = sentence_chunker(full_doc)
    t_sent = (time.perf_counter() - t0) / iters

    words_per_sec_sent = word_count / t_sent
    print(f"\n[1] SentenceChunker (chunk_size=512, overlap=30):")
    print(f"    - Latency per document: {t_sent * 1000:.3f} ms")
    print(f"    - Throughput:           {words_per_sec_sent:,.0f} words/sec ({1 / t_sent:,.1f} docs/sec)")
    print(f"    - Chunks produced:      {len(chunks)} chunks")

    # 2. Benchmark SemanticChunker
    semantic_chunker = akana.SemanticChunker(
        chunk_size=512,
        threshold_mode="percentile",
        threshold_value=0.75,
        min_chunk_size=20,
    )
    # Warmup
    _ = semantic_chunker(full_doc)

    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        chunks = semantic_chunker(full_doc)
    t_sem = (time.perf_counter() - t0) / iters

    words_per_sec_sem = word_count / t_sem
    print(f"\n[2] SemanticChunker (chunk_size=512, mode=percentile, TurboQuant Embeddings):")
    print(f"    - Latency per document: {t_sem * 1000:.3f} ms")
    print(f"    - Throughput:           {words_per_sec_sem:,.0f} words/sec ({1 / t_sem:,.1f} docs/sec)")
    print(f"    - Chunks produced:      {len(chunks)} chunks")

    # 3. Benchmark SDPMChunker
    sdpm_chunker = akana.SDPMChunker(
        chunk_size=512,
        threshold_mode="percentile",
        threshold_value=0.75,
        merge_threshold=0.65,
    )
    # Warmup
    _ = sdpm_chunker(full_doc)

    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        chunks = sdpm_chunker(full_doc)
    t_sdpm = (time.perf_counter() - t0) / iters

    words_per_sec_sdpm = word_count / t_sdpm
    print(f"\n[3] SDPMChunker (chunk_size=512, merge_threshold=0.65, 2-Pass Merge):")
    print(f"    - Latency per document: {t_sdpm * 1000:.3f} ms")
    print(f"    - Throughput:           {words_per_sec_sdpm:,.0f} words/sec ({1 / t_sdpm:,.1f} docs/sec)")
    print(f"    - Chunks produced:      {len(chunks)} chunks")

    print("\n" + "=" * 70)
    print(" ✅ All chunking benchmarks completed successfully.")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()
