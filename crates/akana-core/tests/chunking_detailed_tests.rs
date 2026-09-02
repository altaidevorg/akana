use akana_core::chunking::{
    SDPMChunker, SemanticChunker, SentenceChunker, ThresholdMode,
};

fn get_char_slice(text: &str, start: usize, end: usize) -> String {
    text.chars().skip(start).take(end - start).collect()
}

#[test]
fn test_edge_cases_empty_whitespace_newlines() {
    let chunker = SemanticChunker::new(512, ThresholdMode::Percentile(0.75));
    assert!(chunker.chunk("").is_empty());
    assert!(chunker.chunk(" ").is_empty());
    assert!(chunker.chunk("\n\n\t  \r\n").is_empty());

    let sentence_chunker = SentenceChunker::new(512, 0, 1);
    assert!(sentence_chunker.chunk("").is_empty());
    assert!(sentence_chunker.chunk("   ").is_empty());

    let sdpm_chunker = SDPMChunker::new(512, ThresholdMode::Percentile(0.75), 0.65);
    assert!(sdpm_chunker.chunk("").is_empty());
}

#[test]
fn test_edge_cases_single_word_and_special_characters() {
    let chunker = SemanticChunker::new(512, ThresholdMode::Percentile(0.75));
    let text = "Merhaba!";
    let chunks = chunker.chunk(text);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, "Merhaba!");
    assert_eq!(chunks[0].start_index, 0);
    assert_eq!(chunks[0].end_index, text.chars().count());

    // With emojis and complex unicode
    let unicode_text = "🚀 Türkiye uzay ajansı ilk astronotunu uzaya gönderdi! 🌌 Bilimsel deneyler başarıyla tamamlandı. 🇹🇷";
    let u_chunks = chunker.chunk(unicode_text);
    assert!(!u_chunks.is_empty());
    for c in &u_chunks {
        assert_eq!(get_char_slice(unicode_text, c.start_index, c.end_index), c.text);
    }
}

#[test]
fn test_abbreviations_and_decimals_boundary_integrity() {
    let text = "Prof. Dr. Ahmet Yılmaz ve Doç. Dr. Ayşe Kaya saat 14.30'da 3. katta toplantı yaptı. \
                Toplantıda T.C. kanunları ve AB standartları görüşüldü. \
                Sonuç bildirgesi www.tubitak.gov.tr adresinde yayımlandı.";

    let chunker = SemanticChunker::new(512, ThresholdMode::Percentile(0.75)).with_min_chunk_size(5);
    let chunks = chunker.chunk(text);

    assert!(!chunks.is_empty());
    for c in &chunks {
        assert_eq!(get_char_slice(text, c.start_index, c.end_index), c.text);
        assert!(c.token_count > 0);
    }
}

#[test]
fn test_large_multi_topic_document_segmentation() {
    let text = "Kuantum elektrodinamiği ışık ile maddenin etkileşimini kuantum mekaniksel olarak açıklar. \
                Fotonlar elektromanyetik alanın kuantaları olarak kabul edilir ve Feynman diyagramları ile hesaplamalar yapılır. \
                Renormalizasyon tekniği sayesinde sonsuzluk problemleri çözülmüş ve modern parçacık fiziğinin temeli atılmıştır. \
                \
                Osmanlı Devleti 1453 yılında İstanbul'u fethederek Orta Çağ'ı kapatıp Yeni Çağ'ı başlatmıştır. \
                Fatih Sultan Mehmet döneminde askeri, hukuki ve kültürel alanlarda köklü reformlar gerçekleştirilmiştir. \
                Topkapı Sarayı devletin idare merkezi olmuş ve Divan-ı Hümayun düzenli olarak toplanmıştır. \
                \
                Gastronomi dünyasında Türk mutfağı zengin baharatları ve tencere yemekleriyle öne çıkar. \
                Kayseri mantısı, Antep baklavası ve Adana kebabı coğrafi işaret tescili almış seçkin lezzetlerdir. \
                Zeytinyağlı enginar ve yaprak sarması ise Ege ve Akdeniz mutfak kültürünün vazgeçilmez parçalarıdır. \
                \
                Küresel finans piyasalarında merkez bankalarının faiz kararları likiditeyi doğrudan etkiler. \
                Enflasyonla mücadele kapsamında uygulanan sıkı para politikaları tahvil getirilerini yükseltmiştir. \
                Borsa endeksleri ve emtia fiyatları makroekonomik veriler ışığında dalgalı bir seyir izlemektedir.";

    let chunker = SemanticChunker::new(512, ThresholdMode::Percentile(0.75))
        .with_min_chunk_size(15)
        .with_min_sentences_per_chunk(1);

    let chunks = chunker.chunk(text);

    // Expect at least 3 distinct topic chunks
    assert!(chunks.len() >= 3, "Expected at least 3 topic chunks, got {}", chunks.len());

    // Verify all character offsets and token counts
    for c in &chunks {
        assert_eq!(get_char_slice(text, c.start_index, c.end_index), c.text);
        assert!(c.token_count > 0);
        assert!(!c.sentences.is_empty());
    }

    // Verify topic separation
    assert!(chunks[0].text.contains("Kuantum"));
    assert!(chunks.iter().any(|c| c.text.contains("Osmanlı")));
    assert!(chunks.iter().any(|c| c.text.contains("Gastronomi") || c.text.contains("kebap")));
    assert!(chunks.last().unwrap().text.contains("finans") || chunks.last().unwrap().text.contains("para"));
}

#[test]
fn test_sdpm_hierarchical_merging_behavior() {
    let text = "Yapay sinir ağları katmanlı düğümlerden meydana gelir. \
                Aktivasyon fonksiyonları doğrusal olmayan ilişkileri öğrenmeyi sağlar. \
                Ağırlıklar geriye yayılım algoritması ile optimize edilir. \
                \
                Güneş sisteminde sekiz adet gezegen yer almaktadır. \
                Merkür ve Venüs güneşe en yakın iç gezegenlerdir. \
                Jüpiter ve Satürn ise gaz devi gezegenler sınıfına girer.";

    // High merge threshold -> few merges
    let strict_sdpm = SDPMChunker::new(512, ThresholdMode::Similarity(0.80), 0.85);
    let strict_chunks = strict_sdpm.chunk(text);

    // Low merge threshold -> aggressive merges
    let lenient_sdpm = SDPMChunker::new(512, ThresholdMode::Similarity(0.80), 0.40);
    let lenient_chunks = lenient_sdpm.chunk(text);

    assert!(lenient_chunks.len() <= strict_chunks.len());
    for c in &lenient_chunks {
        assert_eq!(get_char_slice(text, c.start_index, c.end_index), c.text);
        assert!(c.token_count <= 512);
    }
}

#[test]
fn test_sentence_chunker_exact_overlaps() {
    let text = "Cümle 1: Bilgisayar bilimi algoritmaları inceler. \
                Cümle 2: Veri yapıları bilginin verimli saklanmasını sağlar. \
                Cümle 3: Graf teorisi ağ bağlantılarını modeller. \
                Cümle 4: Otomata teorisi hesaplanabilirliğin sınırlarını belirler.";

    // Chunk size 35 with overlap 15 tokens ensures sentences group and overlap
    let chunker = SentenceChunker::new(35, 15, 1);
    let chunks = chunker.chunk(text);

    assert!(chunks.len() >= 2);
    for i in 0..chunks.len() {
        assert_eq!(get_char_slice(text, chunks[i].start_index, chunks[i].end_index), chunks[i].text);
        if i > 0 {
            assert!(chunks[i].start_index < chunks[i - 1].end_index, "chunk[{}].start ({}) should be < chunk[{}].end ({})", i, chunks[i].start_index, i-1, chunks[i - 1].end_index);
        }
    }
}
