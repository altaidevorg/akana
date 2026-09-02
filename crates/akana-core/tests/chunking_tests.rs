use akana_core::chunking::{
    calculate_quantile, calculate_threshold, moving_average_filter,
    savitzky_golay_filter, SDPMChunker, SemanticChunker, SentenceChunker, ThresholdMode,
};

fn get_char_slice(text: &str, start: usize, end: usize) -> String {
    text.chars().skip(start).take(end - start).collect()
}

#[test]
fn test_sentence_chunker_spans_and_offsets() {
    let text = "Ankara Türkiye'nin başkentidir. İstanbul en kalabalık şehridir. İzmir ise Ege kıyısındadır. Antalya turizmin merkezidir.";
    let chunker = SentenceChunker::new(20, 0, 1);
    let chunks = chunker.chunk(text);

    assert!(!chunks.is_empty());
    for chunk in &chunks {
        assert_eq!(get_char_slice(text, chunk.start_index, chunk.end_index), chunk.text);
        assert!(chunk.token_count > 0);
        assert!(!chunk.sentences.is_empty());
    }

    assert_eq!(chunks[0].start_index, 0);
    assert_eq!(chunks.last().unwrap().end_index, text.chars().count());
}

#[test]
fn test_sentence_chunker_empty_and_single() {
    let chunker = SentenceChunker::new(512, 0, 1);
    assert!(chunker.chunk("").is_empty());
    assert!(chunker.chunk("   \n\t  ").is_empty());

    let single = "Bu tek bir cümledir.";
    let chunks = chunker.chunk(single);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, single);
    assert_eq!(chunks[0].start_index, 0);
    assert_eq!(chunks[0].end_index, single.chars().count());
}

#[test]
fn test_sentence_chunker_overlap() {
    let text = "Cümle 1 burada. Cümle 2 burada. Cümle 3 burada. Cümle 4 burada. Cümle 5 burada.";
    let chunker = SentenceChunker::new(10, 4, 1);
    let chunks = chunker.chunk(text);

    assert!(chunks.len() >= 2);
    for chunk in &chunks {
        assert_eq!(get_char_slice(text, chunk.start_index, chunk.end_index), chunk.text);
    }
}

#[test]
fn test_semantic_chunker_topic_boundaries() {
    let chunker = SemanticChunker::new(512, ThresholdMode::Percentile(0.75))
        .with_min_chunk_size(5)
        .with_min_sentences_per_chunk(1);

    let text = "Kuantum mekaniği atomik düzeydeki parçacıkların davranışını açıklar. Dalga fonksiyonunun çökmesi ve süperpozisyon kuantum teorisinin temel kavramlarıdır. \
                Fenerbahçe futbol takımı dün akşam derbi maçında üstün bir oyun sergiledi. Teknik direktör taktik değişikliklerle galibiyeti getirdi. \
                Geleneksel Türk mutfağında zeytinyağlı yemekler ve kebaplar çok meşhurdur. Özellikle baklava ve sütlaç tatlı olarak sıkça tercih edilir.";

    let chunks = chunker.chunk(text);

    assert!(chunks.len() >= 2, "Expected multiple chunks for distinct topics, got {}", chunks.len());

    // Verify chunk slicing and offsets
    for chunk in &chunks {
        assert_eq!(get_char_slice(text, chunk.start_index, chunk.end_index), chunk.text);
        assert!(chunk.token_count > 0);
    }

    // First chunk should be physics-related
    assert!(chunks[0].text.contains("Kuantum"));
    // Last chunk should be food-related
    assert!(chunks.last().unwrap().text.contains("mutfağında") || chunks.last().unwrap().text.contains("baklava"));
}

#[test]
fn test_semantic_chunker_threshold_modes() {
    let text = "Yapay zeka modelleri metin üretiminde ilerleme kaydetti. Doğal dil işleme araştırmaları devam ediyor. \
                Astronomi alanında yeni galaksiler keşfedildi. James Webb uzay teleskobu derin uzayı görüntülüyor.";

    for mode in [
        ThresholdMode::Similarity(0.70),
        ThresholdMode::Percentile(0.75),
        ThresholdMode::StandardDeviation(0.8),
        ThresholdMode::Interquartile(1.0),
        ThresholdMode::Auto,
    ] {
        let chunker = SemanticChunker::new(512, mode).with_min_chunk_size(5);
        let chunks = chunker.chunk(text);
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert_eq!(get_char_slice(text, chunk.start_index, chunk.end_index), chunk.text);
        }
    }
}

#[test]
fn test_sdpm_chunker_double_pass_merge() {
    let chunker = SDPMChunker::new(512, ThresholdMode::Similarity(0.85), 0.50);
    let text = "Makine öğrenmesi algoritmaları veri ile eğitilir. Denetimli öğrenme en yaygın yöntemdir. \
                Denetimsiz öğrenme ise etiketsiz verilerden örüntü çıkarır. \
                Güneş sistemindeki en büyük gezegen Jüpiter'dir. Satürn ise halkalarıyla bilinir.";

    let chunks = chunker.chunk(text);
    assert!(!chunks.is_empty());
    for chunk in &chunks {
        assert_eq!(get_char_slice(text, chunk.start_index, chunk.end_index), chunk.text);
        assert!(chunk.token_count <= 512);
    }
}

#[test]
fn test_similarity_math_utilities() {
    let raw = vec![0.9, 0.85, 0.3, 0.88, 0.25, 0.95];
    let smoothed_ma = moving_average_filter(&raw, 3);
    assert_eq!(smoothed_ma.len(), raw.len());

    let smoothed_sg = savitzky_golay_filter(&raw);
    assert_eq!(smoothed_sg.len(), raw.len());

    let q = calculate_quantile(&raw, 0.5);
    assert!(q > 0.0 && q <= 1.0);

    let thresh_auto = calculate_threshold(&raw, &ThresholdMode::Auto);
    assert!(thresh_auto > 0.0);

    // Test mean_pool_vectors safety
    use akana_core::chunking::mean_pool_vectors;
    assert!(mean_pool_vectors(&[]).is_empty());
    let v1 = vec![0.5, 0.5];
    let v2 = vec![0.5, 0.5];
    let pooled = mean_pool_vectors(&[v1, v2]);
    assert_eq!(pooled.len(), 2);
    let norm = (pooled[0] * pooled[0] + pooled[1] * pooled[1]).sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
}

#[test]
fn test_sdpm_multibyte_turkish_merging() {
    let chunker = SDPMChunker::new(512, ThresholdMode::Similarity(0.70), 0.30);
    let text = "Şiir ve edebiyat; Türkçe'nin zenginliğini, inceliğini ve çağrışım gücünü gösterir. \
                Öykücülük ve romancılık ise toplumsal dönüşümleri derinlemesine işler. \
                İçerik çözümlemesi dilbilimsel yöntemlerle yürütülür.";

    let chunks = chunker.chunk(text);
    assert!(!chunks.is_empty());
    for chunk in &chunks {
        assert_eq!(get_char_slice(text, chunk.start_index, chunk.end_index), chunk.text);
    }
}
