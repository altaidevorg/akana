use akana_core::tokenization::*;

#[test]
fn test_turkish_tokenizer() {
    let text = "Prof. Dr. Ahmet, İstanbul'da saat 14:30'da buluştu. Web: https://akana.org e-posta: info@akana.org #türkçe";
    let tokens = TurkishTokenizer::tokenize(text);

    let token_texts: Vec<&str> = tokens.iter().map(|t| t.text).collect();
    assert!(token_texts.contains(&"Prof."));
    assert!(token_texts.contains(&"Dr."));
    assert!(token_texts.contains(&"İstanbul'da"));
    assert!(token_texts.contains(&"14:30"));
    assert!(token_texts.contains(&"https://akana.org"));
    assert!(token_texts.contains(&"info@akana.org"));
    assert!(token_texts.contains(&"#türkçe"));
}

#[test]
fn test_sentence_segmenter() {
    let text = "Ak Ana, Türk mitolojisinde deniz tanrıçasıdır. Prof. Dr. Ayşe Hanım geldi! Sen de gelecek misin? Evet...";
    let sentences = SentenceSegmenter::segment(text);

    assert_eq!(sentences.len(), 4);
    assert_eq!(sentences[0].text, "Ak Ana, Türk mitolojisinde deniz tanrıçasıdır.");
    assert_eq!(sentences[1].text, "Prof. Dr. Ayşe Hanım geldi!");
    assert_eq!(sentences[2].text, "Sen de gelecek misin?");
    assert_eq!(sentences[3].text, "Evet...");
}
