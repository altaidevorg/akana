use akana_core::tokenization::*;

#[test]
fn test_turkish_tokenizer() {
    let text = "Prof. Dr. Ahmet, İstanbul'da saat 14:30'da buluştu. Web: https://akana.org e-posta: info@akana.org #türkçe";
    let tokens = TurkishTokenizer::tokenize(text);

    let token_texts: Vec<&str> = tokens.iter().map(|t| t.text).collect();
    assert!(token_texts.contains(&"Prof."));
    assert!(token_texts.contains(&"Dr."));
    assert!(token_texts.contains(&"İstanbul'da"));
    // A time or number keeps its suffixes after the apostrophe, as a name does.
    assert!(token_texts.contains(&"14:30'da"));
    assert!(token_texts.contains(&"https://akana.org"));
    assert!(token_texts.contains(&"info@akana.org"));
    assert!(token_texts.contains(&"#türkçe"));
}

#[test]
fn numbers_and_abbreviations_keep_their_suffixes() {
    let words = |text| TurkishTokenizer::tokenize_words(text);
    assert_eq!(words("1924'te kuruldu"), ["1924'te", "kuruldu"]);
    assert_eq!(
        words("21.10.2026'da ve 17.00'ye"),
        ["21.10.2026'da", "ve", "17.00'ye"]
    );
    let numbers = TurkishTokenizer::tokenize("%50'si");
    assert_eq!(numbers[1].text, "50'si");
    assert_eq!(numbers[1].token_type, TokenType::Number);
    // Court decisions write an abbreviation's suffixes after a period.
    assert_eq!(
        words("TCK.nun 53. maddesi, H.U.M.K.nun 427. maddesi"),
        [
            "TCK.nun",
            "53",
            ".",
            "maddesi",
            ",",
            "H.U.M.K.nun",
            "427",
            ".",
            "maddesi"
        ]
    );
    assert_eq!(words("Ltd. Şti. ve San."), ["Ltd.", "Şti.", "ve", "San."]);
    // With a space after the period, the abbreviation is a token of its own.
    assert_eq!(words("TCK. Bu"), ["TCK.", "Bu"]);
}

#[test]
fn test_sentence_segmenter() {
    let text = "Ak Ana, Türk mitolojisinde deniz tanrıçasıdır. Prof. Dr. Ayşe Hanım geldi! Sen de gelecek misin? Evet...";
    let sentences = SentenceSegmenter::segment(text);

    assert_eq!(sentences.len(), 4);
    assert_eq!(
        sentences[0].text,
        "Ak Ana, Türk mitolojisinde deniz tanrıçasıdır."
    );
    assert_eq!(sentences[1].text, "Prof. Dr. Ayşe Hanım geldi!");
    assert_eq!(sentences[2].text, "Sen de gelecek misin?");
    assert_eq!(sentences[3].text, "Evet...");
}
