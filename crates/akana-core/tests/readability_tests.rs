use akana_core::readability::*;

#[test]
fn test_simple_text_readability() {
    let simple_text = "\
        Ali top oynadı. \
        Ayşe eve geldi ve yemek yedi. \
        Güneş açtı, hava çok güzel oldu. \
        Çocuklar bahçede neşeyle koştu.";

    let report = analyze_readability(simple_text);

    // Verify statistics
    assert_eq!(report.statistics.total_sentences, 4);
    assert!(report.statistics.total_words >= 15);
    assert!(report.statistics.total_syllables > 30);

    // Ateşman should classify as Kolay / Çok Kolay (> 70)
    println!("Simple Text Ateşman Score: {}", report.legacy.atesman.score);
    assert!(report.legacy.atesman.score > 70.0);
    assert!(report.legacy.atesman.grade_level.contains("Kolay"));

    // Kalyoncu scores exist and are valid numbers
    assert!(report.kalyoncu.formula1.score > 0.0);
    assert!(report.kalyoncu.formula4.score > 0.0);
}

#[test]
fn test_complex_academic_text_readability() {
    let academic_text = "\
        Türkçenin morfolojik karmaşıklığı ve sondan eklemeli yapısı, doğal dil işleme modellerinde \
        sözcük kökü ile biçimbirim dizilimlerinin derinlemesine ayrıştırılmasını zorunlu kılmaktadır. \
        Bu bağlamda geliştirilen algoritmalar, bağlamsal çokanlamlılığı giderme süreçlerinde \
        olasılıksal geçiş matrisleri ve sentaktik bağımlılık ağaçlarından istifade etmektedir.";

    let report = analyze_readability(academic_text);

    assert_eq!(report.statistics.total_sentences, 2);
    // Unfamiliar words ratio should be significantly higher in academic text
    assert!(report.statistics.unfamiliar_word_ratio > 30.0);
    // Average sentence length should be long (> 15 words/sent)
    assert!(report.statistics.words_per_sentence > 15.0);

    // Ateşman should classify as Zor / Çok Zor (< 50)
    println!("Academic Ateşman Score: {}", report.legacy.atesman.score);
    assert!(report.legacy.atesman.score < 50.0);
    assert!(report.legacy.atesman.grade_level.contains("Zor"));

    // Kalyoncu Formula 1 score should be higher (harder) for academic text than simple text
    let simple_report = analyze_readability("Ali top oynadı. Ayşe eve geldi.");
    assert!(report.kalyoncu.formula1.score > simple_report.kalyoncu.formula1.score);
}
