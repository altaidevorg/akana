use akana_core::analysis::{TurkishKeywordExtractor, TurkishSummarizer};
use akana_core::morphology::{TurkishStemmer, TurkishStopwords};
use akana_core::ner::TurkishNER;
use akana_core::phonology::TurkishSyllabifier;

#[test]
fn test_syllabifier_unit() {
    assert_eq!(TurkishSyllabifier::syllabify("Türkçe"), vec!["Türk", "çe"]);
    assert_eq!(
        TurkishSyllabifier::syllabify("araba"),
        vec!["a", "ra", "ba"]
    );
    assert_eq!(
        TurkishSyllabifier::hyphenate("bilgisayar", "-"),
        "bil-gi-sa-yar"
    );
    assert_eq!(TurkishSyllabifier::count_syllables("öğretmenlerimiz"), 6);
}

#[test]
fn test_stopwords_and_stemmer_unit() {
    let sw = TurkishStopwords::new();
    assert!(sw.is_stopword("ve"));
    assert!(sw.is_stopword("için"));
    assert!(!sw.is_stopword("bilgisayar"));

    let stemmer = TurkishStemmer::new();
    let root = stemmer.stem("kitaplarımızda");
    assert!(root == "kitap" || root == "kitab");

    // The word is analysed as written (review 2 MORPH2-11, RANK2-10): the capital keeps the
    // name (Miller, not mil-ler), and a Roman numeral keeps its root.
    assert_eq!(stemmer.stem("Miller"), "miller");
    assert_eq!(stemmer.stem("miller"), "mil");
    assert_eq!(stemmer.stem("XIX."), "XIX.");
    assert_eq!(stemmer.stem("II'nci"), "II");
}

#[test]
fn test_ner_unit() {
    let text = "Prof. Dr. Ahmet Yılmaz 16 Ağustos 2026 tarihinde İstanbul Üniversitesi bünyesinde 500 TL ödeme yaptı.";
    let entities = TurkishNER::extract_entities(text);
    assert!(!entities.is_empty());
    let labels: Vec<String> = entities.iter().map(|e| e.label.clone()).collect();
    assert!(labels.contains(&"PER".to_string()) || labels.contains(&"DATE".to_string()));
    assert!(labels.contains(&"MONEY".to_string()));
}

#[test]
fn test_keywords_and_summarizer_unit() {
    let text = "Türkiye Cumhuriyeti 1923 yılında kuruldu. Başkenti Ankara'dır. \
        Türkçe, Türk dilleri ailesine ait sondan eklemeli zengin bir dildir. \
        Doğal dil işleme ve morfolojik analiz algoritmaları bu zengin yapıyı çözümler. \
        Günümüzde milyonlarca insan tarafından konuşulmaktadır.";

    let extractor = TurkishKeywordExtractor::new();
    let kw = extractor.extract_keywords(text, 5);
    assert!(!kw.is_empty());

    let summarizer = TurkishSummarizer::new();
    let summary = summarizer.summarize(text, 2);
    assert_eq!(summary.len(), 2);
}

#[test]
fn keywords_are_the_same_on_every_call_and_skip_number_suffixes() {
    // Review 3 NORM3-03: phrases tied on score came in hash order, top_k cut inside the tie, and
    // a number's suffixes became words (1923'te: te).
    let text = "1923'te kurulan Cumhuriyet, %50'sini 3'er ay arayla ödedi. Dava açıldı. \
        Karar verildi. Bilirkişi geldi. Tanık dinlendi. Rapor alındı.";
    let extractor = TurkishKeywordExtractor::new();
    let words = |kw: Vec<akana_core::analysis::KeywordScore>| -> Vec<String> {
        kw.into_iter().map(|k| k.keyword).collect()
    };
    let first = words(extractor.extract_keywords(text, 4));
    for _ in 0..20 {
        assert_eq!(
            words(TurkishKeywordExtractor::new().extract_keywords(text, 4)),
            first
        );
    }
    let all = words(extractor.extract_keywords(text, 50)).join(" ");
    for fragment in ["te", "sin", "er"] {
        assert!(
            !all.split(' ').any(|w| w == fragment),
            "{fragment} in {all}"
        );
    }
}
