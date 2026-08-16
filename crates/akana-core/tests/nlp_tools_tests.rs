use akana_core::phonology::TurkishSyllabifier;
use akana_core::morphology::{TurkishStopwords, TurkishStemmer};
use akana_core::ner::TurkishNER;
use akana_core::analysis::{TurkishKeywordExtractor, TurkishSummarizer};

#[test]
fn test_syllabifier_unit() {
    assert_eq!(TurkishSyllabifier::syllabify("Türkçe"), vec!["Türk", "çe"]);
    assert_eq!(TurkishSyllabifier::syllabify("araba"), vec!["a", "ra", "ba"]);
    assert_eq!(TurkishSyllabifier::hyphenate("bilgisayar", "-"), "bil-gi-sa-yar");
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
