use akana_core::normalization::*;

#[test]
fn test_asciifier() {
    let text = "Türkçe, Çağdaş, Şiir, Ağaç, Gözlük, Işık";
    let ascii = TurkishAsciifier::asciify(text);
    assert_eq!(ascii, "Turkce, Cagdas, Siir, Agac, Gozluk, Isik");
}

#[test]
fn test_deasciifier() {
    let text = "turkce nlp cok hizli calisiyor ve ogrenci kitap okuyor";
    let deasciified = TurkishDeasciifier::deasciify(text);
    assert_eq!(deasciified, "türkçe nlp çok hızlı çalışıyor ve öğrenci kitap okuyor");
}

#[test]
fn test_spellchecker() {
    let checker = TurkishSpellChecker::new();
    assert!(checker.is_correct("kitap"));
    assert!(checker.is_correct("kitaplar"));
    assert!(checker.is_correct("türkiye"));
    assert!(!checker.is_correct("ktap"));

    let suggestions = checker.suggest("ktap", 2, 3);
    assert!(!suggestions.is_empty());
    assert_eq!(suggestions[0].word, "kitap");
}

#[test]
fn test_informal_normalizer() {
    assert_eq!(TurkishInformalNormalizer::deduplicate_repeated_chars("çooookkkk"), "çookk");
    assert_eq!(TurkishInformalNormalizer::normalize_word("yapcam"), "yapacağım");
    assert_eq!(TurkishInformalNormalizer::normalize_word("geliyom"), "geliyorum");
    assert_eq!(TurkishInformalNormalizer::normalize_word("noldu"), "ne oldu");
}
