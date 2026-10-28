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
    assert_eq!(
        deasciified,
        "türkçe nlp çok hızlı çalışıyor ve öğrenci kitap okuyor"
    );
}

#[test]
fn the_deasciifier_reads_lower_case_as_written() {
    // Plan step D7 (review 2 GN2-04): bu is no BÜ and iki no Iki, so the lower-case words
    // stay; a capitalised name is a name (Mugla: Muğla). (kisi and oldugu still become
    // kışı and öldüğü, words with more diacritics: the deasciifier weighs no word frequency.)
    assert_eq!(TurkishDeasciifier::deasciify_word("bu"), "bu");
    assert_eq!(TurkishDeasciifier::deasciify_word("iki"), "iki");
    assert_eq!(TurkishDeasciifier::deasciify_word("Mugla"), "Muğla");
}

#[test]
fn test_spellchecker() {
    let checker = TurkishSpellChecker::new();
    assert!(checker.is_correct("kitap"));
    assert!(checker.is_correct("kitaplar"));
    assert!(checker.is_correct("Türkiye"));
    assert!(checker.is_correct("TBMM"));
    assert!(!checker.is_correct("ktap"));
    // Review 3 NORM3-01: a name or an abbreviation in lower case is no spelling, and no
    // suggestion for a misspelt common word.
    assert!(!checker.is_correct("türkiye"));
    assert!(!checker.is_correct("varmı"));
    assert!(!checker.is_correct("cbk"));
    // (The edit distance counts bytes, so c for ç costs 2.)
    let cok = checker.suggest("cok", 2, 500);
    assert!(cok.iter().any(|s| s.word == "çok"), "{cok:?}");
    assert!(cok.iter().all(|s| s.word != "cbk"), "{cok:?}");

    let suggestions = checker.suggest("ktap", 2, 5);
    assert!(!suggestions.is_empty());
    assert!(suggestions.iter().any(|s| s.word == "kitap"));
}

#[test]
fn misspellings_the_lexicon_lists_are_not_correct() {
    // Review 3 LEX3-01, NORM3-01: Zemberek's non-TDK misspellings and the corpus "names" that
    // were common words without their Turkish letters made these correct.
    let checker = TurkishSpellChecker::new();
    for word in [
        "şöför",
        "döküman",
        "orjinal",
        "ünvan",
        "herşey",
        "motorsiklet",
        "egitim",
    ] {
        assert!(!checker.is_correct(word), "{word}");
    }
    for word in [
        "şoför",
        "doküman",
        "orijinal",
        "unvan",
        "motosiklet",
        "eğitim",
        "hükümet",
    ] {
        assert!(checker.is_correct(word), "{word}");
    }
    // Review 3 NORM3-04, MORPH3-10: what the analyzer reads only as text is found (a name
    // without its apostrophe, I for İ, a numeral as one word) or through a rare productive path
    // is no spelling; a foreign acronym with its Latin I is.
    for word in [
        "Türkiyede",
        "Ankarada",
        "KITAPLAR",
        "Istanbulda",
        "onbeş",
        "yirmidört",
        "kedilen",
        "maddemsi",
        "tarafınsan",
        "varasında",
    ] {
        assert!(!checker.is_correct(word), "{word}");
    }
    for word in [
        "Türkiye'de",
        "KİTAPLAR",
        "on beş",
        "FBI",
        "IMF'nin",
        "edilen",
        "yeşilimsi",
        "benimsin",
    ] {
        assert!(word.split(' ').all(|w| checker.is_correct(w)), "{word}");
    }
}

#[test]
fn test_informal_normalizer() {
    assert_eq!(
        TurkishInformalNormalizer::deduplicate_repeated_chars("çooookkkk"),
        "çok"
    );
    assert_eq!(
        TurkishInformalNormalizer::normalize_word("yapcam"),
        "yapacağım"
    );
    assert_eq!(
        TurkishInformalNormalizer::normalize_word("geliyom"),
        "geliyorum"
    );
    assert_eq!(
        TurkishInformalNormalizer::normalize_word("noldu"),
        "ne oldu"
    );
}

#[test]
fn test_number_converter() {
    assert_eq!(
        TurkishNumberConverter::number_to_words(1923),
        "bin dokuz yüz yirmi üç"
    );
    assert_eq!(TurkishNumberConverter::ordinal_to_words(1), "birinci");
    assert_eq!(TurkishNumberConverter::ordinal_to_words(4), "dördüncü");
    assert_eq!(
        TurkishNumberConverter::currency_to_words(1250.50, "TL"),
        "bin iki yüz elli lira elli kuruş"
    );
    assert_eq!(
        TurkishNumberConverter::words_to_number("bin dokuz yüz yirmi üç").unwrap(),
        1923
    );
}
