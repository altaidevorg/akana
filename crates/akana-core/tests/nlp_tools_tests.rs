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
fn a_name_that_is_also_a_word_is_no_person_at_a_sentences_start_alone() {
    // At a sentence's start a capital says nothing: Kaya, Barış, Ateş, Tatlı there are words
    // (the names the lexicon gained from PII's lists and court decisions made them persons).
    let persons = |text: &str| -> Vec<String> {
        TurkishNER::extract_entities(text)
            .into_iter()
            .filter(|e| e.label == "PER")
            .map(|e| e.text)
            .collect()
    };
    for text in [
        "Kaya parçaları yola düştü.",
        "Barış antlaşması imzalandı.",
        "Ateş düştüğü yeri yakar.",
        "Tatlı yemeyi çok severim.",
        "Yazıcı yine bozuldu. Balıkçı teknesi battı.",
        // The rules made these persons: names rare enough as words (0.9.0 before the tagger).
        "Erdem insanın en güzel süsüdür.",
        "Onaran ekip yolu açtı.",
        "Esen rüzgâr serinletti.",
    ] {
        assert!(persons(text).is_empty(), "{text}: {:?}", persons(text));
    }
    // A full name, a title, a name inside a sentence, or the person named in full before.
    assert_eq!(persons("Ateş Yılmaz geldi."), ["Ateş Yılmaz"]);
    assert_eq!(persons("Sayın Kaya konuştu."), ["Kaya"]);
    assert_eq!(persons("Dün Kaya ile görüştük."), ["Kaya"]);
    assert_eq!(
        persons("Mehmet Yılmaz açıklama yaptı. Yılmaz, kararın doğru olduğunu söyledi."),
        ["Mehmet Yılmaz", "Yılmaz"]
    );
    // A name that is no word is a person at a sentence's start too.
    assert_eq!(persons("Ahmet eve geldi."), ["Ahmet"]);
}

#[test]
fn places_are_no_persons_and_a_name_ends_before_its_suffix() {
    let ents = |text: &str| -> Vec<String> {
        TurkishNER::extract_entities(text)
            .into_iter()
            .map(|e| format!("{}/{}", e.text, e.label))
            .collect()
    };
    // Every proper noun of the lexicon was a person's name: Ankara, Almanya.
    assert_eq!(ents("Ankara güzel bir şehir."), ["Ankara/LOC"]);
    assert_eq!(
        ents("Almanya ve Fransa anlaştı."),
        ["Almanya/LOC", "Fransa/LOC"]
    );
    // A suffix after the apostrophe is no part of the name, which it no longer hides.
    assert_eq!(ents("Dün Ankara'ya gittik."), ["Ankara/LOC"]);
    assert_eq!(ents("Dün Kaya'nın evine gittik."), ["Kaya/PER"]);
    // A person's name ends before a place.
    assert_eq!(
        ents("Ahmet Yılmaz Ankara'da yaşıyor."),
        ["Ahmet Yılmaz/PER", "Ankara/LOC"]
    );
    // Translated news: Bay Gorbaçov; a nationality is no person (Türk gençleri).
    assert_eq!(ents("Bay Gorbaçov konuştu."), ["Gorbaçov/PER"]);
    assert!(ents("Türk gençleri sahneye çıktı.").is_empty());
}

#[test]
fn organisations_and_patterns_beside_names() {
    let ents = |text: &str| -> Vec<String> {
        TurkishNER::extract_entities(text)
            .into_iter()
            .map(|e| format!("{}/{}", e.text, e.label))
            .collect()
    };
    // Organisations are the tagger's, with names that end in no listed word.
    assert_eq!(
        ents("Orta Doğu Teknik Üniversitesi öğrencileri Ankara'da toplandı."),
        ["Orta Doğu Teknik Üniversitesi/ORG", "Ankara/LOC"]
    );
    assert_eq!(
        ents("Prof. Dr. Ahmet Yılmaz 16 Ağustos 2026 tarihinde İstanbul Üniversitesi bünyesinde 500 TL ödeme yaptı."),
        [
            "Ahmet Yılmaz/PER",
            "16 Ağustos 2026/DATE",
            "İstanbul Üniversitesi/ORG",
            "500 TL/MONEY"
        ]
    );
    // Streets, squares and buildings, which the tagger leaves out, are places by their last
    // word; the place takes in the name before it, and a title there.
    assert_eq!(
        ents("Dün Kızılay Meydanı'nda buluştuk."),
        ["Kızılay Meydanı/LOC"]
    );
    assert_eq!(
        ents("Uçak İstanbul Havalimanı'na indi."),
        ["İstanbul Havalimanı/LOC"]
    );
    assert_eq!(
        ents("Mustafa Kemal Paşa Caddesi kapandı."),
        ["Mustafa Kemal Paşa Caddesi/LOC"]
    );
    assert_eq!(ents("Ahmet'in köyü uzak."), ["Ahmet/PER"]);
    // Dates, amounts and percentages are the patterns'; a name the tagger finds over one of
    // them is left out.
    assert_eq!(
        ents("Cuma günü Ahmet geldi."),
        ["Cuma günü/DATE", "Ahmet/PER"]
    );
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
