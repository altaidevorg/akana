//! Review 4's smaller analyzer and lexicon findings (MORPH4-10): names the lexicon lacked,
//! lower-case words read as names, abbreviations of a phrase with a possessive, abbreviations of
//! one word, a tie between two spellings of one reading, and single letters read as numerals;
//! and the names and abbreviations the pre-release audit found apart from their word.

use akana_core::embeddings::MorphologicalSparseEncoder;
use akana_core::morphology::*;

fn first(word: &str) -> MorphParse {
    TurkishMorphology::new()
        .analyze(word)
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("{word}: no reading"))
}

#[test]
fn given_names_the_lexicon_lacked_are_names() {
    // Serpil read as serp-il (serpmek), Gökçen and Özgen with a P2sg, Halide as hali-de; Gürsel
    // as gür, Solmaz as solmak (pre-release audit).
    for name in ["Serpil", "Gökçen", "Özgen", "Halide", "Gürsel", "Solmaz"] {
        let p = first(name);
        assert_eq!(
            (p.lemma.as_str(), p.secondary_pos),
            (name.to_lowercase().as_str(), SecondaryPos::ProperNoun),
            "{name}"
        );
    }
    let tokens = ["Tanık", "Serpil", "Hanım", "geldi", "."];
    let parses = MorphologicalDisambiguator::new().disambiguate(&tokens);
    assert_eq!(parses[1].lemma, "serpil");
    // Kâni Vrana: the adjective kani (its lemma folds the circumflex as the adjective's does).
    let p = first("Kâni");
    assert_eq!(
        (p.lemma.as_str(), p.secondary_pos),
        ("kani", SecondaryPos::ProperNoun)
    );
}

#[test]
fn court_surnames_are_names_and_a_heading_word_is_none() {
    // The pre-release audit: PII's COMMON_SURNAMES never reached the lexicon (Basri BAĞCI read
    // bağ, Sönmez sönmek), nor had court decisions' surnames a name reading (Mehmet ERTEN: erte,
    // Serruh KALELİ: kale, Sacit ADALI: ada).
    let tokens = [
        "Üyeler", ":", "Basri", "BAĞCI", ",", "Mehmet", "ERTEN", ",", "Serruh", "KALELİ", ",",
        "Sacit", "ADALI", ".",
    ];
    let parses = MorphologicalDisambiguator::new().disambiguate(&tokens);
    for (i, lemma) in [(3, "bağcı"), (6, "erten"), (9, "kaleli"), (12, "adalı")] {
        assert_eq!(
            (parses[i].lemma.as_str(), parses[i].secondary_pos),
            (lemma, SecondaryPos::ProperNoun),
            "{}",
            tokens[i]
        );
    }
    assert_eq!(first("Sönmez").lemma, "sönmez");
    // Temel heads a phrase far more often than it follows a given name (Temel Hak ve
    // Özgürlükler): it has no name reading.
    assert!(TurkishMorphology::new()
        .analyze("Temel")
        .iter()
        .all(|p| p.secondary_pos != SecondaryPos::ProperNoun));
}

#[test]
fn a_word_in_lower_case_is_no_name_where_it_is_a_word() {
    // mimarlık and yayıncılık were company names of the corpus, ranked first in lower case.
    assert_eq!(first("mimarlık").lemma, "mimar");
    assert_eq!(first("yayıncılık").lemma, "yayın");
    for word in ["güler", "kadri"] {
        assert_ne!(
            first(word).secondary_pos,
            SecondaryPos::ProperNoun,
            "{word}"
        );
    }
    // A name with no other reading stays one, and the capitalised word is the name.
    assert_eq!(first("ankara").secondary_pos, SecondaryPos::ProperNoun);
    assert_eq!(first("Güler").secondary_pos, SecondaryPos::ProperNoun);
}

#[test]
fn an_abbreviation_of_a_phrase_takes_the_phrase_cases() {
    // Through a silent P3sg, which the tags leave out as a CompoundP3sg noun's
    // (cumhurbaşkanına): no P2sg.
    for (word, tags) in [
        ("HMK'na", ["Noun", "Dat"]),
        ("CBS'nca", ["Noun", "Equ"]),
        ("Şti'nden", ["Noun", "Abl"]),
    ] {
        assert_eq!(first(word).morpheme_tags, tags, "{word}");
    }
    // The bare abbreviation and its genitive keep their one reading.
    let morphology = TurkishMorphology::new();
    assert_eq!(morphology.analyze("HMK").len(), 1);
    assert_eq!(first("HMK'nın").morpheme_tags, ["Noun", "Gen"]);
}

#[test]
fn an_abbreviation_of_one_word_is_that_word() {
    for (word, lemma) in [
        ("md.", "madde"),
        ("mad.", "madde"),
        ("Av.", "avukat"),
        ("Sok.", "sokak"),
        ("San.", "sanayi"),
        // The pre-release audit: company names, titles, addresses (A.Ş. was aş "food").
        ("Tic.", "ticaret"),
        ("Ltd.", "limited"),
        ("Şti.", "şirket"),
        ("A.Ş.", "şirket"),
        ("İth.", "ithalat"),
        ("İhr.", "ihracat"),
        ("Dr.", "doktor"),
        ("Prof.", "profesör"),
        ("Doç.", "doçent"),
        ("Yrd.", "yardımcı"),
        ("Müh.", "mühendis"),
        ("Mah.", "mahalle"),
        ("Cad.", "cadde"),
        ("par.", "paragraf"),
    ] {
        assert_eq!(first(word).lemma, lemma, "{word}");
    }
    let encoder = MorphologicalSparseEncoder::new();
    let terms = |text: &str| encoder.encode_sparse(text).terms;
    assert_eq!(terms("TCK md. 5"), terms("TCK madde 5"));
    assert!(!terms("Av. Ayşe Kaya").contains(&"av".to_string()));
    assert_eq!(
        terms("Deniz Tic. Ltd. Şti."),
        terms("Deniz ticaret limited şirket")
    );
    assert!(!terms("Deniz Gıda A.Ş.").contains(&"aş".to_string()));
}

#[test]
fn one_reading_for_the_evidential_converb() {
    // gelmişken: Evid+While, not also the participle made a verb (0.475 each).
    let parses = TurkishMorphology::new().analyze("gelmişken");
    assert_eq!(parses.len(), 1);
    assert_eq!(parses[0].morpheme_tags, ["Verb", "Evid", "While"]);
}

#[test]
fn a_single_letter_c_d_l_or_m_is_no_numeral() {
    let morphology = TurkishMorphology::new();
    for word in ["L.", "C.", "D.", "M."] {
        assert!(
            morphology
                .analyze(word)
                .iter()
                .all(|p| p.primary_pos != PrimaryPos::Num),
            "{word}"
        );
    }
    assert_eq!(first("V.").primary_pos, PrimaryPos::Num);
    assert_eq!(first("XIX.").primary_pos, PrimaryPos::Num);
}
