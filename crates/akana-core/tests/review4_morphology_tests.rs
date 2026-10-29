//! Review 4's smaller analyzer and lexicon findings (MORPH4-10): names the lexicon lacked,
//! lower-case words read as names, abbreviations of a phrase with a possessive, abbreviations of
//! one word, a tie between two spellings of one reading, and single letters read as numerals.

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
    // Serpil read as serp-il (serpmek), Gökçen and Özgen with a P2sg, Halide as hali-de.
    for name in ["Serpil", "Gökçen", "Özgen", "Halide"] {
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
    ] {
        assert_eq!(first(word).lemma, lemma, "{word}");
    }
    let encoder = MorphologicalSparseEncoder::new();
    let terms = |text: &str| encoder.encode_sparse(text).terms;
    assert_eq!(terms("TCK md. 5"), terms("TCK madde 5"));
    assert!(!terms("Av. Ayşe Kaya").contains(&"av".to_string()));
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
