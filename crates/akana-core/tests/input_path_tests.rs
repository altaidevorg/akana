//! The input path of the sparse and hybrid vectors as review 4 left it (MORPH4-01..06): the
//! tokens, sentences and terms of court-style text.

use akana_core::embeddings::MorphologicalSparseEncoder;
use akana_core::tokenization::{SentenceSegmenter, TurkishTokenizer};

fn terms(encoder: &MorphologicalSparseEncoder, text: &str) -> Vec<String> {
    let mut terms = encoder.encode_sparse(text).terms;
    terms.sort();
    terms
}

fn sentences(text: &str) -> Vec<&str> {
    SentenceSegmenter::segment(text)
        .into_iter()
        .map(|s| s.text)
        .collect()
}

#[test]
fn numerals_abbreviations_and_initials_give_one_term_each() {
    let encoder = MorphologicalSparseEncoder::new();
    let t = |text: &str| terms(&encoder, text);
    // A numeral's term has no period; one letter is no term (MORPH4-01).
    assert_eq!(t("Kanunun I. maddesi"), ["kanun", "madde"]);
    assert_eq!(t("XIX. yüzyıl"), t("XIX yüzyıl"));
    // An abbreviation shaped like a Roman numeral is the abbreviation (MORPH4-01).
    assert_eq!(t("Yargıtay 9. CD. kararı"), ["cd", "karar", "yargıtay"]);
    assert_eq!(t("TCK MD. 5"), t("TCK md 5"));
    // Initials are one token and their letters, bare or with a suffix (MORPH4-04).
    assert_eq!(
        TurkishTokenizer::tokenize_words("H.M.K. ile"),
        ["H.M.K.", "ile"]
    );
    assert_eq!(t("H.M.K. ile"), t("H.M.K.'nın"));
    assert_eq!(t("K.A.nın ifadesi"), ["ifade", "ka"]);
    assert_eq!(t("M.E.N.nin"), ["men"]);
}

#[test]
fn a_name_has_one_term_wherever_it_stands() {
    // MORPH4-02: at a sentence's start, inside it and in capitals.
    let encoder = MorphologicalSparseEncoder::new();
    let has = |text: &str, term: &str| terms(&encoder, text).contains(&term.to_string());
    for text in [
        "Bozboğa geldi.",
        "Davalı Bozboğa geldi.",
        "DAVALI BOZBOĞA GELDİ.",
    ] {
        assert!(has(text, "bozboğa"), "{text}");
    }
    // Inside a sentence and in capitals the capital makes the name; at the start it does not,
    // and the word gives the stem the same word in lower case gives too.
    assert_eq!(
        terms(&encoder, "Davalı Bozboğa geldi."),
        ["bozboğa", "davalı", "gelmek"]
    );
    assert!(has("Bu makrofajlar çoğalır.", "makrofaj"));
    assert!(has("Makrofajlar çoğalır.", "makrofaj"));
    assert!(has("Makrofajlar çoğalır.", "makrofajlar"));
    assert!(has("(Makrofajlar) çoğalır.", "makrofaj"));
}

#[test]
fn suffixes_written_apart_are_no_terms() {
    // MORPH4-03: after a number, an acronym's period, an ellipsis, a glued period, a quote.
    let encoder = MorphologicalSparseEncoder::new();
    let t = |text: &str| terms(&encoder, text);
    assert_eq!(t("Kanunun 65 inci maddesi"), ["65", "kanun", "madde"]);
    assert_eq!(t("TTK. nun 760. maddesinde"), ["760", "madde", "ttk"]);
    assert_eq!(t("Davacı ... ın vekili"), ["davacı", "vekil"]);
    assert_eq!(t("%20 si"), ["20"]);
    assert_eq!(t("Şti.nin"), ["şirket"]);
    assert_eq!(t("\"Oy\"unda"), ["oy"]);
}

#[test]
fn decomposed_text_gives_the_terms_of_composed_text() {
    // MORPH4-05: s + U+0327 is ş.
    let encoder = MorphologicalSparseEncoder::new();
    assert_eq!(
        terms(&encoder, "s\u{327}irketin c\u{327}alıs\u{327}anları"),
        terms(&encoder, "şirketin çalışanları")
    );
}

#[test]
fn court_sentences_are_not_cut_inside() {
    // MORPH4-06: a redaction's ellipsis, an ellipsis before a comma, a quotation that goes on
    // in lower case, a Roman ordinal, an acronym's period, an amount's dash, a time range.
    for text in [
        "Tire ... Asliye Ceza Mahkemesine ... Esas sayılı dosya geldi.",
        "E:..., K:... sayılı karar verildi.",
        "kuralı belirler.'' şeklinde yazıldı.",
        "II. Dünya Savaşı bitti.",
        "TTK. nun 760. maddesi uygulandı.",
        "12.500.-TL ödendi.",
    ] {
        assert_eq!(sentences(text), [text], "{text}");
    }
    assert_eq!(
        sentences("Davacı geldi... Sonra gitti."),
        ["Davacı geldi...", "Sonra gitti."]
    );
    assert_eq!(
        TurkishTokenizer::tokenize_words("saat 15.30-17.00 arası"),
        ["saat", "15.30", "-", "17.00", "arası"]
    );
}
