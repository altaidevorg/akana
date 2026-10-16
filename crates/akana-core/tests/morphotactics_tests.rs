//! Suffixes ported from Zemberek's TurkishMorphotactics that the graph lacked. Each word below
//! had no analysis before (EchoCraft's Akana findings, 2026-10-16).
use akana_core::morphology::*;
use akana_core::parser::*;

/// Whether some parse has this root and contains these tags in this order.
fn parses_as(word: &str, root: &str, tags: &[&str]) -> bool {
    TurkishMorphology::new().analyze(word).iter().any(|p| {
        p.root == root && {
            let mut rest = p.morpheme_tags.iter();
            tags.iter().all(|t| rest.any(|m| m == t))
        }
    })
}

#[test]
fn question_particle_takes_copula_and_person() {
    assert!(parses_as("miyiz", "mi", &["Q", "CopPres1pl"]));
    assert!(parses_as("misin", "mi", &["Q", "CopPres2sg"]));
    assert!(parses_as("musunuz", "mu", &["Q", "CopPres2pl"]));
    assert!(parses_as("miydim", "mi", &["Q", "CopPast", "A1sg"]));
    assert!(parses_as("mıymış", "mı", &["Q", "CopEvid"]));
}

#[test]
fn while_converb_ken() {
    assert!(parses_as("bakarken", "bak", &["Aor", "While"]));
    assert!(parses_as("geçerken", "geç", &["Aor", "While"]));
    assert!(parses_as("beklerken", "bekle", &["Aor", "While"]));
    assert!(parses_as("gelmezken", "gel", &["Neg", "Aor", "While"]));
    assert!(parses_as("çocukken", "çocuk", &["While"]));
    assert!(parses_as("evdeyken", "ev", &["Loc", "While"]));
}

#[test]
fn conditional_copula_after_tense() {
    assert!(parses_as("verirsen", "ver", &["Aor", "CopCond", "A2sg"]));
    assert!(parses_as(
        "bitirebilirsek",
        "bitir",
        &["Abil", "Aor", "CopCond", "A1pl"]
    ));
    assert!(parses_as("geldiyse", "gel", &["Past", "CopCond"]));
}

#[test]
fn progressive_after_vowel_final_stems() {
    // bekle-iyor -> bekliyor (Zemberek's ProgressiveVowelDrop).
    assert!(parses_as("özlüyoruz", "özle", &["Prog", "A1pl"]));
    assert!(parses_as("bekliyoruz", "bekle", &["Prog", "A1pl"]));
    assert!(parses_as("başlıyor", "başla", &["Prog"]));
    assert!(parses_as("diyor", "de", &["Prog"]));
}

#[test]
fn since_and_without_converbs() {
    assert!(parses_as("başlayalı", "başla", &["SinceDoingSo"]));
    assert!(parses_as("gelmeden", "gel", &["WithoutHavingDoneSo"]));
}

#[test]
fn verb_forms_after_negation_and_ability() {
    assert!(parses_as(
        "kalmamamı",
        "kal",
        &["Neg", "ActN", "P1sg", "Acc"]
    ));
    assert!(parses_as("gelmemek", "gel", &["Neg", "Inf"]));
    assert!(parses_as("gelmeyince", "gel", &["Neg", "AdvInce"]));
    assert!(parses_as("yapabilmek", "yap", &["Abil", "Inf"]));
    assert!(parses_as("bulamamak", "bul", &["NegAbil", "Inf"]));
}

#[test]
fn apostrophe_suffixes_on_names() {
    assert!(parses_as("Antalya'da", "antalya", &["Loc"]));
    assert!(parses_as("İstanbul'un", "istanbul", &["Gen"]));
    // A name the lexicon lacks still gets a proper-noun analysis.
    let guessed = TurkishMorphology::new().analyze("Zorluoğlu'na");
    assert!(guessed
        .iter()
        .any(|p| p.secondary_pos == SecondaryPos::ProperNoun
            && p.morpheme_tags.iter().any(|t| t == "Dat")));
}

#[test]
fn degil_is_a_negative_copula() {
    assert!(parses_as("değilim", "değil", &["CopPres1sg"]));
    assert!(parses_as("değildi", "değil", &["CopPast"]));
}

#[test]
fn derivations_from_zemberek() {
    assert!(parses_as("izleyici", "izle", &["Agt"]));
    assert!(parses_as("haksızlık", "hak", &["Without", "Ness"]));
    assert!(parses_as("duygusal", "duygu", &["Related"]));
    assert!(parses_as("açıkça", "açık", &["Ly"]));
    assert!(parses_as("ailece", "aile", &["Equ"]));
    assert!(parses_as("olsun", "ol", &["Imp"]));
    assert!(parses_as("okuyacağı", "oku", &["FutPart"]));
}

#[test]
fn derived_nouns_take_the_full_paradigm() {
    assert!(parses_as(
        "sabırsızlıkla",
        "sabır",
        &["Without", "Ness", "Ins"]
    ));
    assert!(parses_as("kitaplığı", "kitap", &["Ness", "P3sg"]));
    assert!(parses_as("yolcumuz", "yol", &["Agt", "P1pl"]));
}

#[test]
fn question_particle_attaches_to_its_host() {
    let parser = TurkishDependencyParser::new();
    for (sentence, host) in [
        ("Ayşe mi okula geldi", 1),
        ("Ayşe okula mı geldi", 2),
        ("Ayşe okula geldi mi", 3),
        ("Oraya varabilir miyiz", 2),
    ] {
        let tokens: Vec<&str> = sentence.split_whitespace().collect();
        let nodes = parser.parse(&tokens).nodes;
        let q = nodes
            .iter()
            .find(|n| n.form.starts_with('m'))
            .expect("question particle");
        assert_eq!(q.upos, "AUX", "{sentence}");
        assert_eq!(q.deprel, "aux:q", "{sentence}");
        assert_eq!(q.head, host, "{sentence}");
        let root = nodes.iter().find(|n| n.deprel == "root").expect("root");
        assert_eq!(root.upos, "VERB", "{sentence}");
    }
}
