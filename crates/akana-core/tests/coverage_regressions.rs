//! Frequent words 0.7.0 left without any analysis (found by diffing the analyses of RagTurk
//! Wikipedia chunks and Turkuaz news against the pre-0.7.0 build), each with the Zemberek rule it
//! now follows, and the forms those rules must still reject.
use akana_core::morphology::*;

fn has_tags(word: &str, root: &str, tags: &[&str]) -> bool {
    TurkishMorphology::new().analyze(word).iter().any(|p| {
        p.root == root
            && p.morpheme_tags
                .iter()
                .map(String::as_str)
                .eq(tags.iter().copied())
    })
}

fn best(word: &str) -> String {
    TurkishMorphology::new()
        .analyze(word)
        .first()
        .map(|p| p.formatted.clone())
        .unwrap_or_default()
}

#[test]
fn vowel_drop_and_voicing_together() {
    // kayıp -> kayb-ı, akit -> akd-i, nakit -> nakd-i: roots marked VowelDrop and Voicing.
    assert!(has_tags("kaybı", "kayıp", &["Noun", "P3sg"]));
    assert!(has_tags("kaybına", "kayıp", &["Noun", "P3sg", "Dat"]));
    assert!(has_tags("akdi", "akit", &["Noun", "P3sg"]));
    assert!(has_tags("akdini", "akit", &["Noun", "P3sg", "Acc"]));
    assert!(has_tags("nakdi", "nakit", &["Noun", "P3sg"]));
    assert!(has_tags("ahdi", "ahit", &["Noun", "P3sg"]));
    // Each change still needs its own attribute: kitap voices but keeps its vowel.
    assert!(!has_tags("kitbı", "kitap", &["Noun", "P3sg"]));
}

#[test]
fn derived_adjectives_take_the_nominal_paradigm() {
    // adjZeroDeriv_S -> noun_S: görev-li-si, görev-li-ye, gel-en-e, gel-en-in.
    assert!(has_tags("görevlisi", "görev", &["Noun", "With", "P3sg"]));
    assert!(has_tags("görevliye", "görev", &["Noun", "With", "Dat"]));
    assert!(has_tags("görevlinin", "görev", &["Noun", "With", "Gen"]));
    assert!(has_tags("görevliyle", "görev", &["Noun", "With", "Ins"]));
    assert!(has_tags("gelene", "gel", &["Verb", "PresPart", "Dat"]));
    assert!(has_tags("gelenin", "gel", &["Verb", "PresPart", "Gen"]));
    // -DIk / -AcAk participles keep their possessive reading first, with no bare case beside
    // it: ol-duğ-u is P3sg, not Acc.
    assert!(!has_tags("olduğu", "ol", &["Verb", "PastPart", "Acc"]));
    assert!(!has_tags("geldiğin", "gel", &["Verb", "PastPart", "Gen"]));
    assert!(!has_tags("yapacağı", "yap", &["Verb", "FutPart", "Acc"]));
    assert!(has_tags("olduğu", "ol", &["Verb", "PastPart", "P3sg"]));
}

#[test]
fn postpositions_take_the_nominal_copula() {
    // postpZero_S -> nVerb_S: ait-tir, ait-ti, gibi-yim, kadar-dı.
    assert!(has_tags("aittir", "ait", &["Postp", "CopPres3sg"]));
    assert!(has_tags("aitti", "ait", &["Postp", "CopPast"]));
    assert!(has_tags("gibiyim", "gibi", &["Postp", "CopPres1sg"]));
    assert!(has_tags("kadardı", "kadar", &["Postp", "CopPast"]));
    // Still no case or possessive on a postposition.
    assert!(!has_tags("aite", "ait", &["Postp", "Dat"]));
    assert!(!has_tags("içinim", "için", &["Postp", "P1sg"]));
}

#[test]
fn oyle_boyle_soyle_take_the_copula() {
    // advForVerbDeriv_ST -> nVerb_S; other adverbs stay terminal.
    assert!(has_tags("şöyledir", "şöyle", &["Adv", "CopPres3sg"]));
    assert!(has_tags("şöyleydi", "şöyle", &["Adv", "CopPast"]));
    assert!(TurkishMorphology::new()
        .analyze("hemendir")
        .iter()
        .all(|p| p.primary_pos != PrimaryPos::Adv));
}

#[test]
fn persons_after_a_tense_take_the_copula() {
    // vA3pl_ST -> PastAfterTense / NarrAfterTense / Cond / Cop / While, vA1sg_ST.. -> Cop.
    assert!(has_tags("yapıyorlardı", "yap", &["Verb", "Prog", "A3pl", "CopPast"]));
    assert!(has_tags("gelmişlerdi", "gel", &["Verb", "Evid", "A3pl", "CopPast"]));
    assert!(has_tags("vermişlerdir", "ver", &["Verb", "Evid", "A3pl", "Cop"]));
    assert!(has_tags("gelirlermiş", "gel", &["Verb", "Aor", "A3pl", "CopEvid"]));
    assert!(has_tags("geliyorlarsa", "gel", &["Verb", "Prog", "A3pl", "CopCond"]));
    assert!(has_tags("gelirlerken", "gel", &["Verb", "Aor", "A3pl", "While"]));
    assert!(has_tags("gelmişimdir", "gel", &["Verb", "Evid", "A1sg", "Cop"]));
    assert!(has_tags("olmuşlardır", "ol", &["Verb", "Evid", "A3pl", "Cop"]));
    // Not after the past tense, not after a nominal copula, and only -DIr after a non-A3pl
    // person.
    let morph = TurkishMorphology::new();
    for word in ["geldilerdi", "geldilerdir", "evdeyimdi", "gelmişimdi"] {
        assert!(morph.analyze(word).is_empty(), "{word}: {}", best(word));
    }
}

#[test]
fn negative_aorist_participle() {
    // vAorPartNeg_S -> adjAfterVerb_ST: bil-me-z-lik.
    assert!(has_tags("bilmezlik", "bil", &["Verb", "Neg", "AorPart", "Ness"]));
    assert!(has_tags("gelmezlik", "gel", &["Verb", "Neg", "AorPart", "Ness"]));
    // The aorist tense stays the first reading of the bare form.
    assert!(best("bilmez").contains("Neg+Aor"), "{}", best("bilmez"));
}
