//! Analyses the morphotactics used to invent, with the rule each one now follows.
use akana_core::morphology::*;

fn roots(word: &str) -> Vec<String> {
    TurkishMorphology::new()
        .analyze(word)
        .into_iter()
        .map(|p| p.root)
        .collect()
}

fn has_tags(word: &str, root: &str, tags: &[&str]) -> bool {
    TurkishMorphology::new().analyze(word).iter().any(|p| {
        p.root == root
            && p.morpheme_tags
                .iter()
                .map(String::as_str)
                .eq(tags.iter().copied())
    })
}

#[test]
fn k_drop_only_before_the_diminutive() {
    // ben is not bek-n, diyor is not dik-yor, konuda is not konuk-da.
    assert!(!roots("ben").contains(&"bek".to_string()));
    assert!(!roots("diyor").contains(&"dik".to_string()));
    assert!(!roots("konuda").contains(&"konuk".to_string()));
    // küçü-cük still has its diminutive analysis.
    assert!(roots("küçücük").contains(&"küçük".to_string()));
}

#[test]
fn closed_classes_take_no_suffixes() {
    // yo-la (Interj + Ins), he-m (Adv/Interj + P1sg), fır-ın-ın (Adv + P2sg + Gen).
    for word in ["yola", "hem", "fırının", "çoğu"] {
        for p in TurkishMorphology::new().analyze(word) {
            let closed = matches!(
                p.primary_pos,
                PrimaryPos::Adv
                    | PrimaryPos::Conj
                    | PrimaryPos::Interj
                    | PrimaryPos::Det
                    | PrimaryPos::Postp
            );
            assert!(
                !(closed && p.morpheme_tags.len() > 1),
                "{word}: {}",
                p.formatted
            );
        }
    }
}

#[test]
fn lexical_derivations_attach_to_the_bare_root() {
    assert!(has_tags("sevgi", "sev", &["Verb", "DevGi"]));
    // kork-t-um is not Caus + DevN; değ-il-im is not Pass + DevN.
    assert!(!has_tags("korktum", "kork", &["Verb", "Caus", "DevN"]));
    assert!(!roots("değilim").contains(&"değ".to_string()));
}

#[test]
fn passive_allomorphs_follow_the_stem() {
    assert!(has_tags("okundu", "oku", &["Verb", "Pass", "Past"]));
    assert!(has_tags("bilinir", "bil", &["Verb", "Pass", "Aor"]));
    assert!(has_tags("yapıldı", "yap", &["Verb", "Pass", "Past"]));
    // A bare -l is never a passive (ka-l-dı-m used to be kak-l-dı-m).
    assert!(!roots("kaldım").contains(&"kak".to_string()));
}

#[test]
fn de_and_ye_narrow_before_y() {
    assert!(has_tags("diyecek", "de", &["Verb", "Fut"]));
    assert!(has_tags("diyerek", "de", &["Verb", "AdvErek"]));
    assert!(has_tags("yiyen", "ye", &["Verb", "PresPart"]));
}

#[test]
fn bare_verb_stems_are_imperatives() {
    assert!(has_tags("gel", "gel", &["Verb", "Imp", "A2sg"]));
    let tokens = ["Bu", "sabah", "işe", "geç", "kaldım"];
    let chosen = MorphologicalDisambiguator::new().disambiguate(&tokens);
    assert_eq!(chosen[2].root, "iş");
    assert_eq!(chosen[4].primary_pos, PrimaryPos::Verb);
}

#[test]
fn predicates_are_not_attributive() {
    // "yoğundu" closes a clause, so "ki" is the conjunction, not k-ı after an adjective.
    let sentence = "Bu sabah trafik o kadar yoğundu ki işe bir saat geç kaldım";
    let tokens: Vec<&str> = sentence.split_whitespace().collect();
    let chosen = MorphologicalDisambiguator::new().disambiguate(&tokens);
    let ki = chosen.iter().find(|p| p.surface == "ki").expect("ki");
    assert_eq!(ki.primary_pos, PrimaryPos::Conj);
    let kaldim = chosen
        .iter()
        .find(|p| p.surface == "kaldım")
        .expect("kaldım");
    assert_eq!(kaldim.pos, PrimaryPos::Verb);
}
