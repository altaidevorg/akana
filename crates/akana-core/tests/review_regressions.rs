//! Regressions for the code-review items on fix/morphology-echocraft-findings that held up
//! (see each test); items that did not are documented in the review notes.
use akana_core::morphology::*;

#[test]
fn past_copula_after_back_vowel_tenses() {
    // Review item 4: the VerbTense -> CopPast templates listed ydu twice and lacked ydı.
    let morph = TurkishMorphology::new();
    let has = |word: &str, tags: &[&str]| {
        morph.analyze(word).iter().any(|p| {
            let mut rest = p.morpheme_tags.iter();
            tags.iter().all(|t| rest.any(|m| m == t))
        })
    };
    assert!(has("olsaydı", &["Cond", "CopPast"]));
    assert!(has("kalsaydık", &["Cond", "CopPast", "A1pl"]));
    assert!(has("yapmalıydı", &["Nec", "CopPast"]));
    assert!(has("gelseydi", &["Cond", "CopPast"]));
}

#[test]
fn passive_has_no_bare_l() {
    // Review item 1 proposed accepting a bare -l passive after consonants. Turkish has none
    // (Zemberek: In / InIl / +nIl / Il); yap-l-dı is not a word.
    let morph = TurkishMorphology::new();
    assert!(morph.analyze("yapldı").is_empty());
    assert!(morph
        .analyze("yapıldı")
        .iter()
        .any(|p| p.morpheme_tags.iter().any(|t| t == "Pass")));
}

#[test]
fn lexicon_buckets_hold_no_identical_items() {
    // Review item 2: "zam" was listed twice in the built-in list (and again in the generated
    // lexicon).
    let lexicon = RootLexicon::new();
    for root in ["zam", "kitap", "ekmek", "hak", "ayşe"] {
        let items = lexicon.get_items(root).expect(root);
        for (i, a) in items.iter().enumerate() {
            assert!(!items[i + 1..].contains(a), "{root}: duplicate item {a:?}");
        }
    }
}

#[test]
fn readings_differing_only_in_secondary_pos_are_kept_once_each() {
    // Review item 3: deduplication keyed on the analysis string alone, among adjacent
    // entries, dropped a proper-noun reading when it tied with the common noun (Ekmek then,
    // whose name entry is gone since; Toprak here).
    let morph = TurkishMorphology::new();
    let parses = morph.analyze("Toprak");
    let bare: Vec<_> = parses
        .iter()
        .filter(|p| p.formatted == "[toprak:Noun] toprak:Noun+A3sg+Nom")
        .collect();
    assert!(bare
        .iter()
        .any(|p| p.secondary_pos == SecondaryPos::ProperNoun));
    assert!(bare.iter().any(|p| p.secondary_pos == SecondaryPos::None));
    for word in ["Toprak", "zammı", "abi", "İstanbulun", "akarsu"] {
        let parses = morph.analyze(word);
        for (i, a) in parses.iter().enumerate() {
            assert!(
                !parses[i + 1..]
                    .iter()
                    .any(|b| b.formatted == a.formatted && b.secondary_pos == a.secondary_pos),
                "{word}: repeated reading {}",
                a.formatted
            );
        }
    }
}

#[test]
fn names_keep_their_reading_when_a_common_homograph_is_kept_too() {
    // With both readings of "Ali" kept (the name and the adjective "ali"), the capitalised
    // name still wins at the start of the sentence.
    let tokens = ["Ali", "güzel", "kitap", "okudu"];
    let chosen = MorphologicalDisambiguator::new().disambiguate(&tokens);
    assert_eq!(chosen[0].lemma, "ali");
    assert_eq!(chosen[0].secondary_pos, SecondaryPos::ProperNoun);
}
