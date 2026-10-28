//! Review 4's downstream findings that move no vector: the de/da rule beside names
//! (DOWN4-11), the deasciifier on words it should keep (DOWN4-12) and readability's leftovers
//! (DOWN4-13).

use akana_core::grammar::{GrammarRule, TurkishGrammarChecker};
use akana_core::normalization::TurkishDeasciifier;
use akana_core::readability::analyze_readability;

#[test]
fn the_de_da_rule_tells_a_name_from_words() {
    let checker = TurkishGrammarChecker::new();
    let corrected = |text: &str| {
        checker
            .check(text)
            .findings
            .iter()
            .filter(|f| f.rule == GrammarRule::DeDaAfterVerb)
            .map(|f| f.replacement.clone())
            .collect::<Vec<_>>()
    };
    // A suffix after an apostrophe continues a foreign name.
    assert!(corrected("Rossii po futbolu'da oynadı.").is_empty());
    // After a known name or a word the clitic is split as everywhere.
    assert_eq!(corrected("Dün Ayşe bunu'da yaptı."), ["bunu da"]);
    assert_eq!(corrected("Sonra Ali gitti'de geldi."), ["gitti de"]);
    // At a sentence's start a capital before a name does not make a name.
    assert_eq!(corrected("Gerçektende Konya iyi oynadı."), ["Gerçekten de"]);
}

#[test]
fn the_deasciifier_keeps_what_is_written_turkish_or_as_a_name() {
    for word in ["AİHM", "İSKİ", "İnsan", "iPhone", "McDonald"] {
        assert_eq!(TurkishDeasciifier::deasciify(word), word);
    }
    assert_eq!(TurkishDeasciifier::deasciify("cocuk"), "çocuk");
    // A word in capitals with its suffix is still read (it is no name written so).
    assert_eq!(TurkishDeasciifier::deasciify("ISTANBUL'da"), "İSTANBUL'da");
}

#[test]
fn readability_counts_words_and_sentences_and_keeps_its_scores() {
    // A heading's letter is no word, nor an unfamiliar one.
    let with = analyze_readability("A. Başvuru kabul edildi.");
    let without = analyze_readability("Başvuru kabul edildi.");
    assert_eq!(with.statistics.total_words, without.statistics.total_words);
    assert_eq!(
        with.statistics.unfamiliar_words,
        without.statistics.unfamiliar_words
    );
    // No text has no sentence.
    assert_eq!(analyze_readability("").statistics.total_sentences, 0);
    // An endless sentence gives no negative score (K1 gave -5,236).
    let long = vec!["karar"; 8_000].join(" ");
    let k = analyze_readability(&long).kalyoncu;
    for f in [&k.formula1, &k.formula2, &k.formula3, &k.formula4] {
        assert!(f.score >= 0.0, "{}", f.score);
    }
}
