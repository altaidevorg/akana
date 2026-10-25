//! Gold paradigms, misspellings and disambiguated sentences for the morphological analyzer.
//!
//! `data/morphology/paradigms.tsv` lists forms that must have an analysis with a given root and
//! exact tag sequence, `data/morphology/reject.txt` forms that must have none, and
//! `data/morphology/disambiguation.tsv` the reading the disambiguator should pick for each word
//! of a sentence. Every analysis of a dictionary word in `paradigms.tsv` and
//! `frequent_types.tsv`, and every analysis of the 800 frequent court words of
//! `roundtrip_words.txt`, must also generate back from its root (or lemma) and tags. The
//! analyzer does not meet all of them yet:
//! `data/morphology/known_failures.txt` lists the ones it gets wrong, and this test fails when
//! that list is not exactly the current failures, so it shrinks as fixes land and grows only on
//! purpose. `AKANA_GOLD_WRITE=1` rewrites the list.

use akana_core::morphology::{MorphologicalDisambiguator, TurkishMorphology};
use std::collections::BTreeSet;

const PARADIGMS: &str = include_str!("data/morphology/paradigms.tsv");
const REJECT: &str = include_str!("data/morphology/reject.txt");
const DISAMBIGUATION: &str = include_str!("data/morphology/disambiguation.tsv");
const FREQUENT_TYPES: &str = include_str!("data/morphology/frequent_types.tsv");
const ROUNDTRIP_WORDS: &str = include_str!("data/morphology/roundtrip_words.txt");
const KNOWN_FAILURES: &str = include_str!("data/morphology/known_failures.txt");

fn data_lines(text: &str) -> impl Iterator<Item = &str> {
    text.lines()
        .map(str::trim_end)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
}

/// Whether a gold word is a dictionary word written as the generator writes it: its root (or
/// infinitive) is in the lexicon (not a number or Roman numeral), and it has no apostrophe
/// (Ankara'ya).
fn generable(morphology: &TurkishMorphology, word: &str, root: &str) -> bool {
    let in_lexicon = |r: &str| !morphology.root_items(r).is_empty();
    !word.contains(['\'', '’', '`'])
        && (in_lexicon(root)
            || root
                .strip_suffix("mek")
                .or_else(|| root.strip_suffix("mak"))
                .is_some_and(in_lexicon))
}

/// Every gold item the analyzer currently gets wrong, as `accept<TAB>word<TAB>root<TAB>tags`,
/// `reject<TAB>word`, `disamb<TAB>word<TAB>sentence` or `generate<TAB>word<TAB>root<TAB>tags`
/// (a gold or round-trip reading the generator cannot produce).
fn current_failures() -> BTreeSet<String> {
    let morphology = TurkishMorphology::new();
    let mut failures = BTreeSet::new();
    // The generator returns a word the analyzer reads back with the same item and tags; another
    // spelling of the reading (gelin and geliniz) counts as a round trip.
    let generates = |root: &str, tags: &str| {
        let tags: Vec<&str> = tags.split('+').collect();
        morphology.generate(root, &tags).is_some()
    };
    for line in data_lines(PARADIGMS) {
        let fields: Vec<&str> = line.split('\t').collect();
        let [word, root, tags] = fields[..] else {
            panic!("malformed paradigms.tsv line: {line:?}");
        };
        let found = morphology
            .analyze(word)
            .iter()
            .any(|p| p.root == root && p.morpheme_tags.join("+") == tags);
        if !found {
            failures.insert(format!("accept\t{word}\t{root}\t{tags}"));
        } else if generable(&morphology, word, root) && !generates(root, tags) {
            failures.insert(format!("generate\t{word}\t{root}\t{tags}"));
        }
    }
    for line in data_lines(FREQUENT_TYPES) {
        let fields: Vec<&str> = line.split('\t').collect();
        let [word, lemma, _, tags] = fields[..] else {
            panic!("malformed frequent_types.tsv line: {line:?}");
        };
        if generable(&morphology, word, lemma) && !generates(lemma, tags) {
            failures.insert(format!("generate\t{word}\t{lemma}\t{tags}"));
        }
    }
    for word in data_lines(ROUNDTRIP_WORDS) {
        for p in morphology.analyze(word) {
            let tags = p.morpheme_tags.join("+");
            if generable(&morphology, word, &p.root) && !generates(&p.root, &tags) {
                failures.insert(format!("generate\t{word}\t{}\t{tags}", p.root));
            }
        }
    }
    for word in data_lines(REJECT) {
        if !morphology.analyze(word).is_empty() {
            failures.insert(format!("reject\t{word}"));
        }
    }
    let disambiguator = MorphologicalDisambiguator::new();
    let mut lines = data_lines(DISAMBIGUATION).peekable();
    while let Some(line) = lines.next() {
        let sentence = line
            .strip_prefix("> ")
            .unwrap_or_else(|| panic!("expected a sentence line: {line:?}"));
        let tokens: Vec<&str> = sentence.split(' ').collect();
        let chosen = disambiguator.disambiguate(&tokens);
        for (token, parse) in tokens.iter().zip(&chosen) {
            if token.chars().all(|c| c.is_ascii_punctuation()) {
                continue;
            }
            let expected = lines
                .next_if(|l| !l.starts_with("> "))
                .unwrap_or_else(|| panic!("no gold line for {token:?} in {sentence:?}"));
            let fields: Vec<&str> = expected.split('\t').collect();
            assert!(
                fields[0] == *token && fields.len() % 2 == 1,
                "gold line {expected:?} does not match {token:?} in {sentence:?}"
            );
            let tags = parse.morpheme_tags.join("+");
            let right = fields[1..]
                .chunks(2)
                .any(|pair| parse.root == pair[0] && tags == pair[1]);
            if !right {
                failures.insert(format!("disamb\t{token}\t{sentence}"));
            }
        }
    }
    failures
}

#[test]
fn gold_failures_match_known_list() {
    let current = current_failures();
    let known: BTreeSet<String> = data_lines(KNOWN_FAILURES).map(str::to_string).collect();
    if std::env::var_os("AKANA_GOLD_WRITE").is_some() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/morphology/known_failures.txt"
        );
        let mut text = String::from(
            "# Gold items the analyzer gets wrong today (see tests/morphology_gold.rs).\n",
        );
        for f in &current {
            text.push_str(f);
            text.push('\n');
        }
        std::fs::write(path, text).expect("write known_failures.txt");
        return;
    }
    let fixed: Vec<&String> = known.difference(&current).collect();
    let broken: Vec<&String> = current.difference(&known).collect();
    let morphology = TurkishMorphology::new();
    let mut report = String::new();
    let disambiguator = MorphologicalDisambiguator::new();
    for f in &broken {
        let fields: Vec<&str> = f.split('\t').collect();
        let got: Vec<String> = if fields[0] == "generate" {
            let tags: Vec<&str> = fields[3].split('+').collect();
            vec![format!("{:?}", morphology.generate(fields[2], &tags))]
        } else if fields[0] == "disamb" {
            let tokens: Vec<&str> = fields[2].split(' ').collect();
            let chosen = disambiguator.disambiguate(&tokens);
            tokens
                .iter()
                .zip(chosen)
                .filter(|(t, _)| **t == fields[1])
                .map(|(_, p)| p.formatted)
                .collect()
        } else {
            morphology
                .analyze(fields[1])
                .iter()
                .map(|p| p.formatted.clone())
                .collect()
        };
        report.push_str(&format!("  newly failing: {f}\n      got: {got:?}\n"));
    }
    for f in &fixed {
        report.push_str(&format!(
            "  now passing (remove from known_failures.txt): {f}\n"
        ));
    }
    assert!(
        fixed.is_empty() && broken.is_empty(),
        "gold results differ from known_failures.txt ({} known, {} now):\n{report}",
        known.len(),
        current.len()
    );
}
