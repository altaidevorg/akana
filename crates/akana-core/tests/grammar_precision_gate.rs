//! Precision gate for the grammar checker's default rules.
//!
//! `data/grammar_clean_court.txt` holds 1,500 correct, edited sentences from Turkish court
//! decisions (built by `scripts/build_grammar_clean_corpus.py`). A default rule may fire on at
//! most 1 in 1,000 of them, and `correct` may change at most `MAX_CHANGED` of them; anything
//! more is a rule damaging correct Turkish. On failure the message gives every default rule's
//! count and the sentences it fired on. To see how a rule does on more text, run
//! `cargo run --release -p akana-core --example grammar_rule_report`.

use akana_core::grammar::{GrammarFinding, GrammarRule, RuleSet, TurkishGrammarChecker};
use std::collections::BTreeMap;
use std::fmt::Write;

const CORPUS: &str = include_str!("data/grammar_clean_court.txt");

/// Clean sentences `correct` may change (0.2 %).
const MAX_CHANGED: usize = 3;

fn clean_sentences() -> Vec<&'static str> {
    CORPUS
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .collect()
}

/// `f` on every sentence, spread over the machine's threads (the test profile is lightly
/// optimised, and 1,500 checks one after the other take a while).
fn check_all(
    checker: &TurkishGrammarChecker,
    sentences: &[&'static str],
) -> Vec<(Vec<GrammarFinding>, String)> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = sentences.len().div_ceil(threads);
    std::thread::scope(|s| {
        let handles: Vec<_> = sentences
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || {
                    part.iter()
                        .map(|text| {
                            let res = checker.check(text);
                            (res.findings, res.corrected)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    })
}

#[test]
fn default_rules_stay_quiet_on_clean_text() {
    let sentences = clean_sentences();
    assert!(sentences.len() >= 1000, "{} sentences", sentences.len());
    let checker = TurkishGrammarChecker::new();
    let budget = (sentences.len() / 1000).max(1);

    let results = check_all(&checker, &sentences);
    let mut fired: BTreeMap<GrammarRule, Vec<String>> = BTreeMap::new();
    let mut changed: Vec<String> = Vec::new();
    for (text, (findings, corrected)) in sentences.iter().zip(&results) {
        let mut seen: Vec<GrammarRule> = Vec::new();
        for f in findings {
            if !seen.contains(&f.rule) {
                seen.push(f.rule);
                fired.entry(f.rule).or_default().push(format!(
                    "{:?} -> {:?} in: {text}",
                    f.original_text, f.replacement
                ));
            }
        }
        if corrected != text {
            changed.push(format!("{text}\n      -> {corrected}"));
        }
    }

    let over: Vec<GrammarRule> = fired
        .iter()
        .filter(|(_, hits)| hits.len() > budget)
        .map(|(&rule, _)| rule)
        .collect();
    if over.is_empty() && changed.len() <= MAX_CHANGED {
        return;
    }

    let mut report = format!(
        "{} clean sentences; a default rule may fire on {budget}, `correct` may change {MAX_CHANGED}.\n\
         Sentences each default rule fired on:\n",
        sentences.len()
    );
    for rule in RuleSet::default().iter() {
        let hits = fired.get(&rule).map_or(0, Vec::len);
        let mark = if hits > budget {
            "  <-- over budget"
        } else {
            ""
        };
        writeln!(report, "  {:<24} {hits}{mark}", rule.id()).unwrap();
    }
    for rule in &over {
        writeln!(report, "\n{}:", rule.id()).unwrap();
        for hit in fired[rule].iter().take(10) {
            writeln!(report, "  {hit}").unwrap();
        }
    }
    writeln!(report, "\n`correct` changed {} sentences:", changed.len()).unwrap();
    for c in changed.iter().take(15) {
        writeln!(report, "  {c}").unwrap();
    }
    panic!("{report}");
}

#[test]
fn experimental_rules_are_off_by_default() {
    let default = RuleSet::default();
    for &rule in GrammarRule::ALL {
        assert_eq!(default.contains(rule), rule.is_default(), "{}", rule.id());
    }
    assert!(!default.contains(GrammarRule::QuantityPlural));
    assert!(TurkishGrammarChecker::new().rules() == default);
}
