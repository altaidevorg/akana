//! Measures every grammar rule on clean text and on GECTurk.
//!
//! ```text
//! cargo run --release -p akana-core --example grammar_rule_report -- \
//!     --clean court=crates/akana-core/tests/data/grammar_clean_court.txt \
//!     --clean ragturk=ragturk.txt --gec gecturk=gecturk_test.tsv [--examples 8]
//! ```
//!
//! A clean file has one correct sentence per line (`#` lines are skipped). A GECTurk file has
//! one row per line, `labels<TAB>tokens`: the labels comma-separated, the tokens joined by
//! single spaces (`scripts/build_grammar_clean_corpus.py --local` writes both).
//!
//! Each rule runs alone. For clean text the report gives the sentences it fires on, per 1,000;
//! for GECTurk, the share of its findings that overlap a token labelled as an error
//! (precision) and how often it fires on the sentences with no error. It ends with how many
//! clean sentences `correct` changes with the default rules and with all of them, and the
//! per-label recall of both sets on GECTurk. `--context RULE` (repeatable) also prints every
//! finding of that rule in its sentence, to judge it by hand.

use akana_core::grammar::{GrammarFinding, GrammarRule, RuleSet, TurkishGrammarChecker};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};

struct GecRow {
    text: String,
    /// byte span and label of each token
    tokens: Vec<(usize, usize, u32)>,
}

fn read_clean(path: &str) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect()
}

fn read_gec(path: &str) -> Vec<GecRow> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    text.lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            let (labels, toks) = line.split_once('\t').expect("labels<TAB>tokens");
            let labels: Vec<u32> = labels.split(',').map(|l| l.parse().unwrap()).collect();
            let words: Vec<&str> = toks.split(' ').collect();
            assert_eq!(labels.len(), words.len(), "{line}");
            let mut tokens = Vec::with_capacity(words.len());
            let mut pos = 0;
            for (w, &l) in words.iter().zip(&labels) {
                tokens.push((pos, pos + w.len(), l));
                pos += w.len() + 1;
            }
            GecRow {
                text: toks.to_string(),
                tokens,
            }
        })
        .collect()
}

/// `f(i)` for every index below `n`, spread over the machine's threads.
fn par_map<T: Send, F: Fn(usize) -> T + Sync>(n: usize, f: F) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = AtomicUsize::new(0);
    let mut out: Vec<(usize, T)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break mine;
                        }
                        mine.push((i, f(i)));
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    });
    out.sort_by_key(|(i, _)| *i);
    out.into_iter().map(|(_, t)| t).collect()
}

/// `text` with the finding's span marked, cut to about 60 characters on each side.
fn in_context(text: &str, f: &GrammarFinding) -> String {
    let before: String = text[..f.start_offset].chars().rev().take(60).collect();
    let before: String = before.chars().rev().collect();
    let after: String = text[f.end_offset..].chars().take(60).collect();
    format!(
        "{before}[[{}]]{after}  =>  {:?}",
        &text[f.start_offset..f.end_offset],
        f.replacement
    )
}

fn example(f: &GrammarFinding) -> String {
    format!("{:?} -> {:?}", f.original_text, f.replacement)
}

fn top(counts: &HashMap<String, usize>, n: usize) -> String {
    let mut v: Vec<_> = counts.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    v.iter()
        .take(n)
        .map(|(k, c)| format!("{k} ×{c}"))
        .collect::<Vec<_>>()
        .join(" | ")
}

fn main() {
    let mut clean: Vec<(String, Vec<String>)> = Vec::new();
    let mut gec: Vec<(String, Vec<GecRow>)> = Vec::new();
    let mut examples = 6;
    let mut context: Vec<GrammarRule> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--clean" | "--gec" => {
                let v = args.next().expect("NAME=PATH");
                let (name, path) = v.split_once('=').expect("NAME=PATH");
                if a == "--clean" {
                    clean.push((name.to_string(), read_clean(path)));
                } else {
                    gec.push((name.to_string(), read_gec(path)));
                }
            }
            "--examples" => examples = args.next().unwrap().parse().unwrap(),
            "--context" => context.push(args.next().unwrap().parse().unwrap()),
            _ => panic!("unknown argument {a}"),
        }
    }

    let checker = TurkishGrammarChecker::new();
    let rules = GrammarRule::ALL;

    // clean text: per rule, the sentences it fires on and what it does there
    println!("# Clean text: sentences a rule fires on (each rule alone)\n");
    let mut clean_rates: BTreeMap<(GrammarRule, String), (usize, usize)> = BTreeMap::new();
    for (name, sents) in &clean {
        let per_sentence: Vec<Vec<GrammarFinding>> = par_map(sents.len(), |i| {
            rules
                .iter()
                .flat_map(|&r| {
                    checker
                        .check_with_rules(&sents[i], RuleSet::empty().with(r))
                        .findings
                })
                .collect()
        });
        println!("## {name}: {} sentences\n", sents.len());
        for &rule in rules {
            let mut fired = 0;
            let mut ex: HashMap<String, usize> = HashMap::new();
            for fs in &per_sentence {
                let mine: Vec<_> = fs.iter().filter(|f| f.rule == rule).collect();
                if !mine.is_empty() {
                    fired += 1;
                }
                for f in mine {
                    *ex.entry(example(f)).or_default() += 1;
                }
            }
            clean_rates.insert((rule, name.clone()), (fired, sents.len()));
            if context.contains(&rule) {
                for (s, fs) in sents.iter().zip(&per_sentence) {
                    for f in fs.iter().filter(|f| f.rule == rule) {
                        println!("    CTX {} | {}", rule.id(), in_context(s, f));
                    }
                }
            }
            if fired > 0 {
                println!(
                    "- {:<24} {:>4} ({:.2}/1000)  {}",
                    rule.id(),
                    fired,
                    1000.0 * fired as f64 / sents.len() as f64,
                    top(&ex, examples)
                );
            }
        }
        for (label, set) in [("default", RuleSet::default()), ("all", RuleSet::all())] {
            let changed: Vec<bool> = par_map(sents.len(), |i| {
                checker.correct_with_rules(&sents[i], set) != sents[i]
            });
            let n = changed.iter().filter(|&&c| c).count();
            println!(
                "- correct() with the {label} rules changes {n} sentences ({:.2}%)",
                100.0 * n as f64 / sents.len() as f64
            );
        }
        println!();
    }

    // GECTurk: per rule, precision of its findings
    println!("# GECTurk: precision of each rule alone\n");
    let mut gec_stats: BTreeMap<(GrammarRule, String), (usize, usize, usize, usize)> =
        BTreeMap::new();
    for (name, rows) in &gec {
        let per_row: Vec<Vec<GrammarFinding>> = par_map(rows.len(), |i| {
            rules
                .iter()
                .flat_map(|&r| {
                    checker
                        .check_with_rules(&rows[i].text, RuleSet::empty().with(r))
                        .findings
                })
                .collect()
        });
        let clean_rows = rows
            .iter()
            .filter(|r| r.tokens.iter().all(|t| t.2 == 0))
            .count();
        println!(
            "## {name}: {} rows, {clean_rows} without an error\n",
            rows.len()
        );
        for &rule in rules {
            let (mut tp, mut fp, mut clean_fired) = (0, 0, 0);
            let mut labels: BTreeMap<u32, usize> = BTreeMap::new();
            let mut fps: HashMap<String, usize> = HashMap::new();
            for (row, fs) in rows.iter().zip(&per_row) {
                let mine: Vec<_> = fs.iter().filter(|f| f.rule == rule).collect();
                if !mine.is_empty() && row.tokens.iter().all(|t| t.2 == 0) {
                    clean_fired += 1;
                }
                for f in mine {
                    let hit: Vec<u32> = row
                        .tokens
                        .iter()
                        .filter(|t| t.0 < f.end_offset && f.start_offset < t.1 && t.2 > 0)
                        .map(|t| t.2)
                        .collect();
                    if context.contains(&rule) {
                        let mark = if hit.is_empty() { "FP" } else { "TP" };
                        println!(
                            "    CTX {} {mark} | {}",
                            rule.id(),
                            in_context(&row.text, f)
                        );
                    }
                    if hit.is_empty() {
                        fp += 1;
                        *fps.entry(example(f)).or_default() += 1;
                    } else {
                        tp += 1;
                        *labels.entry(hit[0]).or_default() += 1;
                    }
                }
            }
            gec_stats.insert((rule, name.clone()), (tp, fp, clean_fired, clean_rows));
            if tp + fp > 0 {
                let labels: Vec<String> = labels.iter().map(|(l, c)| format!("{l}:{c}")).collect();
                println!(
                    "- {:<24} P={:5.1}% of {:>5}  clean rows fired {:>4} ({:.2}/1000)  labels [{}]\n    FP: {}",
                    rule.id(),
                    100.0 * tp as f64 / (tp + fp) as f64,
                    tp + fp,
                    clean_fired,
                    1000.0 * clean_fired as f64 / clean_rows.max(1) as f64,
                    labels.join(" "),
                    top(&fps, examples)
                );
            }
        }
        for (label, set) in [("default", RuleSet::default()), ("all", RuleSet::all())] {
            let results: Vec<Vec<GrammarFinding>> = par_map(rows.len(), |i| {
                checker.check_with_rules(&rows[i].text, set).findings
            });
            let mut gold: BTreeMap<u32, (usize, usize)> = BTreeMap::new();
            let (mut tp, mut fp, mut clean_flagged) = (0, 0, 0);
            for (row, fs) in rows.iter().zip(&results) {
                if !fs.is_empty() && row.tokens.iter().all(|t| t.2 == 0) {
                    clean_flagged += 1;
                }
                for &(s, e, l) in &row.tokens {
                    let covered = fs.iter().any(|f| s < f.end_offset && f.start_offset < e);
                    if l > 0 {
                        let g = gold.entry(l).or_default();
                        g.1 += 1;
                        if covered {
                            g.0 += 1;
                            tp += 1;
                        }
                    } else if covered {
                        fp += 1;
                    }
                }
            }
            let (hit, all): (usize, usize) =
                gold.values().fold((0, 0), |(a, b), &(h, g)| (a + h, b + g));
            println!(
                "\n{label} rules: token P={:.1}% R={:.1}%, rows without an error flagged {clean_flagged}/{clean_rows} ({:.1}%)",
                100.0 * tp as f64 / (tp + fp).max(1) as f64,
                100.0 * hit as f64 / all.max(1) as f64,
                100.0 * clean_flagged as f64 / clean_rows.max(1) as f64
            );
            let per_label: Vec<String> = gold
                .iter()
                .map(|(l, (h, g))| format!("{l}:{h}/{g}"))
                .collect();
            println!("  recall per label: {}", per_label.join(" "));
        }
        println!();
    }

    // one line per rule, for the table
    println!("# Summary (per 1000 clean sentences; GECTurk precision)\n");
    let mut header = format!("{:<24} {:>7}", "rule", "default");
    for (name, _) in &clean {
        header += &format!(" {name:>12}");
    }
    for (name, _) in &gec {
        header += &format!(
            " {:>16} {:>12}",
            format!("{name} P"),
            format!("{name} clean")
        );
    }
    println!("{header}");
    for &rule in rules {
        let mut line = format!("{:<24} {:>7}", rule.id(), rule.is_default());
        for (name, _) in &clean {
            let (f, n) = clean_rates[&(rule, name.clone())];
            line += &format!(
                " {:>12}",
                format!("{f} ({:.2})", 1000.0 * f as f64 / n as f64)
            );
        }
        for (name, _) in &gec {
            let (tp, fp, cf, cn) = gec_stats[&(rule, name.clone())];
            let p = if tp + fp > 0 {
                format!("{:.1}% /{}", 100.0 * tp as f64 / (tp + fp) as f64, tp + fp)
            } else {
                "-".to_string()
            };
            line += &format!(
                " {p:>16} {:>12}",
                format!("{:.2}", 1000.0 * cf as f64 / cn.max(1) as f64)
            );
        }
        println!("{line}");
    }
}
