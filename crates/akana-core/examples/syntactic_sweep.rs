//! Measures the syntactic analyzer (inflectional groups) on the fixed corpus samples that
//! `scripts/morph_corpus_sample.py` writes, as `morph_sweep` does for the morphological one.
//!
//! ```text
//! cargo run --release -p akana-core --example syntactic_sweep -- --sample .cache/morph_sample \
//!     [--json out.json]
//! ```
//!
//! For the court decisions and RagTurk separately it reports the unanalysed token and type
//! rates (letter tokens, lower-case letter tokens, tokens with digits), parses and inflectional
//! groups per token, and the tokens the syntactic analyzer and the morphological one disagree on
//! (one analyses a word the other does not). It times `analyze` on running text (no cache).
//! `--json` saves the numbers and the most frequent words only one of the two analyses.

use akana_core::morphology::TurkishMorphology;
use akana_core::syntactic_morphology::TurkishSyntacticMorphology;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

const COURT: &[&str] = &["yargitay", "danistay", "emsal", "aym_bb", "aym_norm"];
const RAGTURK: &str = "ragturk";
/// Running-text tokens timed per corpus.
const TIMED_PER_CORPUS: usize = 100_000;
/// Words listed per disagreement kind in the JSON.
const LISTED: usize = 200;

fn read_counts(path: &Path) -> Vec<(String, u64)> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines()
        .filter_map(|l| {
            let (w, c) = l.split_once('\t')?;
            Some((w.to_string(), c.parse().ok()?))
        })
        .collect()
}

/// What the sweep keeps of one word's analyses.
#[derive(Clone, Copy)]
struct Summary {
    parses: usize,
    groups: usize,
    morph_parses: usize,
}

#[derive(Default)]
struct Rates {
    tokens: u64,
    types: u64,
    unk_tokens: u64,
    unk_types: u64,
    lower_tokens: u64,
    lower_unk_tokens: u64,
    parse_sum: u64,
    group_sum: u64,
    num_tokens: u64,
    num_unk_tokens: u64,
    only_syntactic_tokens: u64,
    only_morphological_tokens: u64,
}

impl Rates {
    fn to_json(&self) -> Value {
        let r = |a: u64, b: u64| if b == 0 { 0.0 } else { a as f64 / b as f64 };
        json!({
            "tokens": self.tokens, "types": self.types,
            "unanalysed_token_rate": r(self.unk_tokens, self.tokens),
            "unanalysed_type_rate": r(self.unk_types, self.types),
            "lowercase_unanalysed_token_rate": r(self.lower_unk_tokens, self.lower_tokens),
            "parses_per_token": r(self.parse_sum, self.tokens),
            "groups_per_parse": r(self.group_sum, self.parse_sum),
            "number_tokens": self.num_tokens,
            "number_unanalysed_token_rate": r(self.num_unk_tokens, self.num_tokens),
            "only_syntactic_token_rate": r(self.only_syntactic_tokens, self.tokens),
            "only_morphological_token_rate": r(self.only_morphological_tokens, self.tokens),
        })
    }
}

fn is_lower(word: &str) -> bool {
    word.chars().next().is_some_and(|c| c.is_lowercase())
}

fn main() {
    let mut sample = PathBuf::from(".cache/morph_sample");
    let mut json_out: Option<PathBuf> = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut value = || PathBuf::from(it.next().expect("missing value"));
        match a.as_str() {
            "--sample" => sample = value(),
            "--json" => json_out = Some(value()),
            other => panic!("unknown argument {other}"),
        }
    }
    let eval = sample.join("eval");

    let load = Instant::now();
    let syntactic = TurkishSyntacticMorphology::new();
    let first = syntactic.analyze("geldiğimizde").len();
    let load_ms = load.elapsed().as_secs_f64() * 1000.0;
    let morphology = TurkishMorphology::new();

    let mut cache: HashMap<String, Summary> = HashMap::new();
    let mut analyse = |w: &str| -> Summary {
        if let Some(&s) = cache.get(w) {
            return s;
        }
        let parses = syntactic.analyze(w);
        let s = Summary {
            parses: parses.len(),
            groups: parses.iter().map(|p| p.inflectional_groups.len()).sum(),
            morph_parses: morphology.analyze(w).len(),
        };
        cache.insert(w.to_string(), s);
        s
    };

    let mut groups: BTreeMap<&str, Rates> = BTreeMap::new();
    let mut only_syntactic: HashMap<String, u64> = HashMap::new();
    let mut only_morphological: HashMap<String, u64> = HashMap::new();
    for (group, corpora) in [("court", COURT), ("ragturk", &[RAGTURK][..])] {
        let rates = groups.entry(group).or_default();
        let mut merged: HashMap<String, u64> = HashMap::new();
        for corpus in corpora {
            for (w, c) in read_counts(&eval.join(format!("{corpus}.types.tsv"))) {
                *merged.entry(w).or_default() += c;
            }
            for (w, c) in read_counts(&eval.join(format!("{corpus}.numbers.tsv"))) {
                rates.num_tokens += c;
                if analyse(&w).parses == 0 {
                    rates.num_unk_tokens += c;
                }
            }
        }
        for (w, c) in &merged {
            let s = analyse(w);
            rates.tokens += c;
            rates.types += 1;
            rates.parse_sum += c * s.parses as u64;
            rates.group_sum += c * s.groups as u64;
            if s.parses == 0 {
                rates.unk_tokens += c;
                rates.unk_types += 1;
            }
            if is_lower(w) {
                rates.lower_tokens += c;
                if s.parses == 0 {
                    rates.lower_unk_tokens += c;
                }
            }
            if s.parses > 0 && s.morph_parses == 0 {
                rates.only_syntactic_tokens += c;
                *only_syntactic.entry(w.clone()).or_default() += c;
            }
            if s.parses == 0 && s.morph_parses > 0 {
                rates.only_morphological_tokens += c;
                *only_morphological.entry(w.clone()).or_default() += c;
            }
        }
    }

    // Throughput on running text: every token analysed afresh, as `analyze` is called in use.
    let token_re =
        regex::Regex::new(r"[^\W\d_]+(?:['’][^\W\d_]+)?|%?\d+(?:[.,]\d+)*\.?(?:['’][^\W\d_]+)?")
            .unwrap();
    let mut stream: Vec<String> = Vec::new();
    for corpus in COURT.iter().chain(std::iter::once(&RAGTURK)) {
        let text = std::fs::read_to_string(eval.join(format!("{corpus}.txt"))).unwrap_or_default();
        stream.extend(
            token_re
                .find_iter(&text)
                .take(TIMED_PER_CORPUS)
                .map(|m| m.as_str().to_string()),
        );
    }
    let start = Instant::now();
    let mut analyses = 0usize;
    for tok in &stream {
        analyses += syntactic.analyze(tok).len();
    }
    let secs = start.elapsed().as_secs_f64();
    let words_per_sec = stream.len() as f64 / secs;

    println!("{:<36} {:>12} {:>12}", "", "court", "ragturk");
    let g: Vec<Value> = ["court", "ragturk"]
        .iter()
        .map(|k| groups[k].to_json())
        .collect();
    for key in [
        "tokens",
        "types",
        "unanalysed_token_rate",
        "unanalysed_type_rate",
        "lowercase_unanalysed_token_rate",
        "parses_per_token",
        "groups_per_parse",
        "number_tokens",
        "number_unanalysed_token_rate",
        "only_syntactic_token_rate",
        "only_morphological_token_rate",
    ] {
        let fmt = |v: &Value| match v {
            Value::Number(n) if n.is_u64() => n.to_string(),
            Value::Number(n) if key.ends_with("per_token") || key.ends_with("per_parse") => {
                format!("{:.3}", n.as_f64().unwrap())
            }
            Value::Number(n) => format!("{:.3}%", 100.0 * n.as_f64().unwrap()),
            v => v.to_string(),
        };
        println!("{key:<36} {:>12} {:>12}", fmt(&g[0][key]), fmt(&g[1][key]));
    }
    println!(
        "throughput: {} tokens in {secs:.2}s = {words_per_sec:.0} words/s ({:.2} analyses/token)",
        stream.len(),
        analyses as f64 / stream.len() as f64
    );
    println!("first analysis after {load_ms:.0} ms ({first} parses)");

    let top = |m: &HashMap<String, u64>| -> Vec<Value> {
        let mut v: Vec<(&String, &u64)> = m.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        v.into_iter()
            .take(LISTED)
            .map(|(w, c)| json!([w, c]))
            .collect()
    };
    for (name, m) in [
        ("only syntactic", &only_syntactic),
        ("only morphological", &only_morphological),
    ] {
        let words: Vec<String> = top(m)
            .iter()
            .take(15)
            .map(|v| format!("{} {}", v[0].as_str().unwrap_or(""), v[1]))
            .collect();
        println!("{name}: {} types; top {}", m.len(), words.join(", "));
    }
    if let Some(path) = json_out {
        let out = json!({
            "court": g[0], "ragturk": g[1],
            "throughput_words_per_sec": words_per_sec,
            "timed_tokens": stream.len(),
            "analyses_per_token": analyses as f64 / stream.len() as f64,
            "first_analysis_ms": load_ms,
            "only_syntactic": top(&only_syntactic),
            "only_morphological": top(&only_morphological),
        });
        std::fs::write(&path, serde_json::to_string_pretty(&out).unwrap())
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    }
}
