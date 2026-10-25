//! Measures the morphological analyzer on the fixed corpus samples that
//! `scripts/morph_corpus_sample.py` writes.
//!
//! ```text
//! cargo run --release -p akana-core --example morph_sweep -- --sample .cache/morph_sample \
//!     [--gold crates/akana-core/tests/data/morphology/frequent_types.tsv] \
//!     [--json out.json] [--diff previous.json]
//! ```
//!
//! For the court decisions and RagTurk separately it reports the unanalysed token and type
//! rates (letter tokens, lower-case letter tokens, tokens with digits), the average number of
//! analyses per token, and the share of tokens that have a reading of the kinds the 2026-10-20
//! review found to be junk. It times `analyze` on running text (no cache) and, with `--gold`,
//! checks the best analysis of hand-checked frequent words. `--json` saves the numbers and each
//! frequent type's best analysis; `--diff` compares them with an earlier run and lists the most
//! frequent words whose analysis changed, so a change in the rates can be explained.

use akana_core::morphology::{MorphParse, TurkishMorphology};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

const COURT: &[&str] = &["yargitay", "danistay", "emsal", "aym_bb", "aym_norm"];
const RAGTURK: &str = "ragturk";
/// Running-text tokens timed per corpus.
const TIMED_PER_CORPUS: usize = 100_000;
/// Types saved in the JSON for `--diff`: those seen at least this often.
const SAVED_MIN_COUNT: u64 = 3;

struct Args {
    sample: PathBuf,
    gold: Option<PathBuf>,
    json: Option<PathBuf>,
    diff: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut args = Args {
        sample: PathBuf::from(".cache/morph_sample"),
        gold: None,
        json: None,
        diff: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut value = || PathBuf::from(it.next().expect("missing value"));
        match a.as_str() {
            "--sample" => args.sample = value(),
            "--gold" => args.gold = Some(value()),
            "--json" => args.json = Some(value()),
            "--diff" => args.diff = Some(value()),
            other => panic!("unknown argument {other}"),
        }
    }
    args
}

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
#[derive(Clone)]
struct Summary {
    parses: usize,
    best: String,
    junk_verb: bool,
    junk_dev: bool,
    junk_opt: bool,
    junk_ki: bool,
}

fn summarize(parses: &[MorphParse]) -> Summary {
    let has = |f: &dyn Fn(&MorphParse) -> bool| parses.iter().any(f);
    let tags = |p: &MorphParse, t: &str| p.morpheme_tags.iter().any(|x| x == t);
    Summary {
        parses: parses.len(),
        best: parses
            .first()
            .map(|p| p.formatted.clone())
            .unwrap_or_default(),
        junk_verb: has(&|p| tags(p, "NounToVerb")),
        junk_dev: has(&|p| {
            ["DevN", "DevGi", "DevGin", "DevGen"]
                .iter()
                .any(|t| tags(p, t))
        }),
        junk_opt: has(&|p| {
            p.morpheme_tags
                .windows(2)
                .any(|w| w[0] == "Opt" && w[1] == "A2sg")
        }),
        junk_ki: has(&|p| {
            p.primary_pos.as_str() == "Noun"
                && p.morpheme_tags.get(1).is_some_and(|t| t == "RelClitic")
        }),
    }
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
    junk_verb: u64,
    junk_dev: u64,
    junk_opt: u64,
    junk_ki: u64,
    num_tokens: u64,
    num_unk_tokens: u64,
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
            "junk_nountoverb_token_rate": r(self.junk_verb, self.tokens),
            "junk_devnoun_token_rate": r(self.junk_dev, self.tokens),
            "junk_opt_a2sg_token_rate": r(self.junk_opt, self.tokens),
            "junk_bare_ki_token_rate": r(self.junk_ki, self.tokens),
            "number_tokens": self.num_tokens,
            "number_unanalysed_token_rate": r(self.num_unk_tokens, self.num_tokens),
        })
    }
}

fn is_lower(word: &str) -> bool {
    word.chars().next().is_some_and(|c| c.is_lowercase())
}

fn main() {
    let args = parse_args();
    let morphology = TurkishMorphology::new();
    let eval = args.sample.join("eval");

    let mut cache: HashMap<String, Summary> = HashMap::new();
    let mut analyse = |w: &str| -> Summary {
        if let Some(s) = cache.get(w) {
            return s.clone();
        }
        let s = summarize(&morphology.analyze(w));
        cache.insert(w.to_string(), s.clone());
        s
    };

    let mut groups: BTreeMap<&str, Rates> = BTreeMap::new();
    let mut all_counts: HashMap<String, u64> = HashMap::new();
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
            for (flag, n) in [
                (s.junk_verb, &mut rates.junk_verb),
                (s.junk_dev, &mut rates.junk_dev),
                (s.junk_opt, &mut rates.junk_opt),
                (s.junk_ki, &mut rates.junk_ki),
            ] {
                if flag {
                    *n += c;
                }
            }
            *all_counts.entry(w.clone()).or_default() += c;
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
        analyses += morphology.analyze(tok).len();
    }
    let secs = start.elapsed().as_secs_f64();
    let words_per_sec = stream.len() as f64 / secs;

    // Best analysis of hand-checked frequent words: word, root, lemma, POS, tags.
    let mut gold_json = Value::Null;
    if let Some(gold) = &args.gold {
        let text = std::fs::read_to_string(gold).expect("gold file");
        // A word may have several accepted (lemma, POS, tags) lines; any of them counts.
        let mut expected: Vec<(String, Vec<(String, String, String)>)> = Vec::new();
        for line in text
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
        {
            let f: Vec<&str> = line.split('\t').collect();
            let [word, lemma, pos, tags] = f[..] else {
                panic!("bad gold line {line:?}")
            };
            let answer = (lemma.to_string(), pos.to_string(), tags.to_string());
            match expected.last_mut() {
                Some((w, answers)) if w == word => answers.push(answer),
                _ => expected.push((word.to_string(), vec![answer])),
            }
        }
        let (mut n, mut lemma_ok, mut pos_ok, mut exact) = (0, 0, 0, 0);
        let mut wrong = Vec::new();
        for (word, answers) in &expected {
            n += 1;
            let parses = morphology.analyze(word);
            let best = parses.first();
            let l = answers
                .iter()
                .any(|(lemma, _, _)| best.is_some_and(|p| &p.lemma == lemma));
            let pp = answers
                .iter()
                .any(|(_, pos, _)| best.is_some_and(|p| p.primary_pos.as_str() == pos));
            let e = answers.iter().any(|(lemma, pos, tags)| {
                best.is_some_and(|p| {
                    &p.lemma == lemma
                        && p.primary_pos.as_str() == pos
                        && &p.morpheme_tags.join("+") == tags
                })
            });
            lemma_ok += l as u32;
            pos_ok += pp as u32;
            exact += e as u32;
            if !e {
                let (lemma, pos, tags) = &answers[0];
                wrong.push(format!(
                    "{word}: expected [{lemma}:{pos}] {tags}, got {}",
                    best.map_or("nothing".to_string(), |p| p.formatted.clone())
                ));
            }
        }
        println!("gold best analyses: {exact}/{n} exact, lemma {lemma_ok}/{n}, POS {pos_ok}/{n}");
        for w in &wrong {
            println!("  {w}");
        }
        gold_json =
            json!({"items": n, "exact": exact, "lemma": lemma_ok, "pos": pos_ok, "wrong": wrong});
    }

    println!("{:<44} {:>12} {:>12}", "", "court", "ragturk");
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
        "junk_nountoverb_token_rate",
        "junk_devnoun_token_rate",
        "junk_opt_a2sg_token_rate",
        "junk_bare_ki_token_rate",
        "number_tokens",
        "number_unanalysed_token_rate",
    ] {
        let fmt = |v: &Value| match v {
            Value::Number(n) if n.is_u64() => n.to_string(),
            Value::Number(n) if key == "parses_per_token" => format!("{:.3}", n.as_f64().unwrap()),
            Value::Number(n) => format!("{:.3}%", 100.0 * n.as_f64().unwrap()),
            v => v.to_string(),
        };
        println!("{key:<44} {:>12} {:>12}", fmt(&g[0][key]), fmt(&g[1][key]));
    }
    println!(
        "throughput: {} tokens in {secs:.2}s = {words_per_sec:.0} words/s ({:.2} analyses/token)",
        stream.len(),
        analyses as f64 / stream.len() as f64
    );

    let mut saved: BTreeMap<String, Value> = BTreeMap::new();
    for (w, c) in &all_counts {
        if *c >= SAVED_MIN_COUNT {
            let s = &cache[w];
            saved.insert(w.clone(), json!([c, s.parses, s.best]));
        }
    }
    let out = json!({
        "court": g[0], "ragturk": g[1],
        "throughput_words_per_sec": words_per_sec,
        "timed_tokens": stream.len(),
        "gold": gold_json,
        "types": saved,
    });

    if let Some(prev_path) = &args.diff {
        let prev: Value =
            serde_json::from_str(&std::fs::read_to_string(prev_path).expect("diff file"))
                .expect("diff json");
        diff_report(&prev["types"], &out["types"]);
    }
    if let Some(path) = &args.json {
        std::fs::write(path, serde_json::to_string(&out).unwrap()).expect("write json");
    }
}

/// Lists the most frequent words that gained or lost every analysis, or whose best one changed.
fn diff_report(prev: &Value, now: &Value) {
    let (Some(prev), Some(now)) = (prev.as_object(), now.as_object()) else {
        return;
    };
    let mut lost = Vec::new();
    let mut gained = Vec::new();
    let mut changed = Vec::new();
    for (w, v) in now {
        let Some(p) = prev.get(w) else { continue };
        let count = v[0].as_u64().unwrap_or(0);
        let (before, after) = (p[1].as_u64().unwrap_or(0), v[1].as_u64().unwrap_or(0));
        if before > 0 && after == 0 {
            lost.push((count, w.clone(), p[2].to_string()));
        } else if before == 0 && after > 0 {
            gained.push((count, w.clone(), v[2].to_string()));
        } else if p[2] != v[2] {
            changed.push((count, w.clone(), format!("{} -> {}", p[2], v[2])));
        }
    }
    for (title, mut list) in [
        ("now unanalysed", lost),
        ("now analysed", gained),
        ("best analysis changed", changed),
    ] {
        list.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let total: u64 = list.iter().map(|x| x.0).sum();
        println!(
            "\n{title}: {} types, {total} tokens; most frequent:",
            list.len()
        );
        for (c, w, s) in list.iter().take(40) {
            println!("  {c:>7} {w}  {s}");
        }
    }
}
