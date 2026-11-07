//! Parity of the `espeak-compat` phonemiser with eSpeak NG's reference fixtures.
//!
//! ```text
//! cargo run --release -p akana-core --example espeak_compat_parity -- \
//!     --fixtures DIR --splits validation [--out report.json] [--examples 40]
//! ```
//!
//! `DIR/espeak-plan.jsonl` holds one row per text: `{id, split, text, tokens: [{ph, stress, word,
//! lengthen, mark, ...}]}`. For each split it reports the sentence exact-match rate, the rate of
//! exact phoneme sequences, per-field token agreement after aligning the phoneme sequences
//! (longest common subsequence), confusion counts for `ph` and `stress`, the same agreement for
//! the tokens of words never seen in the `train` split, and the word-level exact rate of those
//! words and of the words outside the profile's lexicon (read by the letter rules and the
//! suffix model alone). `--fixtures` defaults to `AKANA_ESPEAK_FIXTURES`.

use akana_core::speech::{EspeakToken, Phonemizer, Profile, StressSource, Utterance};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::io::BufRead;

struct Obs {
    ph: String,
    stress: u64,
    word: u64,
    lengthen: u64,
    mark: String,
}

fn obs_tokens(row: &Value) -> Vec<Obs> {
    row["tokens"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|t| Obs {
                    ph: t["ph"].as_str().unwrap_or("").to_string(),
                    stress: t["stress"].as_u64().unwrap_or(0),
                    word: t["word"].as_u64().unwrap_or(0),
                    lengthen: t["lengthen"].as_u64().unwrap_or(0),
                    mark: t["mark"].as_str().unwrap_or("").to_string(),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Index pairs of a longest common subsequence of the two symbol sequences.
fn align(a: &[&str], b: &[&str]) -> Vec<(usize, usize)> {
    let (n, m) = (a.len(), b.len());
    let mut dp = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i][j] = if a[i] == b[j] {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < n && j < m {
        if a[i] == b[j] {
            out.push((i, j));
            i += 1;
            j += 1;
        } else if dp[i + 1][j] >= dp[i][j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

#[derive(Default)]
struct Tally {
    sentences: usize,
    exact: usize,
    ph_exact: usize,
    tokens: usize,
    agree: BTreeMap<&'static str, usize>,
    unseen_tokens: usize,
    unseen_agree: BTreeMap<&'static str, usize>,
    unseen_words: usize,
    unseen_words_exact: usize,
    rules_words: usize,
    rules_words_exact: usize,
    ph_conf: BTreeMap<(String, String), usize>,
    stress_conf: BTreeMap<(u64, u64), usize>,
    examples: Vec<Value>,
}

fn fields(o: &Obs, p: &EspeakToken) -> [(&'static str, bool); 4] {
    [
        ("stress", o.stress == p.stress as u64),
        ("word", o.word == p.word as u64),
        ("lengthen", o.lengthen == p.lengthen as u64),
        ("mark", o.mark == p.mark.to_string()),
    ]
}

/// The predicted token index → word index (`None` for pauses).
fn token_words(u: &Utterance) -> Vec<Option<usize>> {
    let mut out = vec![None];
    for (wi, w) in u.words.iter().enumerate() {
        out.extend(std::iter::repeat_n(None, w.pauses_before.len()));
        out.extend(std::iter::repeat_n(Some(wi), w.phones.len()));
    }
    out.push(None);
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut fixtures = std::env::var("AKANA_ESPEAK_FIXTURES").ok();
    let mut splits = vec!["validation".to_string()];
    let mut out_path: Option<String> = None;
    let mut max_examples = 40usize;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--fixtures" => fixtures = args.next(),
            "--splits" => {
                splits = args
                    .next()
                    .unwrap_or_default()
                    .split(',')
                    .map(str::to_string)
                    .collect()
            }
            "--out" => out_path = args.next(),
            "--examples" => max_examples = args.next().and_then(|s| s.parse().ok()).unwrap_or(40),
            _ => {
                eprintln!("unknown argument {a}");
                std::process::exit(2);
            }
        }
    }
    let Some(dir) = fixtures else {
        eprintln!("no fixtures: pass --fixtures DIR or set AKANA_ESPEAK_FIXTURES");
        std::process::exit(2);
    };
    let plan = std::path::Path::new(&dir).join("espeak-plan.jsonl");
    let file = std::fs::File::open(&plan).unwrap_or_else(|e| {
        eprintln!("{}: {e}", plan.display());
        std::process::exit(2)
    });
    let rows: Vec<Value> = std::io::BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter_map(|l| serde_json::from_str(&l).ok())
        .collect();
    let p = Phonemizer::new(Profile::EspeakCompat);
    let mut u = Utterance::default();
    // the train split's readings, for the unseen-word subset
    let mut train_words: HashSet<String> = HashSet::new();
    for r in rows.iter().filter(|r| r["split"] == "train") {
        let _ = p.phonemize_into(r["text"].as_str().unwrap_or(""), &mut u);
        for w in &u.words {
            train_words.insert(w.reading.clone());
        }
    }
    let mut report = serde_json::Map::new();
    report.insert("profile".into(), json!(Profile::EspeakCompat.version()));
    report.insert("fixtures".into(), json!(plan.display().to_string()));
    report.insert("data".into(), json!(p.data_provenance()));
    for split in &splits {
        let mut t = Tally::default();
        for r in rows.iter().filter(|r| r["split"] == split.as_str()) {
            let text = r["text"].as_str().unwrap_or("");
            let _ = p.phonemize_into(text, &mut u);
            let obs = obs_tokens(r);
            let pred = &u.tokens;
            t.sentences += 1;
            let a: Vec<&str> = obs.iter().map(|o| o.ph.as_str()).collect();
            let b: Vec<&str> = pred.iter().map(|q| q.ph.symbol()).collect();
            let same_ph = a == b;
            t.ph_exact += usize::from(same_ph);
            let exact = same_ph
                && obs
                    .iter()
                    .zip(pred.iter())
                    .all(|(o, q)| fields(o, q).iter().all(|f| f.1));
            t.exact += usize::from(exact);
            t.tokens += obs.len();
            let tw = token_words(&u);
            let pairs = align(&a, &b);
            for &(i, j) in &pairs {
                let unseen = tw
                    .get(j)
                    .copied()
                    .flatten()
                    .is_some_and(|w| !train_words.contains(&u.words[w].reading));
                *t.agree.entry("ph").or_default() += 1;
                if unseen {
                    *t.unseen_agree.entry("ph").or_default() += 1;
                }
                for (f, ok) in fields(&obs[i], &pred[j]) {
                    if ok {
                        *t.agree.entry(f).or_default() += 1;
                        if unseen {
                            *t.unseen_agree.entry(f).or_default() += 1;
                        }
                    }
                }
                *t.stress_conf
                    .entry((obs[i].stress, pred[j].stress as u64))
                    .or_default() += 1;
            }
            // ph confusions: unaligned stretches paired in order
            let (mut i0, mut j0) = (0, 0);
            for &(i, j) in pairs.iter().chain(std::iter::once(&(a.len(), b.len()))) {
                let (xa, xb) = (&a[i0..i], &b[j0..j]);
                for k in 0..xa.len().max(xb.len()) {
                    let oa = xa.get(k).copied().unwrap_or("-").to_string();
                    let ob = xb.get(k).copied().unwrap_or("-").to_string();
                    *t.ph_conf.entry((oa, ob)).or_default() += 1;
                }
                i0 = i + 1;
                j0 = j + 1;
            }
            // unseen tokens and words (word level needs equal word counts)
            for (j, w) in tw.iter().enumerate() {
                if let Some(w) = w {
                    if !train_words.contains(&u.words[*w].reading) {
                        let _ = j;
                        t.unseen_tokens += 1;
                    }
                }
            }
            let mut ow: Vec<Vec<&Obs>> = Vec::new();
            for o in &obs {
                if o.ph.starts_with('_') {
                    continue;
                }
                if o.word == 1 || ow.is_empty() {
                    ow.push(Vec::new());
                }
                if let Some(last) = ow.last_mut() {
                    last.push(o);
                }
            }
            if ow.len() == u.words.len() {
                for (w, o) in u.words.iter().zip(&ow) {
                    let same = w.phones.len() == o.len()
                        && w.phones.iter().zip(o.iter()).all(|(ph, o)| {
                            ph.phoneme.symbol() == o.ph
                                && ph.stress as u64 == o.stress
                                && ph.lengthen as u64 == o.lengthen
                        });
                    // words the profile's lexicon does not have: letter rules and suffix model
                    if w.stress_source != StressSource::Lexicon {
                        t.rules_words += 1;
                        t.rules_words_exact += usize::from(same);
                    }
                    if train_words.contains(&w.reading) {
                        continue;
                    }
                    t.unseen_words += 1;
                    t.unseen_words_exact += usize::from(same);
                }
            }
            if !exact && t.examples.len() < max_examples {
                let fmt_o = |o: &[Obs]| {
                    o.iter()
                        .map(|o| {
                            format!(
                                "{}{}",
                                o.ph,
                                if o.ph.starts_with('_') {
                                    String::new()
                                } else {
                                    o.stress.to_string()
                                }
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                };
                let fmt_p = |p: &[EspeakToken]| {
                    p.iter()
                        .map(|q| {
                            format!(
                                "{}{}",
                                q.ph.symbol(),
                                if q.ph.is_pause() {
                                    String::new()
                                } else {
                                    q.stress.to_string()
                                }
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                };
                let words: Vec<Value> = u
                    .words
                    .iter()
                    .map(|w| json!({"reading": w.reading, "stress": w.stress_source.label(), "segments": w.segment_source.label()}))
                    .collect();
                t.examples.push(json!({
                    "id": r["id"], "text": text,
                    "espeak": fmt_o(&obs), "akana": fmt_p(pred), "words": words,
                }));
            }
        }
        let n = t.tokens.max(1) as f64;
        let un = t.unseen_tokens.max(1) as f64;
        let mut agree = serde_json::Map::new();
        for f in ["ph", "stress", "word", "lengthen", "mark"] {
            agree.insert(f.into(), json!(*t.agree.get(f).unwrap_or(&0) as f64 / n));
        }
        let mut unseen = serde_json::Map::new();
        for f in ["ph", "stress", "word", "lengthen", "mark"] {
            unseen.insert(
                f.into(),
                json!(*t.unseen_agree.get(f).unwrap_or(&0) as f64 / un),
            );
        }
        let mut phc: Vec<(&(String, String), &usize)> = t.ph_conf.iter().collect();
        phc.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        let mut stc: Vec<(&(u64, u64), &usize)> =
            t.stress_conf.iter().filter(|(k, _)| k.0 != k.1).collect();
        stc.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        let summary = json!({
            "sentences": t.sentences,
            "sentence_exact": t.exact as f64 / t.sentences.max(1) as f64,
            "ph_sequence_exact": t.ph_exact as f64 / t.sentences.max(1) as f64,
            "tokens": t.tokens,
            "token_agreement": agree,
            "unseen_word_tokens": t.unseen_tokens,
            "unseen_word_token_agreement": unseen,
            "unseen_words": t.unseen_words,
            "unseen_words_exact": t.unseen_words_exact as f64 / t.unseen_words.max(1) as f64,
            "outside_lexicon_words": t.rules_words,
            "outside_lexicon_words_exact": t.rules_words_exact as f64 / t.rules_words.max(1) as f64,
            "ph_confusions": phc.iter().take(30).map(|((a, b), c)| json!([a, b, c])).collect::<Vec<_>>(),
            "stress_confusions": stc.iter().take(30).map(|((a, b), c)| json!([a, b, c])).collect::<Vec<_>>(),
            "examples": t.examples,
        });
        println!(
            "{split}: {} sentences, exact {:.4}, ph sequence exact {:.4}, agreement {}, unseen-word tokens {} agreement {}, unseen words {} exact {:.4}, outside-lexicon words {} exact {:.4}",
            t.sentences,
            t.exact as f64 / t.sentences.max(1) as f64,
            t.ph_exact as f64 / t.sentences.max(1) as f64,
            serde_json::to_string(&agree).unwrap_or_default(),
            t.unseen_tokens,
            serde_json::to_string(&unseen).unwrap_or_default(),
            t.unseen_words,
            t.unseen_words_exact as f64 / t.unseen_words.max(1) as f64,
            t.rules_words,
            t.rules_words_exact as f64 / t.rules_words.max(1) as f64
        );
        report.insert(split.clone(), summary);
    }
    if let Some(path) = out_path {
        let text = serde_json::to_string_pretty(&Value::Object(report)).unwrap_or_default();
        if let Err(e) = std::fs::write(&path, text) {
            eprintln!("{path}: {e}");
            std::process::exit(1);
        }
    }
}
