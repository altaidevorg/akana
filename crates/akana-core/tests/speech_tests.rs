//! The `espeak-compat` phonemiser against eSpeak NG's outputs.
//!
//! - `golden_sentences_match_espeak`: 60 committed CC0 sentences (see
//!   `tests/data/speech/README.md`).
//! - `fixture_parity`: the full reference fixtures, when `AKANA_ESPEAK_FIXTURES` names their
//!   directory (it skips otherwise). It checks the `train` and `validation` splits against the
//!   acceptance targets; the `test` split is reported once per stage by the
//!   `espeak_compat_parity` example, not checked here.

use akana_core::speech::{EspeakToken, Phonemizer, Profile, Utterance};
use serde_json::Value;
use std::io::BufRead;

fn same(t: &EspeakToken, o: &Value) -> bool {
    t.ph.symbol() == o[0].as_str().unwrap_or("")
        && u64::from(t.stress) == o[1].as_u64().unwrap_or(99)
        && u64::from(t.word) == o[2].as_u64().unwrap_or(99)
        && u64::from(t.lengthen) == o[3].as_u64().unwrap_or(99)
        && t.mark.to_string() == o[4].as_str().unwrap_or("")
}

fn show(tokens: &[EspeakToken]) -> String {
    tokens
        .iter()
        .map(|t| format!("{}{}", t.ph.symbol(), t.stress))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn golden_sentences_match_espeak() {
    let p = Phonemizer::new(Profile::EspeakCompat);
    let data = include_str!("data/speech/espeak_compat_golden.jsonl");
    let mut n = 0;
    for line in data.lines() {
        let row: Value = serde_json::from_str(line).expect("golden row");
        let text = row["text"].as_str().expect("text");
        let want = row["tokens"].as_array().expect("tokens");
        let got = p.phonemize(text);
        let ok =
            got.tokens.len() == want.len() && got.tokens.iter().zip(want).all(|(t, o)| same(t, o));
        assert!(ok, "{}: {text}\n  akana {}", row["id"], show(&got.tokens));
        n += 1;
    }
    assert_eq!(n, 60);
}

#[test]
fn behaviours_seen_in_the_probes() {
    let p = Phonemizer::new(Profile::EspeakCompat);
    let syms = |u: &Utterance| -> Vec<String> {
        u.tokens
            .iter()
            .map(|t| format!("{}{}", t.ph.symbol(), t.stress))
            .collect()
    };
    // stressed adverbs and place names as eSpeak has them (initial-accent adverbs final)
    let u = p.phonemize("Ankara, sonra.");
    assert_eq!(
        syms(&u),
        [
            "_0", "a7", "n1", "k1", "a1", "*1", "a1", "_:0", "_0", "s1", "O1", "n7", "R7", "a7",
            "_0"
        ]
    );
    // a word with an apostrophe: final stress, then the pause pair, which replaces the short
    // pause before ve
    let u = p.phonemize("Kemal'e ve Ali'ye");
    let s = syms(&u);
    assert!(s.contains(&"_:1".to_string()));
    assert!(!s.iter().any(|t| t.starts_with("_!")));
    // brackets pause like quotes
    let u = p.phonemize("dedi (sessizce) ve gitti");
    let pauses: Vec<&str> = u
        .tokens
        .iter()
        .filter(|t| t.ph.is_pause())
        .map(|t| t.ph.symbol())
        .collect();
    assert_eq!(pauses, ["_", "_:", "_:", "_:", "_:", "_"]);
    // every word carries its provenance
    let u = p.phonemize("Bilmiyorlardı ve gülüverdiler.");
    for w in &u.words {
        assert!(!w.stress_source.label().is_empty());
        assert!(!w.segment_source.label().is_empty());
    }
}

struct Score {
    sentences: usize,
    exact: usize,
    ph_exact: usize,
    tokens: usize,
    agree: [usize; 5],
}

fn score(p: &Phonemizer, rows: &[Value], split: &str) -> Score {
    let mut s = Score {
        sentences: 0,
        exact: 0,
        ph_exact: 0,
        tokens: 0,
        agree: [0; 5],
    };
    let mut u = Utterance::default();
    for r in rows.iter().filter(|r| r["split"] == split) {
        let text = r["text"].as_str().unwrap_or("");
        p.phonemize_into(text, &mut u).expect("phonemize");
        let obs = r["tokens"].as_array().cloned().unwrap_or_default();
        s.sentences += 1;
        s.tokens += obs.len();
        let ph_same = u.tokens.len() == obs.len()
            && u.tokens
                .iter()
                .zip(&obs)
                .all(|(t, o)| t.ph.symbol() == o["ph"].as_str().unwrap_or(""));
        s.ph_exact += usize::from(ph_same);
        if !ph_same {
            continue;
        }
        let mut all = true;
        for (t, o) in u.tokens.iter().zip(&obs) {
            let f = [
                true,
                u64::from(t.stress) == o["stress"].as_u64().unwrap_or(99),
                u64::from(t.word) == o["word"].as_u64().unwrap_or(99),
                u64::from(t.lengthen) == o["lengthen"].as_u64().unwrap_or(99),
                t.mark.to_string() == o["mark"].as_str().unwrap_or(""),
            ];
            for (k, ok) in f.iter().enumerate() {
                s.agree[k] += usize::from(*ok);
            }
            all &= f.iter().all(|x| *x);
        }
        s.exact += usize::from(all);
    }
    s
}

#[test]
fn fixture_parity() {
    let Ok(dir) = std::env::var("AKANA_ESPEAK_FIXTURES") else {
        eprintln!("skipped: AKANA_ESPEAK_FIXTURES is not set");
        return;
    };
    let plan = std::path::Path::new(&dir).join("espeak-plan.jsonl");
    let file = std::fs::File::open(&plan).expect("espeak-plan.jsonl");
    let rows: Vec<Value> = std::io::BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter_map(|l| serde_json::from_str(&l).ok())
        .collect();
    let p = Phonemizer::new(Profile::EspeakCompat);
    // (split, sentence exact, ph sequence exact, per-token ph stress word lengthen mark);
    // tokens of sentences whose phoneme sequence differs count as disagreeing
    for (split, exact, ph_exact, field) in [
        ("train", 0.999, 0.999, [0.999, 0.999, 0.999, 0.999, 0.999]),
        ("validation", 0.97, 0.98, [0.98, 0.98, 0.98, 0.98, 0.98]),
    ] {
        let s = score(&p, &rows, split);
        let rate = |x: usize, n: usize| x as f64 / n.max(1) as f64;
        eprintln!(
            "{split}: {} sentences, exact {:.4}, ph sequence exact {:.4}",
            s.sentences,
            rate(s.exact, s.sentences),
            rate(s.ph_exact, s.sentences)
        );
        assert!(s.sentences > 0, "no {split} rows");
        assert!(rate(s.exact, s.sentences) >= exact, "{split} exact");
        assert!(rate(s.ph_exact, s.sentences) >= ph_exact, "{split} ph");
        for (k, min) in field.iter().enumerate() {
            assert!(rate(s.agree[k], s.tokens) >= *min, "{split} field {k}");
        }
    }
    // the probes (numbers, symbols, letters): no panic, edges present
    for r in rows.iter().filter(|r| r["split"] == "probe") {
        let u = p.phonemize(r["text"].as_str().unwrap_or(""));
        assert!(u.tokens.len() >= 2);
    }
}
