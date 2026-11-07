//! Latency of the `espeak-compat` phonemiser per sentence, warm, in-process.
//!
//! ```text
//! cargo run --release -p akana-core --example espeak_compat_bench -- [--fixtures DIR] [--reps 20]
//! ```
//!
//! Texts: the reference fixtures' corpus rows (`DIR/espeak-plan.jsonl`, default
//! `AKANA_ESPEAK_FIXTURES`), or else the committed golden sentences. It times every call
//! separately after a warm-up pass and reports the median, 90th and 99th percentiles for
//! sentences of 50 to 70 characters (the brief's "about 60") and for all texts, and the
//! real-time factor against speech at eSpeak's rate setting of 175 words per minute.

use akana_core::speech::{Phonemizer, Profile, Utterance};
use std::io::BufRead;
use std::time::Instant;

fn pct(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let i = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[i.min(sorted.len() - 1)]
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut fixtures = std::env::var("AKANA_ESPEAK_FIXTURES").ok();
    let mut reps = 20usize;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--fixtures" => fixtures = args.next(),
            "--reps" => reps = args.next().and_then(|s| s.parse().ok()).unwrap_or(20),
            _ => {
                eprintln!("unknown argument {a}");
                std::process::exit(2);
            }
        }
    }
    let texts: Vec<String> = match fixtures {
        Some(dir) => {
            let plan = std::path::Path::new(&dir).join("espeak-plan.jsonl");
            let file = std::fs::File::open(&plan).unwrap_or_else(|e| {
                eprintln!("{}: {e}", plan.display());
                std::process::exit(2)
            });
            std::io::BufReader::new(file)
                .lines()
                .map_while(Result::ok)
                .filter_map(|l| serde_json::from_str::<serde_json::Value>(&l).ok())
                .filter(|v| v["split"] != "probe")
                .filter_map(|v| v["text"].as_str().map(str::to_string))
                .collect()
        }
        None => include_str!("../tests/data/speech/espeak_compat_golden.jsonl")
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .filter_map(|v| v["text"].as_str().map(str::to_string))
            .collect(),
    };
    let t0 = Instant::now();
    let p = Phonemizer::new(Profile::EspeakCompat);
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut u = Utterance::default();
    // warm-up
    for t in &texts {
        let _ = p.phonemize_into(t, &mut u);
    }
    let mut all: Vec<f64> = Vec::new();
    let mut mid: Vec<f64> = Vec::new();
    let mut mid_words = 0usize;
    let mut mid_texts = 0usize;
    for t in &texts {
        let chars = t.chars().count();
        let is_mid = (50..=70).contains(&chars);
        if is_mid {
            mid_words += t.split_whitespace().count();
            mid_texts += 1;
        }
        for _ in 0..reps {
            let s = Instant::now();
            let _ = p.phonemize_into(t, &mut u);
            let us = s.elapsed().as_secs_f64() * 1e6;
            all.push(us);
            if is_mid {
                mid.push(us);
            }
        }
    }
    all.sort_by(f64::total_cmp);
    mid.sort_by(f64::total_cmp);
    let words_per_text = mid_words as f64 / mid_texts.max(1) as f64;
    let speech_s = words_per_text * 60.0 / 175.0;
    let median_mid_s = pct(&mid, 0.5) / 1e6;
    println!(
        "texts {} (50-70 chars: {}), reps {reps}, data load {load_ms:.1} ms",
        texts.len(),
        mid_texts
    );
    println!(
        "50-70 chars: median {:.1} us, p90 {:.1} us, p99 {:.1} us, max {:.1} us",
        pct(&mid, 0.5),
        pct(&mid, 0.9),
        pct(&mid, 0.99),
        mid.last().copied().unwrap_or(f64::NAN)
    );
    println!(
        "all texts:   median {:.1} us, p90 {:.1} us, p99 {:.1} us, max {:.1} us",
        pct(&all, 0.5),
        pct(&all, 0.9),
        pct(&all, 0.99),
        all.last().copied().unwrap_or(f64::NAN)
    );
    println!(
        "real time: {words_per_text:.1} words per 50-70-char sentence = {speech_s:.2} s of speech at 175 wpm; \
         median call = {:.0}x faster than real time",
        speech_s / median_mid_s
    );
}
