//! Latency of the phonemiser per sentence, warm, in-process.
//!
//! ```text
//! cargo run --release -p akana-core --example phonemizer_bench -- [--texts FILE.jsonl] [--reps 20] [--profile tts-v2]
//! ```
//!
//! Texts: the `text` fields of a JSONL file (rows whose `split` is `probe` are left out), or else
//! the committed golden sentences. It times every call separately after a warm-up pass and
//! reports the median, 90th and 99th percentiles for sentences of 50 to 70 characters and for all
//! texts, and the real-time factor against speech at 175 words per minute.

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
    let mut texts_file: Option<String> = None;
    let mut reps = 20usize;
    let mut profile = Profile::TtsV1;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--texts" => texts_file = args.next(),
            "--reps" => reps = args.next().and_then(|s| s.parse().ok()).unwrap_or(20),
            "--profile" => {
                let name = args.next().unwrap_or_default();
                profile = Profile::from_name(&name).unwrap_or_else(|| {
                    eprintln!("unknown profile {name:?}");
                    std::process::exit(2)
                });
            }
            _ => {
                eprintln!("unknown argument {a}");
                std::process::exit(2);
            }
        }
    }
    let texts: Vec<String> = match texts_file {
        Some(file) => {
            let plan = std::path::PathBuf::from(file);
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
        None => include_str!("../tests/data/speech/tts_v1_golden.jsonl")
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .filter_map(|v| v["text"].as_str().map(str::to_string))
            .collect(),
    };
    let t0 = Instant::now();
    let p = Phonemizer::new(profile);
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
        "profile {}, texts {} (50-70 chars: {}), reps {reps}, data load {load_ms:.1} ms",
        profile.name(),
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
