//! The phonemiser against its own frozen output.
//!
//! - `tts_v1_golden_is_frozen`: `tts-v1`'s tokens for 60 CC0 sentences, as frozen in stage 2
//!   (EchoCraft's shipped voice was trained on this profile; the file is never rewritten).
//! - `tts_v2_golden_is_frozen`: `tts-v2`'s tokens for the same sentences and the verbaliser's
//!   inputs (`verbalize_inputs.txt`), frozen with `tts-v2.0`.
//! - `verbalize_golden_is_frozen`: the verbaliser's readings and spans for those inputs.
//!
//! After a deliberate change to `tts-v2` or the verbaliser, bump the profile's version and rewrite
//! the files with `cargo test -p akana-core --test speech_tests write_goldens -- --ignored`.

use akana_core::speech::{verbalize, Phonemizer, Profile, Token};
use serde_json::Value;

fn token_row(t: &Token) -> Value {
    serde_json::json!([
        t.ph_symbol(),
        t.stress,
        u8::from(t.word),
        u8::from(t.lengthen),
        t.mark.to_string(),
        t.profile.class(t.ph).as_str(),
        t.ipa(),
    ])
}

fn golden_texts() -> Vec<String> {
    include_str!("data/speech/tts_v1_golden.jsonl")
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter_map(|v| v["text"].as_str().map(str::to_string))
        .collect()
}

fn verbalize_inputs() -> Vec<&'static str> {
    include_str!("data/speech/verbalize_inputs.txt")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect()
}

fn check(profile: Profile, data: &str, n: usize) {
    let p = Phonemizer::new(profile);
    let mut rows = 0;
    for line in data.lines() {
        let row: Value = serde_json::from_str(line).expect("golden row");
        assert_eq!(
            row["version"],
            profile.version(),
            "golden file of another version"
        );
        let text = row["text"].as_str().expect("text");
        let got: Vec<Value> = p.phonemize(text).tokens.iter().map(token_row).collect();
        assert_eq!(Value::Array(got), row["tokens"], "{text}");
        rows += 1;
    }
    assert_eq!(rows, n);
}

#[test]
fn tts_v1_golden_is_frozen() {
    check(
        Profile::TtsV1,
        include_str!("data/speech/tts_v1_golden.jsonl"),
        60,
    );
}

#[test]
fn tts_v2_golden_is_frozen() {
    let n = golden_texts().len() + verbalize_inputs().len();
    check(
        Profile::TtsV2,
        include_str!("data/speech/tts_v2_golden.jsonl"),
        n,
    );
}

fn verbal_row(text: &str) -> Value {
    let v = verbalize(text);
    serde_json::json!({
        "text": text,
        "spoken": v.text,
        "segments": v.segments.iter().map(|s| serde_json::json!([
            &text[s.span.clone()], s.spoken, s.kind,
        ])).collect::<Vec<_>>(),
    })
}

#[test]
fn verbalize_golden_is_frozen() {
    let data = include_str!("data/speech/verbalize_golden.jsonl");
    let rows: Vec<Value> = data
        .lines()
        .map(|l| serde_json::from_str(l).expect("golden row"))
        .collect();
    assert_eq!(rows.len(), verbalize_inputs().len());
    for row in rows {
        let text = row["text"].as_str().expect("text");
        assert_eq!(verbal_row(text), row, "{text}");
    }
}

#[test]
#[ignore = "rewrites tests/data/speech/tts_v2_golden.jsonl and verbalize_golden.jsonl"]
fn write_goldens() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/speech");
    let p = Phonemizer::new(Profile::TtsV2);
    let mut out = String::new();
    let texts = golden_texts();
    for text in texts.iter().map(String::as_str).chain(verbalize_inputs()) {
        let tokens: Vec<Value> = p.phonemize(text).tokens.iter().map(token_row).collect();
        let o = serde_json::json!({
            "version": Profile::TtsV2.version(),
            "text": text,
            "tokens": tokens,
        });
        out.push_str(&serde_json::to_string(&o).expect("json"));
        out.push('\n');
    }
    std::fs::write(dir.join("tts_v2_golden.jsonl"), out).expect("write");
    let mut out = String::new();
    for text in verbalize_inputs() {
        out.push_str(&serde_json::to_string(&verbal_row(text)).expect("json"));
        out.push('\n');
    }
    std::fs::write(dir.join("verbalize_golden.jsonl"), out).expect("write");
}

#[test]
fn profiles_are_deterministic_and_distinct() {
    let text = "Serkan mı bavulu hazırladı? Ankara'ya gittik, sonra döndük.";
    for profile in [Profile::TtsV1, Profile::TtsV2] {
        let a = Phonemizer::new(profile).phonemize(text);
        let b = Phonemizer::new(profile).phonemize(text);
        assert_eq!(a.tokens, b.tokens);
        assert_eq!(Profile::from_name(profile.name()), Some(profile));
    }
    let v1 = Phonemizer::new(Profile::TtsV1).phonemize(text);
    let v2 = Phonemizer::new(Profile::TtsV2).phonemize(text);
    // the same words, pauses and stress; tts-v2 writes IPA and the literature's segments
    assert_eq!(v1.tokens.len(), v2.tokens.len());
    let stress = |u: &akana_core::speech::Utterance| -> Vec<u8> {
        u.tokens.iter().map(|t| t.stress).collect()
    };
    assert_eq!(stress(&v1), stress(&v2));
    let ph = |u: &akana_core::speech::Utterance| -> String {
        u.tokens
            .iter()
            .map(|t| t.ph_symbol())
            .collect::<Vec<_>>()
            .join(" ")
    };
    assert!(ph(&v1).starts_with("_ s E R k a n"));
    assert!(ph(&v2).starts_with("_ s æ r̝̊ k a n"));
    assert_eq!(Profile::default(), Profile::TtsV1);
}

#[test]
fn every_tts_v2_symbol_follows_the_contract() {
    let p = Phonemizer::new(Profile::TtsV2);
    for text in golden_texts()
        .iter()
        .map(String::as_str)
        .chain(verbalize_inputs())
    {
        for t in p.phonemize(text).tokens {
            let ph = t.ph_symbol();
            assert_eq!(ph.starts_with('_'), t.ph.is_pause(), "{ph} in {text}");
            assert_eq!(t.ipa().is_none(), t.ph.is_pause());
        }
    }
}
