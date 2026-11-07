//! Behaviour tests of the `espeak-compat` profile that need no fixtures.

use super::*;

fn syms(u: &Utterance) -> String {
    u.tokens
        .iter()
        .map(|t| t.ph.symbol())
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn empty_and_odd_inputs_give_edges_and_never_panic() {
    let p = Phonemizer::new(Profile::EspeakCompat);
    for text in [
        "",
        " ",
        "...",
        "123",
        "\u{0302}",
        "a\u{0302}b",
        "-i",
        "'",
        "ğ",
        "Ş",
        "x",
        "İ̇",
    ] {
        let u = p.phonemize(text);
        assert_eq!(
            u.tokens.first().map(|t| t.ph),
            Some(Phoneme::Pause),
            "{text:?}"
        );
        assert_eq!(
            u.tokens.last().map(|t| t.ph),
            Some(Phoneme::Pause),
            "{text:?}"
        );
    }
    assert_eq!(syms(&p.phonemize("")), "_ _");
    let u = p.phonemize("123");
    assert!(u.warnings.is_empty());
    let readings: Vec<&str> = u.words.iter().map(|w| w.reading.as_str()).collect();
    assert_eq!(readings, ["yüz", "yirmi", "üç"]);
    assert!(u.words.iter().all(|w| w.normalized == Some("number")));
    assert_eq!(p.phonemize("a # b").warnings.len(), 1);
}

#[test]
fn output_is_deterministic_and_buffers_are_reused() {
    let p = Phonemizer::new(Profile::EspeakCompat);
    let text = "Ömer iki kadehten fazla içmemişti, fakat başı dönüyordu.";
    let a = p.phonemize(text);
    let mut b = Utterance::default();
    p.phonemize_into("başka bir metin", &mut b).unwrap();
    p.phonemize_into(text, &mut b).unwrap();
    assert_eq!(a.tokens, b.tokens);
    assert_eq!(
        serde_json::to_string(&a).unwrap(),
        serde_json::to_string(&b).unwrap()
    );
}

#[test]
fn too_long_text_is_an_error() {
    let p = Phonemizer::new(Profile::EspeakCompat);
    let text = "a".repeat(MAX_TEXT_BYTES + 1);
    let mut u = Utterance::default();
    assert!(matches!(
        p.phonemize_into(&text, &mut u),
        Err(PhonemizeError::TooLong { .. })
    ));
}

#[test]
fn token_schema_serialises_like_the_fixtures() {
    let p = Phonemizer::new(Profile::EspeakCompat);
    let u = p.phonemize("Dağ.");
    let json = serde_json::to_string(&u.tokens).unwrap();
    assert!(
        json.starts_with(
            r#"[{"ph":"_","stress":0,"word":0,"lengthen":0,"mark":".","type":"pause","ipa":null}"#
        ),
        "{json}"
    );
    assert!(
        json.contains(
            r#""ph":"a","stress":7,"word":0,"lengthen":1,"mark":".","type":"vowel","ipa":"aː""#
        ),
        "{json}"
    );
}
