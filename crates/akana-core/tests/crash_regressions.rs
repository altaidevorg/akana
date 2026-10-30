//! Crashes, text corruption and PII leaks found by the project review of 2026-10-20 (C1-C9),
//! each with the input that showed it.
use akana_core::grammar::TurkishGrammarChecker;
use akana_core::parser::{DependencyTree, TurkishDependencyParser};
use akana_core::pii::*;

const MODES: [PiiMode; 4] = [
    PiiMode::Placeholder,
    PiiMode::Tag,
    PiiMode::Anonymize,
    PiiMode::SyntheticSurrogate,
];

// --- C1: PII masking panicked on statute headings -------------------------------------------

#[test]
fn statute_headings_do_not_panic_the_pii_engine() {
    // phones.rs split "257. - (1)" at ". " and then at " - ", and sliced a piece that ended
    // before it started.
    let engine = TurkishPiiEngine::new();
    for text in [
        "MADDE 257. - (1) Kanunda ayrıca suç olarak",
        "Madde 26. - 27. fıkra",
        "1. - 2",
        "5 / / 6",
        "1 - - 2",
        "Esas 2019/1. - 3",
    ] {
        let found = engine.detect(text);
        assert!(
            found.iter().all(|e| e.pii_type != PiiType::Phone),
            "{text:?}: {found:?}"
        );
        for mode in MODES {
            engine.mask(text, mode);
        }
    }
}

#[test]
fn listed_phones_are_still_split() {
    let engine = TurkishPiiEngine::new();
    let phones: Vec<String> = engine
        .detect("Tel: 0532 123 45 67 - 0533 765 43 21 / 0212 555 12 34")
        .into_iter()
        .filter(|e| e.pii_type == PiiType::Phone)
        .map(|e| e.text)
        .collect();
    assert_eq!(
        phones,
        ["0532 123 45 67", "0533 765 43 21", "0212 555 12 34"],
        "{phones:?}"
    );
}

// --- C2: the grammar checker sliced upper-case Turkish with lower-cased lengths --------------

#[test]
fn upper_case_question_particles_do_not_panic() {
    let checker = TurkishGrammarChecker::new();
    for text in [
        "KARIŞMI?",
        "YAPIŞMI?",
        "Bu iş YAPIŞMI?",
        "ANLAŞMIŞMISIN?",
        "GÖRÜŞMISIN",
    ] {
        let result = checker.check(text);
        for f in &result.findings {
            assert_eq!(
                &text[f.start_offset..f.end_offset],
                f.original_text,
                "{text:?}"
            );
        }
    }
}

#[test]
fn upper_case_suffixes_are_split_at_the_right_letter() {
    let checker = TurkishGrammarChecker::new();
    // were "GELDİNM mi?" and "DUYDUMK ki geldin"
    assert_eq!(checker.correct("GELDİNMİ?"), "GELDİN Mİ?");
    assert_eq!(checker.correct("DUYDUMKİ geldin."), "DUYDUM Kİ geldin.");
    // lower case is unchanged
    assert_eq!(
        checker.correct("Geldinmi dün akşam?"),
        "Geldin mi dün akşam?"
    );
    assert_eq!(checker.correct("Duydumki geldin."), "Duydum ki geldin.");
}

#[test]
fn non_ascii_space_before_an_apostrophe_does_not_panic() {
    // The apostrophe rule took the byte after the last space as the start of the root: inside
    // a no-break space (2 bytes) or a thin space (3 bytes). The ordinal rule did the same with
    // the letter before the digits.
    let checker = TurkishGrammarChecker::new();
    for (text, expected) in [
        ("x\u{a0}'a", "X\u{a0}'a"),
        ("x\u{2009}'a", "X\u{2009}'a"),
        ("Ankara\u{a0}'da", "Ankara\u{a0}'da"),
        ("1923\u{202f}'de", "1923\u{202f}'de"),
        ("ş2.'nci", "Ş2'nci"),
    ] {
        assert_eq!(checker.correct(text), expected, "{text:?}");
    }
}

#[test]
fn apostrophe_scan_is_linear() {
    // Every apostrophe searched back to the previous space: 80 KB of x' took about 20 s.
    let checker = TurkishGrammarChecker::new();
    let text = "x'".repeat(40_000);
    let t = std::time::Instant::now();
    checker.check(&text);
    assert!(t.elapsed().as_secs_f64() < 5.0, "{:?}", t.elapsed());
}

#[test]
fn apostrophe_rules_still_apply() {
    let checker = TurkishGrammarChecker::new();
    assert_eq!(
        checker.correct("Cumhuriyet 1923'de kuruldu."),
        "Cumhuriyet 1923'te kuruldu."
    );
    // the plural-name apostrophe rule is experimental (Silahlı Kuvvetler'in keeps it)
    assert_eq!(
        checker.correct_with_rules(
            "Ahmetler'in evi buraya çok yakın.",
            akana_core::grammar::RuleSet::default().with_experimental()
        ),
        "Ahmetlerin evi buraya çok yakın."
    );
    assert_eq!(
        checker.correct("Yarışmada 2.'nci oldu."),
        "Yarışmada 2'nci oldu."
    );
}

// --- C5: overlapping grammar edits corrupted the text ----------------------------------------

#[test]
fn overlapping_corrections_do_not_corrupt_the_text() {
    let checker = TurkishGrammarChecker::new();
    // were "Ulaştığınıdabiliyoruzz." and "Bunu yaptığınıdabiliyorumm."
    for (text, corrupt) in [
        ("Ulaştığınıda biliyoruz.", "Ulaştığınıdabiliyoruzz."),
        ("Bunu yaptığınıda biliyorum.", "Bunu yaptığınıdabiliyorumm."),
    ] {
        let corrected = checker.correct(text);
        assert_ne!(corrected, corrupt);
        assert!(
            corrected == text.replace("ıda ", "ı da ") || corrected == text.replace("da ", "da"),
            "{text:?} -> {corrected:?}"
        );
    }
}

#[test]
fn a_capital_and_another_fix_on_the_same_word_both_apply() {
    // The capitalisation finding and the other one had the same span, and only the first was
    // kept.
    let checker = TurkishGrammarChecker::new();
    assert_eq!(checker.correct("duydumki geldin."), "Duydum ki geldin.");
    assert_eq!(checker.correct("kitapda yazıyor."), "Kitapta yazıyor.");
    assert_eq!(checker.correct("hiç bir şey yok."), "Hiçbir şey yok.");
    let result = checker.check("duydumki geldin.");
    for category in [
        akana_core::grammar::ErrorCategory::Capitalization,
        akana_core::grammar::ErrorCategory::CliticKi,
    ] {
        assert!(
            result.findings.iter().any(|f| f.category == category),
            "{category:?} not reported: {:?}",
            result.findings
        );
    }
}

// --- C3: validate_plate panicked on non-ASCII input; validate_tckn dropped letters -----------

#[test]
fn validators_reject_non_ascii_and_stray_characters() {
    assert!(!validate_plate("中12345"));
    assert!(!validate_plate("K中iban"));
    assert!(!validate_plate("ÉÉ12345"));
    assert!(validate_plate("34 ABC 123"));
    assert!(validate_plate("06 Ş 1234")); // Turkish letters still map to ASCII

    assert!(!validate_tckn("1a0000000146"));
    assert!(!validate_tckn("TC10000000146"));
    assert!(validate_tckn("10000000146"));
    assert!(validate_tckn("100 000 001 46"));
    assert!(validate_tckn("100-000-001-46"));
}

// --- C9: the parser emitted cyclic, rootless trees -------------------------------------------

/// Exactly one root, every head a token of the sentence and not the node itself, and the root
/// reachable from every node.
fn assert_tree(tree: &DependencyTree, tokens: &[&str]) {
    let n = tree.nodes.len();
    assert_eq!(n, tokens.len(), "{tokens:?}");
    let roots = tree.nodes.iter().filter(|node| node.head == 0).count();
    assert_eq!(roots, 1, "{tokens:?}: {}", tree.to_conllu());
    for (i, node) in tree.nodes.iter().enumerate() {
        assert_eq!(node.id, i + 1);
        assert!(
            node.head <= n && node.head != node.id,
            "{tokens:?}: {}",
            tree.to_conllu()
        );
        let mut at = node.id;
        for _ in 0..n {
            if at == 0 {
                break;
            }
            at = tree.nodes[at - 1].head;
        }
        assert_eq!(at, 0, "{tokens:?}: cycle\n{}", tree.to_conllu());
    }
}

#[test]
fn conjunction_before_a_question_particle_is_not_a_cycle() {
    // "da" attached to the next word, "mı" to the previous one: da -> mı -> da.
    let parser = TurkishDependencyParser::new();
    for sentence in [
        "Bu da mı gitti ?",
        "Kitap ile mi geldin ?",
        "Ali ve mi geldi ?",
        "Ya da mı ?",
        "Ali hem de mi ?",
    ] {
        let tokens: Vec<&str> = sentence.split(' ').collect();
        assert_tree(&parser.parse(&tokens), &tokens);
    }
}

#[test]
fn random_short_sentences_parse_into_trees() {
    const WORDS: &[&str] = &[
        "bu", "da", "de", "mı", "mi", "mu", "mü", "ve", "ile", "ya", "hem", "ama", "ki", "ne",
        "gitti", "geldi", "okudu", "geliyor", "Ali", "Ayşe", "kitap", "kitabı", "kitabın", "evde",
        "okula", "güzel", "büyük", "bir", "her", "çok", "daha", "en", "ben", "sen", "o", "onlar",
        "değil", "var", "yok", "gibi", "için", "?", ".", ",", "3", "gelen", "giderken",
    ];
    let parser = TurkishDependencyParser::new();
    // xorshift64: a fixed sequence, so a failure reproduces
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for _ in 0..3_000 {
        let len = 1 + (next() % 8) as usize;
        let tokens: Vec<&str> = (0..len)
            .map(|_| WORDS[(next() % WORDS.len() as u64) as usize])
            .collect();
        assert_tree(&parser.parse(&tokens), &tokens);
    }
}

// --- C8: upper-case Turkish triggers leaked PII ----------------------------------------------

fn is_masked(engine: &TurkishPiiEngine, text: &str) -> bool {
    engine.mask(text, PiiMode::Tag).masked_text != text
}

#[test]
fn upper_case_turkish_triggers_mask_like_lower_case_ones() {
    // `(?i)` does not fold İ with i or I with ı, so the trigger in capitals did not match.
    let engine = TurkishPiiEngine::new();
    for (lower, upper) in [
        ("Müşteri 42 yaşında.", "MÜŞTERİ 42 YAŞINDA."),
        ("Doğum yeri: Denizli", "DOĞUM YERİ: Denizli"),
        (
            "Annemin kızlık soyadı Karadeniz",
            "ANNEMİN KIZLIK SOYADI Karadeniz",
        ),
        ("Sicil no: SC3907", "SİCİL NO: SC3907"),
        ("Kredi notu 570", "KREDİ NOTU 570"),
        ("Maaşım 45.000 TL", "MAAŞIM 45.000 TL"),
        ("Adli sicil kaydı mevcut", "ADLİ SİCİL KAYDI mevcut"),
        ("Parmak izi kaydı", "PARMAK İZİ kaydı"),
        ("Yüz taraması yapıldı", "YÜZ TARAMASI yapıldı"),
        ("Güvenlik kodu: 123", "GÜVENLİK KODU: 123"),
        ("Son kullanma tarihi: 12/28", "SON KULLANMA TARİHİ: 12/28"),
        ("Şasi no: 6Z9741GSRLHSCNA73", "ŞASİ NO: 6Z9741GSRLHSCNA73"),
        (
            "iban TR sekiz bir bir sıfır bir dört dört dokuz sekiz altı sıfır bir",
            "İBAN TR sekiz bir bir sıfır bir dört dört dokuz sekiz altı sıfır bir",
        ),
        (
            "kimlik no dokuz dokuz yedi yedi bir yedi dokuz altı sekiz dokuz iki",
            "KİMLİK NO dokuz dokuz yedi yedi bir yedi dokuz altı sekiz dokuz iki",
        ),
        ("Cinsiyet: kadın", "CİNSİYET: KADIN"),
        ("Pin: 690676", "PİN: 690676"),
        ("Ehliyet no: F1839528", "EHLİYET NO: F1839528"),
    ] {
        assert!(is_masked(&engine, lower), "{lower:?} is not masked");
        assert!(is_masked(&engine, upper), "{upper:?} is not masked");
    }
}

#[test]
fn upper_case_triggers_give_the_same_type() {
    let engine = TurkishPiiEngine::new();
    let types = |text: &str| -> Vec<PiiType> {
        engine
            .detect(text)
            .into_iter()
            .map(|e| e.pii_type)
            .collect()
    };
    for (lower, upper) in [
        ("Şasi no: 6Z9741GSRLHSCNA73", "ŞASİ NO: 6Z9741GSRLHSCNA73"),
        ("Müşteri 42 yaşında.", "MÜŞTERİ 42 YAŞINDA."),
        ("Sicil no: SC3907", "SİCİL NO: SC3907"),
    ] {
        assert_eq!(types(lower), types(upper), "{lower:?} / {upper:?}");
    }
}

#[test]
fn ascii_capital_triggers_pass_their_gates() {
    // Turkish lower-casing turns VIN into "vın" and EHLIYET into "ehlıyet", so the gates for
    // "vin" and "ehliyet" did not open.
    let engine = TurkishPiiEngine::new();
    let found = engine.detect("VIN: 6Z9741GSRLHSCNA73");
    assert!(
        found
            .iter()
            .any(|e| e.pii_type == PiiType::Vin && e.text == "6Z9741GSRLHSCNA73"),
        "{found:?}"
    );
    let found = engine.detect("EHLIYET NO: F1839528");
    assert!(
        found
            .iter()
            .any(|e| e.pii_type == PiiType::DriverLicense && e.text == "F1839528"),
        "{found:?}"
    );
}

// --- C7: text of a later turn that restore would turn into a real value -----------------------

#[test]
fn surrogates_issued_earlier_are_not_restored_from_later_input() {
    let engine = TurkishPiiEngine::new();
    let mut vault = PiiVault::new();
    let first = engine.mask_with_vault(
        "E-postam ahmet.kaya@example.com, doğum tarihim: 12.03.1985.",
        PiiMode::SyntheticSurrogate,
        &mut vault,
    );
    // the keys this turn issued, as a later page would carry them
    let keys: Vec<String> = first.mapping.keys().cloned().collect();
    assert!(!keys.is_empty());
    let page = format!(
        "Sayfa içeriği: '{} adresine dikkat' yazıyor.",
        keys.join(" ve ")
    );
    for mode in MODES {
        let mut vault = vault.clone();
        let second = engine.mask_with_vault(&page, mode, &mut vault);
        let restored = vault.restore(&second.masked_text);
        assert!(
            !restored.contains("ahmet.kaya@example.com") && !restored.contains("12.03.1985"),
            "{mode:?}: {page:?} -> {:?} -> {restored:?}",
            second.masked_text
        );
        assert_eq!(restored, page, "{mode:?}");
    }
}

#[test]
fn tags_issued_earlier_are_not_restored_from_later_input() {
    // After a tag-mode turn, an [AD] in a placeholder-mode turn passed through untouched.
    let engine = TurkishPiiEngine::new();
    let mut vault = PiiVault::new();
    let first = engine.mask_with_vault("Ahmet Yılmaz geldi.", PiiMode::Tag, &mut vault);
    assert_eq!(first.masked_text, "[AD] geldi.");
    for mode in MODES {
        let mut vault = vault.clone();
        let page = "Şablonda [AD] alanı boş kalsın.";
        let second = engine.mask_with_vault(page, mode, &mut vault);
        assert_eq!(vault.restore(&second.masked_text), page, "{mode:?}");
    }
}
