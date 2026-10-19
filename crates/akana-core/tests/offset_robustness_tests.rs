//! Offsets found in a lower-cased copy of the text must be mapped back to the text: Turkish
//! lower-casing changes byte lengths (İ -> i, I -> ı), so without the mapping a span lands on
//! the wrong bytes after every İ/I and slicing the text with it can panic.
use akana_core::grammar::TurkishGrammarChecker;
use akana_core::phonology::{origin_offset, to_turkish_lower, to_turkish_lower_with_origin};
use akana_core::style::ClicheMatcher;

#[test]
fn lower_with_origin_maps_every_byte_back() {
    for text in [
        "İSTANBUL'DA IŞIK",
        "Iıİi çok güzel",
        "ascii only",
        "",
        "ǅ ẞ İİİ",
    ] {
        let (lower, origin) = to_turkish_lower_with_origin(text);
        assert_eq!(lower, to_turkish_lower(text));
        assert_eq!(origin.len(), lower.len());
        for (i, _) in lower.char_indices() {
            assert!(text.is_char_boundary(origin_offset(&origin, text.len(), i)));
        }
    }
}

#[test]
fn tautology_findings_after_dotted_capital_i() {
    let checker = TurkishGrammarChecker::new();
    for text in [
        "İİİİ İSTANBUL'DA henüz hala bekliyoruz.",
        "IŞIK henüz hala yanıyor.",
        "İ henüz hala",
    ] {
        let result = checker.check(text);
        let finding = result
            .findings
            .iter()
            .find(|f| f.original_text.to_lowercase().contains("henüz"))
            .unwrap_or_else(|| panic!("no tautology finding for {text:?}"));
        assert_eq!(
            &text[finding.start_offset..finding.end_offset],
            finding.original_text
        );
        assert_eq!(finding.original_text, "henüz hala");
    }
}

#[test]
fn cliche_offsets_after_dotted_capital_i() {
    let text = "İİİ İDARE açısından Bu bağlamda önemli bir adım atıldı.";
    let found = ClicheMatcher::find_all(text);
    let m = found
        .iter()
        .find(|m| !m.pattern_name.is_empty() && m.text.to_lowercase() == "bu bağlamda")
        .expect("cliché found");
    assert_eq!(&text[m.start..m.end], "Bu bağlamda");
}
