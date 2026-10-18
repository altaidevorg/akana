//! TR IBANs, TCKNs and VKNs by their shape, for the values the checksum detectors in mod.rs do
//! not take: an IBAN with a wrong check digit or in another grouping (TR3333 3333 ...), partly
//! starred (TR33 6370 7659 **** **** **), and a TCKN or VKN after its trigger words even when
//! the checksum fails or other words come between (TC no alanına 60920-884-639, vatandaşlık
//! numarası 08577-580-979, vergi no 753347****, mükellef numarası 1654184088). For a gateway a
//! mistyped IBAN is as sensitive as a valid one. A bare 10- or 11-digit number without a
//! trigger still needs a valid checksum (mod.rs).

use super::normalize::turkish_lower_with_origin;
use super::{PiiEntity, PiiType};
use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    /// "TR", two check digits, then digits or stars in groups of any size.
    static ref TR_IBAN: Regex = Regex::new(r"\btr\s?\d{2}(?:[\s-]?[\d*]){14,24}").unwrap();

    /// A TCKN or VKN trigger, up to three words, and a value of digits and stars.
    static ref ID_AFTER_TRIGGER: Regex = Regex::new(
        r"(?x)
        \b(?:
            (?P<tckn>
              t\.?\s?c\.?(?:\s*kimlik)?(?:\s*(?:no|numarası|numarasi|numaram|nosu))?
              |tckn
              |kimlik\s*(?:no|numarası|numarasi|numaram|nosu|bilgisi)
              |vatandaşlık\s*(?:no|numarası|numaram)?|vatandaslik\s*(?:no|numarasi|numaram)?
            )
            |
            (?P<vkn>
              vergi\s*(?:kimlik\s*)?(?:no|numarası|numarasi|numaram|nosu)
              |vkn
              |mükellef\s*(?:no|numarası)|mukellef\s*(?:no|numarasi)
            )
        )\b
        (?:[\s:=-]+(?:\p{L}+[\s:=-]+){0,3}?|[\s:=-]*)
        (?P<val>[\d*](?:[\x20.\-]?[\d*]){8,13})"
    ).unwrap();
}

fn counts(value: &str) -> (usize, usize) {
    let digits = value.bytes().filter(u8::is_ascii_digit).count();
    let stars = value.bytes().filter(|&b| b == b'*').count();
    (digits, stars)
}

fn push(
    text: &str,
    start: usize,
    end: usize,
    pii_type: PiiType,
    confidence: f32,
    out: &mut Vec<PiiEntity>,
) {
    let span = &text[start..end];
    out.push(PiiEntity {
        text: span.to_string(),
        label: pii_type.as_str().to_string(),
        pii_type,
        start,
        end,
        confidence,
        stem: span.to_string(),
        suffix: None,
    });
}

/// Detects IBAN, TCKN and VKN shapes; see the module documentation.
pub(super) fn detect_ids(text: &str, out: &mut Vec<PiiEntity>) {
    if !text.bytes().any(|b| b.is_ascii_digit()) {
        return;
    }
    let (lower, origin) = turkish_lower_with_origin(text);
    let orig = |i: usize| origin.get(i).copied().unwrap_or(text.len());

    for m in TR_IBAN.find_iter(&lower) {
        let value = m.as_str().trim_end_matches([' ', '-']);
        let (digits, stars) = counts(&value[2..]);
        // a TR IBAN has 24 digits after TR (a group more or less is a typo); a starred one
        // may be cut short
        let full = (22..=26).contains(&(digits + stars)) && digits >= 16;
        let starred = stars > 0 && (16..=24).contains(&(digits + stars)) && digits >= 10;
        if full || starred {
            let (s, e) = (orig(m.start()), orig(m.start() + value.len()));
            push(text, s, e, PiiType::Iban, 0.85, out);
        }
    }

    for cap in ID_AFTER_TRIGGER.captures_iter(&lower) {
        let Some(v) = cap.name("val") else { continue };
        let value = v.as_str().trim_end_matches([' ', '.', '-']);
        // not the start of a longer number
        if lower[v.start() + value.len()..].starts_with(|c: char| c.is_ascii_digit()) {
            continue;
        }
        let (digits, stars) = counts(value);
        let n = digits + stars;
        let pii_type = if cap.name("tckn").is_some() && n == 11 {
            PiiType::Tckn
        } else if cap.name("vkn").is_some() && n == 10 {
            PiiType::Vkn
        } else {
            continue;
        };
        if digits < 6 {
            continue;
        }
        let (s, e) = (orig(v.start()), orig(v.start() + value.len()));
        push(text, s, e, pii_type, 0.9, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn found(text: &str) -> Vec<(String, PiiType)> {
        let mut out = Vec::new();
        detect_ids(text, &mut out);
        out.into_iter().map(|e| (e.text, e.pii_type)).collect()
    }

    #[test]
    fn shapes_and_triggers() {
        for (text, value, t) in [
            (
                "iade için IBAN alanına TR3333 3333 3333 3333 3333 3333 55 girildi",
                "TR3333 3333 3333 3333 3333 3333 55",
                PiiType::Iban,
            ),
            (
                "IBAN TR33 6370 7659 **** **** ** kaydedildi",
                "TR33 6370 7659 **** **** **",
                PiiType::Iban,
            ),
            (
                "Ödeme TR77 8463 7933 5591 9395 5575 71 IBAN numarasına",
                "TR77 8463 7933 5591 9395 5575 71",
                PiiType::Iban,
            ),
            (
                "TC kimlik no hanesinde 54128950340 bulunuyor",
                "54128950340",
                PiiType::Tckn,
            ),
            (
                "Uyelik dogrulamada TC no alanina 60920-884-639 yazildi",
                "60920-884-639",
                PiiType::Tckn,
            ),
            (
                "vatandaşlık numarası 08577-580-979 kaydedildi",
                "08577-580-979",
                PiiType::Tckn,
            ),
            (
                "Bayi kayıt formunda vergi no 753347**** olarak",
                "753347****",
                PiiType::Vkn,
            ),
            (
                "vergi numarası 792 6479 219 olarak güncellendi",
                "792 6479 219",
                PiiType::Vkn,
            ),
            (
                "Tedarikci profilinde mukellef numarasi 1654184088 bilgisi",
                "1654184088",
                PiiType::Vkn,
            ),
        ] {
            assert!(
                found(text).iter().any(|(v, ty)| v == value && *ty == t),
                "{text}: {:?}",
                found(text)
            );
        }
    }

    #[test]
    fn other_numbers_are_left_alone() {
        for text in [
            "Sipariş no 12345678901 kargoda",
            "TC kimlik numarasını yarın 3 kez soracağım",
            "vergi dairesi 2024 yılında 1234 dosya inceledi",
            "TRT 1 kanalı 2024 yılında 100 yıl belgeseli",
        ] {
            assert!(found(text).is_empty(), "{text}: {:?}", found(text));
        }
    }
}
