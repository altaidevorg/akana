//! Postal addresses as people type them: abbreviated, in lower case or capitals, in any order
//! (gümüşköy-erzurum sedefli mh bayır sk. bina 89 D: 22, ordu mavikent yakamoz mh zeytin cad
//! no:109 d 2, Atatürk Sokak No: 62 Talas Mah. Mamak Ankara). The address regex in mod.rs needs
//! "Mahallesi" or "Mah." and capitalised words, and takes one street part.
//!
//! An address is anchored on a mahalle (X mahallesi / mahalle / mah. / mh) followed by a street,
//! a building number or both, or on a street (X sokak / sk. / cadde / cad. / bulvar) followed by
//! a number tag (no, numara, bina). From the anchor it takes the apartment, street, number, unit
//! (daire, d, kat, kapı, blok) and district/province parts that follow, and the province and
//! district words right before it. It works on the Turkish lower-cased text, so İ and capitals
//! do not matter, and maps the spans back.

use super::gazetteer::{DISTRICTS, PROVINCES};
use super::normalize::turkish_lower_with_origin;
use super::{PiiEntity, PiiType};
use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    /// "<name> mahallesi" and what follows it.
    static ref MAHALLE: Regex = Regex::new(
        r"(?x)
        \b\p{L}[\p{L}\d-]*\.?\s+(?:mahallesi|mahalle|mah\.|mah\b|mh\.|mh\b)
        (?P<rest>
          (?:[\s,]+(?:\p{L}+\s+){1,3}(?:apt\.|apt\b|apartmanı|sitesi|blok)\b)?
          (?P<street>[\s,]+(?:[\p{L}\d.]+\s+){1,3}?(?:sokağı|sokak|sok\.|sok\b|sk\.|sk\b|caddesi|cadde|cad\.|cad\b|cd\.|cd\b|bulvarı|bulvar|blv\.|blv\b|bulv\.|yolu|meydanı)\b\.?)?
          (?P<number>[\s,]+(?:(?:no|numara|nr|bina|kapı\s+no)\b\.?\s*:?\s*)?\d+[a-z]?(?:\s*/\s*\d+)?\b)?
          (?:[\s,]+(?:daire|d|kat|kapı|blok)\b\.?\s*:?\s*\d+)*
        )"
    ).unwrap();

    /// "<name> sokak no 12" and what follows it, for addresses without a mahalle. One name word:
    /// a longer name is taken by its capitals (`extend_name_left`), since the leftmost match of
    /// several words began in the prose before it (yeni yer Atatürk Sokak).
    static ref STREET: Regex = Regex::new(
        r"(?x)
        \b\p{L}[\p{L}\d.]*\s+(?:sokağı|sokak|sok\.|sk\.|caddesi|cadde|cad\.|cd\.|bulvarı|bulvar|blv\.)
        \s+(?:no|numara|nr|bina)\b\.?\s*:?\s*\d+[a-z]?(?:\s*/\s*\d+)?\b
        (?:[\s,]+(?:daire|d|kat|kapı|blok)\b\.?\s*:?\s*\d+)*
        (?:[\s,]+\p{L}+\s+(?:mahallesi|mahalle|mah\.|mah\b|mh\.|mh\b))?"
    ).unwrap();

    /// A district and province after the rest (akpınar/kırşehir, kadıköy / istanbul).
    static ref PLACE_PAIR: Regex = Regex::new(r"^[\s,]+\p{L}+\s*/\s*\p{L}+\b").unwrap();
}

fn is_place(word: &str) -> bool {
    let w = word.trim_matches(|c: char| !c.is_alphanumeric());
    !w.is_empty()
        && (PROVINCES.contains(w)
            || DISTRICTS.contains(w)
            || w.split('-')
                .filter(|p| !p.is_empty())
                .all(|p| PROVINCES.contains(p) || DISTRICTS.contains(p))
            || w.split('-').any(|p| PROVINCES.contains(p)))
}

/// The start of the word before `at` in `lower`, and the word.
fn word_before(lower: &str, at: usize) -> (usize, &str) {
    let before = lower[..at].trim_end_matches([' ', '\t', ',']);
    let start = before
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_whitespace() || *c == ',')
        .map_or(0, |(i, c)| i + c.len_utf8());
    (start, &before[start..])
}

/// The start of the province and district words right before `start` in `lower`: a place word
/// (kadıköy, gümüşköy-erzurum), or a province and the word after it, the district in "il ilçe"
/// order whether or not the list knows it (ordu mavikent, istanbul dereova).
fn extend_left(lower: &str, start: usize) -> usize {
    let (s1, w1) = word_before(lower, start);
    if w1.is_empty() {
        return start;
    }
    let (s2, w2) = word_before(lower, s1);
    if !w2.is_empty() && PROVINCES.contains(w2.trim_matches(|c: char| !c.is_alphanumeric())) {
        return s2;
    }
    if is_place(w1) {
        return s1;
    }
    start
}

/// The start of up to two capitalised words before `start` (in `text`'s spelling, read through
/// `origin`): the rest of a name of several words (Hacı Bektaş Veli Sokak, Fatih Sultan Mehmet
/// Mahallesi).
fn extend_name_left(lower: &str, text: &str, origin: &[usize], start: usize) -> usize {
    let mut at = start;
    for _ in 0..2 {
        let (s, w) = word_before(lower, at);
        if w.is_empty() {
            break;
        }
        let o = origin.get(s).copied().unwrap_or(text.len());
        let capital = text[o..].chars().next().is_some_and(char::is_uppercase);
        // a label (Adres:, Adresim) is not part of the name
        if !capital || is_place(w) || w.ends_with(':') || w.starts_with("adres") {
            break;
        }
        at = s;
    }
    at
}

/// The word after `at` in `lower`: its end, and the word.
fn word_after(lower: &str, at: usize) -> (usize, String) {
    let rest = &lower[at..];
    let skip = rest.len() - rest.trim_start_matches([' ', '\t', ',']).len();
    let word: String = rest[skip..]
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '-')
        .collect();
    (at + skip + word.len(), word)
}

/// The end of the district and province after `end` in `lower`: a pair (akpınar/kırşehir),
/// a district and its province (Mamak Ankara, Yenişehir Mersin; the district need not be in
/// the list), or one place word.
fn extend_right(lower: &str, end: usize) -> usize {
    if let Some(m) = PLACE_PAIR.find(&lower[end..]) {
        return end + m.end();
    }
    let (e1, w1) = word_after(lower, end);
    if w1.is_empty() || e1 == end {
        return end;
    }
    let (e2, w2) = word_after(lower, e1);
    if !w2.is_empty() && PROVINCES.contains(w2.as_str()) {
        return e2;
    }
    if is_place(&w1) {
        return e1;
    }
    end
}

/// Detects postal addresses; see the module documentation.
pub(super) fn detect_addresses(text: &str, out: &mut Vec<PiiEntity>) {
    let (lower, origin) = turkish_lower_with_origin(text);
    if !["mah", "mh", "sok", "sk", "cad", "cd", "bulv", "blv"]
        .iter()
        .any(|k| lower.contains(k))
    {
        return;
    }
    let orig = |i: usize| origin.get(i).copied().unwrap_or(text.len());
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for cap in MAHALLE.captures_iter(&lower) {
        // a mahalle alone is a word (mahalle halkı); with a street or a number it is an address
        if cap.name("street").is_none() && cap.name("number").is_none() {
            continue;
        }
        let m = cap.get(0).unwrap();
        spans.push((m.start(), m.end()));
    }
    for m in STREET.find_iter(&lower) {
        spans.push((m.start(), m.end()));
    }
    spans.sort();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (s, e) in spans {
        let s = extend_name_left(&lower, text, &origin, s);
        let (s, e) = (extend_left(&lower, s), extend_right(&lower, e));
        match merged.last_mut() {
            Some(last) if s <= last.1 => last.1 = last.1.max(e),
            _ => merged.push((s, e)),
        }
    }
    for (s, e) in merged {
        let (start, end) = (orig(s), orig(e));
        let span = text[start..end].trim_end_matches(['.', ',', ' ']);
        let end = start + span.len();
        out.push(PiiEntity {
            text: span.to_string(),
            label: PiiType::Address.as_str().to_string(),
            pii_type: PiiType::Address,
            start,
            end,
            confidence: 0.9,
            stem: span.to_string(),
            suffix: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn found(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        detect_addresses(text, &mut out);
        out.into_iter().map(|e| e.text).collect()
    }

    #[test]
    fn informal_addresses() {
        for (text, address) in [
            (
                "teslim noktası gümüşköy-erzurum sedefli mh bayır sk. bina 89 D: 22 kargo",
                "gümüşköy-erzurum sedefli mh bayır sk. bina 89 D: 22",
            ),
            (
                "doğru konum ordu mavikent yakamoz mh zeytin cad no:109 d 2 lütfen",
                "ordu mavikent yakamoz mh zeytin cad no:109 d 2",
            ),
            (
                "konum akdeniz MH GÜNEŞ Sk 127/32 akpınar/kırşehir ama",
                "akdeniz MH GÜNEŞ Sk 127/32 akpınar/kırşehir",
            ),
            (
                "yeni yer Atatürk Sokak No: 62 Talas Mah. Mamak Ankara.",
                "Atatürk Sokak No: 62 Talas Mah. Mamak Ankara",
            ),
            (
                "Adres: Örnek Mahallesi, Veri Sokak No: 14 Kat: 2 Yenişehir Mersin şeklinde",
                "Örnek Mahallesi, Veri Sokak No: 14 Kat: 2 Yenişehir Mersin",
            ),
            (
                "yer edirne altınova köknar MH Yıldız apt rüzgar cadde no: 87 kapı 21 diye",
                "edirne altınova köknar MH Yıldız apt rüzgar cadde no: 87 kapı 21",
            ),
            (
                "Adresim Hacı Bektaş Veli Sokak No: 5 Daire 3 Çankaya/Ankara olarak kayıtlı",
                "Hacı Bektaş Veli Sokak No: 5 Daire 3 Çankaya/Ankara",
            ),
        ] {
            assert_eq!(found(text), [address], "{text}");
        }
    }

    #[test]
    fn prose_is_not_an_address() {
        for text in [
            "mahalle halkı toplandı",
            "bu sokakta çocuklar oynuyor",
            "caddede trafik var",
            "Mahalle muhtarı açıklama yaptı.",
        ] {
            assert!(found(text).is_empty(), "{text}: {:?}", found(text));
        }
    }
}
