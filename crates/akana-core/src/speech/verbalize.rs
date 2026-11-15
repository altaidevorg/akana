//! Written Turkish to its spoken form for a TTS front end: everything a screen reader meets. Numbers (with the suffix written after them re-harmonised to the word they now end
//! in), ordinals, Roman numerals, decimals, fractions, ranges, dates, times, money, percentages,
//! units, telephone numbers, abbreviations and initialisms, URLs and e-mail addresses, symbols,
//! letters spelled with hyphens, and foreign words, brands and letters read Turkish-style.
//!
//! Each verbalisation keeps the byte span of the text it reads ([`Verbalized::segments`]); the
//! text between them is kept as written. Without the word rules ([`verbalize_with`] with
//! `words` false, as `tts-v1` reads), only stretches with a digit or a symbol are verbalised:
//! text of letters and punctuation comes back unchanged.
//!
//! With the word rules, numbers and codes are also read as Turkish service text reads them
//! (`tts-v1` keeps its earlier readings): telephone numbers in their groups with a pause between
//! them; identifier numbers (five or more digits next to a word such as `kod`, `numara`,
//! `sipariş`, `hesap`) digit by digit, while quantities stay numbers; letters in codes and
//! initialisms one letter name each (`KDV` ka de ve, `30 MM 6653` otuz me me altmış altı elli
//! üç); codes of letters and digits and IBANs in their groups; doses (`2x1` iki çarpı bir); a
//! half as `buçuk`; a decimal written with a dot as a decimal; `16.10 tarihinde` as a date.
//!
//! The rules are Akana's own, written from Turkish usage (TDK's spelling guide for ordinals,
//! dates, times and abbreviations; Göksel & Kerslake 2005 for suffix harmony). Initialisms are
//! read as Akana's morphology reads them (Zemberek's letter names and pronunciation guesser, in
//! the lexicon's `pr=` attributes), so the suffixes written after them harmonise as the analyser
//! expects. normalizer-tr (Apache-2.0) was used only as a black box to compare readings; no code
//! or data of it is used.

use crate::morphology::pronunciation::{guess_abbreviation, letter_pronunciation, roman_value};
use crate::phonology::to_turkish_lower;
use lazy_static::lazy_static;
use regex::Regex;
use std::ops::Range;

/// One verbalised stretch of the text.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct VerbalSegment {
    /// Byte span in the input.
    pub span: Range<usize>,
    /// Byte span of its reading in [`Verbalized::text`].
    pub spoken_span: Range<usize>,
    /// The reading (`üçte` for `3'te`).
    pub spoken: String,
    /// What it reads: `cardinal`, `ordinal`, `decimal`, `fraction`, `range`, `date`, `time`,
    /// `money`, `percent`, `unit`, `telephone`, `roman`, `abbreviation`, `initialism`,
    /// `identifier` (a code read digit by digit or letter by letter), `electronic`, `symbol`,
    /// `letters`, `foreign` or `word` (a word written with an apostrophe it does not take).
    pub kind: &'static str,
    /// Where the text kept as written resumes in [`Verbalized::text`] (after a space the reading
    /// may need before a letter).
    #[serde(skip)]
    pub(crate) resume: usize,
}

/// A text and its spoken form.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct Verbalized {
    /// The spoken text: the input with every segment replaced by its reading.
    pub text: String,
    /// The verbalised stretches, in order.
    pub segments: Vec<VerbalSegment>,
}

impl Verbalized {
    /// The input span and kind of a span of the spoken text: a span inside a verbalised segment
    /// maps to that segment's whole input span; text kept as written maps one to one.
    pub(crate) fn source(
        &self,
        r: Range<usize>,
        src_len: usize,
    ) -> (Range<usize>, Option<&'static str>) {
        let segs = &self.segments;
        let map = |pos: usize, is_end: bool| -> (usize, Option<&'static str>) {
            // the last segment starting before pos (for an end, at or before pos - 1)
            let k = segs.partition_point(|s| {
                if is_end {
                    s.spoken_span.start < pos
                } else {
                    s.spoken_span.start <= pos
                }
            });
            if k > 0 {
                let s = &segs[k - 1];
                let inside = if is_end {
                    pos > s.spoken_span.start && pos <= s.spoken_span.end
                } else {
                    pos >= s.spoken_span.start && pos < s.spoken_span.end
                };
                if inside {
                    return (if is_end { s.span.end } else { s.span.start }, Some(s.kind));
                }
            }
            // kept text: shifted as after the last segment that ends at or before pos
            let done = segs.partition_point(|s| s.spoken_span.end <= pos);
            let delta: isize = match done.checked_sub(1).map(|d| &segs[d]) {
                Some(s) => s.span.end as isize - s.resume as isize,
                None => 0,
            };
            let p = (pos as isize + delta).clamp(0, src_len as isize) as usize;
            (p, None)
        };
        let (start, k1) = map(r.start, false);
        let (end, k2) = map(r.end, true);
        (start..end.max(start), k1.or(k2))
    }
}

const VOWELS: &str = "aeıioöuüâîûAEIİOÖUÜÂÎÛ";

fn is_vowel(c: char) -> bool {
    VOWELS.contains(c)
}

// ---------------------------------------------------------------------------------------------
// numbers

const ONES: [&str; 10] = [
    "", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz",
];
const TENS: [&str; 10] = [
    "", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan",
];
const POWERS: [&str; 6] = ["", "bin", "milyon", "milyar", "trilyon", "katrilyon"];

/// A cardinal in words, separated by spaces (`1990` -> `bin dokuz yüz doksan`). Leading zeros
/// are read as `sıfır` (`007` -> `sıfır sıfır yedi`), and a number beyond the largest power
/// digit by digit.
pub(crate) fn cardinal(digits: &str) -> String {
    let trimmed = digits.trim_start_matches('0');
    let mut words: Vec<&str> = vec!["sıfır"; digits.len() - trimmed.len()];
    if trimmed.is_empty() {
        return words.join(" ");
    }
    if trimmed.len() > 3 * POWERS.len() {
        for c in trimmed.chars() {
            let d = c.to_digit(10).unwrap_or(0) as usize;
            words.push(if d == 0 { "sıfır" } else { ONES[d] });
        }
        return words.join(" ");
    }
    let groups = trimmed.len().div_ceil(3);
    let first = trimmed.len() - 3 * (groups - 1);
    let mut start = 0;
    for g in 0..groups {
        let end = if g == 0 { first } else { start + 3 };
        let value: usize = trimmed[start..end].parse().unwrap_or(0);
        let power = groups - 1 - g;
        if value != 0 {
            // bin, not bir bin; but bir milyon
            if !(power == 1 && value == 1) {
                let (h, rest) = (value / 100, value % 100);
                if h > 1 {
                    words.push(ONES[h]);
                }
                if h > 0 {
                    words.push("yüz");
                }
                if rest / 10 > 0 {
                    words.push(TENS[rest / 10]);
                }
                if rest % 10 > 0 {
                    words.push(ONES[rest % 10]);
                }
            }
            if power > 0 {
                words.push(POWERS[power]);
            }
        }
        start = end;
    }
    words.join(" ")
}

/// The ordinal of a number in words (`dört` -> `dördüncü`): the last word takes the suffix.
fn ordinal(words: &str) -> String {
    let (head, last) = match words.rfind(' ') {
        Some(k) => (&words[..=k], &words[k + 1..]),
        None => ("", words),
    };
    let stem = if last == "dört" { "dörd" } else { last };
    let v = high_vowel(harmony_vowel(last));
    if last.ends_with(is_vowel) {
        // altı -> altıncı, iki -> ikinci
        format!("{head}{stem}nc{v}")
    } else {
        format!("{head}{stem}{v}nc{v}")
    }
}

/// The high vowel that harmonises with `v` (ı i u ü).
fn high_vowel(v: char) -> char {
    match (front(v), round(v)) {
        (false, false) => 'ı',
        (true, false) => 'i',
        (false, true) => 'u',
        (true, true) => 'ü',
    }
}

/// Digits after a decimal comma: read as a number, leading zeros as `sıfır` (`05` -> `sıfır beş`).
fn decimals(digits: &str) -> String {
    cardinal(digits)
}

// ---------------------------------------------------------------------------------------------
// suffix harmony

/// Words whose suffixes take front vowels against their last vowel (`saate`).
const INVERSE_HARMONY: &[&str] = &["saat", "santigrat"];

fn harmony_vowel(word: &str) -> char {
    if INVERSE_HARMONY.iter().any(|w| word.ends_with(w)) {
        return 'e';
    }
    word.chars().rev().find(|&c| is_vowel(c)).unwrap_or('e')
}

fn front(v: char) -> bool {
    matches!(v, 'e' | 'i' | 'ö' | 'ü' | 'î' | 'E' | 'İ' | 'Ö' | 'Ü')
}

fn round(v: char) -> bool {
    matches!(v, 'o' | 'ö' | 'u' | 'ü' | 'û' | 'O' | 'Ö' | 'U' | 'Ü')
}

fn voiceless_end(word: &str) -> bool {
    word.chars()
        .last()
        .is_some_and(|c| matches!(c, 'f' | 's' | 't' | 'k' | 'ç' | 'ş' | 'h' | 'p'))
}

/// A suffix as written (`'ye`, `'den`, `'üncü`) re-harmonised to the word it now follows:
/// vowel harmony and `d`/`t`, `c`/`ç` after a voiceless consonant. With `rebuffer` (the reading
/// ends in another word than the one the writer harmonised to: `₺850'ye`, `5 kg'ye`), also the
/// buffer consonant a vowel-final word needs (`y`, the `n` of the genitive, the `s` of the
/// possessive) or a consonant-final word drops: `sekiz yüz elli liraya`, `beş kilograma`.
pub(crate) fn harmonize(word: &str, suffix: &str, rebuffer: bool) -> String {
    let mut prev = harmony_vowel(word);
    let mut out = String::with_capacity(suffix.len() + 2);
    let word_vowel_end = word.ends_with(is_vowel);
    let mut chars: Vec<char> = suffix.chars().collect();
    if rebuffer && word_vowel_end {
        if chars.first().is_some_and(|&c| is_vowel(c)) {
            let buffer = if chars.len() == 2 && chars[1] == 'n' {
                // genitive: -(n)In
                'n'
            } else if chars.len() >= 3 && chars[1] == 'n' {
                // possessive and a case: -(s)InA, -(s)InI, -(s)InDA
                's'
            } else {
                'y'
            };
            chars.insert(0, buffer);
        }
    } else if rebuffer
        && chars.len() >= 2
        && matches!(chars[0], 'y' | 'n' | 's')
        && is_vowel(chars[1])
    {
        // a buffer written for a vowel-final reading, which a consonant-final word drops
        chars.remove(0);
    }
    for (k, &c) in chars.iter().enumerate() {
        let c2 = match c {
            'd' | 't' if k == 0 => {
                if voiceless_end(word) {
                    't'
                } else {
                    'd'
                }
            }
            'c' | 'ç' if k == 0 => {
                if voiceless_end(word) {
                    'ç'
                } else {
                    'c'
                }
            }
            'a' | 'e' => {
                let v = if front(prev) { 'e' } else { 'a' };
                prev = v;
                v
            }
            'ı' | 'i' | 'u' | 'ü' => {
                let v = high_vowel(prev);
                prev = v;
                v
            }
            c => c,
        };
        out.push(c2);
    }
    out
}

/// A number's words with the suffix written after the number: `dört` voices before a vowel
/// (`dördü`).
pub(crate) fn attach(words: &str, suffix: &str) -> String {
    attach_with(words, suffix, false)
}

/// As [`attach`], re-buffering the suffix (see [`harmonize`]).
fn attach_with(words: &str, suffix: &str, rebuffer: bool) -> String {
    if suffix.is_empty() {
        return words.to_string();
    }
    let (head, last) = match words.rfind(' ') {
        Some(k) => (&words[..=k], &words[k + 1..]),
        None => ("", words),
    };
    let suffix = harmonize(last, suffix, rebuffer);
    // dört, çeyrek and buçuk voice before a vowel (dördü, çeyreği, buçuğu)
    let stem = match last {
        "dört" if suffix.starts_with(is_vowel) => "dörd",
        "çeyrek" if suffix.starts_with(is_vowel) => "çeyreğ",
        "buçuk" if suffix.starts_with(is_vowel) => "buçuğ",
        _ => last,
    };
    format!("{head}{stem}{suffix}")
}

// ---------------------------------------------------------------------------------------------
// tables

/// Abbreviations written with a full stop, and the words they stand for.
struct Abbrev {
    written: &'static str,
    spoken: &'static str,
    place: Place,
}

/// Where an abbreviation is read as one: some are also words (`Av.` av, `Bul.` bul).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Place {
    /// Anywhere (`bkz.`, `T.C.`); a capital after it is a name, not a new sentence.
    Any,
    /// Anywhere, at the end of a phrase (`vb.`, `vs.`); its full stop also ends a sentence
    /// before a capital.
    End,
    /// After a capitalised name (`Koç Holding A.Ş.`); its full stop also ends a sentence before a
    /// capital.
    Company,
    /// Before a capitalised name or another title (`Dr. Ayşe`, `Prof. Dr.`); its full stop never
    /// ends a sentence.
    Title,
    /// After a capitalised name or a numbered street (`Atatürk Bul.`, `5. Sok.`).
    Address,
}

macro_rules! abbrevs {
    ($( $w:literal => $s:literal $(, $t:ident)? ;)*) => {
        &[$( Abbrev { written: $w, spoken: $s, place: abbrevs!(@t $($t)?) } ),*]
    };
    (@t title) => { Place::Title };
    (@t end) => { Place::End };
    (@t company) => { Place::Company };
    (@t address) => { Place::Address };
    (@t) => { Place::Any };
}

/// TDK's list of common abbreviations, the titles and the address words.
const ABBREVIATIONS: &[Abbrev] = abbrevs! {
    "Prof. Dr." => "profesör doktor", title;
    "Doç. Dr." => "doçent doktor", title;
    "Yrd. Doç. Dr." => "yardımcı doçent doktor", title;
    "Dr. Öğr. Üyesi" => "doktor öğretim üyesi", title;
    "Öğr. Gör. Dr." => "öğretim görevlisi doktor", title;
    "Öğr. Gör." => "öğretim görevlisi", title;
    "Arş. Gör. Dr." => "araştırma görevlisi doktor", title;
    "Arş. Gör." => "araştırma görevlisi", title;
    "Ord. Prof." => "ordinaryüs profesör", title;
    "Prof." => "profesör", title;
    "Doç." => "doçent", title;
    "Yrd." => "yardımcı", title;
    "Dr." => "doktor", title;
    "Av." => "avukat", title;
    "Op." => "operatör", title;
    "Uzm." => "uzman", title;
    "Müh." => "mühendis", title;
    "Mim." => "mimar", title;
    "Ecz." => "eczacı", title;
    "Hz." => "hazreti", title;
    "Sn." => "sayın", title;
    "Org." => "orgeneral", title;
    "Korg." => "korgeneral", title;
    "Tümg." => "tümgeneral", title;
    "Tuğg." => "tuğgeneral", title;
    "Alb." => "albay", title;
    "Yb." => "yarbay", title;
    "Bnb." => "binbaşı", title;
    "Yzb." => "yüzbaşı", title;
    "Ütğm." => "üsteğmen", title;
    "Tğm." => "teğmen", title;
    "Asb." => "astsubay", title;
    "Mr." => "mister", title;
    "Mrs." => "misiz", title;
    "vb." => "ve benzeri", end;
    "vs." => "vesaire", end;
    "vd." => "ve diğerleri", end;
    "bkz." => "bakınız";
    "Bkz." => "bakınız";
    "örn." => "örneğin";
    "Örn." => "örneğin";
    "yy." => "yüzyıl", end;
    "M.Ö." => "milattan önce";
    "M.S." => "milattan sonra";
    "T.C." => "te ce";
    "A.Ş." => "anonim şirketi", company;
    "Ltd. Şti." => "limited şirketi", company;
    "Şti." => "şirketi", company;
    "Ltd." => "limited", company;
    "Tic." => "ticaret", address;
    "San." => "sanayi", address;
    "Cad." => "caddesi", address;
    "Sok." => "sokağı", address;
    "Sk." => "sokak", address;
    "Mah." => "mahallesi", address;
    "Mh." => "mahallesi", address;
    "Bul." => "bulvarı", address;
    "Blv." => "bulvarı", address;
    "Apt." => "apartmanı", address;
    "çev." => "çeviren";
    "ort." => "ortalama";
    "maks." => "maksimum";
    "min." => "minimum";
    "a.g.e." => "adı geçen eser", end;
};

/// Abbreviations read only before a number (`No: 5`, `s. 12`).
const NUMBER_ABBREVIATIONS: &[(&str, &str)] = &[
    ("No.", "numara"),
    ("No:", "numara"),
    ("no.", "numara"),
    ("no:", "numara"),
    ("Nr.", "numara"),
    ("Tel.", "telefon"),
    ("Tel:", "telefon"),
    ("Tel", "telefon"),
    ("s.", "sayfa"),
    ("S.", "sayfa"),
    ("sf.", "sayfa"),
    ("md.", "madde"),
    ("Md.", "madde"),
];

/// Units after a number: (written, reading). The longest written form is tried first.
const UNITS: &[(&str, &str)] = &[
    ("km²", "kilometrekare"),
    ("km2", "kilometrekare"),
    ("m²", "metrekare"),
    ("m2", "metrekare"),
    ("cm²", "santimetrekare"),
    ("cm2", "santimetrekare"),
    ("mm²", "milimetrekare"),
    ("m³", "metreküp"),
    ("m3", "metreküp"),
    ("cm³", "santimetreküp"),
    ("km", "kilometre"),
    ("cm", "santimetre"),
    ("mm", "milimetre"),
    ("nm", "nanometre"),
    ("µm", "mikrometre"),
    ("μm", "mikrometre"),
    ("m", "metre"),
    ("ha", "hektar"),
    ("kg", "kilogram"),
    ("mg", "miligram"),
    ("µg", "mikrogram"),
    ("gr", "gram"),
    ("g", "gram"),
    ("lt", "litre"),
    ("ml", "mililitre"),
    ("cl", "santilitre"),
    ("dl", "desilitre"),
    ("L", "litre"),
    ("l", "litre"),
    ("cc", "santimetreküp"),
    ("sa", "saat"),
    ("dk", "dakika"),
    ("sn", "saniye"),
    ("ms", "milisaniye"),
    ("°C", "derece"),
    ("℃", "derece"),
    ("°F", "fahrenhayt derece"),
    ("°", "derece"),
    ("kHz", "kilohertz"),
    ("MHz", "megahertz"),
    ("GHz", "gigahertz"),
    ("Hz", "hertz"),
    ("kWh", "kilovat saat"),
    ("Wh", "vat saat"),
    ("kW", "kilovat"),
    ("MW", "megavat"),
    ("GW", "gigavat"),
    ("W", "vat"),
    ("kV", "kilovolt"),
    ("mV", "milivolt"),
    ("V", "volt"),
    ("mAh", "miliamper saat"),
    ("mA", "miliamper"),
    ("Ah", "amper saat"),
    ("Ω", "ohm"),
    ("dB", "desibel"),
    ("KB", "kilobayt"),
    ("kB", "kilobayt"),
    ("MB", "megabayt"),
    ("GB", "gigabayt"),
    ("TB", "terabayt"),
    ("Kbps", "kilobit"),
    ("Mbps", "megabit"),
    ("Gbps", "gigabit"),
    ("px", "piksel"),
    ("kcal", "kilokalori"),
    ("cal", "kalori"),
    ("mmHg", "milimetre cıva"),
    ("hPa", "hektopaskal"),
    ("kPa", "kilopaskal"),
    ("Pa", "paskal"),
    ("rpm", "devir"),
];

/// The time unit of a rate (`km/sa`): its locative (`saatte`).
const PER: &[(&str, &str)] = &[
    ("sa", "saatte"),
    ("saat", "saatte"),
    ("h", "saatte"),
    ("dk", "dakikada"),
    ("sn", "saniyede"),
    ("s", "saniyede"),
];

/// Currencies: (written, unit, subunit).
const CURRENCIES: &[(&str, &str, &str)] = &[
    ("₺", "lira", "kuruş"),
    ("TL", "lira", "kuruş"),
    ("TRY", "lira", "kuruş"),
    ("$", "dolar", "sent"),
    ("USD", "dolar", "sent"),
    ("€", "avro", "sent"),
    ("EUR", "avro", "sent"),
    ("£", "sterlin", "peni"),
    ("GBP", "sterlin", "peni"),
    ("¥", "yen", ""),
    ("JPY", "yen", ""),
    ("CHF", "frank", ""),
    ("₽", "ruble", ""),
    ("RUB", "ruble", ""),
    ("₹", "rupi", ""),
    ("₿", "bitkoin", ""),
    ("BTC", "bitkoin", ""),
];

/// Initialisms read with English letter names or as a word, against what the letter rules
/// would give (`CNN` si en en, `UNICEF` yunisef).
const INITIALISMS: &[(&str, &str)] = &[
    ("BBC", "bibisi"),
    ("CNN", "sienen"),
    ("FBI", "efbiay"),
    ("CIA", "siayey"),
    ("IMF", "ayemef"),
    ("NBA", "enbiey"),
    ("HBO", "eyçbio"),
    ("CEO", "siio"),
    ("DJ", "dicey"),
    ("CD", "sidi"),
    ("DVD", "dividi"),
    ("PC", "pisi"),
    ("PR", "piar"),
    ("VIP", "vip"),
    ("FIFA", "fifa"),
    ("UNICEF", "yunisef"),
    ("UNESCO", "yunesko"),
    ("NASA", "nasa"),
    ("ISO", "iso"),
    ("OK", "okey"),
    ("LED", "led"),
    ("RAM", "ram"),
    ("ROM", "rom"),
    ("PIN", "pin"),
    ("IBAN", "iban"),
    ("COVID", "kovid"),
    ("AIDS", "eyds"),
    ("HIV", "haşiv"),
    ("WC", "vece"),
    ("IQ", "aykü"),
    ("AI", "eyay"),
    ("API", "eypiay"),
    ("USB", "usebe"),
    ("SMS", "semese"),
    ("GPS", "cepese"),
];

/// Foreign words, brands and names as Turkish speakers say them (Akana's own transcriptions).
const FOREIGN: &[(&str, &str)] = &[
    ("iphone", "ayfon"),
    ("ipad", "aypet"),
    ("ios", "ayoes"),
    ("imac", "aymek"),
    ("macbook", "mekbuk"),
    ("apple", "epıl"),
    ("google", "gugıl"),
    ("youtube", "yutub"),
    ("facebook", "feysbuk"),
    ("twitter", "tivitır"),
    ("whatsapp", "vatsap"),
    ("microsoft", "maykrosoft"),
    ("windows", "vindovs"),
    ("netflix", "netfliks"),
    ("spotify", "spotifay"),
    ("zoom", "zum"),
    ("skype", "skayp"),
    ("wi-fi", "vayfay"),
    ("wifi", "vayfay"),
    ("bluetooth", "blutut"),
    ("chatgpt", "çetcipiti"),
    ("openai", "openeyay"),
    ("online", "onlayn"),
    ("offline", "oflayn"),
    ("e-mail", "imeyl"),
    ("email", "imeyl"),
    ("mail", "meyl"),
    ("web", "veb"),
    ("software", "softveyr"),
    ("hardware", "hardveyr"),
    ("laptop", "leptop"),
    ("smartphone", "smartfon"),
    ("playstation", "pleysteyşın"),
    ("xbox", "eksboks"),
    ("nike", "nayki"),
    ("starbucks", "starbaks"),
    ("mcdonald's", "mekdonalds"),
    ("coca-cola", "koka kola"),
    ("new", "nyu"),
    ("washington", "vaşington"),
    ("shakespeare", "şekspir"),
    ("world", "vörld"),
    ("okay", "okey"),
    ("show", "şov"),
    ("weekend", "vikend"),
    ("jazz", "caz"),
    ("c++", "si artı artı"),
    ("c#", "si şarp"),
    (".net", "dat net"),
];

/// Symbols read aloud where they stand alone.
const SYMBOLS: &[(char, &str)] = &[
    ('&', "ve"),
    ('+', "artı"),
    ('=', "eşittir"),
    ('<', "küçüktür"),
    ('>', "büyüktür"),
    ('≤', "küçük eşittir"),
    ('≥', "büyük eşittir"),
    ('≠', "eşit değildir"),
    ('±', "artı eksi"),
    ('×', "çarpı"),
    ('÷', "bölü"),
    ('%', "yüzde"),
    ('‰', "binde"),
    ('°', "derece"),
    ('#', "diyez"),
    ('@', "et"),
    ('*', "yıldız"),
    ('§', "paragraf"),
    ('©', "telif hakkı"),
    ('®', "tescilli marka"),
    ('™', "ticari marka"),
    ('~', "yaklaşık"),
    ('½', "yarım"),
    ('¼', "çeyrek"),
    ('¾', "dörtte üç"),
    ('₺', "lira"),
    ('$', "dolar"),
    ('€', "avro"),
    ('£', "sterlin"),
    ('\\', "ters eğik çizgi"),
];

const MONTHS: [&str; 12] = [
    "ocak", "şubat", "mart", "nisan", "mayıs", "haziran", "temmuz", "ağustos", "eylül", "ekim",
    "kasım", "aralık",
];

lazy_static! {
    static ref SUFFIX: Regex = Regex::new(r"^['’]([a-zçğıöşüâîû]+)").expect("suffix regex");
    static ref EMAIL: Regex =
        Regex::new(r"^[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
            .expect("email regex");
    static ref URL: Regex = Regex::new(
        r"^(?:(?:https?://)(?:www\.)?|www\.)[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*(?:/[A-Za-z0-9._~%/?#=&+-]*)?|^[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:com|org|edu|gov|io|info|biz|app)(?:\.tr)?(?:/[A-Za-z0-9._~%/?#=&+-]*)?|^[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:com|net|org|edu|gov|gen|bel|k12|av|web|tv|info|biz)\.tr(?:/[A-Za-z0-9._~%/?#=&+-]*)?"
    )
    .expect("url regex");
    static ref PHONE: Regex = Regex::new(
        r"^(?:\+90[ -]?|0 ?)?(?:\([0-9]{3,4}\)|[0-9]{3,4})[ -]?[0-9]{3}[ -]?[0-9]{2}[ -]?[0-9]{2}"
    )
    .expect("phone regex");
    static ref DATE: Regex =
        Regex::new(r"^([0-9]{1,2})([./-])([0-9]{1,2})([./-])([0-9]{4}|[0-9]{2})").expect("date regex");
    static ref ISO_DATE: Regex = Regex::new(r"^([0-9]{4})-([0-9]{2})-([0-9]{2})").expect("iso regex");
    static ref TIME: Regex =
        Regex::new(r"^([0-9]{1,2})([:.])([0-9]{2})(?::([0-9]{2}))?").expect("time regex");
    static ref AMOUNT: Regex =
        Regex::new(r"^([0-9]{1,3}(?:\.[0-9]{3})+|[0-9]+)(?:,([0-9]+))?").expect("amount regex");
}

// ---------------------------------------------------------------------------------------------
// the scanner

struct Scan<'a> {
    src: &'a str,
    out: Verbalized,
    /// Start of the text not yet copied.
    kept: usize,
    /// Apply the rules for words written in letters (abbreviations, foreign words and letters,
    /// letters spelled with hyphens in small letters).
    words: bool,
}

impl<'a> Scan<'a> {
    fn keep_to(&mut self, upto: usize) {
        if upto > self.kept {
            self.out.text.push_str(&self.src[self.kept..upto]);
            self.kept = upto;
        }
    }

    /// Replaces `start..end` by `spoken`.
    fn emit(&mut self, start: usize, end: usize, spoken: &str, kind: &'static str) {
        self.keep_to(start);
        // keep a space between the reading and a letter or digit written next to it
        if self
            .out
            .text
            .chars()
            .next_back()
            .is_some_and(char::is_alphanumeric)
        {
            self.out.text.push(' ');
        }
        let o = self.out.text.len();
        self.out.text.push_str(spoken);
        let o_end = self.out.text.len();
        if self.src[end..]
            .chars()
            .next()
            .is_some_and(char::is_alphanumeric)
        {
            self.out.text.push(' ');
        }
        self.out.segments.push(VerbalSegment {
            span: start..end,
            spoken_span: o..o_end,
            spoken: spoken.to_string(),
            kind,
            resume: self.out.text.len(),
        });
        self.kept = end;
    }
}

fn prev_char(s: &str, i: usize) -> Option<char> {
    s[..i].chars().next_back()
}

fn next_char(s: &str, i: usize) -> Option<char> {
    s[i..].chars().next()
}

/// The written suffix after a verbalised stretch (`'te`): its letters and the end.
fn suffix_at(s: &str, i: usize) -> Option<(String, usize)> {
    let c = SUFFIX.captures(&s[i..])?;
    let m = c.get(0)?;
    // a suffix ends the word
    if next_char(s, i + m.end()).is_some_and(|c| c.is_alphanumeric()) {
        return None;
    }
    Some((c[1].to_string(), i + m.end()))
}

/// Attaches the suffix written at `i`, if any: returns the reading and the new end.
fn with_suffix(s: &str, spoken: String, i: usize, rebuffer: bool) -> (String, usize) {
    match suffix_at(s, i) {
        Some((suf, end)) => (attach_with(&spoken, &suf, rebuffer), end),
        None => (spoken, i),
    }
}

/// An amount at `i`: integer digits (thousands dots removed) and decimals.
fn amount_at(s: &str, i: usize) -> Option<(String, Option<String>, usize)> {
    let c = AMOUNT.captures(&s[i..])?;
    let m = c.get(0)?;
    let int = c[1].replace('.', "");
    let dec = c.get(2).map(|d| d.as_str().to_string());
    Some((int, dec, i + m.end()))
}

/// A number and its decimals. With the word rules a half after a whole number is `buçuk`
/// (`9,5` dokuz buçuk; `0,5` stays sıfır virgül beş).
fn amount_words(int: &str, dec: Option<&str>, words: bool) -> String {
    match dec {
        Some("5") if words && !int.trim_start_matches('0').is_empty() => {
            format!("{} buçuk", cardinal(int))
        }
        Some(d) => format!("{} virgül {}", cardinal(int), decimals(d)),
        None => cardinal(int),
    }
}

/// With the word rules, one or two digits after a dot that follows `end` (`3.8`, `17.5`, not
/// `1.250` or `2.0.1`): the decimals and their end.
fn dot_decimal_at(s: &str, end: usize) -> Option<(String, usize)> {
    if next_char(s, end) != Some('.') {
        return None;
    }
    let d0 = end + 1;
    let mut d1 = d0;
    while next_char(s, d1).is_some_and(|c| c.is_ascii_digit()) {
        d1 += 1;
    }
    let another =
        next_char(s, d1) == Some('.') && next_char(s, d1 + 1).is_some_and(|c| c.is_ascii_digit());
    ((1..=2).contains(&(d1 - d0)) && !another).then(|| (s[d0..d1].to_string(), d1))
}

/// A unit written at `i` (after optional spaces): its reading and end.
fn unit_at(s: &str, i: usize) -> Option<(&'static str, usize)> {
    let j = skip_space(s, i, 1);
    let rest = &s[j..];
    let mut best: Option<(&str, &str)> = None;
    for &(w, r) in UNITS {
        if rest.starts_with(w)
            && !next_char(rest, w.len()).is_some_and(|c| c.is_alphanumeric())
            && best.is_none_or(|(bw, _)| w.len() > bw.len())
        {
            best = Some((w, r));
        }
    }
    let (w, r) = best?;
    // `°` and `°C` may follow the number directly; letters need a word boundary before
    Some((r, j + w.len()))
}

fn currency_at(s: &str, i: usize) -> Option<(&'static str, &'static str, usize)> {
    let j = skip_space(s, i, 1);
    let rest = &s[j..];
    CURRENCIES.iter().find_map(|&(w, u, sub)| {
        (rest.starts_with(w) && !next_char(rest, w.len()).is_some_and(|c| c.is_alphanumeric()))
            .then_some((u, sub, j + w.len()))
    })
}

/// Skips at most `max` spaces (a space or a no-break space).
fn skip_space(s: &str, mut i: usize, max: usize) -> usize {
    let mut n = 0;
    while n < max {
        match next_char(s, i) {
            Some(c @ (' ' | '\u{a0}' | '\u{202f}')) => {
                i += c.len_utf8();
                n += 1;
            }
            _ => break,
        }
    }
    i
}

/// An all-caps initialism's reading, as Akana's morphology pronounces it (`ABD` abede,
/// `NATO` nato), unless the table has it.
fn initialism(word: &str) -> String {
    if let Some(&(_, r)) = INITIALISMS.iter().find(|(w, _)| *w == word) {
        return r.to_string();
    }
    // a capital I is ı in a Turkish initialism (one with Ç Ğ İ Ö Ş Ü) and i in a foreign one
    let turkish = word.contains(['Ç', 'Ğ', 'İ', 'Ö', 'Ş', 'Ü']);
    let lower = if turkish {
        to_turkish_lower(word)
    } else {
        word.to_lowercase()
    };
    if let Some(pr) = super::tts::abbreviation_pronunciation(&lower) {
        return pr;
    }
    guess_abbreviation(&lower)
}

// ---------------------------------------------------------------------------------------------
// codes, identifiers and initialisms with the word rules

/// The name of H in a code or an initialism: "he" (THY te he ye, HGS he ge se), never "ha",
/// which is out of place in professional speech (the owner's decision, 2026-11-15).
const H_NAME: &str = "he";

/// A letter's name in a code or an initialism, one word each: the Turkish names, K as "ka" as
/// it is said in initialisms (KDV ka de ve, KVKK ka ve ka ka), Q "kû", W "dabılyu", X "iks".
/// Takes a small letter.
fn code_letter(c: char) -> Option<&'static str> {
    Some(match c {
        'a' | 'â' => "a",
        'b' => "be",
        'c' => "ce",
        'ç' => "çe",
        'd' => "de",
        'e' => "e",
        'f' => "fe",
        'g' => "ge",
        'ğ' => "yumuşak ge",
        'h' => H_NAME,
        'ı' => "ı",
        'i' | 'î' => "i",
        'j' => "je",
        'k' => "ka",
        'l' => "le",
        'm' => "me",
        'n' => "ne",
        'o' => "o",
        'ö' => "ö",
        'p' => "pe",
        'r' => "re",
        's' => "se",
        'ş' => "şe",
        't' => "te",
        'u' | 'û' => "u",
        'ü' => "ü",
        'v' => "ve",
        'y' => "ye",
        'z' => "ze",
        'q' => "kû",
        'w' => "dabılyu",
        'x' => "iks",
        _ => return None,
    })
}

/// Capitals in small letters: Turkish rules when the word has a Turkish capital (`İ` is i, `I`
/// is ı), else as in English (`I` is i).
fn code_lower(word: &str) -> String {
    if word.contains(['Ç', 'Ğ', 'İ', 'Ö', 'Ş', 'Ü']) {
        to_turkish_lower(word)
    } else {
        word.to_lowercase()
    }
}

/// Letters by name, one word each (`KDV` ka de ve); digits are skipped.
fn spell_letters(word: &str) -> String {
    code_lower(word)
        .chars()
        .filter_map(code_letter)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Each digit by name (`0482` sıfır dört sekiz iki).
fn digit_words(digits: &str) -> String {
    digits
        .chars()
        .filter_map(|c| c.to_digit(10))
        .map(|d| if d == 0 { "sıfır" } else { ONES[d as usize] })
        .collect::<Vec<_>>()
        .join(" ")
}

/// A code of capitals and digits read in order: letters by name, digits one by one
/// (`X7K9QZ` iks yedi ka dokuz kû ze).
fn spell_code(code: &str) -> String {
    let lower = code_lower(code);
    let mut words: Vec<&str> = Vec::new();
    for c in lower.chars() {
        if let Some(d) = c.to_digit(10) {
            words.push(if d == 0 { "sıfır" } else { ONES[d as usize] });
        } else if let Some(n) = code_letter(c) {
            words.push(n);
        }
    }
    words.join(" ")
}

/// Whether a reading is the letters' names run together (`kedeve` for kdv, `küre` for qr), as
/// the lexicon and the letter rules write a spelled initialism.
fn is_spelled(pr: &str, letters: &[char]) -> bool {
    let Some((&c, rest)) = letters.split_first() else {
        return pr.is_empty();
    };
    let one;
    let names: &[&str] = match c {
        'k' => &["ke", "ka"],
        'h' => &["he", "ha"],
        'q' => &["kü", "ku", "kû"],
        'w' => &["dabılyu", "ve"],
        'ğ' => &["yumuşakge"],
        _ => match code_letter(c) {
            Some(n) => {
                one = [n];
                &one
            }
            None => return false,
        },
    };
    names
        .iter()
        .any(|n| pr.strip_prefix(n).is_some_and(|r| is_spelled(r, rest)))
}

/// Initialisms read as words, against the letter rule (`SMS` esemes).
const SPOKEN_AS_WORDS: &[(&str, &str)] = &[("SMS", "esemes"), ("IBAN", "iban"), ("KOBİ", "kobi")];

/// Common nouns that are sometimes written with an apostrophe before a suffix, as a name would
/// be (`841 lira'lık`); read as the one word they are (`liralık`).
const APOSTROPHE_NOUNS: &[&str] = &[
    "lira",
    "kuruş",
    "dolar",
    "avro",
    "euro",
    "sterlin",
    "metre",
    "kilometre",
    "santimetre",
    "milimetre",
    "kilo",
    "kilogram",
    "gram",
    "litre",
    "mililitre",
    "derece",
    "saat",
    "dakika",
    "saniye",
    "gün",
    "hafta",
    "ay",
    "yıl",
    "adet",
    "tane",
    "kişi",
    "puan",
    "yüzde",
];

/// An initialism with the word rules: its letters by name, one word each (`KDV` ka de ve), when
/// the lexicon or the letter rules spell it; else its reading as a word (`NATO`, `ODTÜ`, `CNN`
/// sienen). The flag says it is spelled.
fn initialism_words(word: &str) -> (String, bool) {
    if let Some(&(_, r)) = SPOKEN_AS_WORDS.iter().find(|(w, _)| *w == word) {
        return (r.to_string(), false);
    }
    let letters: Vec<char> = code_lower(word).chars().collect();
    let pr = initialism(word);
    if is_spelled(&pr, &letters) {
        (spell_letters(word), true)
    } else {
        (pr, false)
    }
}

/// The word before `i` in the same clause (after spaces and at most a colon or a dash), in
/// small letters.
fn word_before(s: &str, i: usize) -> Option<String> {
    let mut t = s[..i].trim_end_matches([' ', '\u{a0}']);
    if let Some(u) = t.strip_suffix([':', '-']) {
        t = u.trim_end_matches([' ', '\u{a0}']);
    }
    let start = t
        .char_indices()
        .rev()
        .take_while(|&(_, c)| c.is_alphabetic() || c == '\'' || c == '’')
        .last()
        .map(|(k, _)| k)?;
    Some(to_turkish_lower(&t[start..]))
}

/// The word after `end` (after an apostrophe suffix and a space), in small letters.
fn word_after(s: &str, end: usize) -> Option<String> {
    word_after_at(s, end).map(|(w, _)| w)
}

/// The word after `end` and where it ends.
fn word_after_at(s: &str, end: usize) -> Option<(String, usize)> {
    let mut j = end;
    if matches!(next_char(s, j), Some('\'' | '’')) {
        j += next_char(s, j).map_or(1, char::len_utf8);
        while next_char(s, j).is_some_and(char::is_alphabetic) {
            j += next_char(s, j).map_or(1, char::len_utf8);
        }
    }
    let k = skip_space(s, j, 1);
    if k == j {
        return None;
    }
    let e = word_end(s, k);
    (e > k).then(|| (to_turkish_lower(&s[k..e]), e))
}

/// Words that name a number as an identifier wherever it stands next to them (`kodunuz 482915`,
/// `36366 koduyla`, `10832 numaralı`).
const ID_WORDS: &[&str] = &["numara", "nolu", "kod", "şifre", "parola"];
/// Words that name the number after them as an identifier (`Siparişiniz 549853`, `Hesap
/// 982532'ye`), unless a count noun follows it.
const ID_HEADS: &[&str] = &[
    "doğrulama",
    "onay",
    "takip",
    "sipariş",
    "fatura",
    "hesap",
    "hesab",
    "dosya",
    "poliçe",
    "referans",
    "müşteri",
    "abone",
    "posta",
    "rezervasyon",
    "kimlik",
    "sicil",
    "sözleşme",
    "talep",
    "başvuru",
    "kargo",
    "gönderi",
    "pnr",
    "iban",
];
/// Nouns that count what the number before them counts (`12500 başvuru`, `12500 puan`).
const COUNT_NOUNS: &[&str] = &[
    "kişi",
    "adet",
    "tane",
    "puan",
    "kez",
    "defa",
    "ürün",
    "kullanıcı",
    "müşteri",
    "başvuru",
    "sipariş",
    "paket",
    "gün",
    "ay",
    "yıl",
    "saat",
    "dakika",
    "saniye",
    "hafta",
    "lira",
    "kuruş",
    "dolar",
    "avro",
    "euro",
    "metre",
    "kilo",
    "gram",
    "litre",
    "sayfa",
    "kelime",
    "karakter",
    "satır",
    "kayıt",
    "işlem",
    "mesaj",
    "dosya",
    "araç",
    "ev",
    "konut",
    "öğrenci",
];

fn names_id(w: &str, stems: &[&str]) -> bool {
    // a plural is a count (siparişleriniz), not one record
    stems.iter().any(|p| w.starts_with(p)) && !w.contains("ler") && !w.contains("lar")
}

/// Possessive endings a record word may take before its number (`Siparişiniz 549853`,
/// `Hesabım 12345`); with a case ending it introduces an amount (`hesabınıza 15000`).
const POSSESSIVES: &[&str] = &[
    "", "ı", "i", "u", "ü", "m", "ım", "im", "um", "üm", "n", "ın", "in", "un", "ün", "sı", "si",
    "su", "sü", "mız", "miz", "muz", "müz", "ımız", "imiz", "umuz", "ümüz", "nız", "niz", "nuz",
    "nüz", "ınız", "iniz", "unuz", "ünüz",
];

/// Record words that, with a case ending, introduce an amount (`hesabınıza 15000 yatırıldı`,
/// `faturanıza 12500 yansıdı`): they name an identifier bare or with a possessive only.
const AMOUNT_HEADS: &[&str] = &["hesap", "hesab", "fatura"];

/// A record word (`siparişinizi`, `doğrulamasında`; `hesap` and `hesabınız` but not
/// `hesabınıza`).
fn record_head(w: &str) -> bool {
    if w.contains("ler") || w.contains("lar") {
        return false;
    }
    ID_HEADS.iter().any(|p| match w.strip_prefix(p) {
        Some(rest) if AMOUNT_HEADS.contains(p) => POSSESSIVES.contains(&rest),
        Some(_) => true,
        None => false,
    })
}

fn counts(w: &str) -> bool {
    COUNT_NOUNS
        .iter()
        .any(|n| w == *n || (n.chars().count() >= 4 && w.starts_with(n)))
}

/// Whether the digits at `i..end` are an identifier: an identifier word before them, or one
/// with its suffix after them (`36366 koduyla`, `10832 numaralı`; a bare `kod` after a number
/// counts, as in `12345 kod satırı`), or a record word before them with no count noun after.
fn id_context(s: &str, i: usize, end: usize) -> bool {
    let before = word_before(s, i);
    let after = word_after(s, end);
    let id_word = |w: &str| {
        matches!(w, "no" | "nolu")
            || w.starts_with("iban")
            || (names_id(w, ID_WORDS) && !ID_WORDS.contains(&w))
    };
    if after.as_deref().is_some_and(id_word) {
        return true;
    }
    // a record word, then the identifier word: 76950 posta kodlu, 117379 IBAN numarası
    if let Some((w, e)) = word_after_at(s, end) {
        if record_head(&w) && word_after(s, e).as_deref().is_some_and(id_word) {
            return true;
        }
    }
    let head = before
        .as_deref()
        .is_some_and(|w| names_id(w, ID_WORDS) || record_head(w) || w == "no");
    head && !after.as_deref().is_some_and(counts)
}

/// A quantity follows `end`: a currency, a unit, a percent sign or a scale word.
fn quantity_follows(s: &str, end: usize) -> bool {
    let j = skip_space(s, end, 1);
    currency_at(s, end).is_some()
        || unit_at(s, end).is_some()
        || next_char(s, j) == Some('%')
        || ["bin", "milyon", "milyar", "trilyon"]
            .iter()
            .any(|w| s[j..].starts_with(w) && j > end)
}

/// Words before a dotted number that make it a version or a section (`Python 3.12`, `madde
/// 12.5`), which keeps its dot (`nokta`).
const VERSION_WORDS: &[&str] = &[
    "sürüm", "versiyon", "python", "java", "android", "ios", "windows", "linux", "macos", "chrome",
    "firefox", "ubuntu", "madde", "bölüm", "kısım", "fıkra", "paragraf", "sayfa", "tablo", "şekil",
    "adım", "ek", "ünite", "konu", "soru", "v", "ver",
];

/// Whether the dotted number at `i..end` is a version or a section number.
fn version_context(s: &str, i: usize, end: usize) -> bool {
    let before_written = s[..i]
        .trim_end()
        .rsplit(|c: char| !c.is_alphanumeric())
        .next()
        .unwrap_or("");
    // a name in capitals or with a capital inside (HDMI 2.1, iOS 17.2)
    let product = before_written.chars().skip(1).any(char::is_uppercase);
    let before = word_before(s, i);
    let after = word_after(s, end);
    product
        || before.as_deref().is_some_and(|w| {
            VERSION_WORDS
                .iter()
                .any(|v| w == *v || (v.len() >= 4 && w.starts_with(v)))
        })
        || after
            .as_deref()
            .is_some_and(|w| w.starts_with("sürüm") || w.starts_with("versiyon"))
}

/// Suffixes a letter's name takes after an apostrophe (`C'den`, `T'nin`, `G'ye`): case,
/// possessive and plural endings. Anything else is no suffix (`N'aber`, `M'Vila`).
const LETTER_SUFFIXES: &[&str] = &[
    "a", "e", "ya", "ye", "ı", "i", "u", "ü", "yı", "yi", "yu", "yü", "ın", "in", "un", "ün",
    "nın", "nin", "nun", "nün", "da", "de", "ta", "te", "dan", "den", "tan", "ten", "la", "le",
    "yla", "yle", "lar", "ler", "sı", "si", "su", "sü", "daki", "deki", "taki", "teki",
];

/// Words near a ten-digit number that make it a telephone number (`8086727078 numaralı hat`,
/// `telefon numaranız 3827842248`).
fn phone_context(s: &str, i: usize, end: usize) -> bool {
    let phone_word = |w: &str| {
        matches!(w, "tel" | "hat" | "cep" | "gsm")
            || ["telefon", "hatt", "faks", "fax"]
                .iter()
                .any(|p| w.starts_with(p))
    };
    let before = s[..i]
        .split(|c: char| !c.is_alphabetic())
        .filter(|w| !w.is_empty())
        .rev()
        .take(4);
    if before.map(to_turkish_lower).any(|w| phone_word(&w)) {
        return true;
    }
    let mut at = end;
    for _ in 0..2 {
        let Some((w, e)) = word_after_at(s, at) else {
            break;
        };
        if phone_word(&w) {
            return true;
        }
        at = e;
    }
    false
}

/// Words after `NxM` that make it a dose (`2x1 tablet`, `3x2 şeklinde`).
const DOSE_WORDS: &[&str] = &[
    "tablet",
    "kapsül",
    "ölçek",
    "ölçü",
    "damla",
    "kaşık",
    "puf",
    "doz",
    "şeklinde",
    "defa",
    "kez",
    "draje",
    "ampul",
    "saşe",
    "fitil",
    "efervesan",
    "pastil",
];

fn dose_context(s: &str, i: usize, end: usize) -> bool {
    word_after(s, end).is_some_and(|w| DOSE_WORDS.iter().any(|d| w.starts_with(d)))
        || word_before(s, i).is_some_and(|w| w == "günde")
}

lazy_static! {
    /// A Turkish IBAN: TR, two check digits and 22 digits, in groups of four or not.
    static ref IBAN: Regex =
        Regex::new(r"^TR ?([0-9]{2})((?: ?[0-9]{4}){5}) ?([0-9]{2})").expect("iban regex");
    /// A plate-shaped code: a province number, one to three capitals and two to four digits
    /// (`34 ABC 123`, `30 MM 6653`).
    static ref PLATE: Regex =
        Regex::new(r"^([0-9]{2}) ([A-ZÇĞİÖŞÜ]{1,3}) ([0-9]{2,4})").expect("plate regex");
    /// A Turkish number from 0 written in groups of any size: 0850 222 0 600.
    static ref SPACED_PHONE: Regex =
        Regex::new(r"^0[23458][0-9]{2}(?: [0-9]{1,4}){2,4}").expect("spaced phone regex");
    /// A nationwide service number written in its groups: 444 0 333.
    static ref CALL_444: Regex =
        Regex::new(r"^444([ -]?)([0-9])([ -]?)([0-9]{3})").expect("444 regex");
    /// Day and month with a dot, no year: 16.10.
    static ref DAY_MONTH: Regex = Regex::new(r"^([0-9]{1,2})\.([0-9]{1,2})").expect("day regex");
}

/// Months in three letters, Turkish and English (`12 OCA 2026`), which are no plate.
const MONTH_ABBREVIATIONS: &[&str] = &[
    "oca", "şub", "mar", "nis", "may", "haz", "tem", "ağu", "eyl", "eki", "kas", "ara", "jan",
    "feb", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
];

/// An IBAN at `i`: TR and the check digits as a number, each group of four digit by digit, the
/// last two digits as a number, a pause between groups.
fn iban_at(s: &str, i: usize) -> Option<(String, usize)> {
    let cap = IBAN.captures(&s[i..])?;
    let end = i + cap.get(0)?.end();
    if next_char(s, end).is_some_and(|c| c.is_alphanumeric()) {
        return None;
    }
    let body: String = cap[2].chars().filter(char::is_ascii_digit).collect();
    let mut groups = vec![format!("te re {}", cardinal(&cap[1]))];
    for k in 0..5 {
        groups.push(digit_words(&body[4 * k..4 * k + 4]));
    }
    groups.push(cardinal(&cap[3]));
    Some((groups.join(", "), end))
}

/// A plate-shaped code at `i`: the province as a number, the letters by name, the last group as
/// a number (four digits in two pairs). Not when the letters are a unit or a currency (`16 GB
/// 2400`) or a quantity follows.
fn plate_at(s: &str, i: usize) -> Option<(String, usize)> {
    let cap = PLATE.captures(&s[i..])?;
    let end = i + cap.get(0)?.end();
    let province: u32 = cap[1].parse().ok()?;
    let letters = &cap[2];
    let tail = &cap[3];
    let after = next_char(s, end);
    let lower = code_lower(letters);
    if !(1..=81).contains(&province)
        || UNITS.iter().any(|&(w, _)| w == letters || w == lower)
        || CURRENCIES.iter().any(|&(w, _, _)| w == letters)
        || MONTH_ABBREVIATIONS.contains(&lower.as_str())
        // a word in capitals (20 YIL 10 AY)
        || (letters.chars().count() >= 2 && super::tts::is_known_word(&to_turkish_lower(letters)))
        || after.is_some_and(|c| c.is_alphanumeric())
        || (matches!(after, Some('.' | ',')) && next_char(s, end + 1).is_some_and(|c| c.is_ascii_digit()))
        || quantity_follows(s, end)
    {
        return None;
    }
    let tail = if tail.len() == 4 {
        format!("{} {}", cardinal(&tail[..2]), cardinal(&tail[2..]))
    } else {
        cardinal(tail)
    };
    Some((
        format!("{} {} {tail}", cardinal(&cap[1]), spell_letters(letters)),
        end,
    ))
}

/// A run of capitals and digits at `i` that is a code: letters and digits interleaved
/// (`X7K9QZ`), or letters and five or more digits (`TR78240865`). Its reading and end.
fn alnum_code_at(s: &str, i: usize) -> Option<(String, usize)> {
    let mut end = i;
    let mut changes = 0;
    let mut prev_digit: Option<bool> = None;
    let mut digits = 0;
    while let Some(c) = next_char(s, end) {
        let digit = c.is_ascii_digit();
        if !digit && !latin_capital(c) {
            break;
        }
        if prev_digit.is_some_and(|p| p != digit) {
            changes += 1;
        }
        prev_digit = Some(digit);
        digits += usize::from(digit);
        end += c.len_utf8();
    }
    // a word goes on in small letters or with a hyphen: not a code
    if next_char(s, end).is_some_and(|c| c.is_alphanumeric())
        || (next_char(s, end) == Some('-')
            && next_char(s, end + 1).is_some_and(char::is_alphanumeric))
    {
        return None;
    }
    let starts_with_letter = next_char(s, i).is_some_and(|c| !c.is_ascii_digit());
    let code = changes >= 2 || (changes == 1 && starts_with_letter && digits >= 5);
    (code && s[i..end].chars().count() >= 3).then(|| (spell_code(&s[i..end]), end))
}

/// Turkish-style letters for a foreign word: w v, q k, x ks, and accented letters without
/// their accents.
fn turkish_letters(word: &str) -> Option<String> {
    let mut out = String::with_capacity(word.len());
    let mut changed = false;
    for c in word.chars() {
        let r: &str = match c {
            'w' => "v",
            'W' => "V",
            'q' => "k",
            'Q' => "K",
            'x' => "ks",
            'X' => "KS",
            'á' | 'à' | 'ä' | 'ã' | 'å' | 'ā' => "a",
            'é' | 'è' | 'ê' | 'ë' | 'ē' => "e",
            'í' | 'ì' | 'ï' => "i",
            'ó' | 'ò' | 'õ' | 'ø' | 'ō' => "o",
            'ú' | 'ù' => "u",
            'ñ' => "ny",
            'ß' => "ss",
            'Á' | 'À' | 'Ä' | 'Ã' | 'Å' => "A",
            'É' | 'È' | 'Ê' | 'Ë' => "E",
            'Í' | 'Ì' | 'Ï' => "İ",
            'Ó' | 'Ò' | 'Õ' | 'Ø' => "O",
            'Ú' | 'Ù' => "U",
            'Ñ' => "NY",
            _ => {
                out.push(c);
                continue;
            }
        };
        out.push_str(r);
        changed = true;
    }
    changed.then_some(out)
}

/// A combining diacritic (U+0300–U+036F).
fn is_mark(c: char) -> bool {
    ('\u{0300}'..='\u{036F}').contains(&c)
}

/// The end of the letters at `i`, with the hyphens and apostrophes between them. Combining marks
/// belong to the letter before them.
fn word_end(s: &str, i: usize) -> usize {
    let mut j = i;
    while let Some(c) = next_char(s, j) {
        let inner = matches!(c, '-' | '\'' | '’')
            && next_char(s, j + c.len_utf8()).is_some_and(char::is_alphabetic);
        if c.is_alphabetic() || is_mark(c) || inner {
            j += c.len_utf8();
        } else {
            break;
        }
    }
    j
}

/// Verbalises `text` with every rule.
pub fn verbalize(text: &str) -> Verbalized {
    verbalize_with(text, true)
}

/// Verbalises `text`. Without `words`, only stretches with a digit or a symbol are verbalised
/// (numbers, money, units, telephone numbers, e-mail, symbols, `A4`, `C++`); abbreviations,
/// initialisms, Roman numerals, foreign words and letters, spelled letters and URLs written in
/// letters alone are kept as written, so text of letters and punctuation comes back unchanged.
/// Profile `tts-v1` reads text so, which keeps its output on such text as frozen in stage 2.
pub fn verbalize_with(text: &str, words: bool) -> Verbalized {
    let mut sc = Scan {
        src: text,
        out: Verbalized::default(),
        kept: 0,
        words,
    };
    let s = text;
    let mut i = 0;
    while i < s.len() {
        let Some(c) = next_char(s, i) else { break };
        let prev = prev_char(s, i);
        // a combining mark continues its word (decomposed text)
        let at_start = !prev.is_some_and(|p| p.is_alphanumeric() || is_mark(p));
        if let Some(end) = if at_start { try_at(&mut sc, i) } else { None } {
            i = end;
            continue;
        }
        if at_start && c.is_alphabetic() {
            // a plain word: skip it
            i = word_end(s, i).max(i + c.len_utf8());
            continue;
        }
        if let Some(end) = symbol_at(&mut sc, i, c) {
            i = end;
            continue;
        }
        i += c.len_utf8();
    }
    sc.keep_to(s.len());
    sc.out
}

/// Tries every rule at a token start `i`.
fn try_at(sc: &mut Scan<'_>, i: usize) -> Option<usize> {
    let s = sc.src;
    let c = next_char(s, i)?;
    if c.is_ascii_digit()
        || (matches!(c, '+' | '-' | '−' | '(')
            && next_char(s, i + c.len_utf8()).is_some_and(|d| d.is_ascii_digit()))
    {
        return number_rules(sc, i);
    }
    if matches!(c, '₺' | '$' | '€' | '£' | '¥' | '₽' | '₹' | '₿') {
        let j = skip_space(s, i + c.len_utf8(), 1);
        if let Some((int, dec, end)) = amount_at(s, j) {
            let (_, u, sub) = CURRENCIES
                .iter()
                .find(|(w, _, _)| w.starts_with(c))
                .copied()?;
            let spoken = money(&int, dec.as_deref(), u, sub, sc.words);
            let (spoken, end) = with_suffix(s, spoken, end, true);
            sc.emit(i, end, &spoken, "money");
            return Some(end);
        }
        return None;
    }
    if matches!(c, '%' | '‰') {
        let j = skip_space(s, i + c.len_utf8(), 1);
        if let Some((int, mut dec, mut end)) = amount_at(s, j) {
            // with the word rules a dot decimal is a decimal (%3.8 yüzde üç virgül sekiz)
            if sc.words && dec.is_none() && !s[j..end].contains('.') {
                if let Some((d, e)) = dot_decimal_at(s, end) {
                    dec = Some(d);
                    end = e;
                }
            }
            let head = if c == '%' { "yüzde" } else { "binde" };
            let spoken = format!("{head} {}", amount_words(&int, dec.as_deref(), sc.words));
            let (spoken, end) = with_suffix(s, spoken, end, false);
            sc.emit(i, end, &spoken, "percent");
            return Some(end);
        }
        return None;
    }
    if c.is_ascii_alphanumeric() {
        if let Some(m) = EMAIL.find(&s[i..]) {
            let end = trim_trailing(s, i, i + m.end());
            let spoken = electronic(&s[i..end]);
            let (spoken, end) = with_suffix(s, spoken, end, true);
            sc.emit(i, end, &spoken, "electronic");
            return Some(end);
        }
        if let Some(m) = URL.find(&s[i..]) {
            let end = trim_trailing(s, i, i + m.end());
            // an address of letters and dots alone is plain text to tts-v1
            let has_symbol =
                s[i..end].contains(|c: char| c.is_ascii_digit() || "/:?=&#%~+_".contains(c));
            if s[i..end].contains('.') && (sc.words || has_symbol) {
                let spoken = electronic(&s[i..end]);
                let (spoken, end) = with_suffix(s, spoken, end, true);
                sc.emit(i, end, &spoken, "electronic");
                return Some(end);
            }
        }
    }
    if c.is_alphabetic() || c == '.' {
        return word_rules(sc, i);
    }
    None
}

/// Drops sentence punctuation a URL or address match took at its end.
fn trim_trailing(s: &str, start: usize, mut end: usize) -> usize {
    while end > start {
        let Some(c) = s[..end].chars().next_back() else {
            break;
        };
        if matches!(c, '.' | ',' | ';' | ':' | '!' | '?' | ')' | '-' | '/') {
            end -= c.len_utf8();
        } else {
            break;
        }
    }
    end
}

/// An amount and its currency. With the word rules a leading zero is not read (`039,50 TL`
/// otuz dokuz lira elli kuruş).
fn money(int: &str, dec: Option<&str>, unit: &str, sub: &str, words: bool) -> String {
    let int = if words && int.len() > 1 {
        match int.trim_start_matches('0') {
            "" => "0",
            t => t,
        }
    } else {
        int
    };
    let cents = dec.map(|d| {
        let mut d = d.to_string();
        while d.len() < 2 {
            d.push('0');
        }
        d
    });
    match cents {
        Some(d) if !sub.is_empty() && d.len() == 2 => {
            let c: usize = d.parse().unwrap_or(0);
            if c == 0 {
                format!("{} {unit}", cardinal(int))
            } else if int.trim_start_matches('0').is_empty() {
                format!("{} {sub}", cardinal(&c.to_string()))
            } else {
                format!(
                    "{} {unit} {} {sub}",
                    cardinal(int),
                    cardinal(&c.to_string())
                )
            }
        }
        Some(d) => format!("{} virgül {} {unit}", cardinal(int), decimals(&d)),
        None => format!("{} {unit}", cardinal(int)),
    }
}

/// Reads an address or a URL: dots, at signs and slashes as words, the parts as words when they
/// read as Turkish syllables and letter by letter otherwise.
fn electronic(t: &str) -> String {
    let t = t
        .strip_prefix("https://")
        .or_else(|| t.strip_prefix("http://"))
        .unwrap_or(t);
    let mut words: Vec<String> = Vec::new();
    let mut part = String::new();
    let flush = |part: &mut String, words: &mut Vec<String>| {
        if part.is_empty() {
            return;
        }
        let lower = to_turkish_lower(part);
        let w = if part.chars().all(|c| c.is_ascii_digit()) {
            cardinal(part)
        } else if lower == "www" {
            "ve ve ve".to_string()
        } else if lower == "com" {
            "kom".to_string()
        } else if lower == "tr" {
            "te re".to_string()
        } else if let Some(&(_, r)) = FOREIGN.iter().find(|(w, _)| *w == lower) {
            r.to_string()
        } else {
            let l = turkish_letters(&lower).unwrap_or(lower);
            guess_abbreviation(&l)
        };
        words.push(w);
        part.clear();
    };
    for c in t.chars() {
        let sym = match c {
            '.' => Some("nokta"),
            '@' => Some("et"),
            '/' => Some("eğik çizgi"),
            '-' => Some("tire"),
            '_' => Some("alt çizgi"),
            ':' => Some("iki nokta"),
            '?' => Some("soru işareti"),
            '=' => Some("eşittir"),
            '&' => Some("ve"),
            '#' => Some("diyez"),
            '%' => Some("yüzde"),
            '+' => Some("artı"),
            '~' => Some("yaklaşık"),
            _ => None,
        };
        if let Some(w) = sym {
            flush(&mut part, &mut words);
            words.push(w.to_string());
        } else if c.is_ascii_digit() != part.chars().last().is_some_and(|p| p.is_ascii_digit())
            && !part.is_empty()
        {
            flush(&mut part, &mut words);
            part.push(c);
        } else {
            part.push(c);
        }
    }
    flush(&mut part, &mut words);
    words.join(" ")
}

/// A telephone number at `i` (a `+`, a `(` or a digit): its reading and end. Turkish numbers
/// only: 11 digits from 0, 10 digits from 2–5 written in groups, or `+90` and 10 digits. With the
/// word rules also a mobile number of ten digits written without spaces (`5321234567`) and a
/// nationwide service number in its groups (`444 0 333`), and the groups are read with a pause
/// (a comma) between them.
fn phone_at(s: &str, i: usize, words: bool) -> Option<(String, usize)> {
    if words {
        // a number from 0 in other groups, eleven digits in all: 0850 222 0 600
        if let Some(m) = SPACED_PHONE.find(&s[i..]) {
            let end = i + m.end();
            let written = m.as_str();
            let digits = written.chars().filter(char::is_ascii_digit).count();
            if digits == 11 && !next_char(s, end).is_some_and(|c| c.is_ascii_digit()) {
                return Some((telephone(written, true), end));
            }
        }
        if let Some(cap) = CALL_444.captures(&s[i..]) {
            let end = i + cap.get(0)?.end();
            let separated = !cap[1].is_empty() || !cap[3].is_empty();
            if separated && !next_char(s, end).is_some_and(|c| c.is_ascii_digit()) {
                let spoken = format!(
                    "dört yüz kırk dört, {}, {}",
                    digit_words(&cap[2]),
                    cardinal(&cap[4])
                );
                return Some((spoken, end));
            }
        }
    }
    let rest = &s[i..];
    let m = PHONE.find(rest)?;
    let end = i + m.end();
    // not when more digits follow (a longer number or a decimal)
    if next_char(s, end).is_some_and(|d| {
        d.is_ascii_digit()
            || (d == ',' && next_char(s, end + 1).is_some_and(|e| e.is_ascii_digit()))
    }) {
        return None;
    }
    let written = &rest[..m.end()];
    let digits: String = written.chars().filter(char::is_ascii_digit).collect();
    let separated = written.contains([' ', '-', '(']);
    let ok = if written.starts_with('+') {
        digits.len() == 12 && digits.starts_with("90")
    } else {
        (digits.len() == 11 && digits.starts_with('0'))
            || (digits.len() == 10
                && separated
                && matches!(digits.chars().next(), Some('5' | '2' | '3' | '4')))
            // a mobile number written whole, not a quantity (5000000000 kişi)
            || (words
                && digits.len() == 10
                && digits.starts_with('5')
                && digits.trim_end_matches('0').len() >= 5
                && !quantity_follows(s, end)
                && !word_after(s, end).as_deref().is_some_and(counts))
            // a landline or service number written whole, next to a telephone word
            || (words
                && digits.len() == 10
                && matches!(digits.chars().next(), Some('2' | '3' | '4' | '8'))
                && phone_context(s, i, end))
    };
    ok.then(|| (telephone(written, words), end))
}

/// The rules for a token starting with a digit or a sign.
fn number_rules(sc: &mut Scan<'_>, i: usize) -> Option<usize> {
    let s = sc.src;
    let c = next_char(s, i)?;
    // a sign before a number
    if matches!(c, '+' | '-' | '−') {
        // +90 telephone numbers
        if c == '+' {
            if let Some((spoken, end)) = phone_at(s, i, sc.words) {
                let (spoken, end) = with_suffix(s, spoken, end, false);
                sc.emit(i, end, &spoken, "telephone");
                return Some(end);
            }
        }
        // a sign at the start, after a space, a bracket or an operator (x=-5); a hyphen after a
        // letter or a digit is not one
        let p = prev_char(s, i);
        if p.is_some_and(|p| !p.is_whitespace() && !"(=[<>:;/".contains(p)) {
            return None;
        }
        let j = i + c.len_utf8();
        let (int, dec, end) = amount_at(s, j)?;
        let sign = if c == '+' { "artı" } else { "eksi" };
        let (body, kind, end) = quantity(s, &int, dec.as_deref(), end, sc.words);
        let (spoken, end) = with_suffix(
            s,
            format!("{sign} {body}"),
            end,
            kind == "money" || kind == "unit",
        );
        sc.emit(i, end, &spoken, kind);
        return Some(end);
    }
    let rest = &s[i..];
    // telephone numbers: (0532) 123 45 67, 0532 123 45 67, 05321234567
    if let Some((spoken, end)) = phone_at(s, i, sc.words) {
        let (spoken, end) = with_suffix(s, spoken, end, false);
        sc.emit(i, end, &spoken, "telephone");
        return Some(end);
    }
    if c == '(' {
        return None;
    }
    if sc.words {
        // a plate-shaped code: 34 ABC 123, 30 MM 6653
        if let Some((spoken, end)) = plate_at(s, i) {
            let (spoken, end) = with_suffix(s, spoken, end, true);
            sc.emit(i, end, &spoken, "identifier");
            return Some(end);
        }
        // day and month before a date word: 16.10 tarihinde (not 16.10'da, a time)
        if let Some(cap) = DAY_MONTH.captures(rest) {
            let end = i + cap.get(0)?.end();
            let more = next_char(s, end).is_some_and(|c| c.is_ascii_digit())
                || (next_char(s, end) == Some('.')
                    && next_char(s, end + 1).is_some_and(|c| c.is_ascii_digit()));
            let dated = matches!(next_char(s, end), Some(' ' | '\u{a0}'))
                && word_after(s, end).is_some_and(|w| w.starts_with("tarih"));
            let d: u32 = cap[1].parse().unwrap_or(0);
            let m: usize = cap[2].parse().unwrap_or(0);
            if !more && dated && (1..=31).contains(&d) && (1..=12).contains(&m) {
                let spoken = format!("{} {}", cardinal(&d.to_string()), MONTHS[m - 1]);
                sc.emit(i, end, &spoken, "date");
                return Some(end);
            }
        }
    }
    // dates: 12.03.2024, 12/03/2024, 2024-03-12; a two-digit year only after a two-digit day
    // and month (12.03.24), not in a version (1.2.24)
    if let Some(cap) = ISO_DATE.captures(rest) {
        let (y, m, d) = (&cap[1], &cap[2], &cap[3]);
        let end = i + cap.get(0)?.end();
        if let Some(spoken) = date(d, m, y) {
            if !next_char(s, end).is_some_and(|c| c.is_ascii_digit()) {
                let (spoken, end) = with_suffix(s, spoken, end, false);
                sc.emit(i, end, &spoken, "date");
                return Some(end);
            }
        }
    }
    if let Some(cap) = DATE.captures(rest) {
        let end = i + cap.get(0)?.end();
        let full_year = cap[5].len() == 4 || (cap[1].len() == 2 && cap[3].len() == 2);
        if cap[2] == cap[4] && full_year && !next_char(s, end).is_some_and(|c| c.is_ascii_digit()) {
            if let Some(spoken) = date(&cap[1], &cap[3], &cap[5]) {
                let (spoken, end) = with_suffix(s, spoken, end, false);
                sc.emit(i, end, &spoken, "date");
                return Some(end);
            }
        }
    }
    // times: 14:30, 09.45, 14:30:15, 24:00
    if let Some(cap) = TIME.captures(rest) {
        let end = i + cap.get(0)?.end();
        let h: u32 = cap[1].parse().ok()?;
        let m: u32 = cap[3].parse().ok()?;
        let sec = cap.get(4).and_then(|x| x.as_str().parse::<u32>().ok());
        let colon = &cap[2] == ":";
        let next = next_char(s, end);
        let after_ok = !next.is_some_and(|c| c.is_ascii_digit())
            && !(next == Some('.') && next_char(s, end + 1).is_some_and(|c| c.is_ascii_digit()))
            && !(next == Some(',') && next_char(s, end + 1).is_some_and(|c| c.is_ascii_digit()));
        // a time with a dot needs a two-digit hour (09.45), `saat` before it or a suffix after it
        // (9.00'da); 3.14 and 3.12 are decimals, 3.50 TL an amount
        let after_saat = s[..i]
            .trim_end()
            .rsplit(|c: char| !c.is_alphabetic())
            .next()
            .is_some_and(|w| to_turkish_lower(w) == "saat");
        let suffixed = matches!(next, Some('\'' | '’'));
        let dot_ok = colon
            || (sec.is_none()
                && currency_at(s, end).is_none()
                && unit_at(s, end).is_none()
                && (cap[1].len() == 2 || after_saat || suffixed));
        let valid = (h <= 23 && m <= 59) || (h == 24 && m == 0);
        if valid && sec.is_none_or(|x| x <= 59) && after_ok && dot_ok {
            let mut spoken = time(h, m, sec);
            let mut end = end;
            // a range of times: 14:00-16:00
            if let Some(d) = next_char(s, end).filter(|&d| d == '-' || d == '–') {
                if let Some(c2) = TIME.captures(&s[end + d.len_utf8()..]) {
                    let h2: u32 = c2[1].parse().unwrap_or(99);
                    let m2: u32 = c2[3].parse().unwrap_or(99);
                    let e2 = end + d.len_utf8() + c2.get(0).map_or(0, |x| x.end());
                    if ((h2 <= 23 && m2 <= 59) || (h2 == 24 && m2 == 0))
                        && !next_char(s, e2).is_some_and(|c| c.is_ascii_digit())
                    {
                        spoken = format!("{spoken} ila {}", time(h2, m2, None));
                        end = e2;
                    }
                }
            }
            let (spoken, end) = with_suffix(s, spoken, end, false);
            sc.emit(i, end, &spoken, "time");
            return Some(end);
        }
    }
    let (int, dec, end) = amount_at(s, i)?;
    let next = next_char(s, end);
    // an amount with a decimal point and its currency: 3.50 TL, 3.5 TL
    if next == Some('.') && dec.is_none() {
        let d0 = end + 1;
        let mut d1 = d0;
        while next_char(s, d1).is_some_and(|c| c.is_ascii_digit()) {
            d1 += 1;
        }
        if (1..=2).contains(&(d1 - d0)) && next_char(s, d1) != Some('.') {
            if let Some((u, sub, e)) = currency_at(s, d1) {
                let spoken = money(&int, Some(&s[d0..d1]), u, sub, sc.words);
                let (spoken, e) = with_suffix(s, spoken, e, true);
                sc.emit(i, e, &spoken, "money");
                return Some(e);
            }
        }
    }
    // with the word rules a dot and one or two digits is a decimal (17.5 on yedi buçuk, 3.14
    // üç virgül on dört, 2.5 kg), not after a version's or a section's name (Python 3.12)
    if sc.words && dec.is_none() && !s[i..end].contains('.') {
        if let Some((d, d1)) = dot_decimal_at(s, end) {
            if !version_context(s, i, d1) {
                let (body, kind, e) = quantity(s, &int, Some(&d), d1, true);
                let kind = if kind == "cardinal" { "decimal" } else { kind };
                let (spoken, e) = with_suffix(s, body, e, kind == "money" || kind == "unit");
                sc.emit(i, e, &spoken, kind);
                return Some(e);
            }
        }
    }
    if next == Some('.')
        && dec.is_none()
        && next_char(s, end + 1).is_some_and(|c| c.is_ascii_digit())
    {
        // a dotted decimal or a version (3.14, 2.0.1), with its unit (2.5 kg)
        let mut e = end;
        let mut words = vec![cardinal(&int)];
        while next_char(s, e) == Some('.')
            && next_char(s, e + 1).is_some_and(|c| c.is_ascii_digit())
        {
            let start = e + 1;
            let mut k = start;
            while next_char(s, k).is_some_and(|c| c.is_ascii_digit()) {
                k += 1;
            }
            words.push("nokta".to_string());
            words.push(decimals(&s[start..k]));
            e = k;
        }
        let mut kind = "decimal";
        if let Some((u, ue)) = unit_at(s, e) {
            words.push(u.to_string());
            e = ue;
            kind = "unit";
        }
        let (spoken, e) = with_suffix(s, words.join(" "), e, kind == "unit");
        sc.emit(i, e, &spoken, kind);
        return Some(e);
    }
    // with the word rules an identifier, five digits or more next to a word that names one,
    // digit by digit (Doğrulama kodunuz 482915, 10832 numaralı); a quantity stays a number
    if sc.words
        && dec.is_none()
        && int.len() >= 5
        && !s[i..end].contains('.')
        && !next.is_some_and(|c| c.is_alphabetic())
        && !matches!(next, Some('/' | '-' | '–' | 'x' | '×'))
        && !quantity_follows(s, end)
        && id_context(s, i, end)
    {
        let (spoken, end) = with_suffix(s, digit_words(&int), end, true);
        sc.emit(i, end, &spoken, "identifier");
        return Some(end);
    }
    // an ordinal: 2. sınıf, 5'inci
    if dec.is_none() && !int.contains('.') && next == Some('.') {
        let after = skip_space(s, end + 1, 3);
        // before a small letter (2. sınıf); before a capital only at the start of the text or
        // of a line or list item (1. Lig'de, 2. Abdülhamid), not after a full stop (Kaç? 3. Ben)
        let item_start = s[..i]
            .trim_end_matches([' ', '\t'])
            .chars()
            .next_back()
            .is_none_or(|c| matches!(c, ':' | '"' | '“' | '(' | '\n' | '\r'));
        // a street number before its address word: 5. Sok.
        let street = ABBREVIATIONS
            .iter()
            .any(|a| a.place == Place::Address && s[after..].starts_with(a.written));
        if after > end + 1
            && next_char(s, after)
                .is_some_and(|c| c.is_lowercase() || ((item_start || street) && c.is_uppercase()))
        {
            let spoken = ordinal(&cardinal(&int));
            sc.emit(i, end + 1, &spoken, "ordinal");
            return Some(end + 1);
        }
    }
    if let Some((suf, send)) = suffix_at(s, end) {
        if dec.is_none() {
            if let Some(rest) = ordinal_suffix(&suf) {
                let spoken = ordinal(&cardinal(&int));
                let spoken = if rest.is_empty() {
                    spoken
                } else {
                    attach(&spoken, rest)
                };
                sc.emit(i, send, &spoken, "ordinal");
                return Some(send);
            }
        }
    }
    if dec.is_none() {
        // a range: 10-15, 1990-95; a score when the second is smaller: 3-1
        if let Some(d) = next.filter(|&d| d == '-' || d == '–') {
            if let Some((int2, dec2, end2)) = amount_at(s, end + d.len_utf8()) {
                if !next_char(s, end2).is_some_and(|c| c.is_ascii_digit() || c.is_alphabetic()) {
                    let a: u128 = int.parse().unwrap_or(0);
                    let b: u128 = int2.parse().unwrap_or(0);
                    // a year range with the century left out: 1990-95
                    let short_year = int.len() == 4 && int2.len() == 2 && b > a % 100;
                    let first = cardinal(&int);
                    let (second, kind, end2) = quantity(s, &int2, dec2.as_deref(), end2, sc.words);
                    let join = if b > a || short_year { " ila " } else { " " };
                    let kind = if kind == "cardinal" { "range" } else { kind };
                    let (spoken, end2) = with_suffix(
                        s,
                        format!("{first}{join}{second}"),
                        end2,
                        kind == "money" || kind == "unit",
                    );
                    sc.emit(i, end2, &spoken, kind);
                    return Some(end2);
                }
            }
        }
        // a fraction (2/3 üçte iki, up to hundredths); a file or case number or 24/7 otherwise
        // (2012/3121 iki bin on iki bölü üç bin yüz yirmi bir)
        if next == Some('/') {
            if let Some((int2, None, end2)) = amount_at(s, end + 1) {
                if !next_char(s, end2).is_some_and(|c| c.is_ascii_digit() || c == '/') {
                    let a: u128 = int.parse().unwrap_or(0);
                    let b: u128 = int2.parse().unwrap_or(0);
                    let fraction = a < b && b <= 100;
                    let spoken = if a == 24 && b == 7 {
                        "yirmi dört yedi".to_string()
                    } else if sc.words && a == 7 && b == 24 {
                        // service hours, not a fraction
                        "yedi yirmi dört".to_string()
                    } else if fraction {
                        let den = cardinal(&int2);
                        format!("{} {}", attach(&den, "de"), cardinal(&int))
                    } else {
                        format!("{} bölü {}", cardinal(&int), cardinal(&int2))
                    };
                    let (spoken, end2) = match suffix_at(s, end2) {
                        // üçte ikisi: a vowel suffix is the possessive
                        Some((suf, e))
                            if fraction
                                && suf.chars().count() == 1
                                && spoken.ends_with(is_vowel) =>
                        {
                            (format!("{spoken}s{}", harmonize(&spoken, &suf, false)), e)
                        }
                        _ => with_suffix(s, spoken, end2, true),
                    };
                    let kind = if fraction { "fraction" } else { "cardinal" };
                    sc.emit(i, end2, &spoken, kind);
                    return Some(end2);
                }
            }
        }
        // dimensions: 3x4 üçe dört; with the word rules a dose: Günde 2x1 tablet, iki çarpı bir
        if matches!(next, Some('x' | '×')) {
            if let Some((int2, dec2, end2)) = amount_at(s, end + next.map_or(1, char::len_utf8)) {
                if !next_char(s, end2).is_some_and(|c| c.is_alphanumeric()) {
                    // a dose: times a day, then the amount and its unit (Günde 2x500 mg)
                    if sc.words && int.len() <= 2 && int2.len() <= 4 {
                        let (second, kind, e2) = quantity(s, &int2, dec2.as_deref(), end2, true);
                        if dose_context(s, i, e2) {
                            let spoken = format!("{} çarpı {second}", cardinal(&int));
                            let (spoken, e2) =
                                with_suffix(s, spoken, e2, kind == "money" || kind == "unit");
                            sc.emit(i, e2, &spoken, "cardinal");
                            return Some(e2);
                        }
                    }
                    let (second, kind, end2) = quantity(s, &int2, dec2.as_deref(), end2, sc.words);
                    let spoken = format!("{} {second}", attach_with(&cardinal(&int), "e", true));
                    let kind = if kind == "cardinal" { "range" } else { kind };
                    let (spoken, end2) =
                        with_suffix(s, spoken, end2, kind == "money" || kind == "unit");
                    sc.emit(i, end2, &spoken, kind);
                    return Some(end2);
                }
            }
        }
        // a code (ten digits or more, or seven or more from a zero): read in groups of three;
        // a shorter number from a zero is read whole (0850 sıfır sekiz yüz elli)
        if (int.len() >= 10 || (int.len() >= 7 && int.starts_with('0'))) && !s[i..end].contains('.')
        {
            let digits = &s[i..end];
            let mut groups = Vec::new();
            let mut k = 0;
            while k < digits.len() {
                let e = (k + 3).min(digits.len());
                groups.push(cardinal(&digits[k..e]));
                k = e;
            }
            let (spoken, end) = with_suffix(s, groups.join(" "), end, false);
            sc.emit(i, end, &spoken, "cardinal");
            return Some(end);
        }
    }
    // a number and what follows it: a percent sign, a currency, a unit, letters
    let (spoken, mut kind, end) = quantity(s, &int, dec.as_deref(), end, sc.words);
    // letters glued to the number: 3D, 4K (capitals are letters), 1990lar (a suffix), 2x
    let (spoken, end) = if let Some(l) = next_char(s, end).filter(|c| c.is_alphabetic()) {
        let we = word_end(s, end);
        let letters = &s[end..we];
        if l.is_uppercase() && sc.words {
            let (r, spelled) = initialism_words(letters);
            if spelled {
                kind = "initialism";
            }
            (format!("{spoken} {r}"), we)
        } else if l.is_uppercase() {
            (format!("{spoken} {}", initialism(letters)), we)
        } else if letters.chars().any(is_vowel) {
            (attach(&spoken, letters), we)
        } else {
            (format!("{spoken} {}", letter_pronunciation(letters)), we)
        }
    } else {
        with_suffix(s, spoken, end, kind == "money" || kind == "unit")
    };
    sc.emit(i, end, &spoken, kind);
    Some(end)
}

/// `-inci`, `-ncı`, `-üncüsü`: the rest of the suffix after the ordinal's.
fn ordinal_suffix(suf: &str) -> Option<&str> {
    let rest = suf.trim_start_matches(['ı', 'i', 'u', 'ü']);
    let rest = rest.strip_prefix('n')?;
    let rest = rest.strip_prefix('c')?;
    let mut it = rest.char_indices();
    let (_, v) = it.next()?;
    matches!(v, 'ı' | 'i' | 'u' | 'ü').then(|| &rest[v.len_utf8()..])
}

/// A number and the percent sign, currency or unit after it.
fn quantity(
    s: &str,
    int: &str,
    dec: Option<&str>,
    end: usize,
    rules: bool,
) -> (String, &'static str, usize) {
    let words = amount_words(int, dec, rules);
    let j = skip_space(s, end, 1);
    if next_char(s, j) == Some('%') {
        return (format!("yüzde {words}"), "percent", j + 1);
    }
    if let Some((u, sub, e)) = currency_at(s, end) {
        return (money(int, dec, u, sub, rules), "money", e);
    }
    // a scale word and a currency: 1,5 milyon TL
    for scale in ["bin", "milyon", "milyar", "trilyon"] {
        let k = skip_space(s, end, 1);
        if s[k..].starts_with(scale) && k > end {
            let k2 = k + scale.len();
            if let Some((u, _, e)) = currency_at(s, k2) {
                if skip_space(s, k2, 1) > k2 {
                    return (format!("{words} {scale} {u}"), "money", e);
                }
            }
        }
    }
    if let Some((u, e)) = unit_at(s, end) {
        // a rate: 90 km/sa -> saatte doksan kilometre
        if next_char(s, e) == Some('/') {
            let k = e + 1;
            let mut best: Option<(&str, &str)> = None;
            for &(w, r) in PER {
                if s[k..].starts_with(w)
                    && !next_char(s, k + w.len()).is_some_and(|c| c.is_alphanumeric())
                    && best.is_none_or(|(bw, _)| w.len() > bw.len())
                {
                    best = Some((w, r));
                }
            }
            if let Some((w, r)) = best {
                return (format!("{r} {words} {u}"), "unit", k + w.len());
            }
            if let Some((u2, e2)) = unit_at(s, k) {
                return (format!("{words} {u} bölü {u2}"), "unit", e2);
            }
        }
        if u == "devir" {
            return (format!("dakikada {words} devir"), "unit", e);
        }
        if matches!(u, "kilobit" | "megabit" | "gigabit") {
            return (format!("saniyede {words} {u}"), "unit", e);
        }
        return (format!("{words} {u}"), "unit", e);
    }
    let kind = if dec.is_some() { "decimal" } else { "cardinal" };
    (words, kind, end)
}

fn date(d: &str, m: &str, y: &str) -> Option<String> {
    let dd: u32 = d.parse().ok()?;
    let mm: usize = m.parse().ok()?;
    if !(1..=31).contains(&dd) || !(1..=12).contains(&mm) {
        return None;
    }
    // a two-digit year without its zero (05 beş), 00 as sıfır sıfır
    let year = if y.len() == 2 && y.trim_start_matches('0').is_empty() {
        cardinal(y)
    } else if y.len() == 2 {
        cardinal(y.trim_start_matches('0'))
    } else {
        cardinal(y)
    };
    Some(format!(
        "{} {} {year}",
        cardinal(&dd.to_string()),
        MONTHS[mm - 1]
    ))
}

fn time(h: u32, m: u32, sec: Option<u32>) -> String {
    let mut out = cardinal(&h.to_string());
    if h == 0 {
        out = "sıfır".to_string();
    }
    if m > 0 || sec.is_some() {
        out.push(' ');
        out.push_str(&if m < 10 && m > 0 {
            format!("sıfır {}", cardinal(&m.to_string()))
        } else if m == 0 {
            "sıfır sıfır".to_string()
        } else {
            cardinal(&m.to_string())
        });
    }
    if let Some(sec) = sec {
        out.push(' ');
        out.push_str(&if sec == 0 {
            "sıfır sıfır".to_string()
        } else if sec < 10 {
            format!("sıfır {}", cardinal(&sec.to_string()))
        } else {
            cardinal(&sec.to_string())
        });
    }
    out
}

/// A telephone number read in its groups: as written, a group of ten or eleven digits split as
/// 3-3-2-2 or 4-3-2-2 (`+90` and ten digits as 2-3-3-2-2); a leading zero as `sıfır`, `+` as
/// `artı`. With `pauses` (the word rules) the groups are separated by commas (`artı doksan, iki
/// yüz on iki, ...`).
fn telephone(t: &str, pauses: bool) -> String {
    let plus = t.starts_with('+');
    let groups: Vec<&str> = t
        .split(|c: char| !c.is_ascii_digit())
        .filter(|g| !g.is_empty())
        .collect();
    let mut words: Vec<String> = Vec::new();
    if plus && !pauses {
        words.push("artı".to_string());
    }
    for (k, g) in groups.iter().enumerate() {
        let sizes: Vec<usize> = match g.len() {
            12 if plus && k == 0 && g.starts_with("90") => vec![2, 3, 3, 2, 2],
            11 if g.starts_with('0') => vec![4, 3, 2, 2],
            10 => vec![3, 3, 2, 2],
            n if n > 4 => vec![3; n.div_ceil(3)],
            n => vec![n],
        };
        let mut at = 0;
        for n in sizes {
            let e = (at + n).min(g.len());
            if e > at {
                words.push(cardinal(&g[at..e]));
            }
            at = e;
        }
    }
    if !pauses {
        return words.join(" ");
    }
    // artı joins the first group (artı doksan, ...)
    if plus {
        if let Some(first) = words.first_mut() {
            first.insert_str(0, "artı ");
        }
    }
    words.join(", ")
}

/// The rules for a token starting with a letter (or a dot: `.NET`).
fn word_rules(sc: &mut Scan<'_>, i: usize) -> Option<usize> {
    let s = sc.src;
    let words = sc.words;
    // abbreviations with a full stop, longest first (word rules: not in tts-v1)
    if words {
        let mut best: Option<&Abbrev> = None;
        for a in ABBREVIATIONS {
            if s[i..].starts_with(a.written)
                && !next_char(s, i + a.written.len()).is_some_and(char::is_alphanumeric)
                && best.is_none_or(|b| a.written.len() > b.written.len())
            {
                best = Some(a);
            }
        }
        let fits = |a: &Abbrev| -> bool {
            let end = i + a.written.len();
            match a.place {
                Place::Any | Place::End => true,
                Place::Title => {
                    let k = skip_space(s, end, 2);
                    k > end && next_char(s, k).is_some_and(char::is_uppercase)
                }
                Place::Address | Place::Company => after_name_or_number(s, i),
            }
        };
        if let Some(a) = best.filter(|a| fits(a)) {
            let end = i + a.written.len();
            let mut spoken = a.spoken.to_string();
            // the full stop ends the text's last sentence, and the sentence of a post-positioned
            // abbreviation or a company suffix before a capital (vb. Sonra, A.Ş. Bugün); not of
            // a title (Dr. Ali), an address word (Cad. No: 5) or T.C. and bkz. (T.C. Kimlik)
            if a.written.ends_with('.') {
                let k = skip_space(s, end, 3);
                let at_end = s[end..].trim().is_empty();
                let before_capital = k > end && next_char(s, k).is_some_and(char::is_uppercase);
                if at_end || (before_capital && matches!(a.place, Place::End | Place::Company)) {
                    spoken.push('.');
                }
            }
            let (spoken, end) = with_suffix(s, spoken, end, true);
            sc.emit(i, end, &spoken, "abbreviation");
            return Some(end);
        }
    }
    // abbreviations before a number: No: 5, s. 45 (both profiles: the text has digits)
    for &(w, r) in NUMBER_ABBREVIATIONS {
        if s[i..].starts_with(w) {
            let k = skip_space(s, i + w.len(), 2);
            if next_char(s, k).is_some_and(|c| c.is_ascii_digit())
                && (k > i + w.len() || w.ends_with(':'))
            {
                sc.emit(i, i + w.len(), r, "abbreviation");
                return Some(i + w.len());
            }
        }
    }
    // with the word rules: an IBAN in its groups, a code of capitals and digits in order
    if words {
        if let Some((spoken, end)) = iban_at(s, i).or_else(|| alnum_code_at(s, i)) {
            let (spoken, end) = with_suffix(s, spoken, end, true);
            sc.emit(i, end, &spoken, "identifier");
            return Some(end);
        }
    }
    let we = word_end(s, i);
    let c = next_char(s, i)?;
    // tech names with symbols: C++, C# (both profiles), .NET (letters and a dot: word rules)
    for (w, r) in FOREIGN.iter().filter(|(w, _)| w.contains(['+', '#', '.'])) {
        if !words && !w.contains(['+', '#']) {
            continue;
        }
        let n = w.len();
        if s.len() >= i + n
            && s.is_char_boundary(i + n)
            && to_turkish_lower(&s[i..i + n]) == *w
            && !next_char(s, i + n).is_some_and(char::is_alphanumeric)
        {
            let (spoken, end) = with_suffix(s, r.to_string(), i + n, true);
            sc.emit(i, end, &spoken, "foreign");
            return Some(end);
        }
    }
    if c == '.' {
        return None;
    }
    let word = &s[i..we];
    // the word and a suffix after an apostrophe
    let (base, suffix) = match word.find(['\'', '’']) {
        Some(k) => (
            &word[..k],
            Some(&word[k + word[k..].chars().next().map_or(1, char::len_utf8)..]),
        ),
        None => (word, None),
    };
    // letters followed by digits: A4, MP3, COVID-19 (both profiles: the text has digits)
    let digit_end = {
        let mut k = we;
        if next_char(s, k) == Some('-') && next_char(s, k + 1).is_some_and(|c| c.is_ascii_digit()) {
            k += 1;
        }
        let start = k;
        while next_char(s, k).is_some_and(|c| c.is_ascii_digit()) {
            k += 1;
        }
        (k > start && !next_char(s, k).is_some_and(char::is_alphabetic)).then_some((start, k))
    };
    if let Some((ds, de)) = digit_end {
        if suffix.is_none() && base.chars().all(latin_capital) && base.chars().count() <= 5 {
            let letters = if words {
                // the letters by name, one word each (MP3 me pe üç), or the initialism's word
                initialism_words(base).0
            } else if base.chars().count() == 1 {
                letter_pronunciation(&to_turkish_lower(base))
            } else {
                initialism(base)
            };
            let spoken = format!("{letters} {}", cardinal(&s[ds..de]));
            let (spoken, end) = with_suffix(s, spoken, de, false);
            sc.emit(i, end, &spoken, "initialism");
            return Some(end);
        }
    }
    // the rest reads words written in letters alone: not in tts-v1, whose output on such text
    // is frozen
    if !words {
        return None;
    }
    // letters spelled with hyphens: Ç-A-Y
    if base.contains('-') {
        let parts: Vec<&str> = base.split('-').collect();
        if parts.len() >= 2 && parts.iter().all(|p| p.chars().count() == 1) {
            let spoken: Vec<String> = parts
                .iter()
                .map(|p| letter_pronunciation(&to_turkish_lower(p)))
                .collect();
            let spoken = spoken.join(" ");
            let spoken = match suffix {
                Some(suf) => attach_with(&spoken, suf, true),
                None => spoken,
            };
            sc.emit(i, we, &spoken, "letters");
            return Some(we);
        }
    }
    let lower = to_turkish_lower(base);
    // foreign words and brands (a whole word with its apostrophe, or the word and a suffix,
    // re-buffered to the reading: Wi-Fi'ye vayfaya)
    let key = to_turkish_lower(word);
    if let Some(&(_, r)) = FOREIGN.iter().find(|(w, _)| *w == key) {
        sc.emit(i, we, r, "foreign");
        return Some(we);
    }
    let ascii_lower = base.to_lowercase();
    if let Some(&(_, r)) = FOREIGN
        .iter()
        .find(|(w, _)| *w == lower || *w == ascii_lower)
    {
        let spoken = match suffix {
            Some(suf) => attach_with(r, suf, true),
            None => r.to_string(),
        };
        sc.emit(i, we, &spoken, "foreign");
        return Some(we);
    }
    // a common noun written with an apostrophe before its suffix (lira'lık): the word as it is
    // read, liralık
    if let Some(suf) = suffix {
        if APOSTROPHE_NOUNS.contains(&base) {
            sc.emit(i, we, &format!("{base}{suf}"), "word");
            return Some(we);
        }
    }
    let n = base.chars().count();
    // Roman numerals
    if let Some(v) = roman_value(base).filter(|_| !NOT_ROMAN.contains(&base)) {
        let spoken = ordinal(&cardinal(&v.to_string()));
        // with a full stop before a word: XIV. Louis, II. Dünya Savaşı, I. Dünya Savaşı. A single
        // letter is an initial (M. Kemal, C. Ronaldo, Ahmet V. Yılmaz) unless it is I or X not
        // after a name, or V before a small letter (V. yüzyıl)
        if suffix.is_none() && next_char(s, we) == Some('.') {
            let k = skip_space(s, we + 1, 2);
            let next_word = next_char(s, k);
            let single_ok = match base {
                "I" | "X" => !roman_context_before(s, i),
                "V" => next_word.is_some_and(char::is_lowercase),
                _ => false,
            };
            if k > we + 1 && next_word.is_some_and(char::is_alphabetic) && (n >= 2 || single_ok) {
                sc.emit(i, we + 1, &spoken, "roman");
                return Some(we + 1);
            }
        }
        // after a name, of I, V and X only: Mehmet II, Louis XIV'ün (not CV, CD, MD)
        if n >= 2 && base.chars().all(|c| matches!(c, 'I' | 'V' | 'X')) && roman_context(s, i, we) {
            let spoken = match suffix {
                Some(suf) => attach_with(&spoken, suf, true),
                None => spoken,
            };
            sc.emit(i, we, &spoken, "roman");
            return Some(we);
        }
    }
    // initialisms: two or more Latin capitals
    if n >= 2 && base.chars().all(latin_capital) {
        // a word in capitals (a heading, shouting) is read as the word
        let known = super::tts::is_known_word(&lower);
        if !(n >= 4 && known) && !shouted(s, i, we, known) {
            let (mut spoken, spelled) = initialism_words(base);
            let vowel = suffix.and_then(|suf| suf.chars().find(|&c| is_vowel(c)));
            // the suffix as written says how a last K was read (TDK'ye: te de keye, SGK'ya: se ge
            // kaya); H stays "he" and its suffix follows it (HGS'ye he ge seye)
            if let (true, Some(v)) = (spelled, vowel) {
                let last = spoken.rsplit(' ').next().unwrap_or("");
                let other = match (last, front(v)) {
                    ("ka", true) => Some("ke"),
                    _ => None,
                };
                if let Some(o) = other {
                    spoken.truncate(spoken.len() - last.len());
                    spoken.push_str(o);
                }
            } else if let Some(v) = vowel {
                if spoken.ends_with("ka") && front(v) {
                    spoken.truncate(spoken.len() - 1);
                    spoken.push('e');
                } else if spoken.ends_with("ke") && !front(v) && base.ends_with('K') {
                    spoken.truncate(spoken.len() - 1);
                    spoken.push('a');
                }
            }
            let spoken = match suffix {
                Some(suf) => attach_with(&spoken, suf, true),
                None => spoken,
            };
            sc.emit(i, we, &spoken, "initialism");
            return Some(we);
        }
    }
    // a single capital consonant used as a label (G bloğu, C vitamini, Plan B): its name. Not
    // an initial before a full stop (M. Kemal), and not a vowel, which reads as written
    if n == 1
        && latin_capital(c)
        && !is_vowel(c)
        && !matches!(c, 'Â' | 'Î' | 'Û')
        && suffix.is_none_or(|suf| LETTER_SUFFIXES.contains(&suf))
    {
        let after = next_char(s, we);
        let label = match after {
            None => true,
            Some(' ' | '\u{a0}') => {
                next_char(s, skip_space(s, we, 1)).is_some_and(char::is_alphabetic)
            }
            Some(',' | ';' | ':' | ')' | '!' | '?' | '"' | '”') => true,
            Some('.') => s[we + 1..].trim().is_empty(),
            _ => false,
        };
        if label {
            if let Some(name) = code_letter(code_lower(base).chars().next().unwrap_or(' ')) {
                let spoken = match suffix {
                    Some(suf) => attach_with(name, suf, true),
                    None => name.to_string(),
                };
                sc.emit(i, we, &spoken, "letters");
                return Some(we);
            }
        }
    }
    // foreign letters (a single letter is spelled by the letter rules: x iks)
    if let Some(t) = turkish_letters(base).filter(|_| n >= 2) {
        let spoken = match suffix {
            Some(suf) => attach_with(&t, suf, true),
            None => t,
        };
        sc.emit(i, we, &spoken, "foreign");
        return Some(we);
    }
    None
}

/// A capital of the Latin alphabet, Turkish ones included (initialisms are read in it).
fn latin_capital(c: char) -> bool {
    c.is_ascii_uppercase() || matches!(c, 'Ç' | 'Ğ' | 'İ' | 'Ö' | 'Ş' | 'Ü' | 'Â' | 'Î' | 'Û')
}

/// Capitals that spell a Roman numeral but are read as initialisms.
const NOT_ROMAN: &[&str] = &[
    "CD", "CV", "DC", "MC", "MD", "DM", "CC", "CM", "MM", "DIL", "MIX",
];

/// After a capitalised name or a number with a full stop (Atatürk Cad., 5. Sok.).
fn after_name_or_number(s: &str, i: usize) -> bool {
    if roman_context_before(s, i) {
        return true;
    }
    let before = s[..i].trim_end();
    before.len() < s[..i].len()
        && before.ends_with('.')
        && before[..before.len() - 1]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_ascii_digit())
}

/// A Roman numeral after a capitalised name (`Mehmet II`, `Louis XIV`), not followed by more
/// letters or digits.
fn roman_context(s: &str, i: usize, we: usize) -> bool {
    roman_context_before(s, i) && !next_char(s, we).is_some_and(char::is_alphanumeric)
}

/// The word before `i` (after a space) is a capitalised name.
fn roman_context_before(s: &str, i: usize) -> bool {
    let before = s[..i].trim_end();
    if before.len() == s[..i].len() {
        return false;
    }
    let prev_word: &str = before
        .rsplit(|c: char| !c.is_alphabetic())
        .next()
        .unwrap_or("");
    let mut pc = prev_word.chars();
    pc.next().is_some_and(char::is_uppercase) && pc.all(char::is_lowercase)
}

/// The word stands among other words in capitals (a heading or shouting): between two, or next
/// to one when it is a word itself (`BU ÇOK`).
fn shouted(s: &str, i: usize, we: usize, known: bool) -> bool {
    let caps = |w: &str| {
        let n = w.chars().filter(|c| c.is_alphabetic()).count();
        n >= 2
            && w.chars()
                .filter(|c| c.is_alphabetic())
                .all(char::is_uppercase)
    };
    let before = s[..i].split_whitespace().next_back().is_some_and(caps);
    let after = s[we..].split_whitespace().next().is_some_and(caps);
    (before && after) || (known && (before || after))
}

/// A symbol standing alone.
fn symbol_at(sc: &mut Scan<'_>, i: usize, c: char) -> Option<usize> {
    let &(_, r) = SYMBOLS.iter().find(|(sym, _)| *sym == c)?;
    let s = sc.src;
    let end = i + c.len_utf8();
    // an asterisk marks emphasis or a footnote unless it stands between numbers or spaces
    if c == '*' {
        let alone =
            |ch: Option<char>| ch.is_none_or(|ch| ch.is_whitespace() || ch.is_ascii_digit());
        if !(alone(prev_char(s, i)) && alone(next_char(s, end))) {
            return None;
        }
    }
    let (spoken, end) = with_suffix(s, r.to_string(), end, true);
    sc.emit(i, end, &spoken, "symbol");
    Some(end)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(t: &str) -> String {
        verbalize(t).text
    }

    #[test]
    fn cardinals_and_suffixes() {
        assert_eq!(cardinal("0"), "sıfır");
        assert_eq!(cardinal("1000"), "bin");
        assert_eq!(cardinal("1990"), "bin dokuz yüz doksan");
        assert_eq!(cardinal("1000000"), "bir milyon");
        assert_eq!(cardinal("007"), "sıfır sıfır yedi");
        assert_eq!(v("Toplantı 3'te başlıyor."), "Toplantı üçte başlıyor.");
        assert_eq!(v("1990'larda"), "bin dokuz yüz doksanlarda");
        assert_eq!(v("4'ü"), "dördü");
        assert_eq!(v("3 kişi"), "üç kişi");
        assert_eq!(v("1.299 kişi"), "bin iki yüz doksan dokuz kişi");
        assert_eq!(v("-5 derece"), "eksi beş derece");
    }

    #[test]
    fn ordinals_and_roman_numerals() {
        assert_eq!(v("2. sınıf"), "ikinci sınıf");
        assert_eq!(v("5'inci kat"), "beşinci kat");
        assert_eq!(v("4'üncü"), "dördüncü");
        assert_eq!(v("6'ncı"), "altıncı");
        assert_eq!(v("XIV. Louis"), "on dördüncü Louis");
        assert_eq!(v("II. Dünya Savaşı"), "ikinci Dünya Savaşı");
        assert_eq!(v("Mehmet II geldi"), "Mehmet ikinci geldi");
        assert_eq!(v("Saat 2. Sonra"), "Saat iki. Sonra");
    }

    #[test]
    fn decimals_fractions_ranges() {
        assert_eq!(v("3,5 kg"), "üç buçuk kilogram");
        assert_eq!(
            verbalize_with("3,5 kg", false).text,
            "üç virgül beş kilogram"
        );
        assert_eq!(v("0,05"), "sıfır virgül sıfır beş");
        // the lexicon's reading
        assert_eq!(v("HDMI 2.1"), "eyçdiemay iki nokta bir");
        assert_eq!(v("2/3'ü"), "üçte ikisi");
        assert_eq!(v("1/2 oranında"), "ikide bir oranında");
        assert_eq!(v("10-15 kişi"), "on ila on beş kişi");
        assert_eq!(v("3-1 yendi"), "üç bir yendi");
        assert_eq!(v("3x4 metre"), "üçe dört metre");
    }

    #[test]
    fn dates_times_money_percent_units() {
        assert_eq!(v("12.03.2024"), "on iki mart iki bin yirmi dört");
        assert_eq!(v("2024-03-12'de"), "on iki mart iki bin yirmi dörtte");
        assert_eq!(v("14:30'da"), "on dört otuzda");
        assert_eq!(v("09.45'te"), "dokuz kırk beşte");
        assert_eq!(v("14:00-16:00"), "on dört ila on altı");
        assert_eq!(
            v("1.299,90 TL"),
            "bin iki yüz doksan dokuz lira doksan kuruş"
        );
        assert_eq!(v("₺850'ye"), "sekiz yüz elli liraya");
        assert_eq!(v("100 TL'ye"), "yüz liraya");
        assert_eq!(v("1,5 milyon TL"), "bir buçuk milyon lira");
        assert_eq!(v("%25'lik"), "yüzde yirmi beşlik");
        assert_eq!(v("%3,5"), "yüzde üç buçuk");
        assert_eq!(v("12 km'den"), "on iki kilometreden");
        assert_eq!(v("5 kg'ye"), "beş kilograma");
        assert_eq!(v("90 km/sa"), "saatte doksan kilometre");
        assert_eq!(v("25 °C"), "yirmi beş derece");
    }

    #[test]
    fn telephones_addresses() {
        assert_eq!(
            v("0532 123 45 67"),
            "sıfır beş yüz otuz iki, yüz yirmi üç, kırk beş, altmış yedi"
        );
        assert_eq!(
            v("+90 212 555 12 34"),
            "artı doksan, iki yüz on iki, beş yüz elli beş, on iki, otuz dört"
        );
        // tts-v1 keeps its reading
        assert_eq!(
            verbalize_with("0532 123 45 67", false).text,
            "sıfır beş yüz otuz iki yüz yirmi üç kırk beş altmış yedi"
        );
        assert_eq!(
            v("ali.veli@ornek.com'dan"),
            "ali nokta veli et ornek nokta komdan"
        );
        assert_eq!(
            v("www.ornek.com.tr adresi"),
            "ve ve ve nokta ornek nokta kom nokta te re adresi"
        );
    }

    #[test]
    fn abbreviations_and_initialisms() {
        assert_eq!(v("Dr. Ahmet Bey"), "doktor Ahmet Bey");
        assert_eq!(v("Prof. Dr. Ayşe"), "profesör doktor Ayşe");
        assert_eq!(v("kitaplar vb. şeyler"), "kitaplar ve benzeri şeyler");
        assert_eq!(v("kitaplar vb. Sonra"), "kitaplar ve benzeri. Sonra");
        assert_eq!(v("TBMM'de"), "te be me mede");
        assert_eq!(v("ABD'ye"), "a be deye");
        // read as the word it is
        assert_eq!(v("NATO'nun"), "NATO'nun");
        assert_eq!(v("CNN'e"), "sienene");
        assert_eq!(v("A4 kâğıt"), "a dört kâğıt");
        assert_eq!(v("BUGÜN ÇOK GÜZEL"), "BUGÜN ÇOK GÜZEL");
    }

    #[test]
    fn symbols_letters_foreign() {
        assert_eq!(v("C++ ve C# dilleri"), "si artı artı ve si şarp dilleri");
        assert_eq!(v("Ar & Ge"), "Ar ve Ge");
        assert_eq!(v("Ç-A-Y diye"), "çe a ye diye");
        assert_eq!(v("iPhone'u aldım"), "ayfonu aldım");
        assert_eq!(v("Facebook'ta"), "feysbukta");
        assert_eq!(v("Wikipedia"), "Vikipedia");
        assert_eq!(v("café"), "cafe");
    }

    #[test]
    fn context_decides() {
        // an abbreviation that is also a word is read as one only in its place
        assert_eq!(v("Herkes öyle der."), "Herkes öyle der.");
        assert_eq!(
            v("Av. Mehmet ile av. yasağı"),
            "avukat Mehmet ile av. yasağı"
        );
        assert_eq!(v("Atatürk Cad. No: 5"), "Atatürk caddesi numara beş");
        assert_eq!(v("Bul. dedi"), "Bul. dedi");
        // the written suffix says how the last letter was read
        assert_eq!(v("TDK'ye göre"), "te de keye göre");
        assert_eq!(v("SGK'ya"), "se ge kaya");
        // ordinals before a capital at the start of a sentence only
        assert_eq!(v("1. Lig'de"), "birinci Lig'de");
        assert_eq!(v("Yaşı 15. Okula gidiyor."), "Yaşı on beş. Okula gidiyor.");
        assert_eq!(
            v("Ahmet V. Yılmaz ile I. Dünya Savaşı"),
            "Ahmet V. Yılmaz ile birinci Dünya Savaşı"
        );
        assert_eq!(v("3.50 TL"), "üç lira elli kuruş");
        assert_eq!(v("y = 2x + 3"), "y eşittir iki iks artı üç");
        assert_eq!(v("24/7 açık"), "yirmi dört yedi açık");
        assert_eq!(v("Toplam 1000000 kişi"), "Toplam bir milyon kişi");
        assert_eq!(v("BU ÇOK ÖNEMLİ"), "BU ÇOK ÖNEMLİ");
        assert_eq!(v("x"), "x");
    }

    /// The findings of the review of 2026-11-08, one assertion each.
    #[test]
    fn review_regressions() {
        // non-ASCII digits never panic and are not read as ASCII numbers
        for t in [
            "1١11",
            "0١١١",
            "1２３４",
            "0２",
            "12.03.٢٠٢٤",
            "Tarih: 1.1.٢٠",
            "12٣4 kişi",
            "1١",
        ] {
            let _ = verbalize(t);
            let _ = verbalize_with(t, false);
        }
        // tts-v1 (no word rules) reads text of letters and punctuation as written
        for t in [
            "TBMM bugün",
            "ABD'ye",
            "II. Dünya Savaşı",
            "Mehmet II",
            "A-B planı",
            "Bu...net bir cevap.",
            "gittim.dev",
            "www.ornek.com",
            "M. Kemal",
            "Ders CD ile",
            "BU ÇOK",
            "vb. online Shakespeare'i",
        ] {
            assert!(verbalize_with(t, false).segments.is_empty(), "{t}");
        }
        // initials and initialisms are not Roman numerals
        assert_eq!(v("M. Kemal Atatürk geldi."), "M. Kemal Atatürk geldi.");
        assert_eq!(v("C. Ronaldo"), "C. Ronaldo");
        assert!(!v("Ahmet CV gönderdi").contains("yüz"));
        assert!(!v("Ders CD ile").contains("yüz"));
        assert_eq!(v("Louis XIV'ün"), "Louis on dördüncünün");
        // telephone numbers in their groups
        assert_eq!(
            v("+905321234567"),
            "artı doksan, beş yüz otuz iki, yüz yirmi üç, kırk beş, altmış yedi"
        );
        assert_eq!(
            v("(0532) 123 45 67"),
            "sıfır beş yüz otuz iki, yüz yirmi üç, kırk beş, altmış yedi"
        );
        assert_eq!(
            v("0850 222 0 600"),
            "sıfır sekiz yüz elli, iki yüz yirmi iki, sıfır, altı yüz"
        );
        // case numbers, dotted decimals and their units
        assert_eq!(v("2012/3121"), "iki bin on iki bölü üç bin yüz yirmi bir");
        assert_eq!(v("3.14"), "üç virgül on dört");
        assert_eq!(verbalize_with("3.14", false).text, "üç nokta on dört");
        assert_eq!(v("Python 3.12"), "Python üç nokta on iki");
        assert_eq!(v("2.5 kg"), "iki buçuk kilogram");
        assert_eq!(v("3.5 TL"), "üç lira elli kuruş");
        // sentence boundaries around abbreviations
        assert_eq!(v("T.C. Kimlik Numarası"), "te ce Kimlik Numarası");
        assert_eq!(v("bkz. Ek 2"), "bakınız Ek iki");
        assert_eq!(
            v("Koç Holding A.Ş. Bugün açıkladı."),
            "Koç Holding anonim şirketi. Bugün açıkladı."
        );
        // suffixes after foreign words and symbols, re-buffered
        assert_eq!(v("Wi-Fi'ye"), "vayfaya");
        assert_eq!(v("Nike'ı"), "naykiyi");
        assert_eq!(v("1/2'si"), "ikide biri");
        assert_eq!(v("¼'ü"), "çeyreği");
        // smaller readings
        assert_eq!(v("1990-95"), "bin dokuz yüz doksan ila doksan beş");
        assert_eq!(v("24:00"), "yirmi dört");
        assert_eq!(v("x=-5"), "x eşittir eksi beş");
        assert_eq!(v("09:05:07"), "dokuz sıfır beş sıfır yedi");
        assert_eq!(v("Kaç? 3. Ben de."), "Kaç? üç. Ben de.");
        assert_eq!(v("1.2.24 sürümü"), "bir nokta iki nokta yirmi dört sürümü");
        assert_eq!(v("gittim.dev"), "gittim.dev");
        assert_eq!(v("Yıldız Mah. 5. Sok."), "Yıldız mahallesi beşinci sokağı.");
    }

    /// Service text with the word rules (tests/data/speech/service_cases.jsonl has the brief's
    /// cases), and the readings that must not change: precision first.
    #[test]
    fn service_conventions_and_their_guards() {
        // telephone numbers: a pause between groups; mobile numbers written whole; 444 numbers
        assert_eq!(
            v("5321234567 numaralı"),
            "beş yüz otuz iki, yüz yirmi üç, kırk beş, altmış yedi numaralı"
        );
        assert_eq!(v("444 0 333"), "dört yüz kırk dört, sıfır, üç yüz otuz üç");
        assert_eq!(
            v("Nüfus 5000000000 kişi"),
            "Nüfus beş yüz sıfır sıfır sıfır sıfır sıfır sıfır sıfır kişi"
        );
        // identifiers digit by digit, by the words around them
        assert_eq!(v("Kod: 12345"), "Kod: bir iki üç dört beş");
        assert_eq!(
            v("76950 posta kodlu"),
            "yedi altı dokuz beş sıfır posta kodlu"
        );
        assert_eq!(
            v("Siparişinizi 293018 ile"),
            "Siparişinizi iki dokuz üç sıfır bir sekiz ile"
        );
        // ... and quantities as numbers
        assert_eq!(
            v("Hesabınıza 15000 yatırıldı."),
            "Hesabınıza on beş bin yatırıldı."
        );
        assert_eq!(
            v("Hesabınıza 12500 puan"),
            "Hesabınıza on iki bin beş yüz puan"
        );
        assert_eq!(
            v("12345 kod satırı"),
            "on iki bin üç yüz kırk beş kod satırı"
        );
        assert_eq!(
            v("Dosya 12345 KB"),
            "Dosya on iki bin üç yüz kırk beş kilobayt"
        );
        assert_eq!(
            v("hesaplanırken 4463192586 oranı"),
            "hesaplanırken dört yüz kırk altı üç yüz on dokuz iki yüz elli sekiz altı oranı"
        );
        // letters by name, one word each; words and English readings stay
        assert_eq!(v("KDV'si"), "ka de vesi");
        assert_eq!(v("THY ve CHP"), "te he ye ve ce he pe");
        // the owner's decisions (2026-11-15): H is "he" even before a back-vowel suffix, PDF
        // and QR by Turkish letter names, never the word küre
        assert_eq!(v("HGS'ye"), "he ge seye");
        assert!(!v("KH'a").contains("ha"), "{}", v("KH'a"));
        assert_eq!(v("PDF ve QR"), "pe de fe ve kû re");
        assert_eq!(v("ODTÜ'de, CNN"), "odtüde, sienen");
        assert_eq!(v("SMS ile"), "esemes ile");
        // a single capital consonant as a label; a vowel, an initial and an elision stay
        assert_eq!(v("C vitamini, Plan B."), "ce vitamini, Plan be.");
        assert_eq!(v("C'den daha hızlı"), "ceden daha hızlı");
        assert_eq!(v("O gün A blokta"), "O gün A blokta");
        assert_eq!(v("Ahmet B. Yılmaz"), "Ahmet B. Yılmaz");
        assert_eq!(v("N'aber? N'oldu? M'Vila"), "N'aber? N'oldu? M'Vila");
        // a landline written whole next to a telephone word
        assert_eq!(
            v("8086727078 numaralı hat"),
            "sekiz yüz sekiz, altı yüz yetmiş iki, yetmiş, yetmiş sekiz numaralı hat"
        );
        assert_eq!(
            v("4958669031 numaralı üretim hatlarında"),
            "dört dokuz beş sekiz altı altı dokuz sıfır üç bir numaralı üretim hatlarında"
        );
        // plates and codes; capitals that are words, units or months are no plate
        assert_eq!(v("34 ABC 123"), "otuz dört a be ce yüz yirmi üç");
        assert_eq!(
            v("16 GB 2400 MHz"),
            "on altı gigabayt iki bin dört yüz megahertz"
        );
        assert_eq!(v("12 OCA 2026"), "on iki oca iki bin yirmi altı");
        assert!(!v("20 YIL 10 AY").contains("ye ı le"));
        assert_eq!(v("X7K9QZ"), "iks yedi ka dokuz kû ze");
        assert_eq!(v("MP3, B2B, G20"), "me pe üç, be iki be, ge yirmi");
        // doses against dimensions
        assert_eq!(v("Günde 2x500 mg"), "Günde iki çarpı beş yüz miligram");
        assert_eq!(v("3x4 metre"), "üçe dört metre");
        assert_eq!(v("3×4 metre"), "üçe dört metre");
        // decimals: a half, dot decimals, versions and sections keep their dot
        assert_eq!(v("1,5'i"), "bir buçuğu");
        assert_eq!(v("0,5 litre"), "sıfır virgül beş litre");
        assert_eq!(v("Madde 12.5"), "Madde on iki nokta beş");
        assert_eq!(v("HDMI 2.1"), "eyçdiemay iki nokta bir");
        // money without its leading zero; a common noun's apostrophe
        assert_eq!(v("Ürün 0,50 TL."), "Ürün elli kuruş.");
        assert_eq!(v("841 lira'lık"), "sekiz yüz kırk bir liralık");
        // tts-v1 keeps every earlier reading of these
        for (t, old) in [
            (
                "Doğrulama kodunuz 482915.",
                "Doğrulama kodunuz dört yüz seksen iki bin dokuz yüz on beş.",
            ),
            ("%1,5", "yüzde bir virgül beş"),
            ("7/24", "yirmi dörtte yedi"),
            ("Günde 2x1 tablet", "Günde ikiye bir tablet"),
            (
                "TR78240865",
                "tere yetmiş sekiz milyon iki yüz kırk bin sekiz yüz altmış beş",
            ),
        ] {
            assert_eq!(verbalize_with(t, false).text, old, "{t}");
        }
    }

    #[test]
    fn long_input_maps_spans_in_linear_time() {
        // 40,000 verbalised numbers: the span map is a binary search, not a scan
        let text = "x 5 ".repeat(40_000);
        let r = verbalize(&text);
        assert_eq!(r.segments.len(), 40_000);
        let last = r
            .segments
            .last()
            .map(|s| s.spoken_span.clone())
            .unwrap_or(0..0);
        assert_eq!(r.source(last, text.len()).0, text.len() - 2..text.len() - 1);
    }

    #[test]
    fn plain_text_is_unchanged() {
        for t in [
            "Bir ağaçta gül de biter, diken de.",
            "Ankara'ya gittik, sonra döndük.",
            "Hâlâ gelmedi... \"Neden?\" dedi.",
            "Serkan mı bavulu hazırladı?",
            "",
        ] {
            let r = verbalize(t);
            assert_eq!(r.text, t);
            assert!(r.segments.is_empty());
        }
    }

    #[test]
    fn spans_map_back() {
        let t = "Fiyat ₺850'ye düştü.";
        let r = verbalize(t);
        assert_eq!(r.segments.len(), 1);
        let seg = &r.segments[0];
        assert_eq!(&t[seg.span.clone()], "₺850'ye");
        assert_eq!(&r.text[seg.spoken_span.clone()], "sekiz yüz elli liraya");
        // a word inside the reading maps to the whole written amount; one after it, one to one
        let k = r.text.find("liraya").unwrap_or(0);
        assert_eq!(r.source(k..k + "liraya".len(), t.len()).0, seg.span);
        let d = r.text.find("düştü").unwrap_or(0);
        assert_eq!(&t[r.source(d..d + "düştü".len(), t.len()).0], "düştü");
    }

    #[test]
    fn never_panics() {
        for t in [
            "₺",
            "%",
            "3'",
            "'te",
            "1.",
            "1..2",
            "12/",
            "0532",
            "+",
            "-",
            "−5",
            "x",
            "ß",
            "@@",
            "a@b",
            "www.",
            "http://",
            "XIV.",
            "12:",
            "99:99",
            "31.02.2024",
            "32.13.2024",
            "1-",
            "-1-",
            "5 km/",
            "5 km/x",
            "iPhone'",
            "Ç-",
            "-A",
            "1/0",
            "0/0",
            "²",
            "°",
            "9999999999999999999999999999",
            "1,",
            ",5",
            "ĞĞ",
            "İSTANBUL",
            "à",
            "\u{0302}",
        ] {
            let r = verbalize(t);
            for s in &r.segments {
                assert!(s.span.end <= t.len() && s.spoken_span.end <= r.text.len());
            }
        }
    }
}
