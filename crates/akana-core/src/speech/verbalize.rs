//! Written Turkish to its spoken form, for the `tts-v1` profile: everything a screen reader
//! meets. Numbers (with the suffix written after them re-harmonised to the word they now end
//! in), ordinals, Roman numerals, decimals, fractions, ranges, dates, times, money, percentages,
//! units, telephone numbers, abbreviations and initialisms, URLs and e-mail addresses, symbols,
//! letters spelled with hyphens, and foreign words, brands and letters read Turkish-style.
//!
//! Each verbalisation keeps the byte span of the text it reads ([`Verbalized::segments`]); the
//! text between them is kept as written, so text without anything to verbalise comes back
//! unchanged and is phonemised exactly as before.
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
    /// `electronic`, `symbol`, `letters` or `foreign`.
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
        let map = |pos: usize, is_end: bool| -> (usize, Option<&'static str>) {
            // the segment containing pos (for an end, the one containing pos - 1)
            let mut delta: isize = 0;
            for s in &self.segments {
                let inside = if is_end {
                    pos > s.spoken_span.start && pos <= s.spoken_span.end
                } else {
                    pos >= s.spoken_span.start && pos < s.spoken_span.end
                };
                if inside {
                    return (if is_end { s.span.end } else { s.span.start }, Some(s.kind));
                }
                if s.spoken_span.end <= pos {
                    delta = s.span.end as isize - s.resume as isize;
                } else {
                    break;
                }
            }
            let p = (pos as isize + delta).clamp(0, src_len as isize) as usize;
            (p, None)
        };
        let (start, k1) = map(r.start, false);
        let (end, k2) = map(r.end, true);
        (start..end.max(start), k1.or(k2))
    }
}

fn is_upper(c: char) -> bool {
    c.is_uppercase()
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
    let stem = if last == "dört" && suffix.starts_with(is_vowel) {
        "dörd"
    } else {
        last
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
    /// Anywhere (`vb.`, `bkz.`); its full stop also ends a sentence before a capital.
    Any,
    /// Before a capitalised name or another title (`Dr. Ayşe`, `Prof. Dr.`); its full stop never
    /// ends a sentence.
    Title,
    /// After a capitalised name (`Atatürk Bul.`, `Yıldız San.`).
    Address,
}

macro_rules! abbrevs {
    ($( $w:literal => $s:literal $(, $t:ident)? ;)*) => {
        &[$( Abbrev { written: $w, spoken: $s, place: abbrevs!(@t $($t)?) } ),*]
    };
    (@t title) => { Place::Title };
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
    "vb." => "ve benzeri";
    "vs." => "vesaire";
    "vd." => "ve diğerleri";
    "bkz." => "bakınız";
    "Bkz." => "bakınız";
    "örn." => "örneğin";
    "Örn." => "örneğin";
    "yy." => "yüzyıl";
    "M.Ö." => "milattan önce";
    "M.S." => "milattan sonra";
    "T.C." => "te ce";
    "A.Ş." => "anonim şirketi", address;
    "Ltd. Şti." => "limited şirketi", address;
    "Şti." => "şirketi", address;
    "Ltd." => "limited", address;
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
    "a.g.e." => "adı geçen eser";
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
        r"^(?:(?:https?://)(?:www\.)?|www\.)[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*(?:/[A-Za-z0-9._~%/?#=&+-]*)?|^[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:com|net|org|edu|gov|io|info|biz|app|dev|ai)(?:\.tr)?(?:/[A-Za-z0-9._~%/?#=&+-]*)?|^[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:com|net|org|edu|gov|gen|bel|k12|av|web|tv|info|biz)\.tr(?:/[A-Za-z0-9._~%/?#=&+-]*)?"
    )
    .expect("url regex");
    static ref PHONE: Regex = Regex::new(
        r"^(?:\+90[ -]?|0 ?)?(?:\(\d{3,4}\)|\d{3,4})[ -]?\d{3}[ -]?\d{2}[ -]?\d{2}"
    )
    .expect("phone regex");
    static ref DATE: Regex =
        Regex::new(r"^(\d{1,2})([./-])(\d{1,2})([./-])(\d{4}|\d{2})").expect("date regex");
    static ref ISO_DATE: Regex = Regex::new(r"^(\d{4})-(\d{2})-(\d{2})").expect("iso regex");
    static ref TIME: Regex =
        Regex::new(r"^(\d{1,2})([:.])(\d{2})(?::(\d{2}))?").expect("time regex");
    static ref AMOUNT: Regex =
        Regex::new(r"^(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?").expect("amount regex");
    static ref ROMAN: Regex = Regex::new(r"^[IVXLCDM]+").expect("roman regex");
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

fn amount_words(int: &str, dec: Option<&str>) -> String {
    match dec {
        Some(d) => format!("{} virgül {}", cardinal(int), decimals(d)),
        None => cardinal(int),
    }
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

/// The end of the letters at `i`, with the hyphens and apostrophes between them.
fn word_end(s: &str, i: usize) -> usize {
    let mut j = i;
    while let Some(c) = next_char(s, j) {
        let inner = matches!(c, '-' | '\'' | '’')
            && next_char(s, j + c.len_utf8()).is_some_and(char::is_alphabetic);
        if c.is_alphabetic() || c == '\u{0302}' || inner {
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

/// Verbalises `text`; without `words`, words written in letters are kept as written (the
/// abbreviations, foreign words, foreign letters and small-letter spellings are not expanded),
/// so text of letters and punctuation alone comes back unchanged. Profile `tts-v1` reads text so,
/// which keeps its output on such text as it was frozen in stage 2.
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
        let at_start = !prev.is_some_and(|p| p.is_alphanumeric());
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
        || (matches!(c, '+' | '-' | '−')
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
            let spoken = money(&int, dec.as_deref(), u, sub);
            let (spoken, end) = with_suffix(s, spoken, end, true);
            sc.emit(i, end, &spoken, "money");
            return Some(end);
        }
        return None;
    }
    if matches!(c, '%' | '‰') {
        let j = skip_space(s, i + c.len_utf8(), 1);
        if let Some((int, dec, end)) = amount_at(s, j) {
            let head = if c == '%' { "yüzde" } else { "binde" };
            let spoken = format!("{head} {}", amount_words(&int, dec.as_deref()));
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
            if s[i..end].contains('.') {
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

fn money(int: &str, dec: Option<&str>, unit: &str, sub: &str) -> String {
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

/// The rules for a token starting with a digit or a sign.
fn number_rules(sc: &mut Scan<'_>, i: usize) -> Option<usize> {
    let s = sc.src;
    let c = next_char(s, i)?;
    // a sign before a number
    if matches!(c, '+' | '-' | '−') {
        // +90 telephone numbers
        if c == '+' {
            if let Some(m) = PHONE.find(&s[i..]) {
                let end = i + m.end();
                if !next_char(s, end).is_some_and(|d| d.is_ascii_digit()) {
                    let spoken = telephone(&s[i..end]);
                    let (spoken, end) = with_suffix(s, spoken, end, false);
                    sc.emit(i, end, &spoken, "telephone");
                    return Some(end);
                }
            }
        }
        // a minus only stands for a sign at the start or after a space
        let p = prev_char(s, i);
        if p.is_some_and(|p| !p.is_whitespace() && p != '(') {
            return None;
        }
        let j = i + c.len_utf8();
        let (int, dec, end) = amount_at(s, j)?;
        let sign = if c == '+' { "artı" } else { "eksi" };
        let (body, kind, end) = quantity(s, &int, dec.as_deref(), end);
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
    // telephone numbers: 10 or 11 digits in groups, from 0 or 5
    if let Some(m) = PHONE.find(rest) {
        let end = i + m.end();
        let digits = rest[..m.end()].chars().filter(char::is_ascii_digit).count();
        let first = rest.chars().find(char::is_ascii_digit);
        if !next_char(s, end).is_some_and(|d| {
            d.is_ascii_digit()
                || d == ',' && next_char(s, end + 1).is_some_and(|e| e.is_ascii_digit())
        }) && ((digits == 11 && first == Some('0'))
            || (digits == 10 && matches!(first, Some('5' | '2' | '3' | '4')))
                && rest[..m.end()].contains([' ', '-', '(']))
        {
            let spoken = telephone(&rest[..m.end()]);
            let (spoken, end) = with_suffix(s, spoken, end, false);
            sc.emit(i, end, &spoken, "telephone");
            return Some(end);
        }
    }
    // dates: 12.03.2024, 12/03/2024, 2024-03-12
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
        if cap[2] == cap[4] && !next_char(s, end).is_some_and(|c| c.is_ascii_digit()) {
            if let Some(spoken) = date(&cap[1], &cap[3], &cap[5]) {
                let (spoken, end) = with_suffix(s, spoken, end, false);
                sc.emit(i, end, &spoken, "date");
                return Some(end);
            }
        }
    }
    // times: 14:30, 09.45, 14:30:15
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
        // a dot time is not an amount with its currency or unit (3.50 TL)
        let dot_ok =
            colon || (sec.is_none() && currency_at(s, end).is_none() && unit_at(s, end).is_none());
        if h <= 23 && m <= 59 && sec.is_none_or(|x| x <= 59) && after_ok && dot_ok {
            let mut spoken = time(h, m, sec);
            let mut end = end;
            // a range of times: 14:00-16:00
            if let Some(d) = next_char(s, end).filter(|&d| d == '-' || d == '–') {
                if let Some(c2) = TIME.captures(&s[end + d.len_utf8()..]) {
                    let h2: u32 = c2[1].parse().unwrap_or(99);
                    let m2: u32 = c2[3].parse().unwrap_or(99);
                    let e2 = end + d.len_utf8() + c2.get(0).map_or(0, |x| x.end());
                    if h2 <= 23 && m2 <= 59 && !next_char(s, e2).is_some_and(|c| c.is_ascii_digit())
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
    // a number glued to more digits after a dot or comma it did not take (1.2.3, 3.14)
    let next = next_char(s, end);
    // an amount with a decimal point and its currency: 3.50 TL
    if next == Some('.') && dec.is_none() {
        let d0 = end + 1;
        let mut d1 = d0;
        while next_char(s, d1).is_some_and(|c| c.is_ascii_digit()) {
            d1 += 1;
        }
        if d1 - d0 == 2 && next_char(s, d1) != Some('.') {
            if let Some((u, sub, e)) = currency_at(s, d1) {
                let spoken = money(&int, Some(&s[d0..d1]), u, sub);
                let (spoken, e) = with_suffix(s, spoken, e, true);
                sc.emit(i, e, &spoken, "money");
                return Some(e);
            }
        }
    }
    if next == Some('.')
        && dec.is_none()
        && next_char(s, end + 1).is_some_and(|c| c.is_ascii_digit())
    {
        // a dotted decimal or a version: 3.14, 2.0.1
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
        let (spoken, e) = with_suffix(s, words.join(" "), e, false);
        sc.emit(i, e, &spoken, "decimal");
        return Some(e);
    }
    // an ordinal: 2. sınıf, 5'inci
    if dec.is_none() && !int.contains('.') && next == Some('.') {
        let after = skip_space(s, end + 1, 3);
        // before a small letter (2. sınıf); before a capital only at the start of a sentence
        // (1. Lig'de, 2. Abdülhamid), where the full stop cannot end one
        let sentence_start = s[..i]
            .trim_end()
            .chars()
            .next_back()
            .is_none_or(|c| matches!(c, '.' | '!' | '?' | ':' | '"' | '“' | '(' | '\n'));
        if after > end + 1
            && next_char(s, after)
                .is_some_and(|c| c.is_lowercase() || (sentence_start && c.is_uppercase()))
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
    // a range: 10-15, 3-1 (a score)
    if dec.is_none() {
        if let Some(d) = next.filter(|&d| d == '-' || d == '–') {
            if let Some((int2, dec2, end2)) = amount_at(s, end + d.len_utf8()) {
                if !next_char(s, end2).is_some_and(|c| c.is_ascii_digit() || c.is_alphabetic()) {
                    let a: u128 = int.parse().unwrap_or(0);
                    let b: u128 = int2.parse().unwrap_or(0);
                    let first = cardinal(&int);
                    let (second, kind, end2) = quantity(s, &int2, dec2.as_deref(), end2);
                    let join = if b > a { " ila " } else { " " };
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
        // a fraction: 2/3 üçte iki; 24/7 yirmi dört bölü yedi
        if next == Some('/') {
            if let Some((int2, None, end2)) = amount_at(s, end + 1) {
                if !next_char(s, end2).is_some_and(|c| c.is_ascii_digit() || c == '/') {
                    let a: u128 = int.parse().unwrap_or(0);
                    let b: u128 = int2.parse().unwrap_or(0);
                    let spoken = if a == 24 && b == 7 {
                        "yirmi dört yedi".to_string()
                    } else if a < b && b > 0 {
                        let den = cardinal(&int2);
                        format!("{} {}", attach(&den, "de"), cardinal(&int))
                    } else {
                        format!("{} bölü {}", cardinal(&int), cardinal(&int2))
                    };
                    let (spoken, end2) = match suffix_at(s, end2) {
                        // üçte ikisi: a vowel suffix is the possessive
                        Some((suf, e))
                            if suf.chars().count() == 1 && spoken.ends_with(is_vowel) =>
                        {
                            (format!("{spoken}s{}", harmonize(&spoken, &suf, false)), e)
                        }
                        _ => with_suffix(s, spoken, end2, false),
                    };
                    sc.emit(i, end2, &spoken, "fraction");
                    return Some(end2);
                }
            }
        }
        // dimensions: 3x4 üçe dört
        if matches!(next, Some('x' | '×')) {
            if let Some((int2, dec2, end2)) = amount_at(s, end + 1) {
                if !next_char(s, end2).is_some_and(|c| c.is_alphanumeric()) {
                    let (second, kind, end2) = quantity(s, &int2, dec2.as_deref(), end2);
                    let spoken = format!("{} {second}", attach_with(&cardinal(&int), "e", true));
                    let kind = if kind == "cardinal" { "range" } else { kind };
                    let (spoken, end2) =
                        with_suffix(s, spoken, end2, kind == "money" || kind == "unit");
                    sc.emit(i, end2, &spoken, kind);
                    return Some(end2);
                }
            }
        }
        // a code (ten digits or more, or a leading zero): read in groups of three
        if (int.len() >= 10 || (int.len() >= 4 && int.starts_with('0'))) && !s[i..end].contains('.')
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
    let (spoken, kind, end) = quantity(s, &int, dec.as_deref(), end);
    // letters glued to the number: 3D, 4K (capitals are letters), 1990lar (a suffix)
    let (spoken, end) = if let Some(l) = next_char(s, end).filter(|c| c.is_alphabetic()) {
        let we = word_end(s, end);
        let letters = &s[end..we];
        if l.is_uppercase() {
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
fn quantity(s: &str, int: &str, dec: Option<&str>, end: usize) -> (String, &'static str, usize) {
    let words = amount_words(int, dec);
    let j = skip_space(s, end, 1);
    if next_char(s, j) == Some('%') {
        return (format!("yüzde {words}"), "percent", j + 1);
    }
    if let Some((u, sub, e)) = currency_at(s, end) {
        return (money(int, dec, u, sub), "money", e);
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
        } else {
            cardinal(&sec.to_string())
        });
    }
    out
}

/// A telephone number read in its written groups, a leading zero as `sıfır`, `+90` as `artı
/// doksan`.
fn telephone(t: &str) -> String {
    let mut words = Vec::new();
    let mut digits: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut plus = false;
    for c in t.chars() {
        if c.is_ascii_digit() {
            cur.push(c);
        } else {
            if c == '+' {
                plus = true;
            }
            if !cur.is_empty() {
                digits.push(std::mem::take(&mut cur));
            }
        }
    }
    if !cur.is_empty() {
        digits.push(cur);
    }
    // an unspaced number: 0532 123 45 67
    if digits.len() == 1 {
        let d = digits.remove(0);
        let sizes: &[usize] = if d.len() == 11 {
            &[4, 3, 2, 2]
        } else {
            &[3, 3, 2, 2]
        };
        let mut k = 0;
        for &n in sizes {
            let e = (k + n).min(d.len());
            digits.push(d[k..e].to_string());
            k = e;
        }
    }
    if plus {
        words.push("artı".to_string());
    }
    for g in digits {
        words.push(cardinal(&g));
    }
    words.join(" ")
}

/// The rules for a token starting with a letter (or a dot: `.NET`).
fn word_rules(sc: &mut Scan<'_>, i: usize) -> Option<usize> {
    let s = sc.src;
    let words = sc.words;
    // abbreviations with a full stop, longest first
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
            Place::Any => true,
            Place::Title => {
                let k = skip_space(s, end, 2);
                k > end && next_char(s, k).is_some_and(char::is_uppercase)
            }
            Place::Address => roman_context_before(s, i),
        }
    };
    if let Some(a) = best.filter(|a| words && fits(a)) {
        let end = i + a.written.len();
        let mut spoken = a.spoken.to_string();
        // a post-positioned abbreviation's full stop also ends a sentence before a capital
        if a.place == Place::Any && a.written.ends_with('.') {
            let k = skip_space(s, end, 3);
            if k == s.len() || (k > end && next_char(s, k).is_some_and(char::is_uppercase)) {
                spoken.push('.');
            }
        }
        let (spoken, end) = with_suffix(s, spoken, end, true);
        sc.emit(i, end, &spoken, "abbreviation");
        return Some(end);
    }
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
    let we = word_end(s, i);
    let c = next_char(s, i)?;
    // tech names with symbols: C++, C#, .NET
    for (w, r) in FOREIGN.iter().filter(|(w, _)| w.contains(['+', '#', '.'])) {
        let n = w.len();
        if s.len() >= i + n
            && s.is_char_boundary(i + n)
            && to_turkish_lower(&s[i..i + n]) == *w
            && !next_char(s, i + n).is_some_and(char::is_alphanumeric)
        {
            let (spoken, end) = with_suffix(s, r.to_string(), i + n, false);
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
    // letters spelled with hyphens: Ç-A-Y
    if base.contains('-') && (words || base.chars().all(|c| c == '-' || c.is_uppercase())) {
        let parts: Vec<&str> = base.split('-').collect();
        if parts.len() >= 2 && parts.iter().all(|p| p.chars().count() == 1) {
            let spoken: Vec<String> = parts
                .iter()
                .map(|p| letter_pronunciation(&to_turkish_lower(p)))
                .collect();
            let spoken = spoken.join(" ");
            let spoken = match suffix {
                Some(suf) => attach(&spoken, suf),
                None => spoken,
            };
            sc.emit(i, we, &spoken, "letters");
            return Some(we);
        }
    }
    // letters followed by digits: A4, MP3, COVID-19
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
        if suffix.is_none() && base.chars().all(is_upper) && base.chars().count() <= 5 {
            let letters = if base.chars().count() == 1 {
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
    let lower = to_turkish_lower(base);
    // foreign words and brands (a whole word with its apostrophe, or the word and a suffix)
    let key = to_turkish_lower(word);
    if let Some(&(_, r)) = FOREIGN.iter().find(|(w, _)| words && *w == key) {
        sc.emit(i, we, r, "foreign");
        return Some(we);
    }
    let ascii_lower = base.to_lowercase();
    if let Some(&(_, r)) = FOREIGN
        .iter()
        .find(|(w, _)| words && (*w == lower || *w == ascii_lower))
    {
        let spoken = match suffix {
            Some(suf) => attach(r, suf),
            None => r.to_string(),
        };
        sc.emit(i, we, &spoken, "foreign");
        return Some(we);
    }
    let n = base.chars().count();
    // Roman numerals: XIV. Louis, II. Dünya Savaşı, Mehmet II
    if let Some(v) = roman_value(base).filter(|_| suffix.is_none()) {
        // a single letter with a full stop is an initial after a name (Ahmet V. Yılmaz)
        let after_name = roman_context_before(s, i);
        if next_char(s, we) == Some('.') && (n >= 2 || !after_name) {
            let k = skip_space(s, we + 1, 2);
            if k > we + 1 && next_char(s, k).is_some_and(char::is_alphabetic) {
                let spoken = ordinal(&cardinal(&v.to_string()));
                sc.emit(i, we + 1, &spoken, "roman");
                return Some(we + 1);
            }
        }
        if n >= 2 && roman_context(s, i, we) {
            let spoken = ordinal(&cardinal(&v.to_string()));
            sc.emit(i, we, &spoken, "roman");
            return Some(we);
        }
    }
    // initialisms: two or more capitals
    if n >= 2 && base.chars().all(is_upper) {
        // a word in capitals (a heading, shouting) is read as the word
        let known = super::tts::is_known_word(&lower);
        if !(n >= 4 && known) && !shouted(s, i, we, known) {
            let mut spoken = initialism(base);
            // the suffix as written says how the last letter was read (TDK'ye: te de ke)
            if let Some(v) = suffix.and_then(|suf| suf.chars().find(|&c| is_vowel(c))) {
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
    // foreign letters (a single letter is spelled by the letter rules: x iks)
    if let Some(t) = turkish_letters(base).filter(|_| words && n >= 2) {
        let spoken = match suffix {
            Some(suf) => attach(&t, suf),
            None => t,
        };
        sc.emit(i, we, &spoken, "foreign");
        return Some(we);
    }
    None
}

/// A Roman numeral after a capitalised name (`Mehmet II`, `Louis XIV`).
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
    sc.emit(i, end, r, "symbol");
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
        assert_eq!(v("3,5 kg"), "üç virgül beş kilogram");
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
        assert_eq!(v("1,5 milyon TL"), "bir virgül beş milyon lira");
        assert_eq!(v("%25'lik"), "yüzde yirmi beşlik");
        assert_eq!(v("%3,5"), "yüzde üç virgül beş");
        assert_eq!(v("12 km'den"), "on iki kilometreden");
        assert_eq!(v("5 kg'ye"), "beş kilograma");
        assert_eq!(v("90 km/sa"), "saatte doksan kilometre");
        assert_eq!(v("25 °C"), "yirmi beş derece");
    }

    #[test]
    fn telephones_addresses() {
        assert_eq!(
            v("0532 123 45 67"),
            "sıfır beş yüz otuz iki yüz yirmi üç kırk beş altmış yedi"
        );
        assert_eq!(
            v("+90 212 555 12 34"),
            "artı doksan iki yüz on iki beş yüz elli beş on iki otuz dört"
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
        assert_eq!(v("TBMM'de"), "tebememede");
        assert_eq!(v("ABD'ye"), "abedeye");
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
        assert_eq!(v("TDK'ye göre"), "tedekeye göre");
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
