//! High-accuracy Turkish De-asciifier powered by morphological candidate validation and phonetic heuristics.

use crate::morphology::pronunciation::roman_value;
use crate::morphology::{PrimaryPos, TurkishMorphology, APOSTROPHES};
use crate::phonology::{to_turkish_lower, to_turkish_title, to_turkish_upper};
use lazy_static::lazy_static;

lazy_static! {
    static ref GLOBAL_MORPHOLOGY: TurkishMorphology = TurkishMorphology::new();
    static ref FREQUENT_DEASCII: std::collections::HashMap<&'static str, &'static str> = {
        let mut m = std::collections::HashMap::new();
        let entries = [
            ("cok", "çok"),
            ("icin", "için"),
            ("cunku", "çünkü"),
            ("eger", "eğer"),
            ("simdi", "şimdi"),
            ("hizli", "hızlı"),
            ("turkce", "türkçe"),
            ("turkiye", "türkiye"),
            ("guzel", "güzel"),
            ("kucuk", "küçük"),
            ("buyuk", "büyük"),
            ("ogrenci", "öğrenci"),
            ("ogretmen", "öğretmen"),
            ("saglik", "sağlık"),
            ("yasam", "yaşam"),
            ("cagdas", "çağdaş"),
            ("yazi", "yazı"),
            ("yazilim", "yazılım"),
            ("degil", "değil"),
            ("dunya", "dünya"),
            ("gun", "gün"),
            ("gunes", "güneş"),
            ("goz", "göz"),
            ("yurek", "yürek"),
            ("gonul", "gönül"),
            ("arkadas", "arkadaş"),
            ("kardes", "kardeş"),
            ("sehir", "şehir"),
            ("ulke", "ülke"),
            ("caliskan", "çalışkan"),
            ("calismak", "çalışmak"),
            ("calisiyor", "çalışıyor"),
        ];
        for (k, v) in entries {
            m.insert(k, v);
        }
        m
    };
}

pub struct TurkishDeasciifier;

impl TurkishDeasciifier {
    /// Deasciifies a word or text, converting ASCII representations back to Turkish letters.
    pub fn deasciify(text: &str) -> String {
        let letter = |c: char| c.is_alphabetic() && !APOSTROPHES.contains(&c);
        let mut result = String::with_capacity(text.len());
        let mut rest = text;
        while let Some(start) = rest.find(letter) {
            let (between, from) = rest.split_at(start);
            let end = from.find(|c: char| !letter(c)).unwrap_or(from.len());
            let (word, after) = from.split_at(end);
            result.push_str(between);
            let restored = match suffix_stem(&result, word) {
                Some(stem) => suffix_after(&stem, word),
                // A word before an at sign, or in lower case after a period or an at sign inside
                // a token, is part of an address or a file name and stays (Orbat.com,
                // ali@gmail, rapor.pdf).
                None if in_address(&result, word, after) => word.to_string(),
                None => {
                    question_particle(&result, word).unwrap_or_else(|| Self::deasciify_word(word))
                }
            };
            result.push_str(&restored);
            rest = after;
        }
        result.push_str(rest);
        result
    }

    /// Deasciifies a single word.
    pub fn deasciify_word(word: &str) -> String {
        if word.is_empty() {
            return String::new();
        }
        // A letter alone (a heading's A., the U of a statistic) and a Roman numeral in capitals
        // (II., XIX) are as written: no word tells which letter was meant (review GN-10).
        if word.chars().nth(1).is_none() || roman_value(word).is_some() {
            return word.to_string();
        }

        let is_upper = word.chars().next().is_some_and(|c| c.is_uppercase());
        let is_all_upper = word.len() > 1 && word.chars().all(|c| c.is_uppercase());

        let lower = to_turkish_lower(word);

        if let Some(&freq) = FREQUENT_DEASCII.get(lower.as_str()) {
            return if is_all_upper {
                to_turkish_upper(freq)
            } else if is_upper {
                to_turkish_title(freq)
            } else {
                freq.to_string()
            };
        }

        // A word that already has Turkish letters, İ among them, and reads, or is written in
        // capitals, is as its writer meant it: olduğu is no öldüğü, AİHM no AIHM, İSKİ no IŞKI,
        // and the output of this function stays as it is (review 3 NORM3-02, review 4
        // DOWN4-12). So is a word with a capital after a small letter (iPhone, McDonald), a
        // name as written; a word in capitals with its suffix (ISTANBUL'da) is not one.
        let turkish_letter = |c: char| matches!(c, 'ç' | 'ğ' | 'ş' | 'ö' | 'ü' | 'ı');
        // Read in the word as written: a capital I lowers to ı and says nothing (KITAP).
        let written_turkish = word
            .chars()
            .any(|c| turkish_letter(c) || matches!(c, 'Ç' | 'Ğ' | 'Ş' | 'Ö' | 'Ü' | 'İ'));
        if written_turkish && (is_all_upper || !GLOBAL_MORPHOLOGY.analyze_standard(word).is_empty())
        {
            return word.to_string();
        }
        let mut letters = word.chars();
        let camel = letters.next().is_some_and(|first| {
            letters
                .scan(first, |before, c| {
                    let lower_then_upper = before.is_lowercase() && c.is_uppercase();
                    *before = c;
                    Some(lower_then_upper)
                })
                .any(|x| x)
        });
        if camel {
            return word.to_string();
        }
        // So is a capitalised word that is a name as written (Ali, Michael, Chicago): its
        // twins with Turkish letters are other names or words (Alı, Mıchael).
        if is_upper
            && !is_all_upper
            && GLOBAL_MORPHOLOGY
                .analyze_standard(word)
                .iter()
                .any(|p| p.secondary_pos == crate::morphology::pos::SecondaryPos::ProperNoun)
        {
            return word.to_string();
        }

        // Candidates are judged in lower case, as common words: a capital at the start of a
        // sentence or in a heading says nothing, and Bu is no Bü (the abbreviation BÜ), Cocuk no
        // Çoçuk. Only when no candidate reads so is one judged as the word is written, a name
        // (Istanbul: İstanbul), but never as an abbreviation in title case (review 3 NORM3-02).
        let cased = |w: &str| {
            if is_all_upper {
                to_turkish_upper(w)
            } else if is_upper {
                to_turkish_title(w)
            } else {
                w.to_string()
            }
        };

        let candidates = spellings(word, &lower);
        if candidates.len() == 1 {
            return word.to_string();
        }
        // A spelling is judged by how likely the corpus prior makes it as a word, not by its
        // Turkish letters: konusu stays, kisi is kişi, not kışı ([`likeliest`]).
        let best = |judge: &dyn Fn(&str) -> Option<i32>| likeliest(&candidates, &lower, judge);
        let common = |w: &str| GLOBAL_MORPHOLOGY.spelling_log_prior(w, |_| true);
        let as_written = |w: &str| {
            GLOBAL_MORPHOLOGY.spelling_log_prior(&cased(w), |p| {
                !(is_upper
                    && !is_all_upper
                    && p.secondary_pos == crate::morphology::pos::SecondaryPos::Abbreviation)
            })
        };
        let best_word = best(&common)
            .or_else(|| {
                (is_upper || is_all_upper)
                    .then(|| best(&as_written))
                    .flatten()
            })
            .unwrap_or_else(|| lower.clone());

        // Reapply casing
        if is_all_upper {
            to_turkish_upper(&best_word)
        } else if is_upper {
            to_turkish_title(&best_word)
        } else {
            best_word
        }
    }
}

/// The lower-case spellings `lower` (the word `word` in lower case) may stand for, itself first:
/// each c, g, s, o, u and i may be its Turkish letter, and so may each ı written as a capital I
/// (KITAP, Istanbul; review 3 NORM3-02). At most 6 letters are tried (64 spellings).
fn spellings(word: &str, lower: &str) -> Vec<String> {
    let typed_i: Vec<bool> = word.chars().map(|c| c == 'I').collect();
    let mut candidates: Vec<String> = vec![lower.to_string()];
    let ambiguous = lower.chars().enumerate().filter_map(|(idx, ch)| {
        let twin = match ch {
            'ı' if typed_i.get(idx) == Some(&true) => 'i',
            'c' => 'ç',
            'g' => 'ğ',
            's' => 'ş',
            'o' => 'ö',
            'u' => 'ü',
            'i' => 'ı',
            _ => return None,
        };
        Some((idx, twin))
    });
    for (idx, twin) in ambiguous.take(6) {
        let twins: Vec<String> = candidates
            .iter()
            .map(|cand| {
                cand.chars()
                    .enumerate()
                    .map(|(i, c)| if i == idx { twin } else { c })
                    .collect()
            })
            .collect();
        candidates.extend(twins);
    }
    candidates
}

/// The likeliest of `candidates` under `judge` (a log prior in rank units; `None` for a spelling
/// that does not read), or `None` when none reads. The input `lower` is among them and wins a
/// tie. Where a bonus for each Turkish letter chose, 2.98 % of the words of correct court text
/// became other words (konusu konuşu, son sön, olup ölüp; review GN-10). A margin the input
/// would keep its place by was measured and is not used: from 400 to 3,232 rank units (0.5 to 4
/// nats) it kept 0.04 to 0.05 % more of the words of correct court text and restored 0.09 to
/// 0.89 % fewer of the words typed without Turkish letters.
fn likeliest(
    candidates: &[String],
    lower: &str,
    judge: &dyn Fn(&str) -> Option<i32>,
) -> Option<String> {
    let mut best: Option<(i32, &String)> = None;
    for cand in candidates {
        let Some(score) = judge(cand) else {
            continue;
        };
        if best.is_none_or(|(b, w)| score > b || (score == b && cand == lower && w != lower)) {
            best = Some((score, cand));
        }
    }
    best.map(|(_, w)| w.clone())
}

/// The word, restored, at the end of the text `before` (which `word` follows) that `word` is
/// the suffix of: `word` follows it after an apostrophe (Kanun'un, 1990'lı) or, in lower case,
/// after the period of an abbreviation in capitals (İİK.nun, TTK.'nun). The suffix of a numeral
/// or of an abbreviation is also written after a space (10 uncu, TTK nun, İİK’ nun), where it is
/// one when it reads so. `None` when `word` is a word of its own.
fn suffix_stem(before: &str, word: &str) -> Option<String> {
    let last = before.chars().next_back()?;
    let body = &before[..before.len() - last.len_utf8()];
    let (body, mark) = match last {
        c if APOSTROPHES.contains(&c) => match body.strip_suffix('.') {
            Some(body) => (body, ".'"),
            None => (body, "'"),
        },
        '.' => (body, "."),
        ' ' => (body.strip_suffix(APOSTROPHES).unwrap_or(body), " "),
        _ => return None,
    };
    let start = body
        .char_indices()
        .rev()
        .take_while(|&(_, c)| c.is_alphanumeric() && !APOSTROPHES.contains(&c))
        .last()?
        .0;
    let stem = &body[start..];
    let abbreviation = stem.chars().nth(1).is_some() && stem.chars().all(char::is_uppercase);
    let numeral = stem.chars().all(|c| c.is_ascii_digit());
    let lower = word.starts_with(char::is_lowercase);
    let of_stem = match mark {
        "'" => true,
        "." => abbreviation && lower,
        ".'" => abbreviation,
        _ => {
            (numeral || abbreviation)
                && lower
                && GLOBAL_MORPHOLOGY
                    .spelling_log_prior(&format!("{stem}'{word}"), |_| true)
                    .is_some()
        }
    };
    of_stem.then(|| stem.to_string())
}

/// Whether `word` comes before an at sign (`after` follows it), or, in lower case, after a
/// period or an at sign that follows a letter or a digit at the end of the text `before`.
fn in_address(before: &str, word: &str, after: &str) -> bool {
    let mut marks = before.chars().rev();
    after.starts_with('@')
        || (word.starts_with(char::is_lowercase)
            && matches!(marks.next(), Some('.' | '@'))
            && marks.next().is_some_and(char::is_alphanumeric))
}

/// The question particle written apart (mi, misin, mudur) and typed without Turkish letters,
/// with the vowel of the word before it, as a suffix takes it (değil mi, var mı, bu mudur): judged
/// alone, mi was the likelier mı and mudur the noun müdür (review GN-10). `None` when `word` is
/// no such particle or no word and space come before it, at the end of the text `before`.
fn question_particle(before: &str, word: &str) -> Option<String> {
    let typed_turkish = |c: char| "çğışöüÇĞŞÖÜİ".contains(c);
    let body = before.trim_end_matches(char::is_whitespace);
    if body.len() == before.len() || word.chars().any(typed_turkish) {
        return None;
    }
    let start = body
        .char_indices()
        .rev()
        .take_while(|&(_, c)| c.is_alphabetic())
        .last()?
        .0;
    let vowel = to_turkish_lower(&body[start..])
        .chars()
        .rev()
        .find_map(|c| match c {
            'a' | 'ı' => Some('ı'),
            'e' | 'i' => Some('i'),
            'o' | 'u' => Some('u'),
            'ö' | 'ü' => Some('ü'),
            _ => None,
        })?;
    let lower = to_turkish_lower(word);
    if !lower.starts_with('m') {
        return None;
    }
    // Only the vowel's own twin may stand for it: i (or a capital I) for i and ı, u for u and ü.
    let particle = lower
        .chars()
        .map(|c| match c {
            'i' | 'ı' if matches!(vowel, 'i' | 'ı') => Some(vowel),
            'u' if matches!(vowel, 'u' | 'ü') => Some(vowel),
            'i' | 'ı' | 'u' => None,
            c => Some(c),
        })
        .collect::<Option<String>>()?;
    GLOBAL_MORPHOLOGY
        .analyze_standard(&particle)
        .iter()
        .any(|p| p.pos == PrimaryPos::Q)
        .then(|| {
            if word.chars().all(char::is_uppercase) {
                to_turkish_upper(&particle)
            } else if word.starts_with(char::is_uppercase) {
                to_turkish_title(&particle)
            } else {
                particle
            }
        })
}

/// A suffix written apart from its word `stem` takes the spelling that makes the word read, or
/// stays as written: alone, nun would be judged a word of its own, and Kanun'un became
/// Kanun'ün (review GN-10).
fn suffix_after(stem: &str, suffix: &str) -> String {
    let lower = to_turkish_lower(suffix);
    let capitals = suffix.chars().all(char::is_uppercase);
    let cased = |w: &str| {
        if capitals {
            to_turkish_upper(w)
        } else {
            w.to_string()
        }
    };
    let judge =
        |w: &str| GLOBAL_MORPHOLOGY.spelling_log_prior(&format!("{stem}'{}", cased(w)), |_| true);
    match likeliest(&spellings(suffix, &lower), &lower, &judge) {
        Some(w) if w != lower => cased(&w),
        _ => suffix.to_string(),
    }
}
