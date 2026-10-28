//! High-accuracy Turkish De-asciifier powered by morphological candidate validation and phonetic heuristics.

use crate::morphology::TurkishMorphology;
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
        let mut result = String::with_capacity(text.len());
        let tokens: Vec<&str> = text.split_inclusive(|c: char| !c.is_alphabetic()).collect();

        for token in tokens {
            let alphabetic_part: String = token.chars().filter(|c| c.is_alphabetic()).collect();
            let non_alphabetic_part: String =
                token.chars().filter(|c| !c.is_alphabetic()).collect();

            if alphabetic_part.is_empty() {
                result.push_str(token);
                continue;
            }

            let restored = Self::deasciify_word(&alphabetic_part);
            result.push_str(&restored);
            result.push_str(&non_alphabetic_part);
        }

        result
    }

    /// Deasciifies a single word.
    pub fn deasciify_word(word: &str) -> String {
        if word.is_empty() {
            return String::new();
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

        // A word that already has Turkish letters and reads is as its writer meant it: olduğu
        // is no öldüğü, and the output of this function stays as it is (review 3 NORM3-02).
        let turkish_letter = |c: char| matches!(c, 'ç' | 'ğ' | 'ş' | 'ö' | 'ü' | 'ı');
        if lower.chars().any(turkish_letter) && !GLOBAL_MORPHOLOGY.analyze_standard(word).is_empty()
        {
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

        // Find candidate letter substitutions: c->ç, g->ğ, s->ş, o->ö, u->ü, i->ı
        let mut ambiguous_positions: Vec<(usize, Vec<char>)> = Vec::new();
        // A capital I typed without the Turkish keyboard may be İ: its lower case ı may be i
        // (KITAP, Istanbul; review 3 NORM3-02).
        let typed_i: Vec<bool> = word.chars().map(|c| c == 'I').collect();
        for (idx, ch) in lower.chars().enumerate() {
            match ch {
                'ı' if typed_i.get(idx) == Some(&true) => {
                    ambiguous_positions.push((idx, vec!['ı', 'i']))
                }
                'c' => ambiguous_positions.push((idx, vec!['c', 'ç'])),
                'g' => ambiguous_positions.push((idx, vec!['g', 'ğ'])),
                's' => ambiguous_positions.push((idx, vec!['s', 'ş'])),
                'o' => ambiguous_positions.push((idx, vec!['o', 'ö'])),
                'u' => ambiguous_positions.push((idx, vec!['u', 'ü'])),
                'i' => ambiguous_positions.push((idx, vec!['i', 'ı'])),
                _ => {}
            }
        }

        if ambiguous_positions.is_empty() {
            return word.to_string();
        }

        // Limit permutation search to at most 6 ambiguous chars (2^6 = 64 branches) to guarantee speed
        if ambiguous_positions.len() > 6 {
            ambiguous_positions.truncate(6);
        }

        let mut candidates: Vec<String> = vec![lower.clone()];

        for (idx, replacements) in &ambiguous_positions {
            let mut new_candidates = Vec::new();
            for cand in &candidates {
                let chars: Vec<char> = cand.chars().collect();
                for &rep in replacements {
                    let mut modified = chars.clone();
                    modified[*idx] = rep;
                    new_candidates.push(modified.into_iter().collect::<String>());
                }
            }
            candidates = new_candidates;
        }

        // Score candidates with morphological validation: the input is one of them, and a
        // tie keeps it.
        let score_of = |cand: &str, parses: &[crate::morphology::MorphParse]| {
            let diacritic_count = cand.chars().filter(|&c| turkish_letter(c)).count();
            // Higher score for candidate matching morphotactics with more proper Turkish characters.
            // A word's parse scores sum to its expected spelling factor (how cleanly it is
            // written), whatever the number of readings.
            let well_formed: f32 = parses.iter().map(|p| p.score).sum();
            1.0 + (diacritic_count as f32 * 0.2) + well_formed
        };
        let best = |judge: &dyn Fn(&str) -> Vec<crate::morphology::MorphParse>| {
            let mut best: Option<(f32, &String)> = None;
            for cand in &candidates {
                let parses = judge(cand);
                if parses.is_empty() {
                    continue;
                }
                let score = score_of(cand, &parses);
                if best
                    .is_none_or(|(b, w)| score > b || (score == b && *cand == lower && *w != lower))
                {
                    best = Some((score, cand));
                }
            }
            best.map(|(_, w)| w.clone())
        };
        let common = |w: &str| GLOBAL_MORPHOLOGY.analyze_standard(w);
        let as_written = |w: &str| {
            let mut parses = GLOBAL_MORPHOLOGY.analyze_standard(&cased(w));
            if is_upper && !is_all_upper {
                parses.retain(|p| {
                    p.secondary_pos != crate::morphology::pos::SecondaryPos::Abbreviation
                });
            }
            parses
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
