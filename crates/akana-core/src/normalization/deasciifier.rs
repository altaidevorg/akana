//! High-accuracy Turkish De-asciifier powered by morphological candidate validation and phonetic heuristics.

use crate::morphology::TurkishMorphology;
use crate::phonology::{to_turkish_lower, to_turkish_upper, to_turkish_title};
use lazy_static::lazy_static;

lazy_static! {
    static ref GLOBAL_MORPHOLOGY: TurkishMorphology = TurkishMorphology::new();

    static ref FREQUENT_DEASCII: std::collections::HashMap<&'static str, &'static str> = {
        let mut m = std::collections::HashMap::new();
        let entries = [
            ("cok", "çok"), ("icin", "için"), ("cunku", "çünkü"), ("eger", "eğer"),
            ("simdi", "şimdi"), ("hizli", "hızlı"), ("turkce", "türkçe"), ("turkiye", "türkiye"),
            ("guzel", "güzel"), ("kucuk", "küçük"), ("buyuk", "büyük"), ("ogrenci", "öğrenci"),
            ("ogretmen", "öğretmen"), ("saglik", "sağlık"), ("yasam", "yaşam"), ("cagdas", "çağdaş"),
            ("yazi", "yazı"), ("yazilim", "yazılım"), ("degil", "değil"), ("dunya", "dünya"),
            ("gun", "gün"), ("gunes", "güneş"), ("goz", "göz"), ("yurek", "yürek"),
            ("gonul", "gönül"), ("arkadas", "arkadaş"), ("kardes", "kardeş"), ("sehir", "şehir"),
            ("ulke", "ülke"), ("caliskan", "çalışkan"), ("calismak", "çalışmak"), ("calisiyor", "çalışıyor"),
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
            let non_alphabetic_part: String = token.chars().filter(|c| !c.is_alphabetic()).collect();

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

        let is_upper = word.chars().next().map_or(false, |c| c.is_uppercase());
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

        // Check if the word as-is is already a valid Turkish word
        let direct_parses = GLOBAL_MORPHOLOGY.analyze(&lower);
        
        // Find candidate letter substitutions: c->ç, g->ğ, s->ş, o->ö, u->ü, i->ı
        let mut ambiguous_positions: Vec<(usize, Vec<char>)> = Vec::new();
        for (idx, ch) in lower.chars().enumerate() {
            match ch {
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

        // Score candidates with morphological validation
        let mut best_word = lower.clone();
        let mut best_score = if direct_parses.is_empty() { -1.0 } else { 0.5 };

        for cand in candidates {
            if cand == lower && !direct_parses.is_empty() {
                continue;
            }
            let parses = GLOBAL_MORPHOLOGY.analyze(&cand);
            if !parses.is_empty() {
                let diacritic_count = cand.chars().filter(|&c| matches!(c, 'ç' | 'ğ' | 'ş' | 'ö' | 'ü' | 'ı')).count();
                // Higher score for candidate matching morphotactics with more proper Turkish characters
                let score = 1.0 + (diacritic_count as f32 * 0.2) + parses[0].score;
                if score > best_score {
                    best_score = score;
                    best_word = cand;
                }
            }
        }

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
