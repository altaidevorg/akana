//! Named Entity Recognition (NER) for Turkish text.
//! Recognizes Person (PER), Location (LOC) and Organization (ORG) with a trained tagger
//! ([`model`]), and Date (DATE), Money (MONEY) and Percent (PERCENT) with patterns over the token
//! stream.

use crate::morphology::APOSTROPHES;
use crate::phonology::to_turkish_lower;
use crate::tokenization::{SentenceSegmenter, Token, TokenType, TurkishTokenizer};
use lazy_static::lazy_static;

mod model;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamedEntity {
    pub text: String,
    pub label: String,
    pub start: usize,
    pub end: usize,
}

lazy_static! {
    // Titles, and the words that end an organisation's or a place's name: lists the tagger's
    // features read.
    static ref TITLE_TRIGGERS: HashSet<&'static str> = {
        let mut set = HashSet::new();
        for &t in &[
            "sayın",
            "prof.",
            "dr.",
            "doç.",
            "yrd.",
            "av.",
            "müh.",
            "bakan",
            "başkan",
            "vali",
            "kaymakam",
            "rektör",
            "dekan",
            "müdür",
            "öğretmen",
            "cumhurbaşkanı",
            "başbakan",
            "general",
            "albay",
            "kaptan",
            "binbaşı",
            "bey",
            "hanım",
            "efendi",
            "paşa",
            // translated news writes a foreign person as Bay Gorbaçov, Bayan Coleman
            "bay",
            "bayan",
            "lord",
            "leydi",
            "sir",
        ] {
            set.insert(t);
        }
        set
    };
    static ref ORG_SUFFIXES: HashSet<&'static str> = {
        let mut set = HashSet::new();
        for &s in &[
            "bakanlığı",
            "müdürlüğü",
            "kurulu",
            "vakfı",
            "derneği",
            "kulübü",
            "üniversitesi",
            "fakültesi",
            "enstitüsü",
            "bankası",
            "holding",
            "şirketi",
            "partisi",
            "ajansı",
            "belediyesi",
            "federasyonu",
            "merkezi",
            "başkanlığı",
            "komisyonu",
            "hastanesi",
            "teşkilatı",
        ] {
            set.insert(s);
        }
        set
    };
    static ref LOC_SUFFIXES: HashSet<&'static str> = {
        let mut set = HashSet::new();
        for &s in &[
            "ili",
            "ilçesi",
            "köyü",
            "dağı",
            "nehri",
            "gölü",
            "denizi",
            "boğazı",
            "körfezi",
            "caddesi",
            "sokağı",
            "mahallesi",
            "meydanı",
            "bulvarı",
            "sarayı",
            "kalesi",
            "havalimanı",
            "havaalanı",
            "köprüsü",
        ] {
            set.insert(s);
        }
        set
    };
    static ref MONTHS: HashSet<&'static str> = {
        let mut set = HashSet::new();
        for &m in &[
            "ocak", "şubat", "mart", "nisan", "mayıs", "haziran", "temmuz", "ağustos", "eylül",
            "ekim", "kasım", "aralık",
        ] {
            set.insert(m);
        }
        set
    };
    static ref DAYS: HashSet<&'static str> = {
        let mut set = HashSet::new();
        for &d in &[
            "pazartesi",
            "salı",
            "çarşamba",
            "perşembe",
            "cuma",
            "cumartesi",
            "pazar",
        ] {
            set.insert(d);
        }
        set
    };
    static ref CURRENCY_KEYWORDS: HashSet<&'static str> = {
        let mut set = HashSet::new();
        for &c in &[
            "tl", "try", "lira", "dolar", "avro", "euro", "sterlin", "$", "€", "₺", "£",
        ] {
            set.insert(c);
        }
        set
    };
}

/// The words that end the name of a street, a square or a building, which the tagger leaves out
/// (Kızılay Meydanı); places of nature and administration it finds itself (Van Gölü, Ağrı Dağı).
const FACILITY_WORDS: [&str; 10] = [
    "caddesi",
    "sokağı",
    "mahallesi",
    "meydanı",
    "bulvarı",
    "sarayı",
    "kalesi",
    "havalimanı",
    "havaalanı",
    "köprüsü",
];

pub struct TurkishNER;

/// A number, date or time token without the suffixes the tokenizer keeps with it after an
/// apostrophe (2026'da, %50'si): an entity ends with the number.
fn without_number_suffix(tok: Token<'_>) -> Token<'_> {
    if matches!(
        tok.token_type,
        TokenType::Number | TokenType::Date | TokenType::Time
    ) {
        if let Some(i) = tok.text.find(['\'', '’']) {
            return Token::new(&tok.text[..i], tok.token_type, tok.start, tok.start + i);
        }
    }
    tok
}

/// Where a name ends in its word's text: before the apostrophe of its suffix, else after its
/// last letter or digit.
fn name_end(text: &str) -> usize {
    text.find(APOSTROPHES)
        .unwrap_or_else(|| text.trim_end_matches(|c: char| !c.is_alphanumeric()).len())
}

impl TurkishNER {
    /// Extracts named entities from a Turkish text string.
    pub fn extract_entities(text: &str) -> Vec<NamedEntity> {
        let mut entities = Vec::new();
        // Persons, places and organisations come from the trained tagger, sentence by sentence;
        // a name ends before the suffix after its apostrophe (Kaya'nın: Kaya).
        let mut names = Vec::new();
        for sentence in SentenceSegmenter::segment(text) {
            let offset = sentence.text.as_ptr() as usize - text.as_ptr() as usize;
            let tokens = TurkishTokenizer::tokenize(sentence.text);
            let words: Vec<&str> = tokens.iter().map(|t| t.text).collect();
            for (kind, a, b) in model::spans(&model::MODEL.tag(&words)) {
                let start = offset + tokens[a].start;
                let end = offset + tokens[b - 1].start + name_end(words[b - 1]);
                if end > start {
                    names.push(NamedEntity {
                        text: text[start..end].to_string(),
                        label: kind.to_string(),
                        start,
                        end,
                    });
                }
            }
        }
        // Amounts, percentages, dates, and streets and buildings from patterns, which a name the
        // tagger finds over them does not override (Cuma günü, 16 Ağustos 2026, Kızılay Meydanı).
        let tokens: Vec<Token> = TurkishTokenizer::tokenize(text)
            .into_iter()
            .map(without_number_suffix)
            .collect();
        let n = tokens.len();

        let mut i = 0;
        while i < n {
            let tok = &tokens[i];
            let clean_tok = tok
                .text
                .trim_matches(|c: char| !c.is_alphanumeric() && c != '\'');
            let lower_tok = to_turkish_lower(clean_tok);

            // 1. Money Pattern Matching (e.g. 500 TL, 100 dolar, 50€, $100)
            if tok.token_type == TokenType::Number {
                if i + 1 < n {
                    let next_clean = tokens[i + 1].text.trim_matches(|c: char| {
                        !c.is_alphanumeric() && c != '$' && c != '€' && c != '₺' && c != '£'
                    });
                    let next_lower = to_turkish_lower(next_clean);
                    if CURRENCY_KEYWORDS.contains(next_lower.as_str()) {
                        let matched_tokens: Vec<&str> =
                            tokens[i..=i + 1].iter().map(|t| t.text).collect();
                        entities.push(NamedEntity {
                            text: matched_tokens.join(" "),
                            label: "MONEY".to_string(),
                            start: tok.start,
                            end: tokens[i + 1].end,
                        });
                        i += 2;
                        continue;
                    }
                }
            } else if tok.text.starts_with('$')
                || tok.text.starts_with('€')
                || tok.text.starts_with('₺')
                || tok.text.starts_with('£')
            {
                entities.push(NamedEntity {
                    text: tok.text.to_string(),
                    label: "MONEY".to_string(),
                    start: tok.start,
                    end: tok.end,
                });
                i += 1;
                continue;
            }

            // 2. Percent Pattern Matching (e.g. %50, yüzde 25)
            if tok.text == "%" && i + 1 < n && tokens[i + 1].token_type == TokenType::Number {
                let matched_tokens: Vec<&str> = tokens[i..=i + 1].iter().map(|t| t.text).collect();
                entities.push(NamedEntity {
                    text: matched_tokens.join(""),
                    label: "PERCENT".to_string(),
                    start: tok.start,
                    end: tokens[i + 1].end,
                });
                i += 2;
                continue;
            } else if tok.text.starts_with('%')
                && tok.text.len() > 1
                && tok.text[1..].chars().any(|c| c.is_ascii_digit())
            {
                entities.push(NamedEntity {
                    text: tok.text.to_string(),
                    label: "PERCENT".to_string(),
                    start: tok.start,
                    end: tok.end,
                });
                i += 1;
                continue;
            } else if lower_tok == "yüzde"
                && i + 1 < n
                && tokens[i + 1].token_type == TokenType::Number
            {
                let matched_tokens: Vec<&str> = tokens[i..=i + 1].iter().map(|t| t.text).collect();
                entities.push(NamedEntity {
                    text: matched_tokens.join(" "),
                    label: "PERCENT".to_string(),
                    start: tok.start,
                    end: tokens[i + 1].end,
                });
                i += 2;
                continue;
            }

            // 3. Date Pattern Matching (e.g. 16 Ağustos 2026, Pazartesi günü)
            if tok.token_type == TokenType::Number && i + 1 < n {
                let next_tok = tokens[i + 1]
                    .text
                    .trim_matches(|c: char| !c.is_alphabetic());
                let next_lower = to_turkish_lower(next_tok);
                if MONTHS.contains(next_lower.as_str()) {
                    let mut step = 2;
                    if i + 2 < n && tokens[i + 2].token_type == TokenType::Number {
                        step = 3;
                    }
                    let matched_tokens: Vec<&str> =
                        tokens[i..i + step].iter().map(|t| t.text).collect();
                    let ent_text = matched_tokens.join(" ");
                    entities.push(NamedEntity {
                        text: ent_text,
                        label: "DATE".to_string(),
                        start: tok.start,
                        end: tokens[i + step - 1].end,
                    });
                    i += step;
                    continue;
                }
            }

            // Day pattern (Pazartesi günü)
            if DAYS.contains(lower_tok.as_str()) {
                let mut step = 1;
                if i + 1 < n && to_turkish_lower(tokens[i + 1].text) == "günü" {
                    step = 2;
                }
                let matched_tokens: Vec<&str> =
                    tokens[i..i + step].iter().map(|t| t.text).collect();
                let ent_text = matched_tokens.join(" ");
                entities.push(NamedEntity {
                    text: ent_text,
                    label: "DATE".to_string(),
                    start: tok.start,
                    end: tokens[i + step - 1].end,
                });
                i += step;
                continue;
            }

            // 4. Streets, squares and buildings (Kızılay Meydanı, Atatürk Bulvarı, İstanbul
            // Havalimanı'na), which the tagger leaves out: WikiNER, which trained it, keeps
            // facilities apart from places, and a tagger trained with them as places lost ORG F1.
            // The place starts with the name the tagger found right before the word (Atatürk
            // Bulvarı), or before a title there (Mustafa Kemal Paşa Caddesi, Ahmet Bey Sokağı),
            // else with the capitalised word before it, and ends before the suffix after its
            // apostrophe.
            let head = lower_tok.split(APOSTROPHES).next().unwrap_or_default();
            if i > 0 && FACILITY_WORDS.contains(&head) {
                let prev = &tokens[i - 1];
                let after_patterns = entities.last().map_or(0, |e: &NamedEntity| e.end);
                if prev.text.starts_with(char::is_uppercase)
                    && !prev.text.contains(APOSTROPHES)
                    && prev.start >= after_patterns
                {
                    let name_before = |t: &Token| {
                        let end = t.start + name_end(t.text);
                        names
                            .binary_search_by_key(&end, |e| e.end)
                            .ok()
                            .map(|k| names[k].start)
                            .filter(|&s| s >= after_patterns)
                    };
                    let title = TITLE_TRIGGERS.contains(to_turkish_lower(prev.text).as_str());
                    let start = name_before(prev)
                        .or_else(|| {
                            (title && i > 1)
                                .then(|| name_before(&tokens[i - 2]))
                                .flatten()
                        })
                        .unwrap_or(prev.start);
                    let end = tok.start + name_end(tok.text);
                    entities.push(NamedEntity {
                        text: text[start..end].to_string(),
                        label: "LOC".to_string(),
                        start,
                        end,
                    });
                }
            }

            i += 1;
        }

        // Both lists run in text order without overlaps, so one sweep finds the names that
        // overlap a pattern's entity.
        let mut k = 0;
        names.retain(|a| {
            while entities.get(k).is_some_and(|b| b.end <= a.start) {
                k += 1;
            }
            !entities.get(k).is_some_and(|b| b.start < a.end)
        });
        entities.append(&mut names);

        // Deduplicate and sort by starting offset
        entities.sort_by_key(|e| e.start);
        entities.dedup_by(|a, b| a.start == b.start && a.end == b.end);

        entities
    }
}
