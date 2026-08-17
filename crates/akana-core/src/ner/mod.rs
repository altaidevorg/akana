//! Named Entity Recognition (NER) for Turkish text.
//! Recognizes Person (PER), Location (LOC), Organization (ORG), Date (DATE), Money (MONEY), and Percent (PERCENT).
//! Accelerated with single-pass zero-copy token stream matching.

use std::collections::HashSet;
use serde::{Deserialize, Serialize};
use lazy_static::lazy_static;
use crate::tokenization::{TurkishTokenizer, TokenType};
use crate::phonology::to_turkish_lower;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamedEntity {
    pub text: String,
    pub label: String,
    pub start: usize,
    pub end: usize,
}

lazy_static! {
    static ref PERSON_NAMES: HashSet<String> = {
        let mut set = HashSet::new();
        let lex_str = include_str!("../morphology/zemberek_lexicon.txt");
        for line in lex_str.lines() {
            if line.contains("ProperNoun") {
                if let Some(first_word) = line.split_whitespace().next() {
                    set.insert(to_turkish_lower(first_word));
                }
            }
        }
        set
    };

    static ref TITLE_TRIGGERS: HashSet<&'static str> = {
        let mut set = HashSet::new();
        for &t in &["sayın", "prof.", "dr.", "doç.", "yrd.", "av.", "müh.", "bakan", "başkan",
                    "vali", "kaymakam", "rektör", "dekan", "müdür", "öğretmen", "cumhurbaşkanı",
                    "başbakan", "general", "albay", "kaptan", "binbaşı", "bey", "hanım", "efendi", "paşa"] {
            set.insert(t);
        }
        set
    };

    static ref ORG_SUFFIXES: HashSet<&'static str> = {
        let mut set = HashSet::new();
        for &s in &["bakanlığı", "müdürlüğü", "kurulu", "vakfı", "derneği", "kulübü",
                    "üniversitesi", "fakültesi", "enstitüsü", "bankası", "holding",
                    "şirketi", "partisi", "ajansı", "belediyesi", "federasyonu",
                    "merkezi", "başkanlığı", "komisyonu", "hastanesi", "teşkilatı"] {
            set.insert(s);
        }
        set
    };

    static ref LOC_SUFFIXES: HashSet<&'static str> = {
        let mut set = HashSet::new();
        for &s in &["ili", "ilçesi", "köyü", "dağı", "nehri", "gölü", "denizi",
                    "boğazı", "körfezi", "caddesi", "sokağı", "mahallesi", "meydanı",
                    "bulvarı", "sarayı", "kalesi", "havalimanı", "havaalanı", "köprüsü"] {
            set.insert(s);
        }
        set
    };

    static ref MONTHS: HashSet<&'static str> = {
        let mut set = HashSet::new();
        for &m in &["ocak", "şubat", "mart", "nisan", "mayıs", "haziran",
                    "temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık"] {
            set.insert(m);
        }
        set
    };

    static ref DAYS: HashSet<&'static str> = {
        let mut set = HashSet::new();
        for &d in &["pazartesi", "salı", "çarşamba", "perşembe", "cuma", "cumartesi", "pazar"] {
            set.insert(d);
        }
        set
    };

    static ref CURRENCY_KEYWORDS: HashSet<&'static str> = {
        let mut set = HashSet::new();
        for &c in &["tl", "try", "lira", "dolar", "avro", "euro", "sterlin", "$", "€", "₺", "£"] {
            set.insert(c);
        }
        set
    };
}

pub struct TurkishNER;

impl TurkishNER {
    /// Extracts named entities from a Turkish text string.
    pub fn extract_entities(text: &str) -> Vec<NamedEntity> {
        let mut entities = Vec::new();
        let tokens = TurkishTokenizer::tokenize(text);
        let n = tokens.len();

        let mut i = 0;
        while i < n {
            let tok = &tokens[i];
            let clean_tok = tok.text.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'');
            let lower_tok = to_turkish_lower(clean_tok);

            // 1. Money Pattern Matching (e.g. 500 TL, 100 dolar, 50€, $100)
            if tok.token_type == TokenType::Number {
                if i + 1 < n {
                    let next_clean = tokens[i + 1].text.trim_matches(|c: char| !c.is_alphanumeric() && c != '$' && c != '€' && c != '₺' && c != '£');
                    let next_lower = to_turkish_lower(next_clean);
                    if CURRENCY_KEYWORDS.contains(next_lower.as_str()) {
                        let matched_tokens: Vec<&str> = tokens[i..=i+1].iter().map(|t| t.text).collect();
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
            } else if tok.text.starts_with('$') || tok.text.starts_with('€') || tok.text.starts_with('₺') || tok.text.starts_with('£') {
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
                let matched_tokens: Vec<&str> = tokens[i..=i+1].iter().map(|t| t.text).collect();
                entities.push(NamedEntity {
                    text: matched_tokens.join(""),
                    label: "PERCENT".to_string(),
                    start: tok.start,
                    end: tokens[i + 1].end,
                });
                i += 2;
                continue;
            } else if tok.text.starts_with('%') && tok.text.len() > 1 && tok.text[1..].chars().any(|c| c.is_ascii_digit()) {
                entities.push(NamedEntity {
                    text: tok.text.to_string(),
                    label: "PERCENT".to_string(),
                    start: tok.start,
                    end: tok.end,
                });
                i += 1;
                continue;
            } else if lower_tok == "yüzde" && i + 1 < n && tokens[i + 1].token_type == TokenType::Number {
                let matched_tokens: Vec<&str> = tokens[i..=i+1].iter().map(|t| t.text).collect();
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
                let next_tok = tokens[i + 1].text.trim_matches(|c: char| !c.is_alphabetic());
                let next_lower = to_turkish_lower(next_tok);
                if MONTHS.contains(next_lower.as_str()) {
                    let mut step = 2;
                    if i + 2 < n && tokens[i + 2].token_type == TokenType::Number {
                        step = 3;
                    }
                    let matched_tokens: Vec<&str> = tokens[i..i+step].iter().map(|t| t.text).collect();
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
                let matched_tokens: Vec<&str> = tokens[i..i+step].iter().map(|t| t.text).collect();
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

            // 4. Multi-token Organization Pattern (Capitalized sequence ending with ORG suffix)
            if clean_tok.chars().next().map_or(false, |c| c.is_uppercase()) {
                let mut j = i;
                let mut org_matched = false;
                while j < n {
                    let cur_clean = tokens[j].text.trim_matches(|c: char| !c.is_alphanumeric());
                    let cur_lower = to_turkish_lower(cur_clean);
                    if ORG_SUFFIXES.contains(cur_lower.as_str()) {
                        let matched_tokens: Vec<&str> = tokens[i..=j].iter().map(|t| t.text).collect();
                        entities.push(NamedEntity {
                            text: matched_tokens.join(" "),
                            label: "ORG".to_string(),
                            start: tokens[i].start,
                            end: tokens[j].end,
                        });
                        i = j + 1;
                        org_matched = true;
                        break;
                    }
                    if !cur_clean.chars().next().map_or(false, |c| c.is_uppercase()) && cur_clean != "ve" && cur_clean != "ile" {
                        break;
                    }
                    j += 1;
                }
                if org_matched {
                    continue;
                }

                // 5. Multi-token Location Pattern (Capitalized sequence ending with LOC suffix)
                let mut loc_matched = false;
                if i + 1 < n {
                    let next_clean = tokens[i + 1].text.trim_matches(|c: char| !c.is_alphanumeric());
                    let next_lower = to_turkish_lower(next_clean);
                    if LOC_SUFFIXES.contains(next_lower.as_str()) {
                        let matched_tokens: Vec<&str> = tokens[i..=i+1].iter().map(|t| t.text).collect();
                        entities.push(NamedEntity {
                            text: matched_tokens.join(" "),
                            label: "LOC".to_string(),
                            start: tok.start,
                            end: tokens[i + 1].end,
                        });
                        i += 2;
                        loc_matched = true;
                    }
                }
                if loc_matched {
                    continue;
                }

                // 6. Person Pattern (Title trigger + Capitalized Word, or in PERSON_NAMES)
                let is_person = PERSON_NAMES.contains(lower_tok.as_str());
                let preceded_by_title = i > 0 && TITLE_TRIGGERS.contains(to_turkish_lower(tokens[i - 1].text.trim_matches(|c: char| !c.is_alphabetic())).as_str());
                let followed_by_title = i + 1 < n && TITLE_TRIGGERS.contains(to_turkish_lower(tokens[i + 1].text.trim_matches(|c: char| !c.is_alphabetic())).as_str());

                if is_person || preceded_by_title || followed_by_title {
                    let mut j = i + 1;
                    while j < n {
                        let next_clean = tokens[j].text.trim_matches(|c: char| !c.is_alphabetic());
                        if !next_clean.is_empty() && next_clean.chars().next().unwrap().is_uppercase() && !ORG_SUFFIXES.contains(to_turkish_lower(next_clean).as_str()) {
                            j += 1;
                        } else {
                            break;
                        }
                    }

                    let matched_tokens: Vec<&str> = tokens[i..j].iter().map(|t| t.text).collect();
                    entities.push(NamedEntity {
                        text: matched_tokens.join(" "),
                        label: "PER".to_string(),
                        start: tok.start,
                        end: tokens[j - 1].end,
                    });
                    i = j;
                    continue;
                }
            }

            i += 1;
        }

        // Deduplicate and sort by starting offset
        entities.sort_by_key(|e| e.start);
        entities.dedup_by(|a, b| a.start == b.start && a.end == b.end);

        entities
    }
}
