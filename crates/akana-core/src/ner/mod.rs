//! Named Entity Recognition (NER) for Turkish text.
//! Recognizes Person (PER), Location (LOC), Organization (ORG), Date (DATE), Money (MONEY), and Percent (PERCENT).
//! Accelerated with single-pass zero-copy token stream matching.

use crate::morphology::{rank_units, SecondaryPos, TurkishMorphology, APOSTROPHES};
use crate::phonology::to_turkish_lower;
use crate::pii::{is_person_name, is_place_name};
use crate::tokenization::{Token, TokenType, TurkishTokenizer};
use lazy_static::lazy_static;
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
    /// The analyzer (its lexicon is shared), to tell a name that is also a common word.
    static ref MORPHOLOGY: TurkishMorphology = TurkishMorphology::new();
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

/// Where the organisation scan from token `i` stops: the first token at or after `i` that is
/// an ORG suffix (`true`), or that is neither capitalised nor `ve`/`ile` (`false`; also when
/// the tokens run out, with `tokens.len()`).
fn org_scan_stop(tokens: &[Token], i: usize) -> (usize, bool) {
    for (j, tok) in tokens.iter().enumerate().skip(i) {
        let cur_clean = tok.text.trim_matches(|c: char| !c.is_alphanumeric());
        if ORG_SUFFIXES.contains(to_turkish_lower(cur_clean).as_str()) {
            return (j, true);
        }
        if !cur_clean.chars().next().is_some_and(|c| c.is_uppercase())
            && cur_clean != "ve"
            && cur_clean != "ile"
        {
            return (j, false);
        }
    }
    (tokens.len(), false)
}

/// A token's word without the punctuation around it (an apostrophe kept).
fn clean(text: &str) -> &str {
    text.trim_matches(|c: char| !c.is_alphanumeric() && !APOSTROPHES.contains(&c))
}

/// A word's part before an apostrophe and whether a suffix follows it (Kaya'nın: Kaya, true).
fn name_part(word: &str) -> (&str, bool) {
    match word.find(APOSTROPHES) {
        Some(i) => (&word[..i], true),
        None => (word, false),
    }
}

/// Where a name ends in its token's text: before the apostrophe of its suffix, else after its
/// last letter.
fn name_end(text: &str) -> usize {
    text.find(APOSTROPHES)
        .unwrap_or_else(|| text.trim_end_matches(|c: char| !c.is_alphanumeric()).len())
}

/// Whether a token can go on a person's name: capitalised, and no place, organisation suffix or
/// nationality that is no name (Ahmet Yılmaz Ankara'da: the name ends before Ankara).
fn name_like(text: &str) -> bool {
    let (word, _) = name_part(clean(text));
    let lower = to_turkish_lower(word);
    word.chars().next().is_some_and(char::is_uppercase)
        && !ORG_SUFFIXES.contains(lower.as_str())
        && (is_person_name(&lower) || !is_place_name(&lower))
}

/// Whether a listed name at a sentence's start, with no other evidence, is a person's: one that is
/// no common word is; one before the head of a compound (Kaya parçaları, Barış antlaşması: P3sg)
/// is the word; otherwise it is a person's where it is rare as a word (Kerem, Hakan), not where it
/// is frequent (Güzel, Demir, Tatlı): the corpus prior of its likeliest common reading is under
/// [`RARE_WORD_LOG_PRIOR`].
fn person_at_start(base: &str, next: Option<&str>) -> bool {
    let Some(word) = MORPHOLOGY.spelling_log_prior(base, |p| {
        !matches!(
            p.secondary_pos,
            SecondaryPos::ProperNoun | SecondaryPos::Abbreviation
        )
    }) else {
        return true;
    };
    let compound_head = next.is_some_and(|t| {
        let (w, _) = name_part(clean(t));
        w.chars().next().is_some_and(char::is_lowercase)
            && MORPHOLOGY
                .analyze(w)
                .first()
                .is_some_and(|p| p.morpheme_tags.iter().any(|m| m == "P3sg"))
    });
    !compound_head && word < RARE_WORD_LOG_PRIOR
}

/// The corpus log prior under which a common word is rare enough to be a name at a sentence's
/// start (-14 nats): chosen on the UD Turkish train and dev sentences, from -8 to -16 nats, by
/// the F1 of persons at a sentence's start (0.889; -8 nats gave 0.720, -11 0.835, -16 0.877).
const RARE_WORD_LOG_PRIOR: i32 = rank_units(-14.0);

/// Whether token `i` begins a sentence: the text's first word, or the first after a period, an
/// exclamation or question mark, an ellipsis or a line break, opening quotes and brackets aside.
fn sentence_start(text: &str, tokens: &[Token], i: usize) -> bool {
    let mut k = i;
    while k > 0 {
        let prev = &tokens[k - 1];
        if text
            .get(prev.end..tokens[k].start)
            .is_some_and(|gap| gap.contains('\n'))
        {
            return true;
        }
        if prev.text.chars().all(|c| {
            matches!(
                c,
                '"' | '\'' | '(' | '[' | '«' | '“' | '‘' | '-' | '–' | '—'
            )
        }) {
            k -= 1;
            continue;
        }
        return prev.text.ends_with(['.', '!', '?', '…']);
    }
    true
}

impl TurkishNER {
    /// Extracts named entities from a Turkish text string.
    pub fn extract_entities(text: &str) -> Vec<NamedEntity> {
        let mut entities = Vec::new();
        let tokens: Vec<Token> = TurkishTokenizer::tokenize(text)
            .into_iter()
            .map(without_number_suffix)
            .collect();
        let n = tokens.len();
        // Last result of `org_scan_stop`: (start, stop token, whether it is an ORG suffix).
        let mut org_stop: Option<(usize, usize, bool)> = None;
        // The words of the full names found so far (Mehmet Yılmaz), each of which names the
        // person alone later (Yılmaz, açıklamasında ...).
        let mut named: HashSet<String> = HashSet::new();

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

            // 4. Multi-token Organization Pattern (Capitalized sequence ending with ORG suffix)
            if clean_tok.chars().next().is_some_and(|c| c.is_uppercase()) {
                // The scan from `i` stops at the same token for every start before it, so that
                // token is looked up once per run: walking to the end of the capitalised run
                // from each of its tokens made 8,000 capitalised words take 2.5 s.
                let (j, is_suffix) = match org_stop {
                    Some((from, j, is_suffix)) if from <= i && i <= j => (j, is_suffix),
                    _ => {
                        let (j, is_suffix) = org_scan_stop(&tokens, i);
                        org_stop = Some((i, j, is_suffix));
                        (j, is_suffix)
                    }
                };
                if is_suffix {
                    let matched_tokens: Vec<&str> = tokens[i..=j].iter().map(|t| t.text).collect();
                    entities.push(NamedEntity {
                        text: matched_tokens.join(" "),
                        label: "ORG".to_string(),
                        start: tokens[i].start,
                        end: tokens[j].end,
                    });
                    i = j + 1;
                    continue;
                }

                // 5. Multi-token Location Pattern (Capitalized sequence ending with LOC suffix)
                let mut loc_matched = false;
                if i + 1 < n {
                    let next_clean = tokens[i + 1]
                        .text
                        .trim_matches(|c: char| !c.is_alphanumeric());
                    let next_lower = to_turkish_lower(next_clean);
                    if LOC_SUFFIXES.contains(next_lower.as_str()) {
                        let matched_tokens: Vec<&str> =
                            tokens[i..=i + 1].iter().map(|t| t.text).collect();
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

                // 6. Persons and places. A person's name is a given name or surname of the PII
                // lexicon, a place a province, a district, or a country, region or city abroad;
                // the lexicon's other proper nouns are neither (Ankara and Almanya were persons).
                // A name is looked up without the suffix after its apostrophe (Kaya'nın), and its
                // entity ends there, as a number's does.
                let (name, suffixed) = name_part(clean_tok);
                let base = to_turkish_lower(name);
                let title_beside = |k: Option<usize>| {
                    k.and_then(|k| tokens.get(k)).is_some_and(|t| {
                        TITLE_TRIGGERS.contains(
                            to_turkish_lower(t.text.trim_matches(|c: char| !c.is_alphabetic()))
                                .as_str(),
                        )
                    })
                };
                let title = title_beside(i.checked_sub(1)) || title_beside(Some(i + 1));
                let next_name = !suffixed && tokens.get(i + 1).is_some_and(|t| name_like(t.text));
                let place = is_place_name(&base);
                // Inside a sentence the capital says a listed name is one. At a sentence's start
                // it says nothing: a listed name that may be a common word there is a person with
                // a suffix after an apostrophe, before another name, beside a title or once the
                // text has named the person in full, and otherwise only when it is rare as a word
                // and does not modify the head of a compound ([`person_at_start`]). A place that
                // is also a name (Aydın, Fatih) is a person only by the same evidence.
                let strong = title || next_name || named.contains(&base);
                let is_person = (is_person_name(&base)
                    && (strong
                        || (!place
                            && (suffixed
                                || !sentence_start(text, &tokens, i)
                                || person_at_start(&base, tokens.get(i + 1).map(|t| t.text))))))
                    || title;

                if is_person {
                    let mut j = i + 1;
                    let mut last_suffixed = suffixed;
                    while j < n && !last_suffixed && name_like(tokens[j].text) {
                        last_suffixed = name_part(clean(tokens[j].text)).1;
                        j += 1;
                    }
                    let last = &tokens[j - 1];
                    let end = last.start + name_end(last.text);
                    if j - i > 1 {
                        named.extend(
                            tokens[i..j]
                                .iter()
                                .map(|t| to_turkish_lower(name_part(clean(t.text)).0)),
                        );
                    }
                    entities.push(NamedEntity {
                        text: text[tok.start..end].to_string(),
                        label: "PER".to_string(),
                        start: tok.start,
                        end,
                    });
                    i = j;
                    continue;
                }
                if place {
                    let end = tok.start + name_end(tok.text);
                    entities.push(NamedEntity {
                        text: text[tok.start..end].to_string(),
                        label: "LOC".to_string(),
                        start: tok.start,
                        end,
                    });
                    i += 1;
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
