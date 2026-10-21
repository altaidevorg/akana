//! Fast, zero-copy, rule-based Turkish tokenizer with SIMD and character-level fast dispatch.

use lazy_static::lazy_static;
use regex::Regex;
use serde::{Deserialize, Serialize};

/// Longest possible email address (RFC 5321 path limit), in bytes.
const MAX_EMAIL_LEN: usize = 254;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenType {
    Word,
    ProperNounWithApostrophe,
    Number,
    Punctuation,
    Url,
    Email,
    Hashtag,
    Mention,
    Abbreviation,
    Time,
    Date,
    Symbol,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Token<'a> {
    pub text: &'a str,
    pub token_type: TokenType,
    pub start: usize,
    pub end: usize,
}

impl<'a> Token<'a> {
    pub fn new(text: &'a str, token_type: TokenType, start: usize, end: usize) -> Self {
        Self {
            text,
            token_type,
            start,
            end,
        }
    }
}

lazy_static! {
    /// Turkish abbreviations that usually end with a period
    static ref ABBREVIATIONS: std::collections::HashSet<&'static str> = {
        let mut set = std::collections::HashSet::new();
        let abbrs = [
            "prof.", "dr.", "doç.", "yrd.", "av.", "alb.", "bknz.", "bkz.", "cad.",
            "co.", "corp.", "çev.", "dak.", "der.", "dz.", "ecz.", "fak.", "gen.",
            "haz.", "hrk.", "hz.", "iö.", "is.", "ist.", "itb.", "ith.", "kd.",
            "kr.", "kur.", "kuv.", "ltd.", "mah.", "madr.", "mak.", "müh.", "müteh.",
            "no.", "num.", "öğr.", "ör.", "pa.", "par.", "ped.", "pk.", "s.",
            "say.", "sf.", "sn.", "soc.", "sok.", "st.", "şb.", "şer.", "şrk.",
            "tbmm.", "tc.", "tck.", "tel.", "ter.", "tes.", "tic.", "tl.", "tug.",
            "tüm.", "üniv.", "uzm.", "vb.", "vd.", "vol.", "vs.", "yy.", "yard.",
            "yön.", "yük.", "yur.", "zool.", "km.", "m.", "cm.", "mm.", "kg.", "gr.",
            "mad.", "madde.", "vol.", "volüm.", "bk.", "sf.", "sy.", "krş."
        ];
        for a in abbrs {
            set.insert(a);
        }
        set
    };

    static ref URL_REGEX: Regex = Regex::new(r"^(https?://[^\s/$.?#].[^\s]*|www\.[^\s/$.?#].[^\s]*)").unwrap();
    static ref EMAIL_REGEX: Regex = Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap();
    static ref HASHTAG_REGEX: Regex = Regex::new(r"^#[a-zA-Z0-9_çğıöşüÇĞİÖŞÜ]+").unwrap();
    static ref MENTION_REGEX: Regex = Regex::new(r"^@[a-zA-Z0-9_çğıöşüÇĞİÖŞÜ]+").unwrap();
    static ref TIME_REGEX: Regex = Regex::new(r"^\d{1,2}[:.]\d{2}([:.]\d{2})?").unwrap();
    static ref DATE_REGEX: Regex = Regex::new(r"^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}").unwrap();
    static ref NUMBER_REGEX: Regex = Regex::new(r"^[-+]?(?:\d+(?:[.,]\d+)*)%?").unwrap();
}

pub struct TurkishTokenizer;

impl TurkishTokenizer {
    /// Tokenizes the input text into an array of zero-copy `Token` references.
    pub fn tokenize<'a>(text: &'a str) -> Vec<Token<'a>> {
        Self::tokens(text).collect()
    }

    /// The tokens of [`tokenize`](Self::tokenize), produced one at a time, so that a caller
    /// that needs only the start of a long text does not tokenize all of it.
    pub(crate) fn tokens(text: &str) -> Tokens<'_> {
        Tokens { text, idx: 0 }
    }

    /// Fast tokenization returning only string slices.
    pub fn tokenize_words(text: &str) -> Vec<&str> {
        Self::tokenize(text).into_iter().map(|t| t.text).collect()
    }
}

/// Iterator over the tokens of a text; see [`TurkishTokenizer::tokens`].
pub(crate) struct Tokens<'a> {
    text: &'a str,
    idx: usize,
}

impl<'a> Iterator for Tokens<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Token<'a>> {
        let text = self.text;
        let len = text.len();

        while self.idx < len {
            let idx = self.idx;
            let remaining = &text[idx..];
            let first_char = remaining.chars().next().unwrap();

            // Skip whitespace fast
            if first_char.is_whitespace() {
                self.idx += first_char.len_utf8();
                continue;
            }

            let start_idx = idx;

            // 1. Hashtag (#)
            if first_char == '#' {
                if let Some(mat) = HASHTAG_REGEX.find(remaining) {
                    let token_text = mat.as_str();
                    self.idx = start_idx + token_text.len();
                    return Some(Token::new(
                        token_text,
                        TokenType::Hashtag,
                        start_idx,
                        self.idx,
                    ));
                }
            }

            // 2. Mention (@)
            if first_char == '@' {
                if let Some(mat) = MENTION_REGEX.find(remaining) {
                    let token_text = mat.as_str();
                    self.idx = start_idx + token_text.len();
                    return Some(Token::new(
                        token_text,
                        TokenType::Mention,
                        start_idx,
                        self.idx,
                    ));
                }
            }

            // 3. URL (http, https, www)
            if first_char == 'h' || first_char == 'H' || first_char == 'w' || first_char == 'W' {
                if let Some(mat) = URL_REGEX.find(remaining) {
                    let token_text = mat.as_str();
                    self.idx = start_idx + token_text.len();
                    return Some(Token::new(token_text, TokenType::Url, start_idx, self.idx));
                }
            }

            // 4. Digits, Numbers, Dates, Times (+/- numbers)
            if first_char.is_ascii_digit() || first_char == '+' || first_char == '-' {
                if let Some(mat) = DATE_REGEX.find(remaining) {
                    let token_text = mat.as_str();
                    self.idx = start_idx + token_text.len();
                    return Some(Token::new(token_text, TokenType::Date, start_idx, self.idx));
                }

                if let Some(mat) = TIME_REGEX.find(remaining) {
                    let token_text = mat.as_str();
                    self.idx = start_idx + token_text.len();
                    return Some(Token::new(token_text, TokenType::Time, start_idx, self.idx));
                }

                if let Some(mat) = NUMBER_REGEX.find(remaining) {
                    let token_text = mat.as_str();
                    if token_text.chars().any(|c| c.is_ascii_digit()) {
                        self.idx = start_idx + token_text.len();
                        return Some(Token::new(
                            token_text,
                            TokenType::Number,
                            start_idx,
                            self.idx,
                        ));
                    }
                }
            }

            // 5. Alphabetic Words, Turkish Proper Nouns, and Abbreviations
            if first_char.is_alphabetic() {
                // Quick check for email: an '@' before the next whitespace, looked for no further
                // than an address can be long (254). Scanning to the next whitespace from every
                // word start made whitespace-free text (base64, minified JSON) quadratic.
                if remaining
                    .char_indices()
                    .take_while(|&(i, c)| i < MAX_EMAIL_LEN && !c.is_whitespace())
                    .any(|(_, c)| c == '@')
                {
                    if let Some(mat) = EMAIL_REGEX.find(remaining) {
                        let token_text = mat.as_str();
                        self.idx = start_idx + token_text.len();
                        return Some(Token::new(
                            token_text,
                            TokenType::Email,
                            start_idx,
                            self.idx,
                        ));
                    }
                }

                // Zero-allocation zero-copy Word Scanner
                let mut word_end = idx;
                let mut has_apostrophe = false;
                let mut is_abbreviation = false;

                let mut chars_iter = remaining.char_indices().peekable();
                while let Some((offset, c)) = chars_iter.next() {
                    if c.is_alphabetic() {
                        word_end = idx + offset + c.len_utf8();
                    } else if c == '\'' || c == '’' {
                        // Turkish proper noun apostrophe (e.g. İstanbul'da)
                        if let Some(&(_, next_c)) = chars_iter.peek() {
                            if next_c.is_alphabetic() {
                                has_apostrophe = true;
                                word_end = idx + offset + c.len_utf8();
                                continue;
                            }
                        }
                        break;
                    } else if c == '.' {
                        // Check if candidate abbreviation (e.g. "Prof.", "Dr.", "vb.")
                        let candidate = &text[start_idx..idx + offset + 1];
                        let cand_lower = super::super::phonology::to_turkish_lower(candidate);
                        if ABBREVIATIONS.contains(cand_lower.as_str()) {
                            is_abbreviation = true;
                            word_end = idx + offset + 1;
                        }
                        break;
                    } else {
                        break;
                    }
                }

                let token_text = &text[start_idx..word_end];
                let token_type = if is_abbreviation {
                    TokenType::Abbreviation
                } else if has_apostrophe {
                    TokenType::ProperNounWithApostrophe
                } else {
                    TokenType::Word
                };

                self.idx = word_end;
                return Some(Token::new(token_text, token_type, start_idx, word_end));
            }

            // 6. Check Punctuation & Symbols
            let char_len = first_char.len_utf8();
            let token_type = if first_char.is_ascii_punctuation()
                || matches!(first_char, '…' | '“' | '”' | '‘' | '’' | '—' | '–')
            {
                TokenType::Punctuation
            } else {
                TokenType::Symbol
            };

            self.idx = start_idx + char_len;
            return Some(Token::new(
                &text[start_idx..self.idx],
                token_type,
                start_idx,
                self.idx,
            ));
        }
        None
    }
}
