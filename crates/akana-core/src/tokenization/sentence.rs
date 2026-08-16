//! Fast sentence boundary detector for Turkish text.

use super::tokenizer::{TurkishTokenizer, TokenType};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sentence<'a> {
    pub text: &'a str,
    pub start: usize,
    pub end: usize,
}

impl<'a> Sentence<'a> {
    pub fn new(text: &'a str, start: usize, end: usize) -> Self {
        Self { text, start, end }
    }
}

pub struct SentenceSegmenter;

impl SentenceSegmenter {
    /// Segments text into Turkish sentences, respecting abbreviations, ellipses, numbers, and quotations.
    pub fn segment<'a>(text: &'a str) -> Vec<Sentence<'a>> {
        let tokens = TurkishTokenizer::tokenize(text);
        if tokens.is_empty() {
            return Vec::new();
        }

        let mut sentences = Vec::new();
        let mut current_start = tokens[0].start;
        let n = tokens.len();
        let mut i = 0;

        while i < n {
            let token = &tokens[i];

            let is_sentence_ender = match token.text {
                "." => {
                    // Make sure it's not preceded by abbreviation or part of number
                    if token.token_type == TokenType::Abbreviation {
                        false
                    } else if i > 0 && tokens[i - 1].token_type == TokenType::Abbreviation {
                        false
                    } else if i > 0 && tokens[i - 1].token_type == TokenType::Number {
                        // Ordinal number check (e.g. "3. madde", "1. kat")
                        if i + 1 < n && tokens[i + 1].token_type == TokenType::Word {
                            let next_first_char = tokens[i + 1].text.chars().next().unwrap();
                            !next_first_char.is_lowercase()
                        } else {
                            true
                        }
                    } else {
                        true
                    }
                }
                "!" | "?" | "…" | "!?" | "?!" => true,
                _ => false,
            };

            if is_sentence_ender {
                let mut end_idx = token.end;
                let mut next_i = i + 1;

                // Consume any additional sentence enders (e.g. multiple dots in "...", "!!", "?!")
                while next_i < n && matches!(tokens[next_i].text, "." | "!" | "?" | "…") {
                    end_idx = tokens[next_i].end;
                    next_i += 1;
                }

                // Consume any trailing quotes / closing brackets
                while next_i < n && matches!(tokens[next_i].text, "\"" | "”" | "’" | "'" | ")" | "]" | "»") {
                    end_idx = tokens[next_i].end;
                    next_i += 1;
                }

                let s_text = text[current_start..end_idx].trim();
                if !s_text.is_empty() {
                    sentences.push(Sentence::new(s_text, current_start, end_idx));
                }

                if next_i < n {
                    current_start = tokens[next_i].start;
                } else {
                    current_start = text.len();
                }

                i = next_i;
            } else {
                i += 1;
            }
        }

        // If leftover tokens exist after loop
        if current_start < text.len() {
            let remaining = text[current_start..].trim();
            if !remaining.is_empty() {
                sentences.push(Sentence::new(remaining, current_start, text.len()));
            }
        }

        sentences
    }
}
