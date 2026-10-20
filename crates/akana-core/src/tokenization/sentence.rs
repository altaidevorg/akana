//! Fast sentence boundary detector for Turkish text.

use super::tokenizer::{Token, TokenType, Tokens, TurkishTokenizer};

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

/// The style auditor's AI-text detector was trained on this behaviour: changing it shifts the
/// detector's inputs and fails `style::ai_detector::tests::outputs_match_training_reference`.
pub struct SentenceSegmenter;

impl SentenceSegmenter {
    /// Segments text into Turkish sentences, respecting abbreviations, ellipses, numbers, and quotations.
    pub fn segment<'a>(text: &'a str) -> Vec<Sentence<'a>> {
        let mut scan = Scan::new(text);
        let mut sentences = Vec::new();
        while let Step::Sentence(s) = scan.next_sentence(usize::MAX) {
            sentences.push(s);
        }
        sentences
    }

    /// Where the text of the first sentence of [`segment`](Self::segment) ends
    /// (`start + text.len()`), or `text.len()` when there is none; but at most `cap`.
    ///
    /// Tokenizes only up to that sentence's end and the tokens that decide it, or up to `cap`,
    /// so asking for the first sentence of a long text does not cost the whole text.
    pub(crate) fn first_sentence_end(text: &str, cap: usize) -> usize {
        match Scan::new(text).next_sentence(cap) {
            Step::Sentence(s) => (s.start + s.text.len()).min(cap),
            Step::Stopped => cap,
            Step::Done => text.len().min(cap),
        }
    }
}

enum Step<'a> {
    Sentence(Sentence<'a>),
    /// The scan reached a token starting at or after the stop offset before a sentence ended.
    Stopped,
    /// No sentence is left.
    Done,
}

/// The segmentation loop over lazily produced tokens, one sentence per call.
struct Scan<'a> {
    text: &'a str,
    rest: Tokens<'a>,
    /// The tokens produced so far.
    tokens: Vec<Token<'a>>,
    /// Index of the next token to look at.
    i: usize,
    current_start: usize,
    done: bool,
}

impl<'a> Scan<'a> {
    fn new(text: &'a str) -> Self {
        let mut rest = TurkishTokenizer::tokens(text);
        let first = rest.next();
        Self {
            text,
            rest,
            current_start: first.as_ref().map_or(0, |t| t.start),
            done: first.is_none(),
            tokens: first.into_iter().collect(),
            i: 0,
        }
    }

    /// Whether token `i` exists; tokenizes up to it.
    fn has(&mut self, i: usize) -> bool {
        while self.tokens.len() <= i {
            match self.rest.next() {
                Some(t) => self.tokens.push(t),
                None => return false,
            }
        }
        true
    }

    /// The next sentence, unless the scan first reaches a token starting at or after `stop`.
    fn next_sentence(&mut self, stop: usize) -> Step<'a> {
        if self.done {
            return Step::Done;
        }
        let text = self.text;
        while self.has(self.i) {
            let i = self.i;
            let token = self.tokens[i].clone();
            if token.start >= stop {
                return Step::Stopped;
            }

            let is_sentence_ender = match token.text {
                "." => {
                    // Make sure it's not preceded by abbreviation or part of number
                    if token.token_type == TokenType::Abbreviation
                        || (i > 0 && self.tokens[i - 1].token_type == TokenType::Abbreviation)
                    {
                        false
                    } else if i > 0 && self.tokens[i - 1].token_type == TokenType::Number {
                        // Ordinal number check (e.g. "3. madde", "1. kat")
                        if self.has(i + 1) && self.tokens[i + 1].token_type == TokenType::Word {
                            let next_first_char = self.tokens[i + 1].text.chars().next().unwrap();
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
                while self.has(next_i) && matches!(self.tokens[next_i].text, "." | "!" | "?" | "…")
                {
                    end_idx = self.tokens[next_i].end;
                    next_i += 1;
                }

                // Consume any trailing quotes / closing brackets
                while self.has(next_i)
                    && matches!(
                        self.tokens[next_i].text,
                        "\"" | "”" | "’" | "'" | ")" | "]" | "»"
                    )
                {
                    end_idx = self.tokens[next_i].end;
                    next_i += 1;
                }

                let start = self.current_start;
                self.current_start = if self.has(next_i) {
                    self.tokens[next_i].start
                } else {
                    text.len()
                };
                self.i = next_i;

                let s_text = text[start..end_idx].trim();
                if !s_text.is_empty() {
                    return Step::Sentence(Sentence::new(s_text, start, end_idx));
                }
            } else {
                self.i += 1;
            }
        }

        // If leftover tokens exist after loop
        self.done = true;
        if self.current_start < text.len() {
            let remaining = text[self.current_start..].trim();
            if !remaining.is_empty() {
                return Step::Sentence(Sentence::new(remaining, self.current_start, text.len()));
            }
        }
        Step::Done
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXTS: &[&str] = &[
        "",
        "   ",
        "Tek cümle",
        "Bir. İki! Üç? Dört…",
        "Dr. Ali geldi. Prof. Ayşe gitti.",
        "3. madde geçerli. 1. Kat burada.",
        "Saat 12.30'da \"Geldi mi?\" dedi. (Evet.) [Hayır!] «Tamam.» Son",
        "Elbette http://a: b. Tabii www.x.com/y: z!",
        "İşte: liste. Not: bitti...  Yeni?! cümle",
        "Gel. 12. gün... vb. şeyler. ali@x.com yazdı. #etiket @ad",
        "a\nb. c",
        "...!!! ???",
    ];

    #[test]
    fn first_sentence_end_matches_segment() {
        for text in TEXTS {
            let first = SentenceSegmenter::segment(text)
                .first()
                .map_or(text.len(), |s| s.start + s.text.len());
            for cap in (0..=text.len() + 1).chain([usize::MAX]) {
                assert_eq!(
                    SentenceSegmenter::first_sentence_end(text, cap),
                    first.min(cap),
                    "{text:?} cap {cap}"
                );
            }
        }
    }

    #[test]
    fn sentence_offsets_increase() {
        // Callers convert or match the offsets in one forward pass (chunkers, label colons).
        for text in TEXTS {
            let mut last = 0;
            for s in SentenceSegmenter::segment(text) {
                assert!(last <= s.start && s.start <= s.end, "{text:?}");
                last = s.end;
            }
        }
    }
}
