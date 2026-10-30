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
/// detector's inputs and fails `style::ai_detector::tests::inputs_match_drift_canary` (and
/// `outputs_match_training_reference` when its sample shows it).
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

    /// Whether token `next` is an apostrophe joined to the text ending at `end` and to the
    /// letters of a suffix after it ("...'nin", "2.'si"): no closing quote.
    fn suffix_apostrophe(&mut self, next: usize, end: usize) -> bool {
        if !self.has(next) {
            return false;
        }
        let t = self.tokens[next].clone();
        t.start == end
            && t.text.chars().count() == 1
            && t.text.starts_with(['\'', '’', '‘', '`', '´', '′', '＇'])
            && self.has(next + 1)
            && self.tokens[next + 1].start == t.end
            && self.tokens[next + 1]
                .text
                .starts_with(|c: char| c.is_alphabetic())
    }

    /// Whether the text goes on after an ellipsis ending at `end`, with token `next`: an
    /// apostrophe joined to it and to the letters of a suffix, a word in lower case, or a comma
    /// or semicolon (E:..., K:...; review 4 MORPH4-06).
    fn continues_after(&mut self, next: usize, end: usize) -> bool {
        if !self.has(next) {
            return false;
        }
        let t = self.tokens[next].clone();
        let lower_word = matches!(
            t.token_type,
            TokenType::Word | TokenType::ProperNounWithApostrophe
        ) && t.text.starts_with(|c: char| c.is_lowercase());
        lower_word || matches!(t.text, "," | ";") || self.suffix_apostrophe(next, end)
    }

    /// Whether token `i` is a word in lower case.
    fn lower_word(&mut self, i: usize) -> bool {
        self.has(i)
            && matches!(
                self.tokens[i].token_type,
                TokenType::Word | TokenType::ProperNounWithApostrophe | TokenType::Abbreviation
            )
            && self.tokens[i].text.starts_with(|c: char| c.is_lowercase())
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
                // A suffix after the period ("2.'si") continues the word (review 3 TOK3-05).
                "." if self.suffix_apostrophe(i + 1, token.end) => false,
                "." => {
                    // Make sure it's not preceded by abbreviation or part of number
                    if token.token_type == TokenType::Abbreviation
                        || (i > 0 && self.tokens[i - 1].token_type == TokenType::Abbreviation)
                    {
                        false
                    } else if i > 0 && self.tokens[i - 1].token_type == TokenType::Number {
                        // An ordinal: "3. madde", "1. kat", and before a capital a number of
                        // one or two digits, as court decisions name courts and paragraphs
                        // ("Ankara 3. Asliye Hukuk Mahkemesi", "2. Mahkeme reddetti", "Yargıtay
                        // 9. HD."); a year or an amount ends the sentence (review 3 TOK3-03).
                        if self.has(i + 1)
                            && matches!(
                                self.tokens[i + 1].token_type,
                                TokenType::Word
                                    | TokenType::Abbreviation
                                    | TokenType::ProperNounWithApostrophe
                            )
                        {
                            let next_first_char = self.tokens[i + 1].text.chars().next().unwrap();
                            let number = self.tokens[i - 1].text;
                            let ordinal =
                                number.len() <= 2 && number.chars().all(|c| c.is_ascii_digit());
                            !next_first_char.is_lowercase() && !ordinal
                        } else {
                            // the dash of an amount right after its period (12.500.-TL)
                            !(self.has(i + 1)
                                && self.tokens[i + 1].start == token.end
                                && self.tokens[i + 1].text == "-")
                        }
                    } else if self.has(i + 1)
                        && token.end == self.tokens[i + 1].start
                        && matches!(self.tokens[i + 1].text, "," | ";" | ":" | "-")
                    {
                        // A comma or semicolon right after the period: no sentence starts with
                        // one (an abbreviation the list lacks); nor with the dash of an amount
                        // (12.500.-TL).
                        false
                    } else if i > 0
                        && self.tokens[i - 1].end == token.start
                        && self.tokens[i - 1].text.chars().nth(1).is_some()
                        && self.tokens[i - 1].text.chars().all(char::is_uppercase)
                        && (self.lower_word(i + 1)
                            || (self.has(i + 1)
                                && self.tokens[i - 1]
                                    .text
                                    .chars()
                                    .all(|c| matches!(c, 'I' | 'V' | 'X' | 'L' | 'C' | 'D' | 'M'))
                                && self.tokens[i - 1].text.chars().count() <= 6
                                && self.tokens[i + 1].text.starts_with(char::is_uppercase)))
                    {
                        // An acronym the abbreviation list lacks, its period and a word in
                        // lower case (TTK. nun, İİK. m. 170), and a Roman ordinal before a
                        // capital (II. Dünya Savaşı): no sentence end (review 4 MORPH4-06).
                        false
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
                let mut dots = usize::from(token.text == ".");
                while self.has(next_i) && matches!(self.tokens[next_i].text, "." | "!" | "?" | "…")
                {
                    end_idx = self.tokens[next_i].end;
                    dots += usize::from(self.tokens[next_i].text == ".");
                    next_i += 1;
                }

                // An ellipsis inside a sentence: a name left out (davacı ...'nin vekili,
                // HÂKİMİ ...'NIN) whose suffix follows, or text that goes on in lower case
                // (davacı ... vekili). Ending the sentence there made the suffix open the
                // next sentence and become a term (review 3 TOK3-01).
                let ellipsis = dots >= 2 || token.text == "…";
                // An ellipsis that no word touches stands for words left out (Tire ... Asliye
                // Ceza Mahkemesine ... Esas sayılı dosya; E:..., K:...): no sentence end. One
                // that ends a word trails off (gelmedi... Sonra).
                let redaction = ellipsis
                    && !(i > 0
                        && self.tokens[i - 1].end == token.start
                        && self.tokens[i - 1]
                            .text
                            .chars()
                            .last()
                            .is_some_and(char::is_alphanumeric));
                if ellipsis && (redaction || self.continues_after(next_i, end_idx)) {
                    self.i = next_i;
                    continue;
                }

                // Consume any trailing quotes / closing brackets, but not an apostrophe that
                // joins a suffix (review 3 TOK3-01).
                let before_quotes = next_i;
                while self.has(next_i)
                    && matches!(
                        self.tokens[next_i].text,
                        "\"" | "”" | "’" | "'" | ")" | "]" | "»"
                    )
                    && !self.suffix_apostrophe(next_i, end_idx)
                {
                    end_idx = self.tokens[next_i].end;
                    next_i += 1;
                }
                // A quotation that ends with its sentence's period and goes on in lower case
                // (... belirler.'' şeklinde): one sentence (review 4 MORPH4-06).
                if next_i > before_quotes && token.text == "." && self.lower_word(next_i) {
                    self.i = next_i;
                    continue;
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
        "Davacı ...'nin vekili geldi. Davalı ... vekili gelmedi... Sonra.",
        "Ankara 3. Asliye Hukuk Mahkemesi'nin 2019/12 E. , 2020/5 K. sayılı kararı. Son 2019. Yeni",
        "A.Ş.'nin ve T.C. Anayasası'nın (d. 1950, ö. 2010) hükmü.",
        "\u{feff}Madde 2.'si uygulanır. Yargıtay 9. HD. E. 2019/1 sayılı. Bitti!'nin sonu",
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
