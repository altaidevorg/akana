//! The tokenizer of the PII tagger's models: BERT WordPiece as Hugging Face's `tokenizers` runs it
//! for the BERTurk vocabulary, with every token's character offsets in the caller's text.
//!
//! The pipeline is the models' `tokenizer.json`, after the text the model reads ([`model_text`]):
//! BertNormalizer (control characters dropped, whitespace made a space, CJK ideographs set apart;
//! no lower-casing, no accent stripping), BertPreTokenizer (words split at whitespace, each
//! punctuation character a word of its own) and WordPiece (greedy longest match, `##` before a
//! word's later pieces, a word that cannot be split or is over 100 characters long becomes
//! `[UNK]`). The character classes come from the `unicode_categories` crate, as in `tokenizers`.
//! rten-text's tokenizer gives the same ids, but it places each `##` piece's start two bytes late,
//! which the span mapping cannot use.

use ahash::AHashMap;
use unicode_categories::UnicodeCategories;

/// A token: its vocabulary id and the characters of the text it covers (character indices, end
/// exclusive).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
    pub id: u32,
    pub start: usize,
    pub end: usize,
}

/// A BERT WordPiece vocabulary and its special tokens.
#[derive(Debug, Clone)]
pub struct WordPiece {
    vocab: AHashMap<String, u32>,
    prefix: String,
    max_chars: usize,
    pub(crate) unk: u32,
    pub(crate) cls: u32,
    pub(crate) sep: u32,
}

impl WordPiece {
    /// Reads a Hugging Face `tokenizer.json` whose model is WordPiece.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let v: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let model = &v["model"];
        if model["type"] != "WordPiece" {
            return Err(format!(
                "a WordPiece model was expected, not {}",
                model["type"]
            ));
        }
        let vocab: AHashMap<String, u32> = model["vocab"]
            .as_object()
            .ok_or("the tokenizer has no vocabulary")?
            .iter()
            .filter_map(|(piece, id)| Some((piece.clone(), u32::try_from(id.as_u64()?).ok()?)))
            .collect();
        let id = |piece: &str| {
            vocab
                .get(piece)
                .copied()
                .ok_or_else(|| format!("{piece} is not in the vocabulary"))
        };
        Ok(Self {
            unk: id(model["unk_token"].as_str().unwrap_or("[UNK]"))?,
            cls: id("[CLS]")?,
            sep: id("[SEP]")?,
            prefix: model["continuing_subword_prefix"]
                .as_str()
                .unwrap_or("##")
                .to_string(),
            max_chars: model["max_input_chars_per_word"]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .unwrap_or(100),
            vocab,
        })
    }

    /// Whether `other` gives every piece the same id (models averaged together must).
    pub fn same_vocabulary(&self, other: &Self) -> bool {
        self.vocab == other.vocab && self.prefix == other.prefix
    }

    /// The tokens of `chars`, the text the model reads (see [`model_text`]).
    pub fn tokenize(&self, chars: &[char]) -> Vec<Token> {
        let mut out = Vec::new();
        let mut word: Vec<(char, usize)> = Vec::new();
        for (i, &c) in chars.iter().enumerate() {
            if c == '\0' || c == '\u{fffd}' || is_control(c) {
                continue; // dropped by the normalizer: the word goes on around it
            }
            if is_whitespace(c) {
                self.pieces(&word, &mut out);
                word.clear();
            } else if is_punctuation(c) || is_chinese_char(c) {
                self.pieces(&word, &mut out);
                word.clear();
                self.pieces(&[(c, i)], &mut out);
            } else {
                word.push((c, i));
            }
        }
        self.pieces(&word, &mut out);
        out
    }

    /// The WordPiece tokens of one word, its characters given with their indices in the text.
    fn pieces(&self, word: &[(char, usize)], out: &mut Vec<Token>) {
        let (Some(first), Some(last)) = (word.first(), word.last()) else {
            return;
        };
        let unknown = Token {
            id: self.unk,
            start: first.1,
            end: last.1 + 1,
        };
        if word.len() > self.max_chars {
            out.push(unknown);
            return;
        }
        let mut pieces = Vec::new();
        let mut piece = String::new();
        let mut start = 0;
        while start < word.len() {
            let mut end = word.len();
            let mut found = None;
            while start < end {
                piece.clear();
                if start > 0 {
                    piece.push_str(&self.prefix);
                }
                piece.extend(word[start..end].iter().map(|(c, _)| c));
                if let Some(&id) = self.vocab.get(piece.as_str()) {
                    found = Some(id);
                    break;
                }
                end -= 1;
            }
            let Some(id) = found else {
                out.push(unknown);
                return;
            };
            pieces.push(Token {
                id,
                start: word[start].1,
                end: word[end - 1].1 + 1,
            });
            start = end;
        }
        out.extend(pieces);
    }
}

/// The text the model reads (scripts/pii_train/tagger.py's `normalize`): variation selectors and
/// emoji or other symbols that the tokenizer would glue to a word, turning the whole word into
/// `[UNK]` ("Yılmaz🎉"), become spaces, one character for one, so that offsets still hold.
pub fn model_text(chars: &[char]) -> Vec<char> {
    chars
        .iter()
        .map(|&c| {
            let symbol = u32::from(c) > 127
                && (c.is_symbol_other()
                    || c.is_symbol_modifier()
                    || c.is_other_private_use()
                    || is_unassigned(c));
            if symbol || is_variation_selector(c) {
                ' '
            } else {
                c
            }
        })
        .collect()
}

fn is_variation_selector(c: char) -> bool {
    matches!(c, '\u{fe00}'..='\u{fe0f}' | '\u{e0100}'..='\u{e01ef}')
}

/// A character in no Unicode category of the crate's tables (Cn).
fn is_unassigned(c: char) -> bool {
    !(c.is_letter()
        || c.is_mark()
        || c.is_number()
        || c.is_punctuation()
        || c.is_symbol()
        || c.is_separator()
        || c.is_other())
}

/// BertNormalizer's control characters: Cc, Cf and Co, except tab, newline and carriage return.
fn is_control(c: char) -> bool {
    !matches!(c, '\t' | '\n' | '\r') && c.is_other()
}

fn is_whitespace(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r') || c.is_whitespace()
}

/// BertPreTokenizer's punctuation: ASCII punctuation and every Unicode punctuation category.
fn is_punctuation(c: char) -> bool {
    c.is_ascii_punctuation() || UnicodeCategories::is_punctuation(c)
}

/// The CJK ideographs BertNormalizer sets apart with spaces.
fn is_chinese_char(c: char) -> bool {
    matches!(u32::from(c),
        0x4E00..=0x9FFF
        | 0x3400..=0x4DBF
        | 0x20000..=0x2A6DF
        | 0x2A700..=0x2B73F
        | 0x2B740..=0x2B81F
        | 0x2B920..=0x2CEAF
        | 0xF900..=0xFAFF
        | 0x2F800..=0x2FA1F)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> WordPiece {
        let vocab = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "ahmet", "Ahmet", "Yıl", "##maz", "'", "ın", "29",
            "/", "9", "geliyor", "##lar", ".", "Türkiye", "##de",
        ];
        let vocab: serde_json::Map<String, serde_json::Value> = vocab
            .iter()
            .enumerate()
            .map(|(i, p)| (p.to_string(), i.into()))
            .collect();
        let json = serde_json::json!({"model": {"type": "WordPiece", "unk_token": "[UNK]",
            "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": vocab}});
        WordPiece::from_json(&json.to_string()).unwrap()
    }

    fn tokens(text: &str) -> Vec<(String, String)> {
        let t = tokenizer();
        let chars: Vec<char> = text.chars().collect();
        let read = model_text(&chars);
        let names: AHashMap<u32, &String> = t.vocab.iter().map(|(p, &id)| (id, p)).collect();
        t.tokenize(&read)
            .iter()
            .map(|tok| {
                (
                    names[&tok.id].clone(),
                    chars[tok.start..tok.end].iter().collect(),
                )
            })
            .collect()
    }

    fn pairs(list: &[(&str, &str)]) -> Vec<(String, String)> {
        list.iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect()
    }

    #[test]
    fn continuation_pieces_cover_their_own_characters() {
        assert_eq!(
            tokens("Ahmet Yılmaz'ın"),
            pairs(&[
                ("Ahmet", "Ahmet"),
                ("Yıl", "Yıl"),
                ("##maz", "maz"),
                ("'", "'"),
                ("ın", "ın")
            ])
        );
    }

    #[test]
    fn punctuation_is_a_word_of_its_own() {
        assert_eq!(
            tokens("29/9."),
            pairs(&[("29", "29"), ("/", "/"), ("9", "9"), (".", ".")])
        );
    }

    #[test]
    fn a_word_without_a_split_is_unknown_as_a_whole() {
        assert_eq!(
            tokens("geliyorlarxyz ahmet"),
            pairs(&[("[UNK]", "geliyorlarxyz"), ("ahmet", "ahmet")])
        );
    }

    #[test]
    fn a_control_character_is_dropped_inside_its_word() {
        // a cp1252 apostrophe read as Latin-1: the normalizer drops it and the word reads on
        assert_eq!(
            tokens("Türkiye\u{92}de"),
            pairs(&[("Türkiye", "Türkiye"), ("##de", "de")])
        );
    }

    #[test]
    fn an_emoji_becomes_a_space_and_offsets_hold() {
        assert_eq!(
            tokens("Ahmet🎉geliyor"),
            pairs(&[("Ahmet", "Ahmet"), ("geliyor", "geliyor")])
        );
        assert_eq!(model_text(&['a', '🎉', 'b']), vec!['a', ' ', 'b']);
    }

    #[test]
    fn a_word_over_the_length_limit_is_unknown() {
        let long = "a".repeat(101);
        assert_eq!(tokens(&long), pairs(&[("[UNK]", &long)]));
    }
}
