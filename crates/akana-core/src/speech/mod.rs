//! Turkish text to phonemes and prosody tokens for a TTS front end: Akana's own rules and
//! morphology, no eSpeak NG data or code.
//!
//! Two profiles:
//! - [`Profile::TtsV1`] (the default): word stress from the phonology literature, a prominence
//!   hierarchy, pauses where a speaker pauses, long vowels and geminates. Its output on text of
//!   letters and punctuation is frozen: EchoCraft's shipped voice was trained on it.
//! - [`Profile::TtsV2`]: `tts-v1` with the segments of the phonology literature (Zimmer & Orgun
//!   1999; Göksel & Kerslake 2005; Yavuz & Balcı 2011), more stress rules, every rule of the
//!   verbaliser, and IPA symbols as `ph`. For the next model.
//!
//! Both verbalise the text first ([`verbalize`]): numbers, dates, money, units, initialisms,
//! symbols and addresses are read out, and each word keeps the span of the text it reads.
//!
//! ```
//! use akana_core::speech::{Phonemizer, Profile};
//! let p = Phonemizer::new(Profile::TtsV1);
//! let u = p.phonemize("Bir ağaçta gül de biter, diken de.");
//! let syms: Vec<&str> = u.tokens.iter().map(|t| t.ph_symbol()).collect();
//! assert_eq!(syms[..6], ["_", "b", "I", "R", "a", "tS"]);
//! ```
//!
//! Every decision that is not the default carries its reason: a word's
//! [`Word::stress_source`] and [`Word::segment_source`] name the lexicon entry, the rule and its
//! source. Output is deterministic: the same text and version give the same tokens.

mod letters;
mod phoneme;
mod text;
mod tts;
mod verbalize;

pub use phoneme::{Phoneme, PhonemeClass, ALL_PHONEMES};
pub use verbalize::{verbalize, verbalize_with, VerbalSegment, Verbalized};

use serde::Serialize;
use std::ops::Range;

/// A phonemiser profile. A profile's output changes only with a version bump and new golden
/// files.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Profile {
    /// Akana's TTS front end as frozen in stage 2 (`tts-v1.0`): stress from the literature and
    /// morphology, checked against the teacher's pitch, a prominence hierarchy, pauses where the
    /// teacher pauses, long vowels and geminates. Its `ph` symbols are the stage-1 mnemonics
    /// (`a`, `E`, `dZ`, `L/`, `*`). See `speech::tts`.
    #[default]
    TtsV1,
    /// The consensus of the Turkish phonology literature (`docs/phonemizer-tts-v2.md`):
    /// - segments: tense vowels inside the word; a word-final open /e/ as [ɛ]; [æ] before a
    ///   tautosyllabic coda sonorant and in *-mAz*; /ɾ/ a tap, devoiced only before a pause;
    ///   palatal k and g when tautosyllabic with a front vowel; circumflex vowels long only in
    ///   open syllables;
    /// - stress: pre-stressing *-en* adverbs, *-CA*, *-iye*; place names by Sezer's rule;
    ///   compounds and intensives on their first element;
    /// - every verbaliser rule, and the phoneme's IPA as `ph`.
    TtsV2,
}

impl Profile {
    /// The profile's name (`tts-v1`).
    pub fn name(self) -> &'static str {
        match self {
            Profile::TtsV1 => "tts-v1",
            Profile::TtsV2 => "tts-v2",
        }
    }

    /// The profile for a name (`tts-v1`, `tts_v1`, `TtsV1`).
    pub fn from_name(name: &str) -> Option<Profile> {
        match name {
            "tts-v1" | "tts_v1" | "TtsV1" => Some(Profile::TtsV1),
            "tts-v2" | "tts_v2" | "TtsV2" => Some(Profile::TtsV2),
            _ => None,
        }
    }

    /// The profile's rules version: it changes whenever its output does.
    pub fn version(self) -> &'static str {
        match self {
            Profile::TtsV1 => "tts-v1.0",
            Profile::TtsV2 => "tts-v2.0",
        }
    }

    /// The profile's symbol for a phoneme: the mnemonic in `tts-v1`, the IPA (without a length
    /// mark) in `tts-v2`; pauses are `_`, `_:` and `_!` in both.
    pub fn symbol(self, ph: Phoneme) -> &'static str {
        match self {
            Profile::TtsV2 if !ph.is_pause() => ph.ipa().unwrap_or(ph.symbol()),
            _ => ph.symbol(),
        }
    }

    /// The profile's class of a phoneme: `tts-v2` counts the tap /ɾ/ a liquid, `tts-v1` keeps
    /// the stage-1 `vstop`.
    pub fn class(self, ph: Phoneme) -> PhonemeClass {
        match (self, ph) {
            (Profile::TtsV2, Phoneme::Tap) => PhonemeClass::Liquid,
            _ => ph.class(),
        }
    }
}

/// A pause token.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum Pause {
    /// `_`: at each edge of the text.
    Edge,
    /// `_:`: at a clause boundary.
    ClauseEnd,
    /// `_!`: a short pause (quotes and brackets, before a conjunction, after an adverbial
    /// clause).
    Short,
}

impl Pause {
    pub fn phoneme(self) -> Phoneme {
        match self {
            Pause::Edge => Phoneme::Pause,
            Pause::ClauseEnd => Phoneme::PauseClause,
            Pause::Short => Phoneme::PauseShort,
        }
    }
}

/// Why a word is stressed where it is.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum StressSource {
    /// No rule or entry applies: the last syllable.
    DefaultFinal,
    /// Spelled letter by letter: the last letter.
    Spelled,
    /// An entry of the exceptional-stress lexicon (`source` names its class and evidence, as in
    /// `speech::tts::STRESS_LEXICON`).
    Lexical { source: &'static str },
    /// A suffix rule on the word's morphological analysis.
    Morphology { rule: &'static str },
    /// Sezer's rule for a place name.
    Sezer,
    /// The first part of a compound the lexicon splits.
    CompoundFirst,
    /// A clitic, unstressed.
    Clitic,
}

impl StressSource {
    /// The provenance label (`lexicon:adverb:L,A,T`, `morphology:pre-accenting(-mA)`).
    pub fn label(&self) -> String {
        match self {
            StressSource::DefaultFinal => "default-final".to_string(),
            StressSource::Spelled => "rule:spelled".to_string(),
            StressSource::Lexical { source } => format!("lexicon:{source}"),
            StressSource::Morphology { rule } => format!("morphology:{rule}"),
            StressSource::Sezer => "sezer".to_string(),
            StressSource::CompoundFirst => "compound:first-part".to_string(),
            StressSource::Clitic => "clitic".to_string(),
        }
    }
}

/// Where a word's phonemes come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SegmentSource {
    /// Spelled letter by letter (a word without vowels).
    Spelled,
    /// `tts-v1`: the letter rules with the profile's long vowels and geminates.
    TtsRules,
    /// The letter rules, with a segment the lexicon's attributes change (the clear l of an
    /// InverseHarmony loan).
    LexiconAttribute,
    /// `tts-v2`: the letter rules with the literature's allophones.
    TtsV2Rules,
}

impl SegmentSource {
    pub fn label(self) -> &'static str {
        match self {
            SegmentSource::Spelled => "rule:spelled",
            SegmentSource::TtsRules => "tts-v1-rule:letters",
            SegmentSource::LexiconAttribute => "lexicon:akana(InverseHarmony)",
            SegmentSource::TtsV2Rules => "tts-v2-rule:letters",
        }
    }
}

/// One phoneme of a word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Phone {
    pub phoneme: Phoneme,
    /// The prominence level of the phoneme's syllable (0–7; see `speech::tts`).
    pub stress: u8,
    /// A long vowel or a geminate.
    pub lengthen: bool,
}

/// A word as read.
#[derive(Clone, Debug, Serialize)]
pub struct Word {
    /// Byte span in the text (a word read from a number or a symbol spans all of it).
    pub span: Range<usize>,
    /// The reading the lexicon is looked up by (lower case, `’` as `'`, without hyphens).
    pub reading: String,
    /// The pauses before the word.
    pub pauses_before: Vec<Pause>,
    pub phones: Vec<Phone>,
    /// The clause's number (for its mark).
    pub clause: usize,
    pub stress_source: StressSource,
    pub segment_source: SegmentSource,
    /// Always false (kept for the stage-1 schema).
    pub promoted: bool,
    /// The word carries the clause's nucleus (7).
    pub tonic: bool,
    /// What the word reads when it is not written as letters: the verbaliser's kind
    /// (`cardinal`, `money`, `initialism` ...), or `number` for digits read without it.
    pub normalized: Option<&'static str>,
    /// The word's prominence: the level of its stressed syllable (0 for a clitic).
    pub prominence: u8,
}

/// A token: a phoneme or a pause, in the schema EchoCraft's TTS reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Token {
    pub ph: Phoneme,
    /// The prominence level of the token's syllable (0–7).
    pub stress: u8,
    /// The first phoneme of a word.
    pub word: bool,
    pub lengthen: bool,
    /// The punctuation ending the token's clause (`,` `.` `?` `!` `;` `:`).
    pub mark: char,
    /// The profile that wrote it, which decides its symbol.
    pub profile: Profile,
}

impl Token {
    /// The token's IPA, with `ː` when long (`None` for pauses).
    pub fn ipa(&self) -> Option<&'static str> {
        if self.lengthen {
            self.ph.ipa_long()
        } else {
            self.ph.ipa()
        }
    }

    /// The token's `ph` symbol in its profile.
    pub fn ph_symbol(&self) -> &'static str {
        self.profile.symbol(self.ph)
    }
}

impl Serialize for Token {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("Token", 7)?;
        s.serialize_field("ph", self.ph_symbol())?;
        s.serialize_field("stress", &self.stress)?;
        s.serialize_field("word", &u8::from(self.word))?;
        s.serialize_field("lengthen", &u8::from(self.lengthen))?;
        s.serialize_field("mark", &self.mark.to_string())?;
        s.serialize_field("type", self.profile.class(self.ph).as_str())?;
        s.serialize_field("ipa", &self.ipa())?;
        s.end()
    }
}

/// Something in the text the profile did not read.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Warning {
    /// Byte span in the text.
    pub span: Range<usize>,
    pub message: String,
}

/// A phonemised text.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Utterance {
    pub words: Vec<Word>,
    /// The token stream, edge pauses included.
    pub tokens: Vec<Token>,
    /// Characters the profile did not read (an emoji, a symbol the verbaliser does not know).
    pub warnings: Vec<Warning>,
}

/// Why a text could not be phonemised.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PhonemizeError {
    #[error("text of {len} bytes is over the {max}-byte limit")]
    TooLong { len: usize, max: usize },
}

/// The longest text one call takes (1 MiB): a screen reader sends sentences.
pub const MAX_TEXT_BYTES: usize = 1 << 20;

/// The phonemiser. Its data (Akana's morphology) is loaded once per process and shared.
#[derive(Clone, Debug, Default)]
pub struct Phonemizer {
    profile: Profile,
}

impl Phonemizer {
    pub fn new(profile: Profile) -> Self {
        // load the morphology now rather than in the first call
        lazy_static::initialize(&tts::NLP);
        Phonemizer { profile }
    }

    pub fn profile(&self) -> Profile {
        self.profile
    }

    /// Phonemises `text` (texts over [`MAX_TEXT_BYTES`] give an empty utterance).
    pub fn phonemize(&self, text: &str) -> Utterance {
        let mut out = Utterance::default();
        let _ = self.phonemize_into(text, &mut out);
        out
    }

    /// Phonemises `text` into `out`, reusing its buffers.
    pub fn phonemize_into(&self, text: &str, out: &mut Utterance) -> Result<(), PhonemizeError> {
        tts::phonemize_tts(text, out, self.profile)
    }
}
