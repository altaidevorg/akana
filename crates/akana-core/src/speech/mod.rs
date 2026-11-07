//! Turkish text to phonemes and prosody tokens for a TTS front end.
//!
//! The first profile, [`Profile::EspeakCompat`], reproduces the token stream eSpeak NG 1.52
//! gives EchoCraft's student TTS (phoneme, stress level 0–7, start of word, length mark, the
//! clause's punctuation), so the student can run on it without retraining. It was built clean
//! room: from eSpeak's observed outputs (reference fixtures, and a CC0 word list run through
//! the same eSpeak build) and the behaviour described in EchoCraft's brief, never from eSpeak's
//! rule files or source (GPL-3.0). eSpeak is the parity target here, not an authority on
//! Turkish; later profiles will follow the phonology literature.
//!
//! ```
//! use akana_core::speech::{Phonemizer, Profile};
//! let p = Phonemizer::new(Profile::EspeakCompat);
//! let u = p.phonemize("Bir ağaçta gül de biter, diken de.");
//! let syms: Vec<&str> = u.tokens.iter().map(|t| t.ph.symbol()).collect();
//! assert_eq!(syms[..6], ["_", "b", "I", "R", "a", "tS"]);
//! ```
//!
//! Every decision that is not the default carries its reason: a word's
//! [`Word::stress_source`] and [`Word::segment_source`] say whether the lexicon, a rule or the
//! suffix model decided it. Output is deterministic: the same text and version give the same
//! tokens.

mod letters;
pub(crate) mod model;
mod phoneme;
mod text;

#[cfg(test)]
mod tests;

pub use phoneme::{Phoneme, PhonemeClass, ALL_PHONEMES};

use letters::{has_vowel, key_to_phonemes, spell, stress_dependent_segments, Seg};
use model::{Mark, Model, MODEL};
use serde::Serialize;
use std::ops::Range;
use text::{clause_marks, segment, TextWord};

/// A phonemiser profile. A profile's output changes only with a version bump and new
/// fixtures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Profile {
    /// eSpeak NG 1.52 Turkish as EchoCraft's student TTS reads it (stage 1, parity).
    EspeakCompat,
}

impl Profile {
    /// The profile's name (`espeak-compat`).
    pub fn name(self) -> &'static str {
        match self {
            Profile::EspeakCompat => "espeak-compat",
        }
    }

    /// The profile for a name (`espeak-compat`, `espeak_compat`).
    pub fn from_name(name: &str) -> Option<Profile> {
        match name {
            "espeak-compat" | "espeak_compat" | "EspeakCompat" => Some(Profile::EspeakCompat),
            _ => None,
        }
    }

    /// The profile's data and rules version: it changes whenever its output does.
    pub fn version(self) -> &'static str {
        match self {
            Profile::EspeakCompat => "espeak-compat-1",
        }
    }
}

/// A pause token.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum Pause {
    /// `_`: at each edge, and after a clause's `_:`.
    Edge,
    /// `_:`: ends a non-final clause; a pair marks a quote or follows a word with an apostrophe.
    ClauseEnd,
    /// `_!`: before ve, veya, yoksa, ama, ki and kime.
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
    /// The word is in the profile's lexicon (seen in eSpeak's outputs).
    Lexicon,
    /// The suffix model, from the longest ending the word shares with lexicon words.
    SuffixModel { ending: String },
    /// After an apostrophe, an ending eSpeak does not stress (`'la`, `'le`, `'dir`, ...).
    ApostropheEnding { ending: String },
    /// After an apostrophe, the last syllable (eSpeak's default there).
    ApostropheFinal,
    /// No ending known: the last syllable.
    DefaultFinal,
    /// Spelled letter by letter: each letter secondary, the last primary.
    Spelled,
}

impl StressSource {
    /// The provenance label (`lexicon:espeak-compat`, `espeak-compat-rule:suffix-model(-ıyordu)`).
    pub fn label(&self) -> String {
        match self {
            StressSource::Lexicon => "lexicon:espeak-compat".to_string(),
            StressSource::SuffixModel { ending } => {
                format!("espeak-compat-rule:suffix-model(-{ending})")
            }
            StressSource::ApostropheEnding { ending } => {
                format!("espeak-compat-rule:apostrophe-ending('{ending})")
            }
            StressSource::ApostropheFinal => "espeak-compat-rule:apostrophe-final".to_string(),
            StressSource::DefaultFinal => "default-final".to_string(),
            StressSource::Spelled => "espeak-compat-rule:spelled".to_string(),
        }
    }
}

/// Where a word's phonemes come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SegmentSource {
    /// The lexicon (eSpeak's phonemes, or the letter rules where they agree).
    Lexicon,
    /// The letter rules.
    Rules,
    /// The letter rules, with a stress-dependent choice (g/J, e/E) changed by the word's stress.
    RulesStressAdjusted,
    /// Spelled letter by letter (a word without vowels).
    Spelled,
}

impl SegmentSource {
    pub fn label(self) -> &'static str {
        match self {
            SegmentSource::Lexicon => "lexicon:espeak-compat",
            SegmentSource::Rules => "espeak-compat-rule:letters",
            SegmentSource::RulesStressAdjusted => "espeak-compat-rule:letters+stress",
            SegmentSource::Spelled => "espeak-compat-rule:spelled",
        }
    }
}

/// One phoneme of a word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Phone {
    pub phoneme: Phoneme,
    /// eSpeak's stress level: 0 diminished, 1 unstressed, 3 secondary, 6 a word's primary, 7
    /// the clause's last primary (tonic). A consonant takes the level of the next vowel in its
    /// clause.
    pub stress: u8,
    /// The length mark: a vowel before ğ, or a doubled stop.
    pub lengthen: bool,
}

/// A word as read.
#[derive(Clone, Debug, Serialize)]
pub struct Word {
    /// Byte span in the text.
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
    /// The primary stress was added because the word ends its clause (`ben`, `bu`, `o` ...) or
    /// the clause had none.
    pub promoted: bool,
    /// The word carries the clause's tonic (7).
    pub tonic: bool,
    /// What the word reads when it is not written as letters (`number`: `yirmi` of `20`).
    pub normalized: Option<&'static str>,
}

/// A token of eSpeak's plan, the schema EchoCraft's student TTS reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EspeakToken {
    pub ph: Phoneme,
    pub stress: u8,
    /// The first phoneme of a word.
    pub word: bool,
    pub lengthen: bool,
    /// The punctuation ending the token's clause (`,` `.` `?` `!` `;` `:`).
    pub mark: char,
}

impl EspeakToken {
    /// eSpeak's IPA for the token (`None` for pauses).
    pub fn ipa(&self) -> Option<&'static str> {
        if self.lengthen {
            self.ph.ipa_long()
        } else {
            self.ph.ipa()
        }
    }
}

impl Serialize for EspeakToken {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("EspeakToken", 7)?;
        s.serialize_field("ph", self.ph.symbol())?;
        s.serialize_field("stress", &self.stress)?;
        s.serialize_field("word", &u8::from(self.word))?;
        s.serialize_field("lengthen", &u8::from(self.lengthen))?;
        s.serialize_field("mark", &self.mark.to_string())?;
        s.serialize_field("type", self.ph.class().as_str())?;
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
    /// The token stream in the fixtures' schema, edge pauses included.
    pub tokens: Vec<EspeakToken>,
    /// Characters the profile skipped (symbols): `espeak-compat` reads letters and digits.
    pub warnings: Vec<Warning>,
}

impl Utterance {
    /// The token stream in the fixtures' schema.
    pub fn espeak_tokens(&self) -> &[EspeakToken] {
        &self.tokens
    }
}

/// Why a text could not be phonemised.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PhonemizeError {
    #[error("text of {len} bytes is over the {max}-byte limit")]
    TooLong { len: usize, max: usize },
}

/// The longest text one call takes (1 MiB): a screen reader sends sentences.
pub const MAX_TEXT_BYTES: usize = 1 << 20;

/// The phonemiser. Its data is loaded once per process and shared.
#[derive(Clone, Debug)]
pub struct Phonemizer {
    profile: Profile,
}

/// Per-word working state.
struct WordState {
    segs: Vec<Seg>,
    marks: Vec<Mark>,
    levels: Vec<u8>,
    stress_source: StressSource,
    segment_source: SegmentSource,
    spelled: bool,
    promoted: bool,
    tonic: bool,
}

/// Function words that become stressed when they end their clause (observed).
const STRESSABLE: [&str; 8] = ["bir", "bu", "o", "ben", "sen", "biz", "siz", "şu"];
/// Open-syllable function words that stay at level 1 after a level-1 syllable (observed).
const OPEN_FUNCTION_WORDS: [&str; 12] = [
    "bu", "o", "şu", "de", "da", "mi", "mı", "mu", "mü", "ki", "ta", "ve",
];

impl Phonemizer {
    pub fn new(profile: Profile) -> Self {
        // load the data now rather than in the first call
        lazy_static::initialize(&MODEL);
        Phonemizer { profile }
    }

    pub fn profile(&self) -> Profile {
        self.profile
    }

    /// The data file's provenance lines.
    pub fn data_provenance(&self) -> Vec<String> {
        MODEL.header.clone()
    }

    /// Phonemises `text` (texts over [`MAX_TEXT_BYTES`] give an empty utterance).
    pub fn phonemize(&self, text: &str) -> Utterance {
        let mut out = Utterance::default();
        let _ = self.phonemize_into(text, &mut out);
        out
    }

    /// Phonemises `text` into `out`, reusing its buffers.
    pub fn phonemize_into(&self, text: &str, out: &mut Utterance) -> Result<(), PhonemizeError> {
        phonemize_with(&MODEL, self.profile, text, out)
    }
}

/// The pipeline with a given model (the shipped one, or one built from part of the fixtures
/// for cross-validation).
pub(crate) fn phonemize_with(
    model: &Model,
    profile: Profile,
    text: &str,
    out: &mut Utterance,
) -> Result<(), PhonemizeError> {
    out.words.clear();
    out.tokens.clear();
    out.warnings.clear();
    if text.len() > MAX_TEXT_BYTES {
        return Err(PhonemizeError::TooLong {
            len: text.len(),
            max: MAX_TEXT_BYTES,
        });
    }
    let mut tw: Vec<TextWord> = Vec::new();
    let final_gap = segment(text, &mut tw);
    let mut states: Vec<WordState> = tw.iter().map(|w| lexical(model, w)).collect();
    prosody(&tw, &mut states);
    // warnings: characters in the gaps that are not spaces or punctuation the profile reads
    let mut warn_at = |range: Range<usize>| {
        for (off, c) in text[range.clone()].char_indices() {
            if c.is_whitespace()
                || matches!(
                    c,
                    ',' | '.'
                        | ';'
                        | ':'
                        | '!'
                        | '?'
                        | '…'
                        | '"'
                        | '“'
                        | '”'
                        | '«'
                        | '»'
                        | '\''
                        | '’'
                        | '-'
                        | '('
                        | ')'
                )
            {
                continue;
            }
            let s = range.start + off;
            out.warnings.push(Warning {
                span: s..s + c.len_utf8(),
                message: format!("{c:?} not read by {}", profile.name()),
            });
        }
    };
    for w in &tw {
        warn_at(w.gap.clone());
    }
    warn_at(final_gap.clone());
    // tokens
    let mut marks = Vec::new();
    clause_marks(text, &mut marks);
    let mut clause_of: Vec<usize> = Vec::new();
    let mut push = |tokens: &mut Vec<EspeakToken>,
                    ph: Phoneme,
                    stress: u8,
                    word: bool,
                    lengthen: bool,
                    clause: usize| {
        tokens.push(EspeakToken {
            ph,
            stress,
            word,
            lengthen,
            mark: '.',
        });
        clause_of.push(clause);
    };
    push(&mut out.tokens, Phoneme::Pause, 0, false, false, 0);
    for (wi, (w, st)) in tw.iter().zip(&states).enumerate() {
        for (pj, p) in w.pauses.iter().enumerate() {
            let ci = if w.boundary && pj < 2 {
                w.clause.saturating_sub(1)
            } else {
                w.clause
            };
            push(&mut out.tokens, p.phoneme(), 0, false, false, ci);
        }
        // a word-final l before a vowel-initial word with no pause between is not coda
        let link = tw.get(wi + 1).is_some_and(|n| n.pauses.is_empty())
            && states
                .get(wi + 1)
                .and_then(|n| n.segs.first())
                .is_some_and(|s| s.ph.is_vowel());
        let mut vi = 0;
        let mut phones = Vec::with_capacity(st.segs.len());
        for (pi, s) in st.segs.iter().enumerate() {
            let mut ph = s.ph;
            if pi + 1 == st.segs.len() && link {
                ph = match ph {
                    Phoneme::LCoda => Phoneme::L,
                    Phoneme::LDarkCoda => Phoneme::LDark,
                    p => p,
                };
            }
            let stress = if ph.is_vowel() {
                let l = st.levels.get(vi).copied().unwrap_or(0);
                vi += 1;
                l
            } else {
                0
            };
            push(&mut out.tokens, ph, stress, pi == 0, s.long, w.clause);
            phones.push(Phone {
                phoneme: ph,
                stress,
                lengthen: s.long,
            });
        }
        out.words.push(Word {
            span: w.span.clone(),
            reading: w.key.clone(),
            pauses_before: w.pauses.to_vec(),
            phones,
            clause: w.clause,
            stress_source: st.stress_source.clone(),
            segment_source: st.segment_source,
            promoted: st.promoted,
            tonic: st.tonic,
            normalized: if w.number { Some("number") } else { None },
        });
    }
    let last_clause = tw.last().map(|w| w.clause).unwrap_or(0);
    push(
        &mut out.tokens,
        Phoneme::Pause,
        0,
        false,
        false,
        last_clause,
    );
    // consonants and pauses take the level of the next vowel in their clause
    let mut next = 0u8;
    for t in out.tokens.iter_mut().rev() {
        if t.ph == Phoneme::Pause {
            t.stress = 0;
            next = 0;
        } else if t.ph.is_vowel() {
            next = t.stress;
        } else {
            t.stress = next;
        }
    }
    // the word phones' consonant levels, from the tokens
    let mut k = 1;
    for (w, word) in tw.iter().zip(out.words.iter_mut()) {
        k += w.pauses.len();
        for phone in word.phones.iter_mut() {
            phone.stress = out.tokens[k].stress;
            k += 1;
        }
    }
    // observed quirk: with an ellipsis ending the last clause or the one before it, the
    // last clause keeps the previous clause's number
    if last_clause > 0 {
        if let Some(first_last) = tw.iter().find(|w| w.clause == last_clause) {
            if text[final_gap.clone()].contains('…') || text[first_last.gap.clone()].contains('…')
            {
                for c in clause_of.iter_mut() {
                    if *c == last_clause {
                        *c = last_clause - 1;
                    }
                }
            }
        }
    }
    for (t, &c) in out.tokens.iter_mut().zip(&clause_of) {
        t.mark = marks.get(c).copied().unwrap_or('.');
    }
    Ok(())
}

/// The word's phonemes and lexical stress marks.
fn lexical(model: &Model, w: &TextWord) -> WordState {
    let letters: String = w.key.chars().filter(|&c| c != '\'').collect();
    let mut segs = Vec::new();
    if !has_vowel(&letters) {
        spell(&letters, &mut segs);
        let n = segs.iter().filter(|s| s.ph.is_vowel()).count();
        let mut marks = vec![Mark::Secondary; n];
        if let Some(m) = marks.last_mut() {
            *m = Mark::Primary;
        }
        return WordState {
            segs,
            marks,
            levels: Vec::new(),
            stress_source: StressSource::Spelled,
            segment_source: SegmentSource::Spelled,
            spelled: true,
            promoted: false,
            tonic: false,
        };
    }
    let entry = model.lexicon.get(&w.key);
    let segment_source = match entry {
        Some(e) => {
            match &e.phonemes {
                Some(p) => segs.extend_from_slice(p),
                None => {
                    key_to_phonemes(&w.key, &mut segs);
                    for s in segs.iter_mut() {
                        s.letter = '\0';
                    }
                }
            }
            SegmentSource::Lexicon
        }
        None => {
            key_to_phonemes(&w.key, &mut segs);
            SegmentSource::Rules
        }
    };
    let nsyl = segs.iter().filter(|s| s.ph.is_vowel()).count();
    let (marks, stress_source) = if w.izafet {
        if entry.is_some() {
            let (mut m, src) = predict_marks(model, &w.key, nsyl);
            // the izafet -i copies its stem's stress (observed: kuvve-i v6 v6 e6 i6)
            let n = m.len();
            if n >= 2 && m[n - 1] == Mark::Primary {
                m[n - 1] = Mark::Unstressed;
                m[n - 2] = Mark::Primary;
            }
            (m, src)
        } else {
            let stem: String = {
                let mut s = w.key.clone();
                s.pop();
                s
            };
            let (mut m, src) = predict_marks(model, &stem, nsyl.saturating_sub(1));
            m.push(Mark::Unstressed);
            (m, src)
        }
    } else {
        predict_marks(model, &w.key, nsyl)
    };
    WordState {
        segs,
        marks,
        levels: Vec::new(),
        stress_source,
        segment_source,
        spelled: false,
        promoted: false,
        tonic: false,
    }
}

/// A word's lexical stress marks: the lexicon's, or the apostrophe rule's, or the suffix
/// model's.
fn predict_marks(model: &Model, key: &str, nsyl: usize) -> (Vec<Mark>, StressSource) {
    if let Some(e) = model.lexicon.get(key) {
        if e.marks.len() == nsyl {
            return (e.marks.clone(), StressSource::Lexicon);
        }
    }
    if nsyl == 0 {
        return (Vec::new(), StressSource::DefaultFinal);
    }
    let (pos, src) = if let Some((_, tail)) = key.rsplit_once('\'') {
        match model.apostrophe.get(tail) {
            Some(&p) => (
                p,
                StressSource::ApostropheEnding {
                    ending: tail.to_string(),
                },
            ),
            None => (1, StressSource::ApostropheFinal),
        }
    } else {
        let (p, ending) = model.suffix_position(key);
        if ending.is_empty() {
            (p, StressSource::DefaultFinal)
        } else {
            (
                p,
                StressSource::SuffixModel {
                    ending: ending.to_string(),
                },
            )
        }
    };
    if pos == 0 {
        return (vec![Mark::Unstressed; nsyl], src);
    }
    let pos = (pos as usize).min(nsyl);
    let k = nsyl - pos;
    (
        (0..nsyl)
            .map(|i| {
                if i == k {
                    Mark::Primary
                } else {
                    Mark::Unstressed
                }
            })
            .collect(),
        src,
    )
}

/// The unstressed syllables after a word's primary stress (observed patterns).
fn post_pattern(n: usize, closed_final: bool, demote_last: bool, out: &mut [u8]) {
    match n {
        0 => {}
        1 => out[0] = 1,
        2 => {
            out[0] = 1;
            out[1] = if closed_final || demote_last { 0 } else { 1 };
        }
        _ => {
            for (i, o) in out.iter_mut().enumerate().take(n) {
                *o = if i == 2 { 1 } else { 0 };
            }
        }
    }
}

/// Stress levels per clause: promotion of clause-final function words, the tonic, and the
/// 0/1 rhythm of unstressed syllables (all as observed in the fixtures).
fn prosody(tw: &[TextWord], states: &mut [WordState]) {
    let mut start = 0;
    while start < tw.len() {
        let mut end = start + 1;
        while end < tw.len() && !tw[end].boundary {
            end += 1;
        }
        clause_levels(&tw[start..end], &mut states[start..end]);
        start = end;
    }
}

fn clause_levels(tw: &[TextWord], st: &mut [WordState]) {
    let n = tw.len();
    let has_primary = |s: &WordState| s.marks.contains(&Mark::Primary);
    // a stressable function word ending its clause is stressed
    if let Some(last) = st.last_mut() {
        if !last.marks.contains(&Mark::Primary)
            && STRESSABLE.contains(&tw[n - 1].key.as_str())
            && !last.marks.is_empty()
        {
            let k = last.marks.len() - 1;
            last.marks[k] = Mark::Primary;
            last.promoted = true;
        }
    }
    // a clause ending in an unstressed word changes the post-stress pattern
    let last_unstressed = !st.last().is_some_and(has_primary);
    if !st.iter().any(has_primary) {
        let mut target = if n == 2 { 0 } else { n - 1 };
        for (i, s) in st.iter().enumerate() {
            if s.marks.contains(&Mark::Secondary) {
                target = i;
            }
        }
        let s = &mut st[target];
        if !s.marks.is_empty() {
            let j = s
                .marks
                .iter()
                .position(|&m| m == Mark::Secondary)
                .unwrap_or(s.marks.len() - 1);
            s.marks[j] = Mark::Primary;
            s.promoted = true;
        }
    }
    let tonic = st.iter().rposition(has_primary);
    let mut prev: Option<u8> = None;
    for i in 0..n {
        let next_short = i + 1 < n && tw[i + 1].pauses.contains(&Pause::Short);
        let is_last = i + 1 == n;
        let prev_promoted = i > 0 && st[i - 1].promoted;
        let s = &mut st[i];
        let nsyl = s.marks.len();
        let mut levels = vec![0u8; nsyl];
        let closed_final = s.segs.last().is_some_and(|x| !x.ph.is_vowel());
        if s.spelled {
            for (j, l) in levels.iter_mut().enumerate() {
                *l = if s.marks[j] == Mark::Secondary {
                    3
                } else if Some(i) == tonic {
                    7
                } else {
                    6
                };
            }
        } else if let Some(k) = s.marks.iter().rposition(|&m| m == Mark::Primary) {
            for (j, l) in levels.iter_mut().enumerate().take(k) {
                *l = if j == 0 && prev != Some(1) { 1 } else { 0 };
            }
            levels[k] = if Some(i) == tonic { 7 } else { 6 };
            post_pattern(
                nsyl - k - 1,
                closed_final,
                next_short || (last_unstressed && !is_last),
                &mut levels[k + 1..],
            );
        } else {
            for (j, (level, mark)) in levels.iter_mut().zip(&s.marks).enumerate() {
                *level = if *mark == Mark::Secondary {
                    3
                } else if j == 0 {
                    match prev {
                        None => 1,
                        Some(1) => {
                            if OPEN_FUNCTION_WORDS.contains(&tw[i].key.as_str())
                                && !is_last
                                && !last_unstressed
                                && !next_short
                            {
                                1
                            } else {
                                0
                            }
                        }
                        Some(p) if p >= 4 && prev_promoted => 0,
                        _ => 1,
                    }
                } else {
                    1
                };
            }
        }
        if tw[i].izafet && nsyl >= 2 {
            levels[nsyl - 1] = levels[nsyl - 2];
        }
        if let Some(&l) = levels.last() {
            prev = Some(l);
        }
        s.tonic = Some(i) == tonic;
        if s.segment_source == SegmentSource::Rules
            && stress_dependent_segments(&mut s.segs, &levels)
        {
            s.segment_source = SegmentSource::RulesStressAdjusted;
        }
        s.levels = levels;
    }
}
