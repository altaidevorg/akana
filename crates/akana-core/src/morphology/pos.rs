//! Part of speech (POS) definitions and morphological feature tags for Turkish.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrimaryPos {
    Noun,
    Verb,
    Adj,
    Adv,
    Pron,
    Num,
    Conj,
    Postp,
    Interj,
    Q,       // Question particle (mi/mı/mu/mü)
    Punc,
    Dup,     // Duplication (ikileme)
    Unknown,
}

impl PrimaryPos {
    pub fn as_str(&self) -> &'static str {
        match self {
            PrimaryPos::Noun => "Noun",
            PrimaryPos::Verb => "Verb",
            PrimaryPos::Adj => "Adj",
            PrimaryPos::Adv => "Adv",
            PrimaryPos::Pron => "Pron",
            PrimaryPos::Num => "Num",
            PrimaryPos::Conj => "Conj",
            PrimaryPos::Postp => "Postp",
            PrimaryPos::Interj => "Interj",
            PrimaryPos::Q => "Q",
            PrimaryPos::Punc => "Punc",
            PrimaryPos::Dup => "Dup",
            PrimaryPos::Unknown => "Unk",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SecondaryPos {
    None,
    ProperNoun,
    Time,
    Ordinal,
    Cardinal,
    Distributive,
    Percentage,
    Demonstrative,
    Personal,
    Reflexive,
    Reciprocal,
    Quantitive,
    Question,
    Relative,
}

impl SecondaryPos {
    pub fn as_str(&self) -> &'static str {
        match self {
            SecondaryPos::None => "",
            SecondaryPos::ProperNoun => "Prop",
            SecondaryPos::Time => "Time",
            SecondaryPos::Ordinal => "Ord",
            SecondaryPos::Cardinal => "Card",
            SecondaryPos::Distributive => "Dist",
            SecondaryPos::Percentage => "Percent",
            SecondaryPos::Demonstrative => "Demons",
            SecondaryPos::Personal => "Pers",
            SecondaryPos::Reflexive => "Reflex",
            SecondaryPos::Reciprocal => "Recip",
            SecondaryPos::Quantitive => "Quant",
            SecondaryPos::Question => "Ques",
            SecondaryPos::Relative => "Rel",
        }
    }
}

bitflags::bitflags! {
    /// Phonological and morphological flags for dictionary items.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct RootAttr: u32 {
        const NONE = 0;
        /// Root mutates consonant when affix starting with vowel is appended (p->b, ç->c, t->d, k->ğ/g)
        const VOICING = 1 << 0;
        /// Root loses its second syllable vowel (burun -> burnu, akıl -> aklı)
        const VOWEL_DROP = 1 << 1;
        /// Root doubles its last consonant (hak -> hakkı, his -> hissi)
        const CONSONANT_DOUBLING = 1 << 2;
        /// Root does not follow regular vowel harmony (loanwords like saat, kalp, alkol)
        const INVERSE_HARMONY = 1 << 3;
        /// Root is a proper noun (capitalized, takes apostrophe)
        const PROPER_NOUN = 1 << 4;
        /// Root ends with a vowel that mutates (e.g. de -> diyen, ye -> yiyen)
        const VOWEL_MUTATION = 1 << 5;
        /// Compound noun with 3rd person possessive ending in base form (e.g. atkuyruğu, zeytinyağı)
        const COMPOUND_P3SG = 1 << 6;
        /// Plural form by default
        const PLURAL = 1 << 7;
    }
}
