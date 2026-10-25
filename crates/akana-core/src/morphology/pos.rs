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
    Q, // Question particle (mi/mı/mu/mü)
    Punc,
    Dup, // Duplication (ikileme)
    Unknown,
    Det, // Determiner (bir, bu, her, çok); takes no suffixes, as in Zemberek
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
            PrimaryPos::Det => "Det",
        }
    }

    /// Parses a lexicon POS column (case-insensitive). Accepts Zemberek's `Ques` for `Q`.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.to_lowercase().as_str() {
            "noun" => PrimaryPos::Noun,
            "verb" => PrimaryPos::Verb,
            "adj" => PrimaryPos::Adj,
            "adv" => PrimaryPos::Adv,
            "pron" => PrimaryPos::Pron,
            "num" => PrimaryPos::Num,
            "conj" => PrimaryPos::Conj,
            "postp" => PrimaryPos::Postp,
            "interj" => PrimaryPos::Interj,
            "q" | "ques" => PrimaryPos::Q,
            "punc" => PrimaryPos::Punc,
            "dup" => PrimaryPos::Dup,
            "det" => PrimaryPos::Det,
            _ => return None,
        })
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
    Abbreviation,
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
            SecondaryPos::Abbreviation => "Abbrv",
        }
    }

    /// Parses a lexicon secondary-POS token (case-insensitive): the short names of
    /// [`SecondaryPos::as_str`] plus the long forms the lexicon files use (`ProperNoun`).
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.to_lowercase().as_str() {
            "prop" | "proper" | "propernoun" => SecondaryPos::ProperNoun,
            "time" => SecondaryPos::Time,
            "ord" | "ordinal" => SecondaryPos::Ordinal,
            "card" | "cardinal" => SecondaryPos::Cardinal,
            "dist" | "distributive" => SecondaryPos::Distributive,
            "percent" => SecondaryPos::Percentage,
            "demons" | "demonstrative" => SecondaryPos::Demonstrative,
            "pers" | "personal" => SecondaryPos::Personal,
            "reflex" | "reflexive" => SecondaryPos::Reflexive,
            "recip" | "reciprocal" => SecondaryPos::Reciprocal,
            "quant" | "quantitive" => SecondaryPos::Quantitive,
            "ques" | "question" => SecondaryPos::Question,
            "rel" | "relative" => SecondaryPos::Relative,
            "abbrv" | "abbreviation" => SecondaryPos::Abbreviation,
            _ => return None,
        })
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
        /// Plural form by default (Zemberek's ImplicitPlural: hayvanat, annemler)
        const PLURAL = 1 << 7;
        /// A root ending in a stop consonant that keeps it before a vowel (saat-i, hukuk-u)
        const NO_VOICING = 1 << 8;
        /// A verb whose aorist is -Ar (yap-ar, et-er); most monosyllables
        const AORIST_A = 1 << 9;
        /// A verb whose aorist is -Ir (gel-ir, oku-r); polysyllables and 13 monosyllables
        const AORIST_I = 1 << 10;
        /// A vowel-final verb that loses its vowel before -Iyor (bekle -> bekl-iyor)
        const PROGRESSIVE_VOWEL_DROP = 1 << 11;
        /// A verb whose passive is -In / -InIl (oku-n, bil-in) rather than -Il
        const PASSIVE_IN = 1 << 12;
        /// A verb whose causative is -t (oku-t, getir-t) rather than -DIr
        const CAUSATIVE_T = 1 << 13;
        /// A verb that is reflexive in itself (kaşınmak)
        const REFLEXIVE = 1 << 14;
        /// A verb that is reciprocal in itself (boğuşmak)
        const RECIPROCAL = 1 << 15;
        /// A verb that ends like a reciprocal but is not one
        const NON_RECIPROCAL = 1 << 16;
        /// A plural that carries a 1st-person possessive (annemler)
        const IMPLICIT_P1SG = 1 << 17;
        /// A plural that carries a 2nd-person possessive (annenler)
        const IMPLICIT_P2SG = 1 << 18;
        /// A noun that is dative in itself (içeri, dışarı)
        const IMPLICIT_DATIVE = 1 << 19;
        /// A family noun that takes no possessive (Zemberek's FamilyMember)
        const FAMILY_MEMBER = 1 << 20;
        /// A word that takes no suffixes
        const NO_SUFFIX = 1 << 21;
        /// The stem of a CompoundP3sg noun without its possessive (cumhurbaşkan- of
        /// cumhurbaşkanı), generated at load time; it cannot stand alone
        const COMPOUND_P3SG_ROOT = 1 << 22;
    }
}
