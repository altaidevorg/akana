//! The 47 phoneme and pause symbols of the `espeak-compat` profile: eSpeak NG's Turkish mnemonics
//! as EchoCraft's student TTS reads them, with the IPA and the phoneme class eSpeak reports for
//! each (observed in the reference fixtures' `phoneme-table.json`, not taken from eSpeak's files).

use serde::{Serialize, Serializer};

/// A phoneme or pause symbol. [`Phoneme::symbol`] is the eSpeak mnemonic the student TTS reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Phoneme {
    /// `a` [a]
    A,
    /// `e` [e], tense
    E,
    /// `E` [ɛ], lax e (word-final or closed syllable)
    ELax,
    /// `&` [æ], e before a tautosyllabic m, n or l
    Ae,
    /// `i` [i], tense
    I,
    /// `I` [ɪ], lax i
    ILax,
    /// `@` [ɯ], the letter ı
    Ih,
    /// `o` [o], tense
    O,
    /// `O` [ɔ], lax o
    OLax,
    /// `u` [u], tense
    U,
    /// `U` [ʊ], lax u
    ULax,
    /// `y` [y], tense ü
    Ue,
    /// `Y` [ø], lax ü
    UeLax,
    /// `W` [œ], the letter ö
    Oe,
    /// `b`
    B,
    /// `c` [c], k in some words ending in -ken
    KPalatal,
    /// `d`
    D,
    /// `dZ` [dʒ], the letter c
    Dzh,
    /// `f`
    F,
    /// `g` [ɡ]
    G,
    /// `J` [ɟ], g before a front vowel
    GPalatal,
    /// `h`
    H,
    /// `Z` [ʒ], the letter j
    Zh,
    /// `k`
    K,
    /// `l`, clear l before a front vowel
    L,
    /// `l/`, clear l closing a syllable
    LCoda,
    /// `L` [ɫ], dark l before a back vowel
    LDark,
    /// `L/` [ɫ], dark l closing a syllable
    LDarkCoda,
    /// `m`
    M,
    /// `n`
    N,
    /// `p`
    P,
    /// `R` [r], trill
    R,
    /// `*` [ɾ], tap between vowels inside a word
    Tap,
    /// `r/` [r], rare word-final variant
    RCoda,
    /// `s`
    S,
    /// `S` [ʃ], the letter ş
    Sh,
    /// `t`
    T,
    /// `tS` [tʃ], the letter ç
    Tsh,
    /// `v`
    V,
    /// `j`, the letter y, and ğ after e or i
    J,
    /// `z`
    Z,
    /// `Q` [ɣ], ğ after a consonant (a typo in the corpus)
    Gh,
    /// `w`, outside the student's vocabulary
    W,
    /// `_`, pause: one at each edge, and after a clause's `_:`
    Pause,
    /// `_:`, pause ending a non-final clause; a pair `_: _:` at quotes and after a word with an
    /// apostrophe
    PauseClause,
    /// `_!`, short pause before ve, veya, yoksa, ama, ki and kime
    PauseShort,
    /// `_|`, outside the student's vocabulary
    PauseBar,
}

/// eSpeak's phoneme class of a symbol, as the fixtures report it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhonemeClass {
    Vowel,
    Stop,
    VoicedStop,
    Fricative,
    VoicedFricative,
    Nasal,
    Liquid,
    Pause,
}

impl PhonemeClass {
    /// The class name the fixtures use (`vowel`, `stop`, `vstop`, ...).
    pub fn as_str(self) -> &'static str {
        match self {
            PhonemeClass::Vowel => "vowel",
            PhonemeClass::Stop => "stop",
            PhonemeClass::VoicedStop => "vstop",
            PhonemeClass::Fricative => "fricative",
            PhonemeClass::VoicedFricative => "vfricative",
            PhonemeClass::Nasal => "nasal",
            PhonemeClass::Liquid => "liquid",
            PhonemeClass::Pause => "pause",
        }
    }
}

use Phoneme::*;

/// Every symbol, for tables and parsing.
pub const ALL_PHONEMES: [Phoneme; 47] = [
    A,
    E,
    ELax,
    Ae,
    I,
    ILax,
    Ih,
    O,
    OLax,
    U,
    ULax,
    Ue,
    UeLax,
    Oe,
    B,
    KPalatal,
    D,
    Dzh,
    F,
    G,
    GPalatal,
    H,
    Zh,
    K,
    L,
    LCoda,
    LDark,
    LDarkCoda,
    M,
    N,
    P,
    R,
    Tap,
    RCoda,
    S,
    Sh,
    T,
    Tsh,
    V,
    J,
    Z,
    Gh,
    W,
    Pause,
    PauseClause,
    PauseShort,
    PauseBar,
];

impl Phoneme {
    /// The eSpeak mnemonic (`a`, `E`, `dZ`, `L/`, `_:` ...).
    pub fn symbol(self) -> &'static str {
        match self {
            A => "a",
            E => "e",
            ELax => "E",
            Ae => "&",
            I => "i",
            ILax => "I",
            Ih => "@",
            O => "o",
            OLax => "O",
            U => "u",
            ULax => "U",
            Ue => "y",
            UeLax => "Y",
            Oe => "W",
            B => "b",
            KPalatal => "c",
            D => "d",
            Dzh => "dZ",
            F => "f",
            G => "g",
            GPalatal => "J",
            H => "h",
            Zh => "Z",
            K => "k",
            L => "l",
            LCoda => "l/",
            LDark => "L",
            LDarkCoda => "L/",
            M => "m",
            N => "n",
            P => "p",
            R => "R",
            Tap => "*",
            RCoda => "r/",
            S => "s",
            Sh => "S",
            T => "t",
            Tsh => "tS",
            V => "v",
            J => "j",
            Z => "z",
            Gh => "Q",
            W => "w",
            Pause => "_",
            PauseClause => "_:",
            PauseShort => "_!",
            PauseBar => "_|",
        }
    }

    /// The symbol for a mnemonic, if it is one of the profile's.
    pub fn from_symbol(s: &str) -> Option<Phoneme> {
        ALL_PHONEMES.iter().copied().find(|p| p.symbol() == s)
    }

    /// eSpeak's IPA for the symbol, `None` for pauses.
    pub fn ipa(self) -> Option<&'static str> {
        Some(match self {
            A => "a",
            E => "e",
            ELax => "ɛ",
            Ae => "æ",
            I => "i",
            ILax => "ɪ",
            Ih => "ɯ",
            O => "o",
            OLax => "ɔ",
            U => "u",
            ULax => "ʊ",
            Ue => "y",
            UeLax => "ø",
            Oe => "œ",
            B => "b",
            KPalatal => "c",
            D => "d",
            Dzh => "dʒ",
            F => "f",
            G => "ɡ",
            GPalatal => "ɟ",
            H => "h",
            Zh => "ʒ",
            K => "k",
            L | LCoda => "l",
            LDark | LDarkCoda => "ɫ",
            M => "m",
            N => "n",
            P => "p",
            R | RCoda => "r",
            Tap => "ɾ",
            S => "s",
            Sh => "ʃ",
            T => "t",
            Tsh => "tʃ",
            V => "v",
            J => "j",
            Z => "z",
            Gh => "ɣ",
            W => "w",
            Pause | PauseClause | PauseShort | PauseBar => return None,
        })
    }

    /// eSpeak's IPA for the symbol with its length mark: `aː`, but `ɯɯ` for a long ı (observed).
    pub fn ipa_long(self) -> Option<&'static str> {
        Some(match self {
            A => "aː",
            E => "eː",
            ELax => "ɛː",
            Ae => "æː",
            I => "iː",
            ILax => "ɪː",
            Ih => "ɯɯ",
            O => "oː",
            OLax => "ɔː",
            U => "uː",
            ULax => "ʊː",
            Ue => "yː",
            UeLax => "øː",
            Oe => "œː",
            B => "bː",
            KPalatal => "cː",
            D => "dː",
            Dzh => "dʒː",
            F => "fː",
            G => "ɡː",
            GPalatal => "ɟː",
            H => "hː",
            Zh => "ʒː",
            K => "kː",
            L | LCoda => "lː",
            LDark | LDarkCoda => "ɫː",
            M => "mː",
            N => "nː",
            P => "pː",
            R | RCoda => "rː",
            Tap => "ɾː",
            S => "sː",
            Sh => "ʃː",
            T => "tː",
            Tsh => "tʃː",
            V => "vː",
            J => "jː",
            Z => "zː",
            Gh => "ɣː",
            W => "wː",
            Pause | PauseClause | PauseShort | PauseBar => return None,
        })
    }

    /// eSpeak's phoneme class of the symbol.
    pub fn class(self) -> PhonemeClass {
        match self {
            A | E | ELax | Ae | I | ILax | Ih | O | OLax | U | ULax | Ue | UeLax | Oe => {
                PhonemeClass::Vowel
            }
            KPalatal | K | P | T | Tsh => PhonemeClass::Stop,
            B | D | Dzh | G | GPalatal | Tap => PhonemeClass::VoicedStop,
            F | H | S | Sh => PhonemeClass::Fricative,
            Zh | V | Z | Gh => PhonemeClass::VoicedFricative,
            M | N => PhonemeClass::Nasal,
            L | LCoda | LDark | LDarkCoda | R | RCoda | J | W => PhonemeClass::Liquid,
            Pause | PauseClause | PauseShort | PauseBar => PhonemeClass::Pause,
        }
    }

    pub fn is_vowel(self) -> bool {
        self.class() == PhonemeClass::Vowel
    }

    pub fn is_pause(self) -> bool {
        self.class() == PhonemeClass::Pause
    }

    /// Whether the student TTS's 45-symbol vocabulary has the symbol (`w` and `_|` it reads as
    /// `_`).
    pub fn in_student_vocabulary(self) -> bool {
        !matches!(self, W | PauseBar)
    }
}

impl Serialize for Phoneme {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.symbol())
    }
}
