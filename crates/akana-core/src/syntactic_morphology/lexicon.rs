//! Lexicon layer for Syntactic Expressive Morphology.
//! Incorporates gold-standard root categorization, zero-derivation elimination, and morphophonemic flags.

use std::collections::HashMap;
use lazy_static::lazy_static;
use crate::phonology::to_turkish_lower;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SyntacticPOS {
    Noun,
    Verb,
    Adj,
    Adv,
    Pron,
    Num,
    Conj,
    Postp,
    Det,
    Interj,
    ProperNoun,
}

impl SyntacticPOS {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Noun => "NN",
            Self::Verb => "VB",
            Self::Adj => "ADJ",
            Self::Adv => "ADV",
            Self::Pron => "PRON",
            Self::Num => "NUM",
            Self::Conj => "CONJ",
            Self::Postp => "POSTP",
            Self::Det => "DET",
            Self::Interj => "INTERJ",
            Self::ProperNoun => "PROP",
        }
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct SyntacticRootFlags: u16 {
        const NONE = 0;
        const VOICING = 1 << 0;             // Consonant softening (kitap -> kitab-)
        const VOWEL_DROP = 1 << 1;          // Epenthetic vowel drop (akıl -> akl-)
        const CONSONANT_DOUBLING = 1 << 2;  // Gemination (hak -> hakk-, his -> hiss-)
        const FRONT_HARMONY_EXCEPTION = 1 << 3; // Foreign loan with front vowel suffixes (saat, rol, alkol, kalp, harf)
        const COMPOUND_P3SG = 1 << 4;       // Possessive compound stem (ayakkabı -> ayakkabı-)
        const DUP_LAST_VOWEL = 1 << 5;
    }
}

#[derive(Debug, Clone)]
pub struct SyntacticRootEntry {
    pub lemma: String,
    pub primary_pos: SyntacticPOS,
    pub secondary_pos: Vec<SyntacticPOS>,
    pub flags: SyntacticRootFlags,
}

lazy_static! {
    /// Exception roots with front vowel harmony (Arabic/Persian/French loans)
    static ref FRONT_VOWEL_EXCEPTIONS: std::collections::HashSet<&'static str> = {
        let mut s = std::collections::HashSet::new();
        for &w in &[
            "saat", "rol", "gol", "alkol", "kalp", "harf", "kontrol", "sembol",
            "petrol", "protokol", "santral", "metal", "kemal", "hilal", "ihtimal",
            "istiklal", "misal", "usul", "kabul", "meşgul", "makul", "mahsul"
        ] {
            s.insert(w);
        }
        s
    };

    /// Curated dual-class roots eliminating zero-derivations
    static ref DUAL_CLASS_ROOTS: HashMap<&'static str, Vec<SyntacticPOS>> = {
        let mut m = HashMap::new();
        m.insert("güzel", vec![SyntacticPOS::Adj, SyntacticPOS::Adv, SyntacticPOS::Noun]);
        m.insert("iyi", vec![SyntacticPOS::Adj, SyntacticPOS::Adv, SyntacticPOS::Noun]);
        m.insert("kötü", vec![SyntacticPOS::Adj, SyntacticPOS::Adv, SyntacticPOS::Noun]);
        m.insert("hızlı", vec![SyntacticPOS::Adj, SyntacticPOS::Adv]);
        m.insert("yavaş", vec![SyntacticPOS::Adj, SyntacticPOS::Adv]);
        m.insert("doğru", vec![SyntacticPOS::Adj, SyntacticPOS::Adv, SyntacticPOS::Postp, SyntacticPOS::Noun]);
        m.insert("yalnız", vec![SyntacticPOS::Adj, SyntacticPOS::Adv, SyntacticPOS::Conj]);
        m.insert("rahat", vec![SyntacticPOS::Adj, SyntacticPOS::Adv, SyntacticPOS::Noun]);
        m.insert("soğuk", vec![SyntacticPOS::Adj, SyntacticPOS::Noun]);
        m.insert("sıcak", vec![SyntacticPOS::Adj, SyntacticPOS::Noun]);
        m.insert("kolay", vec![SyntacticPOS::Adj, SyntacticPOS::Adv]);
        m.insert("zor", vec![SyntacticPOS::Adj, SyntacticPOS::Adv, SyntacticPOS::Noun]);
        m.insert("erken", vec![SyntacticPOS::Adv, SyntacticPOS::Adj]);
        m.insert("geç", vec![SyntacticPOS::Adv, SyntacticPOS::Adj]);
        m.insert("açık", vec![SyntacticPOS::Adj, SyntacticPOS::Adv, SyntacticPOS::Noun]);
        m.insert("kapalı", vec![SyntacticPOS::Adj, SyntacticPOS::Adv]);
        m
    };
}

pub struct SyntacticLexicon {
    entries: HashMap<String, Vec<SyntacticRootEntry>>,
}

impl Default for SyntacticLexicon {
    fn default() -> Self {
        Self::new()
    }
}

impl SyntacticLexicon {
    pub fn new() -> Self {
        let mut lex = Self {
            entries: HashMap::new(),
        };
        lex.load_default_lexicon();
        lex
    }

    fn load_default_lexicon(&mut self) {
        let lex_str = include_str!("google_lexicon.txt");
        for line in lex_str.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            let raw_word = parts[0];
            let lower_word = to_turkish_lower(raw_word);
            let mut pos = SyntacticPOS::Noun;
            let mut flags = SyntacticRootFlags::NONE;

            for part in &parts[1..] {
                match *part {
                    "Verb" => pos = SyntacticPOS::Verb,
                    "Adj" => pos = SyntacticPOS::Adj,
                    "Adv" => pos = SyntacticPOS::Adv,
                    "Pron" => pos = SyntacticPOS::Pron,
                    "Num" => pos = SyntacticPOS::Num,
                    "Conj" => pos = SyntacticPOS::Conj,
                    "Postp" => pos = SyntacticPOS::Postp,
                    "Det" => pos = SyntacticPOS::Det,
                    "Interj" => pos = SyntacticPOS::Interj,
                    "ProperNoun" => pos = SyntacticPOS::ProperNoun,
                    "Voicing" => flags |= SyntacticRootFlags::VOICING,
                    "VowelDrop" => flags |= SyntacticRootFlags::VOWEL_DROP,
                    "Doubling" => flags |= SyntacticRootFlags::CONSONANT_DOUBLING,
                    "CompoundP3sg" => flags |= SyntacticRootFlags::COMPOUND_P3SG,
                    _ => {}
                }
            }

            if FRONT_VOWEL_EXCEPTIONS.contains(lower_word.as_str()) {
                flags |= SyntacticRootFlags::FRONT_HARMONY_EXCEPTION;
            }

            let secondary = DUAL_CLASS_ROOTS.get(lower_word.as_str()).cloned().unwrap_or_default();

            let entry = SyntacticRootEntry {
                lemma: raw_word.to_string(),
                primary_pos: pos,
                secondary_pos: secondary,
                flags,
            };

            self.entries.entry(lower_word).or_default().push(entry);
        }
    }

    pub fn lookup(&self, surface: &str) -> Option<&Vec<SyntacticRootEntry>> {
        self.entries.get(surface)
    }
}
