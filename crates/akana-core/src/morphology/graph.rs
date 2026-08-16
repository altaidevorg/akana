//! Morphotactics state machine and suffix transitions for Turkish.

use super::suffixes::SuffixType;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MorphState {
    // Nominal States
    NounRoot,
    NounPlural,
    NounPossessive,
    NounCase,
    NounCopula,
    NounTerminal,

    // Verbal States
    VerbRoot,
    VerbAbility,
    VerbNegation,
    VerbTense,
    VerbPerson,
    VerbTerminal,

    // Derivation States
    DerivedNoun,
    DerivedAdj,
    DerivedAdv,
    DerivedVerb,
}

#[derive(Debug, Clone)]
pub struct SuffixTransition {
    pub from_state: MorphState,
    pub to_state: MorphState,
    pub suffix_type: SuffixType,
    pub surface_templates: &'static [&'static str],
}

pub struct TurkishMorphotactics;

impl TurkishMorphotactics {
    /// Returns the complete set of valid suffix transitions in standard Turkish.
    pub fn transitions() -> Vec<SuffixTransition> {
        vec![
            // --- NOMINAL MORPHOTACTICS ---
            // NounRoot -> Diminutive (-cik)
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::NounRoot,
                suffix_type: SuffixType::DiminutiveCik,
                surface_templates: &["cik", "cık", "cuk", "cük", "çik", "çık", "çuk", "çük", "ıcık", "icik", "ucuk", "ücük"],
            },

            // NounRoot -> Plural
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::NounPlural,
                suffix_type: SuffixType::Plural,
                surface_templates: &["lar", "ler"],
            },
            // NounRoot -> Possessive
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::NounPossessive,
                suffix_type: SuffixType::Possessive1Sg,
                surface_templates: &["m", "ım", "im", "um", "üm"],
            },
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::NounPossessive,
                suffix_type: SuffixType::Possessive2Sg,
                surface_templates: &["n", "ın", "in", "un", "ün"],
            },
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::NounPossessive,
                suffix_type: SuffixType::Possessive3Sg,
                surface_templates: &["ı", "i", "u", "ü", "sı", "si", "su", "sü"],
            },
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::NounPossessive,
                suffix_type: SuffixType::Possessive1Pl,
                surface_templates: &["mız", "miz", "muz", "müz", "ımız", "imiz", "umuz", "ümüz"],
            },
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::NounPossessive,
                suffix_type: SuffixType::Possessive2Pl,
                surface_templates: &["nız", "niz", "nuz", "nüz", "ınız", "iniz", "unuz", "ünüz"],
            },
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::NounPossessive,
                suffix_type: SuffixType::Possessive3Pl,
                surface_templates: &["ları", "leri"],
            },

            // NounPlural -> Possessive
            SuffixTransition {
                from_state: MorphState::NounPlural,
                to_state: MorphState::NounPossessive,
                suffix_type: SuffixType::Possessive1Sg,
                surface_templates: &["ım", "im"],
            },
            SuffixTransition {
                from_state: MorphState::NounPlural,
                to_state: MorphState::NounPossessive,
                suffix_type: SuffixType::Possessive2Sg,
                surface_templates: &["in", "ın"],
            },
            SuffixTransition {
                from_state: MorphState::NounPlural,
                to_state: MorphState::NounPossessive,
                suffix_type: SuffixType::Possessive3Sg,
                surface_templates: &["i", "ı"],
            },
            SuffixTransition {
                from_state: MorphState::NounPlural,
                to_state: MorphState::NounPossessive,
                suffix_type: SuffixType::Possessive1Pl,
                surface_templates: &["imiz", "ımız"],
            },
            SuffixTransition {
                from_state: MorphState::NounPlural,
                to_state: MorphState::NounPossessive,
                suffix_type: SuffixType::Possessive2Pl,
                surface_templates: &["iniz", "ınız"],
            },

            // NounRoot / NounPlural / NounPossessive -> Case
            // Accusative
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseAcc,
                surface_templates: &["ı", "i", "u", "ü", "yı", "yi", "yu", "yü"],
            },
            SuffixTransition {
                from_state: MorphState::NounPlural,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseAcc,
                surface_templates: &["ı", "i"],
            },
            SuffixTransition {
                from_state: MorphState::NounPossessive,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseAcc,
                surface_templates: &["ı", "i", "u", "ü", "nı", "ni", "nu", "nü"],
            },

            // Dative
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseDat,
                surface_templates: &["a", "e", "ya", "ye"],
            },
            SuffixTransition {
                from_state: MorphState::NounPlural,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseDat,
                surface_templates: &["a", "e"],
            },
            SuffixTransition {
                from_state: MorphState::NounPossessive,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseDat,
                surface_templates: &["a", "e", "na", "ne"],
            },

            // Locative
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseLoc,
                surface_templates: &["da", "de", "ta", "te"],
            },
            SuffixTransition {
                from_state: MorphState::NounPlural,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseLoc,
                surface_templates: &["da", "de"],
            },
            SuffixTransition {
                from_state: MorphState::NounPossessive,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseLoc,
                surface_templates: &["da", "de", "nda", "nde"],
            },

            // Ablative
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseAbl,
                surface_templates: &["dan", "den", "tan", "ten"],
            },
            SuffixTransition {
                from_state: MorphState::NounPlural,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseAbl,
                surface_templates: &["dan", "den"],
            },
            SuffixTransition {
                from_state: MorphState::NounPossessive,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseAbl,
                surface_templates: &["dan", "den", "ndan", "nden"],
            },

            // Genitive
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseGen,
                surface_templates: &["ın", "in", "un", "ün", "nın", "nin", "nun", "nün"],
            },
            SuffixTransition {
                from_state: MorphState::NounPlural,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseGen,
                surface_templates: &["ın", "in"],
            },
            SuffixTransition {
                from_state: MorphState::NounPossessive,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseGen,
                surface_templates: &["ın", "in", "un", "ün", "nın", "nin", "nun", "nün"],
            },

            // Instrumental
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseInst,
                surface_templates: &["la", "le", "yla", "yle"],
            },
            SuffixTransition {
                from_state: MorphState::NounPlural,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseInst,
                surface_templates: &["la", "le"],
            },
            SuffixTransition {
                from_state: MorphState::NounPossessive,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseInst,
                surface_templates: &["la", "le", "yla", "yle"],
            },

            // Relational Clitic (-ki) from Locative / Genitive
            // e.g. ev-de-ki, masa-da-ki, biz-im-ki, okul-un-ki
            SuffixTransition {
                from_state: MorphState::NounCase,
                to_state: MorphState::NounRoot,
                suffix_type: SuffixType::CliticKi,
                surface_templates: &["ki", "kü"], // dünkü, bugünkü
            },

            // Nominal Predicative / Copula
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::NounCopula,
                suffix_type: SuffixType::CopulaPres3Sg,
                surface_templates: &["dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür"],
            },
            SuffixTransition {
                from_state: MorphState::NounCase,
                to_state: MorphState::NounCopula,
                suffix_type: SuffixType::CopulaPres3Sg,
                surface_templates: &["dır", "dir", "dur", "dür"],
            },
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::NounCopula,
                suffix_type: SuffixType::CopulaPast,
                surface_templates: &["dı", "di", "du", "dü", "tı", "ti", "tu", "tü", "ydı", "ydi", "ydu", "ydü"],
            },

            // Nominal Derivations (Noun -> Noun/Adj/Verb)
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::DerivedNoun,
                suffix_type: SuffixType::DerivNess,
                surface_templates: &["lık", "lik", "luk", "lük"],
            },
            SuffixTransition {
                from_state: MorphState::DerivedNoun,
                to_state: MorphState::NounRoot,
                suffix_type: SuffixType::DerivNess,
                surface_templates: &["lık", "lik", "luk", "lük"],
            },
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::DerivedNoun,
                suffix_type: SuffixType::DerivAgent,
                surface_templates: &["cı", "ci", "cu", "cü", "çı", "çi", "çu", "çü"],
            },
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::DerivedAdj,
                suffix_type: SuffixType::DerivWith,
                surface_templates: &["lı", "li", "lu", "lü"],
            },
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::DerivedAdj,
                suffix_type: SuffixType::DerivWithout,
                surface_templates: &["sız", "siz", "suz", "süz"],
            },
            // Noun -> Verb (-le / -leş / -lendir)
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::VerbRoot,
                suffix_type: SuffixType::DerivLe,
                surface_templates: &["le", "la"],
            },
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::VerbRoot,
                suffix_type: SuffixType::DerivLes,
                surface_templates: &["leş", "laş"],
            },
            SuffixTransition {
                from_state: MorphState::DerivedAdj,
                to_state: MorphState::VerbRoot,
                suffix_type: SuffixType::DerivLes,
                surface_templates: &["leş", "laş"],
            },
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::VerbRoot,
                suffix_type: SuffixType::DerivLendir,
                surface_templates: &["lendir", "landır"],
            },

            // --- VERBAL MORPHOTACTICS ---
            // VerbRoot -> Ability
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::VerbAbility,
                suffix_type: SuffixType::VerbAbility,
                surface_templates: &["ebil", "abil", "yebil", "yabil"],
            },
            // VerbRoot -> Negation
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::VerbNegation,
                suffix_type: SuffixType::VerbNeg,
                surface_templates: &["me", "ma"],
            },
            // VerbRoot -> Negative Ability (-(y)eme / -(y)ama : gidemem, yapamam)
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::VerbNegation,
                suffix_type: SuffixType::VerbNegAbility,
                surface_templates: &["eme", "ama", "yeme", "yama"],
            },
            SuffixTransition {
                from_state: MorphState::VerbAbility,
                to_state: MorphState::VerbNegation,
                suffix_type: SuffixType::VerbNeg,
                surface_templates: &["me", "ma"],
            },

            // VerbRoot / VerbNegation -> Tense
            // Past
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::VerbTense,
                suffix_type: SuffixType::TensePast,
                surface_templates: &["di", "dı", "du", "dü", "ti", "tı", "tu", "tü"],
            },
            SuffixTransition {
                from_state: MorphState::VerbNegation,
                to_state: MorphState::VerbTense,
                suffix_type: SuffixType::TensePast,
                surface_templates: &["di", "dı"],
            },
            // Evidential / Reported Past
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::VerbTense,
                suffix_type: SuffixType::TenseEvid,
                surface_templates: &["miş", "mış", "muş", "müş"],
            },
            SuffixTransition {
                from_state: MorphState::VerbNegation,
                to_state: MorphState::VerbTense,
                suffix_type: SuffixType::TenseEvid,
                surface_templates: &["miş", "mış"],
            },
            // Progressive
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::VerbTense,
                suffix_type: SuffixType::TenseProg,
                surface_templates: &["iyor", "ıyor", "uyor", "üyor", "yor"],
            },
            SuffixTransition {
                from_state: MorphState::VerbNegation,
                to_state: MorphState::VerbTense,
                suffix_type: SuffixType::TenseProg,
                surface_templates: &["mıyor", "miyor", "muyor", "müyor"],
            },
            // Future
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::VerbTense,
                suffix_type: SuffixType::TenseFut,
                surface_templates: &["ecek", "acak", "yecek", "yacak"],
            },
            SuffixTransition {
                from_state: MorphState::VerbNegation,
                to_state: MorphState::VerbTense,
                suffix_type: SuffixType::TenseFut,
                surface_templates: &["yecek", "yacak"],
            },
            // Aorist
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::VerbTense,
                suffix_type: SuffixType::TenseAorist,
                surface_templates: &["r", "ar", "er", "ır", "ir", "ur", "ür"],
            },
            SuffixTransition {
                from_state: MorphState::VerbNegation,
                to_state: MorphState::VerbTense,
                suffix_type: SuffixType::TenseAorist,
                surface_templates: &["z", "mez", "maz"],
            },
            // Necessitative
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::VerbTense,
                suffix_type: SuffixType::TenseNec,
                surface_templates: &["meli", "malı"],
            },
            // Optative
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::VerbTense,
                suffix_type: SuffixType::TenseOpt,
                surface_templates: &["e", "a", "ye", "ya"],
            },
            // Conditional
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::VerbTense,
                suffix_type: SuffixType::TenseCond,
                surface_templates: &["se", "sa"],
            },

            // VerbTense -> Person
            SuffixTransition {
                from_state: MorphState::VerbTense,
                to_state: MorphState::VerbPerson,
                suffix_type: SuffixType::Verb1Sg,
                surface_templates: &["m", "ım", "im", "um", "üm", "yım", "yim", "yum", "yüm"],
            },
            SuffixTransition {
                from_state: MorphState::VerbTense,
                to_state: MorphState::VerbPerson,
                suffix_type: SuffixType::Verb2Sg,
                surface_templates: &["n", "sın", "sin", "sun", "sün"],
            },
            SuffixTransition {
                from_state: MorphState::VerbTense,
                to_state: MorphState::VerbPerson,
                suffix_type: SuffixType::Verb1Pl,
                surface_templates: &["k", "ız", "iz", "uz", "üz", "yız", "yiz", "yuz", "yüz"],
            },
            SuffixTransition {
                from_state: MorphState::VerbTense,
                to_state: MorphState::VerbPerson,
                suffix_type: SuffixType::Verb2Pl,
                surface_templates: &["nız", "niz", "nuz", "nüz", "sınız", "siniz", "sunuz", "sünüz"],
            },
            SuffixTransition {
                from_state: MorphState::VerbTense,
                to_state: MorphState::VerbPerson,
                suffix_type: SuffixType::Verb3Pl,
                surface_templates: &["ler", "lar"],
            },

            // Verb Derivations
            // Infinitive (Verb -> Noun)
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::NounRoot,
                suffix_type: SuffixType::DerivInfinitive,
                surface_templates: &["mek", "mak"],
            },
            // Action Noun (Verb -> Noun)
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::NounRoot,
                suffix_type: SuffixType::DerivActNoun,
                surface_templates: &["me", "ma"],
            },
            // Manner (Verb -> Noun)
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::NounRoot,
                suffix_type: SuffixType::DerivManner,
                surface_templates: &["iş", "ış", "uş", "üş", "yiş", "yış", "yuş", "yüş"],
            },
            // Verb -> Noun (-im / -gi)
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::NounRoot,
                suffix_type: SuffixType::DerivIm,
                surface_templates: &["im", "ım", "um", "üm"],
            },
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::NounRoot,
                suffix_type: SuffixType::DerivGi,
                surface_templates: &["gi", "gı", "gu", "gü", "ki", "kı", "ku", "kü"],
            },
            // Verb -> Adj (-gin / -gen)
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::DerivedAdj,
                suffix_type: SuffixType::DerivGin,
                surface_templates: &["gin", "gın", "gun", "gün", "kin", "kın", "kun", "kün"],
            },
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::DerivedAdj,
                suffix_type: SuffixType::DerivGen,
                surface_templates: &["gen", "gan", "ken", "kan"],
            },
            // Participles (Verb -> Adj / Nominalized Noun)
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::DerivedAdj,
                suffix_type: SuffixType::DerivPresPart,
                surface_templates: &["en", "an", "yen", "yan"],
            },
            SuffixTransition {
                from_state: MorphState::DerivedAdj,
                to_state: MorphState::NounRoot,
                suffix_type: SuffixType::Plural,
                surface_templates: &["lar", "ler"],
            },
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::DerivedAdj,
                suffix_type: SuffixType::DerivPastPart,
                surface_templates: &["dik", "dık", "duk", "dük", "tik", "tık", "tuk", "tük", "ceğ", "cağ", "eceğ", "acağ"],
            },
            SuffixTransition {
                from_state: MorphState::DerivedAdj,
                to_state: MorphState::NounPossessive,
                suffix_type: SuffixType::Possessive3Sg,
                surface_templates: &["i", "ı", "u", "ü"],
            },
            SuffixTransition {
                from_state: MorphState::DerivedAdj,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseAbl,
                surface_templates: &["tan", "ten", "dan", "den"],
            },

            // Compound Tense / Copula on Verbs (geliyordu, yapmıştı, gidecekti, oynuyorlardı)
            SuffixTransition {
                from_state: MorphState::VerbTense,
                to_state: MorphState::NounCopula,
                suffix_type: SuffixType::CopulaPast,
                surface_templates: &["du", "dü", "dı", "di", "tu", "tü", "tı", "ti", "ydu", "ydi", "ydu", "ydü"],
            },
            SuffixTransition {
                from_state: MorphState::VerbTense,
                to_state: MorphState::NounCopula,
                suffix_type: SuffixType::CopulaEvid,
                surface_templates: &["muş", "müş", "mış", "miş"],
            },
            SuffixTransition {
                from_state: MorphState::NounCopula,
                to_state: MorphState::VerbPerson,
                suffix_type: SuffixType::Verb3Pl,
                surface_templates: &["lar", "ler"],
            },
            SuffixTransition {
                from_state: MorphState::NounCopula,
                to_state: MorphState::VerbPerson,
                suffix_type: SuffixType::Verb1Sg,
                surface_templates: &["m"],
            },
            SuffixTransition {
                from_state: MorphState::NounCopula,
                to_state: MorphState::VerbPerson,
                suffix_type: SuffixType::Verb1Pl,
                surface_templates: &["k"],
            },

            // Voice Derivations (Passive & Causative on Verb)
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::VerbRoot,
                suffix_type: SuffixType::DerivPassive,
                surface_templates: &["ıl", "il", "ul", "ül", "ın", "in", "un", "ün", "l", "n"],
            },
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::VerbRoot,
                suffix_type: SuffixType::DerivCausative,
                surface_templates: &["dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür", "t", "ıt", "it", "ut", "üt"],
            },

            // Adverbials (Verb -> Adv)
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::DerivedAdv,
                suffix_type: SuffixType::DerivAdvErek,
                surface_templates: &["erek", "arak", "yerek", "yarak"],
            },
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::DerivedAdv,
                suffix_type: SuffixType::DerivAdvInce,
                surface_templates: &["ince", "ınca", "ünce", "unca", "yince", "yınca", "yünce", "yunca"],
            },
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::DerivedAdv,
                suffix_type: SuffixType::DerivAdvIp,
                surface_templates: &["ip", "ıp", "up", "üp", "yip", "yıp", "yup", "yüp"],
            },
        ]
    }
}
