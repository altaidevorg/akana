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

    /// Question particle mI (Zemberek's questionRoot_S): takes copula and person.
    QuesRoot,
    /// After the evidential copula -(y)mIş, which takes the z-paradigm persons (-Im, -sIn).
    NounCopulaEvid,
    /// After the 3rd-person imperative -sIn (gel-sin, gel-sin-ler).
    VerbImp,
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
        let mut transitions = vec![
            // --- NOMINAL MORPHOTACTICS ---
            // NounRoot -> Diminutive (-cik)
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::NounRoot,
                suffix_type: SuffixType::DiminutiveCik,
                surface_templates: &[
                    "cik", "cık", "cuk", "cük", "çik", "çık", "çuk", "çük", "ıcık", "icik", "ucuk",
                    "ücük",
                ],
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
            // Relational Clitic (-ki) from Locative / Genitive or directly from Temporal Nouns (akşamki, dünkü)
            SuffixTransition {
                from_state: MorphState::NounCase,
                to_state: MorphState::NounRoot,
                suffix_type: SuffixType::CliticKi,
                surface_templates: &["ki", "kü"], // dünkü, bugünkü
            },
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::NounRoot,
                suffix_type: SuffixType::CliticKi,
                surface_templates: &["ki", "kü"], // akşamki, dünkü, bugünkü, yarınki
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
                surface_templates: &[
                    "dı", "di", "du", "dü", "tı", "ti", "tu", "tü", "ydı", "ydi", "ydu", "ydü",
                ],
            },
            // Nominal Derivations (Noun -> Noun/Adj/Verb)
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::DerivedNoun,
                suffix_type: SuffixType::DerivNess,
                surface_templates: &["lık", "lik", "luk", "lük", "lığ", "liğ", "luğ", "lüğ"],
            },
            SuffixTransition {
                from_state: MorphState::DerivedNoun,
                to_state: MorphState::NounRoot,
                suffix_type: SuffixType::DerivNess,
                surface_templates: &["lık", "lik", "luk", "lük", "lığ", "liğ", "luğ", "lüğ"],
            },
            SuffixTransition {
                from_state: MorphState::NounRoot,
                to_state: MorphState::DerivedNoun,
                suffix_type: SuffixType::DerivAgent,
                surface_templates: &["cı", "ci", "cu", "cü", "çı", "çi", "çu", "çü"],
            },
            SuffixTransition {
                from_state: MorphState::DerivedNoun,
                to_state: MorphState::DerivedNoun,
                suffix_type: SuffixType::DerivAgent,
                surface_templates: &["cı", "ci", "cu", "cü", "çı", "çi", "çu", "çü"],
            },
            SuffixTransition {
                from_state: MorphState::DerivedNoun,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseDat,
                surface_templates: &["e", "a", "ye", "ya"],
            },
            SuffixTransition {
                from_state: MorphState::DerivedNoun,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseLoc,
                surface_templates: &["de", "da", "te", "ta"],
            },
            SuffixTransition {
                from_state: MorphState::DerivedNoun,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseAbl,
                surface_templates: &["den", "dan", "ten", "tan"],
            },
            SuffixTransition {
                from_state: MorphState::DerivedNoun,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseAcc,
                surface_templates: &["i", "ı", "u", "ü", "yi", "yı", "yu", "yü"],
            },
            SuffixTransition {
                from_state: MorphState::DerivedNoun,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseGen,
                surface_templates: &["in", "ın", "un", "ün", "nin", "nın", "nun", "nün"],
            },
            SuffixTransition {
                from_state: MorphState::DerivedNoun,
                to_state: MorphState::NounPlural,
                suffix_type: SuffixType::Plural,
                surface_templates: &["ler", "lar"],
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
            SuffixTransition {
                from_state: MorphState::VerbAbility,
                to_state: MorphState::VerbTense,
                suffix_type: SuffixType::TenseProg,
                surface_templates: &["iyor", "ıyor", "uyor", "üyor", "yor", "mekte", "makta"],
            },
            SuffixTransition {
                from_state: MorphState::VerbAbility,
                to_state: MorphState::VerbTense,
                suffix_type: SuffixType::TenseAorist,
                surface_templates: &["ir", "ır", "ur", "ür", "er", "ar"],
            },
            SuffixTransition {
                from_state: MorphState::VerbAbility,
                to_state: MorphState::VerbTense,
                suffix_type: SuffixType::TenseFut,
                surface_templates: &["ecek", "acak"],
            },
            SuffixTransition {
                from_state: MorphState::VerbAbility,
                to_state: MorphState::VerbTense,
                suffix_type: SuffixType::TensePast,
                surface_templates: &["di", "dı", "du", "dü"],
            },
            // Passive and Causative recursive verb derivation
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::VerbRoot,
                suffix_type: SuffixType::DerivPassive,
                surface_templates: &["il", "ıl", "ul", "ül", "in", "ın", "un", "ün", "nil", "nıl"],
            },
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::VerbRoot,
                suffix_type: SuffixType::DerivCausative,
                surface_templates: &[
                    "dir", "dır", "dur", "dür", "tir", "tır", "tur", "tür", "t", "it", "ıt", "ut",
                    "üt",
                ],
            },
            // VerbRoot -> Negation
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::VerbNegation,
                suffix_type: SuffixType::VerbNeg,
                surface_templates: &["me", "ma", "mi", "mı", "mu", "mü"],
            },
            // VerbRoot -> Negative Ability (-(y)eme / -(y)ama : gidemem, yapamam, gidemiyor)
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::VerbNegation,
                suffix_type: SuffixType::VerbNegAbility,
                surface_templates: &["eme", "ama", "yeme", "yama", "yemi", "yamı", "emi", "amı"],
            },
            SuffixTransition {
                from_state: MorphState::VerbAbility,
                to_state: MorphState::VerbNegation,
                suffix_type: SuffixType::VerbNeg,
                surface_templates: &["me", "ma", "mi", "mı", "mu", "mü"],
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
                surface_templates: &["di", "dı", "du", "dü"],
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
                surface_templates: &["miş", "mış", "muş", "müş"],
            },
            // Progressive
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::VerbTense,
                suffix_type: SuffixType::TenseProg,
                surface_templates: &["iyor", "ıyor", "uyor", "üyor", "yor", "mekte", "makta"],
            },
            SuffixTransition {
                from_state: MorphState::VerbNegation,
                to_state: MorphState::VerbTense,
                suffix_type: SuffixType::TenseProg,
                surface_templates: &["yor", "mıyor", "miyor", "muyor", "müyor"],
            },
            // Future
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::VerbTense,
                suffix_type: SuffixType::TenseFut,
                surface_templates: &[
                    "ecek", "acak", "yecek", "yacak", "eceğ", "acağ", "yeceğ", "yacağ",
                ],
            },
            SuffixTransition {
                from_state: MorphState::VerbNegation,
                to_state: MorphState::VerbTense,
                suffix_type: SuffixType::TenseFut,
                surface_templates: &["yecek", "yacak", "yeceğ", "yacağ"],
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
                surface_templates: &[
                    "nız", "niz", "nuz", "nüz", "sınız", "siniz", "sunuz", "sünüz",
                ],
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
                from_state: MorphState::VerbNegation,
                to_state: MorphState::DerivedAdj,
                suffix_type: SuffixType::DerivPresPart,
                surface_templates: &["yen", "yan", "en", "an"],
            },
            SuffixTransition {
                from_state: MorphState::DerivedAdj,
                to_state: MorphState::NounPlural,
                suffix_type: SuffixType::Plural,
                surface_templates: &["lar", "ler"],
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
                // The future participle -(y)AcAk used to be listed here as PastPart; it is
                // DerivFutPart below.
                surface_templates: &[
                    "dik", "dık", "duk", "dük", "tik", "tık", "tuk", "tük", "diğ", "dığ", "duğ",
                    "düğ", "tiğ", "tığ", "tuğ", "tüğ",
                ],
            },
            SuffixTransition {
                from_state: MorphState::DerivedAdj,
                to_state: MorphState::NounPossessive,
                suffix_type: SuffixType::Possessive1Sg,
                surface_templates: &["im", "ım", "um", "üm"],
            },
            SuffixTransition {
                from_state: MorphState::DerivedAdj,
                to_state: MorphState::NounPossessive,
                suffix_type: SuffixType::Possessive2Sg,
                surface_templates: &["in", "ın", "un", "ün"],
            },
            SuffixTransition {
                from_state: MorphState::DerivedAdj,
                to_state: MorphState::NounPossessive,
                suffix_type: SuffixType::Possessive3Sg,
                surface_templates: &["i", "ı", "u", "ü"],
            },
            SuffixTransition {
                from_state: MorphState::DerivedAdj,
                to_state: MorphState::NounPossessive,
                suffix_type: SuffixType::Possessive1Pl,
                surface_templates: &["imiz", "ımız", "umuz", "ümüz"],
            },
            SuffixTransition {
                from_state: MorphState::DerivedAdj,
                to_state: MorphState::NounPossessive,
                suffix_type: SuffixType::Possessive2Pl,
                surface_templates: &["iniz", "ınız", "unuz", "ünüz"],
            },
            SuffixTransition {
                from_state: MorphState::DerivedAdj,
                to_state: MorphState::NounPossessive,
                suffix_type: SuffixType::Possessive3Pl,
                surface_templates: &["leri", "ları"],
            },
            SuffixTransition {
                from_state: MorphState::DerivedAdj,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseLoc,
                surface_templates: &["de", "da", "te", "ta"],
            },
            SuffixTransition {
                from_state: MorphState::DerivedAdj,
                to_state: MorphState::NounCase,
                suffix_type: SuffixType::CaseAbl,
                surface_templates: &["tan", "ten", "dan", "den"],
            },
            // Compound Tense / Copula on Verbs (geliyordu, yapmıştı, gidecekti, yapılabilmektedir)
            SuffixTransition {
                from_state: MorphState::VerbTense,
                to_state: MorphState::NounCopula,
                suffix_type: SuffixType::CopulaPres3Sg,
                surface_templates: &["dur", "dür", "dır", "dir", "tur", "tür", "tır", "tir"],
            },
            SuffixTransition {
                from_state: MorphState::VerbTense,
                to_state: MorphState::NounCopula,
                suffix_type: SuffixType::CopulaPast,
                surface_templates: &[
                    "du", "dü", "dı", "di", "tu", "tü", "tı", "ti", "ydu", "ydi", "ydu", "ydü",
                ],
            },
            SuffixTransition {
                from_state: MorphState::VerbTense,
                to_state: MorphState::NounCopulaEvid,
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
                surface_templates: &[
                    "dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür", "t", "ıt", "it", "ut",
                    "üt",
                ],
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
                surface_templates: &[
                    "ince", "ınca", "ünce", "unca", "yince", "yınca", "yünce", "yunca",
                ],
            },
            SuffixTransition {
                from_state: MorphState::VerbRoot,
                to_state: MorphState::DerivedAdv,
                suffix_type: SuffixType::DerivAdvIp,
                surface_templates: &["ip", "ıp", "up", "üp", "yip", "yıp", "yup", "yüp"],
            },
        ];
        transitions.extend(Self::zemberek_transitions());
        transitions
    }

    /// Transitions ported from Zemberek's TurkishMorphotactics that the original graph lacked:
    /// the nominal copula and its persons (nVerb_S), the question particle (questionRoot_S),
    /// tense + -(y)sA / -(y)ken, verb forms after negation and ability, and the -(y)AlI,
    /// -mAdAn, -(y)IcI, -sAl, -CA and imperative -sIn suffixes.
    fn zemberek_transitions() -> Vec<SuffixTransition> {
        use MorphState::*;
        use SuffixType::*;
        let t = |from_state, to_state, suffix_type, surface_templates| SuffixTransition {
            from_state,
            to_state,
            suffix_type,
            surface_templates,
        };

        const PAST_COP: &[&str] = &[
            "dı", "di", "du", "dü", "tı", "ti", "tu", "tü", "ydı", "ydi", "ydu", "ydü",
        ];
        const EVID_COP: &[&str] = &["mış", "miş", "muş", "müş", "ymış", "ymiş", "ymuş", "ymüş"];
        const COND_COP: &[&str] = &["sa", "se", "ysa", "yse"];
        const WHILE: &[&str] = &["ken", "yken"];
        const PRES_3SG: &[&str] = &["dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür"];
        // z-paradigm persons. 1sg/1pl only with the buffer -y- on bare nominals: without it
        // they coincide with the possessives (kitab-ım "my book").
        const P1SG_Y: &[&str] = &["yım", "yim", "yum", "yüm"];
        const P1SG: &[&str] = &["ım", "im", "um", "üm", "yım", "yim", "yum", "yüm"];
        const P2SG: &[&str] = &["sın", "sin", "sun", "sün"];
        const P1PL_Y: &[&str] = &["yız", "yiz", "yuz", "yüz"];
        const P1PL: &[&str] = &["ız", "iz", "uz", "üz", "yız", "yiz", "yuz", "yüz"];
        const P2PL: &[&str] = &["sınız", "siniz", "sunuz", "sünüz"];
        const P3PL: &[&str] = &["lar", "ler"];

        let mut v = Vec::new();

        // Nominal copula on every nominal state (nVerb_S: elma-ydı, elma-ymış, elma-ysa,
        // elma-yken, elma-yım). NounRoot already has -DIr and -DI.
        for from in [
            NounPlural,
            NounPossessive,
            NounCase,
            DerivedAdj,
            DerivedNoun,
        ] {
            v.push(t(from, NounCopula, CopulaPast, PAST_COP));
            v.push(t(from, NounCopula, CopulaPres3Sg, PRES_3SG));
        }
        for from in [
            NounRoot,
            NounPlural,
            NounPossessive,
            NounCase,
            DerivedAdj,
            DerivedNoun,
        ] {
            v.push(t(from, NounCopulaEvid, CopulaEvid, EVID_COP));
            v.push(t(from, NounCopula, CopulaCond, COND_COP));
            v.push(t(from, DerivedAdv, DerivWhile, WHILE));
            let (p1sg, p1pl) = if from == NounCase {
                (P1SG, P1PL)
            } else {
                (P1SG_Y, P1PL_Y)
            };
            v.push(t(from, VerbPerson, CopulaPres1Sg, p1sg));
            v.push(t(from, VerbPerson, CopulaPres2Sg, P2SG));
            v.push(t(from, VerbPerson, CopulaPres1Pl, p1pl));
            v.push(t(from, VerbPerson, CopulaPres2Pl, P2PL));
        }
        // Persons after the past and conditional copula (k-paradigm) and after the
        // evidential copula (z-paradigm). NounCopula already has -m, -k and -lAr.
        v.push(t(NounCopula, VerbPerson, Verb2Sg, &["n"]));
        v.push(t(
            NounCopula,
            VerbPerson,
            Verb2Pl,
            &["nız", "niz", "nuz", "nüz"],
        ));
        v.push(t(NounCopulaEvid, VerbPerson, Verb1Sg, P1SG));
        v.push(t(NounCopulaEvid, VerbPerson, Verb2Sg, P2SG));
        v.push(t(NounCopulaEvid, VerbPerson, Verb1Pl, P1PL));
        v.push(t(NounCopulaEvid, VerbPerson, Verb2Pl, P2PL));
        v.push(t(NounCopulaEvid, VerbPerson, Verb3Pl, P3PL));

        // Question particle mI: mi-yim, mi-sin, mi-yiz, mi-siniz, mi-dir, mi-ydi-m, mi-ymiş.
        v.push(t(QuesRoot, VerbPerson, CopulaPres1Sg, P1SG_Y));
        v.push(t(QuesRoot, VerbPerson, CopulaPres2Sg, P2SG));
        v.push(t(QuesRoot, VerbPerson, CopulaPres1Pl, P1PL_Y));
        v.push(t(QuesRoot, VerbPerson, CopulaPres2Pl, P2PL));
        v.push(t(
            QuesRoot,
            NounCopula,
            CopulaPres3Sg,
            &["dır", "dir", "dur", "dür"],
        ));
        v.push(t(
            QuesRoot,
            NounCopula,
            CopulaPast,
            &["ydı", "ydi", "ydu", "ydü"],
        ));
        v.push(t(
            QuesRoot,
            NounCopulaEvid,
            CopulaEvid,
            &["ymış", "ymiş", "ymuş", "ymüş"],
        ));

        // Tense + conditional copula and + -ken (gelir-se-n, geliyor-sa, geldi-yse, gelir-ken,
        // gelmez-ken, gelecek-ken).
        v.push(t(VerbTense, NounCopula, CopulaCond, COND_COP));
        v.push(t(VerbTense, DerivedAdv, DerivWhile, &["ken"]));

        // Verb forms after negation -mA (vNeg_S): gel-me-mek, gel-me-me, gel-me-yiş,
        // gel-me-diğ-i, gel-me-yecek, gel-me-yince, gel-me-yip, gel-me-yerek, gel-me-sin.
        v.push(t(VerbNegation, NounRoot, DerivInfinitive, &["mek", "mak"]));
        v.push(t(VerbNegation, NounRoot, DerivActNoun, &["me", "ma"]));
        v.push(t(VerbNegation, NounRoot, DerivManner, &["yiş", "yış"]));
        v.push(t(
            VerbNegation,
            DerivedAdj,
            DerivPastPart,
            &["dik", "dık", "diğ", "dığ"],
        ));
        v.push(t(
            VerbNegation,
            DerivedAdj,
            DerivFutPart,
            &["yecek", "yacak", "yeceğ", "yacağ"],
        ));
        v.push(t(
            VerbNegation,
            DerivedAdv,
            DerivAdvInce,
            &["yince", "yınca"],
        ));
        v.push(t(VerbNegation, DerivedAdv, DerivAdvIp, &["yip", "yıp"]));
        v.push(t(
            VerbNegation,
            DerivedAdv,
            DerivAdvErek,
            &["yerek", "yarak"],
        ));
        v.push(t(VerbNegation, VerbImp, TenseImp, &["sin", "sın"]));

        // Verb forms after ability -(y)AbIl (always -il, so front unrounded harmony):
        // yap-abil-mek, yap-abil-en, yap-abil-diğ-i, yap-abil-ince, yap-abil-miş.
        v.push(t(
            MorphState::VerbAbility,
            NounRoot,
            DerivInfinitive,
            &["mek"],
        ));
        v.push(t(MorphState::VerbAbility, NounRoot, DerivActNoun, &["me"]));
        v.push(t(
            MorphState::VerbAbility,
            DerivedAdj,
            DerivPresPart,
            &["en"],
        ));
        v.push(t(
            MorphState::VerbAbility,
            DerivedAdj,
            DerivPastPart,
            &["dik", "diğ"],
        ));
        v.push(t(
            MorphState::VerbAbility,
            DerivedAdj,
            DerivFutPart,
            &["ecek", "eceğ"],
        ));
        v.push(t(
            MorphState::VerbAbility,
            DerivedAdv,
            DerivAdvInce,
            &["ince"],
        ));
        v.push(t(MorphState::VerbAbility, DerivedAdv, DerivAdvIp, &["ip"]));
        v.push(t(
            MorphState::VerbAbility,
            DerivedAdv,
            DerivAdvErek,
            &["erek"],
        ));
        v.push(t(MorphState::VerbAbility, VerbTense, TenseEvid, &["miş"]));
        v.push(t(MorphState::VerbAbility, VerbTense, TenseNec, &["meli"]));

        // Future participle (vFutPart_S): oku-yacağ-ı, gel-ecek.
        v.push(t(
            VerbRoot,
            DerivedAdj,
            DerivFutPart,
            &[
                "ecek", "acak", "yecek", "yacak", "eceğ", "acağ", "yeceğ", "yacağ",
            ],
        ));

        // Converbs -(y)AlI (vSinceDoingSo_S) and -mAdAn (vWithoutHavingDoneSo_S).
        v.push(t(
            VerbRoot,
            DerivedAdv,
            DerivSinceDoingSo,
            &["alı", "eli", "yalı", "yeli"],
        ));
        v.push(t(
            VerbRoot,
            DerivedAdv,
            DerivWithoutHavingDoneSo,
            &["madan", "meden"],
        ));

        // Agent noun -(y)IcI from verbs (vAgt_S): izle-yici, koru-yucu, öldür-ücü.
        v.push(t(
            VerbRoot,
            DerivedNoun,
            DerivAgent,
            &["ıcı", "ici", "ucu", "ücü", "yıcı", "yici", "yucu", "yücü"],
        ));

        // 3rd-person imperative (vImp_S + A3sg -sIn, A3pl -sInlAr): gel-sin, ol-sun-lar.
        v.push(t(
            VerbRoot,
            VerbImp,
            TenseImp,
            &["sın", "sin", "sun", "sün"],
        ));
        v.push(t(VerbImp, VerbPerson, Verb3Pl, P3PL));

        // Adjective derivations: -sIz/-lI + -lIk (haksız-lık), -sAl (duygu-sal), -CA (Ly on
        // adjectives: açık-ça, güzel-ce; Equ on nouns and pronouns: aile-ce, sen-ce). The
        // analyzer allows Ly only after an adjective and Equ only after a non-adjective.
        v.push(t(
            DerivedAdj,
            DerivedNoun,
            DerivNess,
            &["lık", "lik", "luk", "lük", "lığ", "liğ", "luğ", "lüğ"],
        ));
        v.push(t(NounRoot, DerivedAdj, DerivRelated, &["sal", "sel"]));
        // A derived noun (-lIk, -CI, -(y)IcI) takes the whole nominal paradigm, as Zemberek's
        // derivations return to noun_S; it lacked the instrumental and the possessives
        // (sabırsızlık-la, kitaplığ-ı, yolcu-muz).
        v.push(t(
            DerivedNoun,
            NounCase,
            CaseInst,
            &["la", "le", "yla", "yle"],
        ));
        v.push(t(
            DerivedNoun,
            NounPossessive,
            Possessive1Sg,
            &["m", "ım", "im", "um", "üm"],
        ));
        v.push(t(
            DerivedNoun,
            NounPossessive,
            Possessive2Sg,
            &["n", "ın", "in", "un", "ün"],
        ));
        v.push(t(
            DerivedNoun,
            NounPossessive,
            Possessive3Sg,
            &["ı", "i", "u", "ü", "sı", "si", "su", "sü"],
        ));
        v.push(t(
            DerivedNoun,
            NounPossessive,
            Possessive1Pl,
            &["mız", "miz", "muz", "müz", "ımız", "imiz", "umuz", "ümüz"],
        ));
        v.push(t(
            DerivedNoun,
            NounPossessive,
            Possessive2Pl,
            &["nız", "niz", "nuz", "nüz", "ınız", "iniz", "unuz", "ünüz"],
        ));
        v.push(t(
            DerivedNoun,
            NounPossessive,
            Possessive3Pl,
            &["ları", "leri"],
        ));
        for from in [NounRoot, DerivedAdj] {
            v.push(t(from, DerivedAdv, DerivLy, &["ca", "ce", "ça", "çe"]));
        }
        for from in [NounRoot, NounPossessive] {
            v.push(t(from, NounCase, CaseEqu, &["ca", "ce", "ça", "çe"]));
        }

        v
    }
}
