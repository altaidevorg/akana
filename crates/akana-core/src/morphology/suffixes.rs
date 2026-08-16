//! Suffix definitions and morpheme representations for Turkish.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SuffixType {
    // Nominal Inflectional
    Plural,         // -ler, -lar
    Possessive1Sg,  // -m, -ım, -im, -um, -üm
    Possessive2Sg,  // -n, -ın, -in, -un, -ün
    Possessive3Sg,  // -ı, -i, -u, -ü, -sı, -si, -su, -sü
    Possessive1Pl,  // -mız, -miz, -muz, -müz, -ımız...
    Possessive2Pl,  // -nız, -niz, -nuz, -nüz, -ınız...
    Possessive3Pl,  // -leri, -ları
    CaseNom,        // Ø
    CaseAcc,        // -ı, -i, -u, -ü, -yı, -yi, -yu, -yü
    CaseDat,        // -a, -e, -ya, -ye
    CaseLoc,        // -da, -de, -ta, -te
    CaseAbl,        // -dan, -den, -tan, -ten
    CaseGen,        // -ın, -in, -un, -ün, -nın, -nin, -nun, -nün
    CaseInst,       // -la, -le, -yla, -yle
    CaseEqu,        // -ca, -ce, -ça, -çe

    // Predicative / Copula
    CopulaPres1Sg,  // -(y)ım, -(y)im, -(y)um, -(y)üm
    CopulaPres2Sg,  // -sın, -sin, -sun, -sün
    CopulaPres3Sg,  // -dır, -dir, -dur, -dür, -tır, -tir, -tur, -tür
    CopulaPres1Pl,  // -(y)ız, -(y)iz, -(y)uz, -(y)üz
    CopulaPres2Pl,  // -sınız, -siniz, -sunuz, -sünüz
    CopulaPres3Pl,  // -dırlar, -dirler, -durlar, -dürler
    CopulaPast,     // -(y)dı, -(y)di, -(y)du, -(y)dü, -(y)tı, -(y)ti...
    CopulaEvid,     // -(y)mış, -(y)miş, -(y)muş, -(y)müş
    CopulaCond,     // -(y)sa, -(y)se

    // Verbal Inflectional
    VerbNeg,        // -ma, -me
    VerbAbility,    // -(y)abil, -(y)ebil
    TensePast,      // -dı, -di, -du, -dü, -tı, -ti, -tu, -tü
    TenseEvid,      // -mış, -miş, -muş, -müş
    TenseProg,      // -(ı)yor, -(i)yor, -(u)yor, -(ü)yor
    TenseAorist,    // -r, -ar, -er, -ır, -ir, -ur, -ür
    TenseFut,       // -(y)acak, -(y)ecek
    TenseNec,       // -malı, -meli
    TenseOpt,       // -(y)a, -(y)e
    TenseCond,      // -sa, -se

    // Verb Agreement (Person)
    Verb1Sg,        // -m, -ım, -im, -um, -üm, -(y)ım, -(y)im
    Verb2Sg,        // -n, -sın, -sin, -sun, -sün
    Verb3Sg,        // Ø
    Verb1Pl,        // -k, -ız, -iz, -uz, -üz, -(y)ız, -(y)iz
    Verb2Pl,        // -nız, -niz, -nuz, -nüz, -sınız, -siniz...
    Verb3Pl,        // -lar, -ler

    // Derivational
    DerivNess,      // -lık, -lik, -luk, -lük (Noun -> Noun/Adj)
    DerivWith,      // -lı, -li, -lu, -lü (Noun -> Adj)
    DerivWithout,   // -sız, -siz, -suz, -süz (Noun -> Neg Adj)
    DerivAgent,     // -cı, -ci, -cu, -cü, -çı, -çi, -çu, -çü (Noun -> Noun)
    DerivInfinitive,// -mak, -mek (Verb -> Noun)
    DerivActNoun,   // -ma, -me (Verb -> Noun)
    DerivManner,    // -(y)ış, -(y)iş, -(y)uş, -(y)üş (Verb -> Noun)
    DerivPresPart,  // -(y)an, -(y)en (Verb -> Adj)
    DerivPastPart,  // -dık, -dik, -duk, -dük, -tık... (Verb -> Adj)
    DerivFutPart,   // -(y)acak, -(y)ecek (Verb -> Adj)
    DerivAdvErek,   // -(y)arak, -(y)erek (Verb -> Adv)
    DerivAdvInce,   // -(y)ınca, -(y)ince, -(y)unca, -(y)ünce (Verb -> Adv)
    DerivAdvIp,     // -(y)ıp, -(y)ip, -(y)up, -(y)üp (Verb -> Adv)
    DerivCausative, // -dır, -dir, -dur, -dür, -t, -tır... (Verb -> Verb)
    DerivPassive,   // -(ı)l, -(i)l, -(u)l, -(ü)l, -(ı)n, -(i)n... (Verb -> Verb)
}

impl SuffixType {
    pub fn tag(&self) -> &'static str {
        match self {
            SuffixType::Plural => "A3pl",
            SuffixType::Possessive1Sg => "P1sg",
            SuffixType::Possessive2Sg => "P2sg",
            SuffixType::Possessive3Sg => "P3sg",
            SuffixType::Possessive1Pl => "P1pl",
            SuffixType::Possessive2Pl => "P2pl",
            SuffixType::Possessive3Pl => "P3pl",
            SuffixType::CaseNom => "Nom",
            SuffixType::CaseAcc => "Acc",
            SuffixType::CaseDat => "Dat",
            SuffixType::CaseLoc => "Loc",
            SuffixType::CaseAbl => "Abl",
            SuffixType::CaseGen => "Gen",
            SuffixType::CaseInst => "Ins",
            SuffixType::CaseEqu => "Equ",
            SuffixType::CopulaPres1Sg => "CopPres1sg",
            SuffixType::CopulaPres2Sg => "CopPres2sg",
            SuffixType::CopulaPres3Sg => "CopPres3sg",
            SuffixType::CopulaPres1Pl => "CopPres1pl",
            SuffixType::CopulaPres2Pl => "CopPres2pl",
            SuffixType::CopulaPres3Pl => "CopPres3pl",
            SuffixType::CopulaPast => "CopPast",
            SuffixType::CopulaEvid => "CopEvid",
            SuffixType::CopulaCond => "CopCond",
            SuffixType::VerbNeg => "Neg",
            SuffixType::VerbAbility => "Abil",
            SuffixType::TensePast => "Past",
            SuffixType::TenseEvid => "Evid",
            SuffixType::TenseProg => "Prog",
            SuffixType::TenseAorist => "Aor",
            SuffixType::TenseFut => "Fut",
            SuffixType::TenseNec => "Nec",
            SuffixType::TenseOpt => "Opt",
            SuffixType::TenseCond => "Cond",
            SuffixType::Verb1Sg => "A1sg",
            SuffixType::Verb2Sg => "A2sg",
            SuffixType::Verb3Sg => "A3sg",
            SuffixType::Verb1Pl => "A1pl",
            SuffixType::Verb2Pl => "A2pl",
            SuffixType::Verb3Pl => "A3pl",
            SuffixType::DerivNess => "Ness",
            SuffixType::DerivWith => "With",
            SuffixType::DerivWithout => "Without",
            SuffixType::DerivAgent => "Agt",
            SuffixType::DerivInfinitive => "Inf",
            SuffixType::DerivActNoun => "ActN",
            SuffixType::DerivManner => "Manner",
            SuffixType::DerivPresPart => "PresPart",
            SuffixType::DerivPastPart => "PastPart",
            SuffixType::DerivFutPart => "FutPart",
            SuffixType::DerivAdvErek => "AdvErek",
            SuffixType::DerivAdvInce => "AdvInce",
            SuffixType::DerivAdvIp => "AdvIp",
            SuffixType::DerivCausative => "Caus",
            SuffixType::DerivPassive => "Pass",
        }
    }
}
