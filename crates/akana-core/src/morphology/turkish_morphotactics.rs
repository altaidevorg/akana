//! Turkish morphotactics: a port of Zemberek's `TurkishMorphotactics` graph (zemberek-nlp
//! 0.17, `morphology/morphotactics/TurkishMorphotactics.java`).
//!
//! State names and transitions follow Zemberek's, so its documentation and tests apply. Where
//! Zemberek's own graph has a slip it is fixed rather than copied:
//! - conditions name dictionary items by lemma, POS and secondary POS, so items the Java code
//!   looks up under an id the lexicon does not use (`o_Pron_Demons` is `o_Pron_Demons_1`,
//!   `falan_Pron_Pers` is `falan_Pron`) are still found;
//! - the question particle's -DIr before -lAr goes to its own state, not the pronoun's;
//! - the reflexive kendi takes the empty 3rd-person possessive (kendi), not the 2nd-person one;
//! - -(m)sI's buffer m is only for vowel-final stems; Zemberek's `+msI` made ev-si "ev-like";
//! - the past and evidential copula do not follow the simple past and its plural
//!   (geldilerdi);
//! - a -DIk or -(y)AcAk participle takes no case but the ablative and locative without a
//!   possessive (ol-duğ-u is P3sg only; gel-dik-ten stays);
//! - abbreviations take possessives and derivations (TL'si, ABD'li): the analyzer requires the
//!   apostrophe instead.

use super::morphotactics::{Cond, Graph, ItemRef, Morpheme as M, MorphemeSet, StateId, StateSet};
use super::phonetics::PhoneticAttrs as P;
use super::pos::{PrimaryPos, RootAttr as R, SecondaryPos};

/// The states other parts of the analyzer start from or look for.
#[derive(Debug, Clone, Copy)]
pub struct RootStates {
    pub noun: StateId,
    pub noun_compound_root: StateId,
    pub noun_su_root: StateId,
    pub noun_proper: StateId,
    pub noun_abbrv: StateId,
    pub noun_no_suffix: StateId,
    pub noun_last_vowel_drop: StateId,
    pub adj: StateId,
    pub adj_last_vowel_drop: StateId,
    pub num: StateId,
    pub pron_pers: StateId,
    pub pron_pers_mod: StateId,
    pub pron_demons: StateId,
    pub pron_quant: StateId,
    pub pron_quant_modified: StateId,
    pub pron_ques: StateId,
    pub pron_reflex: StateId,
    pub adv: StateId,
    pub adv_for_verb_deriv: StateId,
    pub conj: StateId,
    pub question: StateId,
    pub interj: StateId,
    pub verb: StateId,
    pub verb_vowel_drop: StateId,
    pub verb_last_vowel_drop_mod: StateId,
    pub verb_last_vowel_drop_unmod: StateId,
    pub verb_implicit_recip: StateId,
    pub verb_implicit_reflex: StateId,
    pub verb_de_ye: StateId,
    pub verb_degil: StateId,
    pub imek: StateId,
    pub punc: StateId,
    pub det: StateId,
    pub postp: StateId,
    pub dup: StateId,
    /// POS roots of a nominal predicate group (the copula): its persons and tenses are shown
    /// as CopPres1sg, CopPast, ... (evde-yim, güzel-di).
    pub nominal_verb: StateSet,
}

fn it(lemma: &'static str, pos: PrimaryPos, secondary: Option<SecondaryPos>) -> ItemRef {
    ItemRef {
        lemma,
        pos,
        secondary,
    }
}

fn items(refs: &[ItemRef]) -> Cond {
    Cond::RootIsAny(refs.to_vec())
}

fn has(a: R) -> Cond {
    Cond::HasRootAttr(a)
}

fn ph(a: P) -> Cond {
    Cond::HasPhonetic(a)
}

fn cm(ms: &[M]) -> Cond {
    Cond::ContainsMorpheme(MorphemeSet::of(ms))
}

fn pm(ms: &[M]) -> Cond {
    Cond::PrevMorphemeIsAny(MorphemeSet::of(ms))
}

fn ps(ss: &[StateId]) -> Cond {
    Cond::PrevStateIsAny(StateSet::of(ss))
}

fn ld(ss: &[StateId]) -> Cond {
    Cond::LastDerivationIsAny(StateSet::of(ss))
}

fn cg(ss: &[StateId]) -> Cond {
    Cond::CurrentGroupContainsAny(StateSet::of(ss))
}

fn pg(ss: &[StateId]) -> Cond {
    Cond::PreviousGroupContainsAny(StateSet::of(ss))
}

fn pgm(ms: &[M]) -> Cond {
    Cond::PreviousGroupContainsMorpheme(MorphemeSet::of(ms))
}

const T: Cond = Cond::True;

/// Builds the graph and returns it with the root states.
#[allow(non_snake_case, clippy::too_many_lines)]
pub fn build() -> (Graph, RootStates) {
    use PrimaryPos::{Adv as AdvP, Noun as NounP, Postp as PostpP, Pron as PronP, Verb as VerbP};
    use SecondaryPos::{Demonstrative, Personal, Quantitive, Question, Reflexive, Time};

    let mut g = Graph::new();
    // `s` = non-terminal, `st` = terminal, `d` = derivative, `r` = POS root.
    macro_rules! s {
        ($name:literal, $m:expr) => {
            g.state($name, $m, false, false, false)
        };
    }
    macro_rules! st {
        ($name:literal, $m:expr) => {
            g.state($name, $m, true, false, false)
        };
    }
    macro_rules! d {
        ($name:literal, $m:expr) => {
            g.state($name, $m, false, true, false)
        };
    }
    macro_rules! r {
        ($name:literal, $m:expr) => {
            g.state($name, $m, false, false, true)
        };
    }
    macro_rules! rt {
        ($name:literal, $m:expr) => {
            g.state($name, $m, true, false, true)
        };
    }

    // ---- Items conditions name ----
    let ben = it("ben", PronP, Some(Personal));
    let sen = it("sen", PronP, Some(Personal));
    let o = it("o", PronP, Some(Personal));
    let biz = it("biz", PronP, Some(Personal));
    let siz = it("siz", PronP, Some(Personal));
    let falan = it("falan", PronP, None);
    let falanca = it("falanca", PronP, None);
    let bu = it("bu", PronP, Some(Demonstrative));
    let su = it("şu", PronP, Some(Demonstrative));
    let o_demons = it("o", PronP, Some(Demonstrative));
    let q = |l| it(l, PronP, Some(Quantitive));
    let (birbiri, biri, bazi, bircogu, birkaci) = (
        q("birbiri"),
        q("biri"),
        q("bazı"),
        q("birçoğu"),
        q("birkaçı"),
    );
    let (beriki, cogu, cumlesi, hep, herbiri) = (
        it("beriki", PronP, None),
        q("çoğu"),
        q("cümlesi"),
        q("hep"),
        q("herbiri"),
    );
    let (herkes, hicbiri, hepsi, kimi, kimse) =
        (q("herkes"), q("hiçbiri"), q("hepsi"), q("kimi"), q("kimse"));
    let (oburku, oburu, tumu, topu, umum) = (
        q("öbürkü"),
        q("öbürü"),
        q("tümü"),
        it("topu", PronP, None),
        it("umum", PronP, None),
    );
    let ne = it("ne", PronP, Some(Question));
    let nere = it("nere", PronP, Some(Question));
    let kim = it("kim", PronP, Some(Question));
    let kendi = it("kendi", PronP, Some(Reflexive));
    let degil = it("değil", VerbP, None);
    let dun = it("dün", NounP, Some(Time));
    let gun = it("gün", NounP, Some(Time));
    let bugun = it("bugün", NounP, Some(Time));
    let ileri = it("ileri", NounP, None);
    let geri = it("geri", NounP, None);
    let ote = it("öte", NounP, None);
    let beri = it("beri", NounP, None);
    let gibi = it("gibi", PostpP, None);
    let sonra = it("sonra", PostpP, None);

    // ---- States ----
    let puncRoot_ST = rt!("puncRoot_ST", M::Punc);

    let noun_S = r!("noun_S", M::Noun);
    let nounCompoundRoot_S = r!("nounCompoundRoot_S", M::Noun);
    let nounSuRoot_S = r!("nounSuRoot_S", M::Noun);
    let nounInf1Root_S = r!("nounInf1Root_S", M::Noun);
    let nounActOfRoot_S = r!("nounActOfRoot_S", M::Noun);

    let a3sg_S = s!("a3sg_S", M::A3sg);
    let a3sgSu_S = s!("a3sgSu_S", M::A3sg);
    let a3sgCompound_S = s!("a3sgCompound_S", M::A3sg);
    let a3sgInf1_S = s!("a3sgInf1_S", M::A3sg);
    let a3sgActOf_S = s!("a3sgActOf_S", M::A3sg);
    let a3pl_S = s!("a3pl_S", M::A3pl);
    let a3plActOf_S = s!("a3plActOf_S", M::A3pl);
    let a3plCompound_S = s!("a3plCompound_S", M::A3pl);
    let a3plCompound2_S = s!("a3plCompound2_S", M::A3pl);

    let pnon_S = s!("pnon_S", M::Pnon);
    let pnonCompound_S = s!("pnonCompound_S", M::Pnon);
    let pnonCompound2_S = s!("pnonCompound2_S", M::Pnon);
    let pnonInf1_S = s!("pnonInf1_S", M::Pnon);
    let pnonActOf_S = s!("pnonActOf", M::Pnon);
    let p1sg_S = s!("p1sg_S", M::P1sg);
    let p2sg_S = s!("p2sg_S", M::P2sg);
    let p3sg_S = s!("p3sg_S", M::P3sg);
    let p1pl_S = s!("p1pl_S", M::P1pl);
    let p2pl_S = s!("p2pl_S", M::P2pl);
    let p3pl_S = s!("p3pl_S", M::P3pl);

    let nom_ST = st!("nom_ST", M::Nom);
    let nom_S = s!("nom_S", M::Nom);
    let dat_ST = st!("dat_ST", M::Dat);
    let abl_ST = st!("abl_ST", M::Abl);
    let loc_ST = st!("loc_ST", M::Loc);
    let ins_ST = st!("ins_ST", M::Ins);
    let acc_ST = st!("acc_ST", M::Acc);
    let gen_ST = st!("gen_ST", M::Gen);
    let equ_ST = st!("equ_ST", M::Equ);

    let dim_S = d!("dim_S", M::Dim);
    let ness_S = d!("ness_S", M::Ness);
    let agt_S = d!("agt_S", M::Agt);
    let related_S = d!("related_S", M::Related);
    let rel_S = d!("rel_S", M::Rel);
    let relToPron_S = d!("relToPron_S", M::Rel);
    let with_S = d!("with_S", M::With);
    let without_S = d!("without_S", M::Without);
    let justLike_S = d!("justLike_S", M::JustLike);
    let nounZeroDeriv_S = d!("nounZeroDeriv_S", M::Zero);
    let become_S = d!("become_S", M::Become);
    let acquire_S = d!("acquire_S", M::Acquire);

    let nounLastVowelDropRoot_S = r!("nounLastVowelDropRoot_S", M::Noun);
    let adjLastVowelDropRoot_S = r!("adjLastVowelDropRoot_S", M::Adj);
    let postpLastVowelDropRoot_S = r!("postpLastVowelDropRoot_S", M::Postp);
    let a3PlLastVowelDrop_S = s!("a3PlLastVowelDrop_S", M::A3pl);
    let a3sgLastVowelDrop_S = s!("a3sgLastVowelDrop_S", M::A3sg);
    let pNonLastVowelDrop_S = s!("pNonLastVowelDrop_S", M::Pnon);
    let zeroLastVowelDrop_S = d!("zeroLastVowelDrop_S", M::Zero);

    let nounProper_S = r!("nounProper_S", M::Noun);
    let nounAbbrv_S = r!("nounAbbrv_S", M::Noun);
    let nounNoSuffix_S = r!("nounNoSuffix_S", M::Noun);
    let nounA3sgNoSuffix_S = s!("nounA3sgNoSuffix_S", M::A3sg);
    let nounPnonNoSuffix_S = s!("nounPnonNoSuffix_S", M::Pnon);
    let nounNomNoSuffix_ST = st!("nounNomNoSuffix_ST", M::Nom);

    let adjectiveRoot_ST = rt!("adjectiveRoot_ST", M::Adj);
    let adjAfterVerb_S = r!("adjAfterVerb_S", M::Adj);
    let adjAfterVerb_ST = rt!("adjAfterVerb_ST", M::Adj);
    let adjZeroDeriv_S = d!("adjZeroDeriv_S", M::Zero);
    let aPnon_ST = st!("aPnon_ST", M::Pnon);
    let aP1sg_ST = st!("aP1sg_ST", M::P1sg);
    let aP2sg_ST = st!("aP2sg_ST", M::P2sg);
    let aP3sg_ST = st!("aP3sg_ST", M::P3sg);
    let aP1pl_ST = st!("aP1pl_ST", M::P1pl);
    let aP2pl_ST = st!("aP2pl_ST", M::P2pl);
    let aP3pl_ST = st!("aP3pl_ST", M::P3pl);
    let aLy_S = d!("aLy_S", M::Ly);
    let aAsIf_S = d!("aAsIf_S", M::AsIf);
    let aAgt_S = d!("aAgt_S", M::Agt);

    let numeralRoot_ST = rt!("numeralRoot_ST", M::Num);
    let numZeroDeriv_S = d!("numZeroDeriv_S", M::Zero);

    let nVerb_S = r!("nVerb_S", M::Verb);
    let nVerbDegil_S = r!("nVerbDegil_S", M::Verb);
    let nPresent_S = s!("nPresent_S", M::Pres);
    let nPast_S = s!("nPast_S", M::Past);
    let nNarr_S = s!("nNarr_S", M::Narr);
    let nCond_S = s!("nCond_S", M::Cond);
    let nA1sg_ST = st!("nA1sg_ST", M::A1sg);
    let nA2sg_ST = st!("nA2sg_ST", M::A2sg);
    let nA1pl_ST = st!("nA1pl_ST", M::A1pl);
    let nA2pl_ST = st!("nA2pl_ST", M::A2pl);
    let nA3sg_ST = st!("nA3sg_ST", M::A3sg);
    let nA3sg_S = s!("nA3sg_S", M::A3sg);
    let nA3pl_ST = st!("nA3pl_ST", M::A3pl);
    let nCop_ST = st!("nCop_ST", M::Cop);
    let nCopBeforeA3pl_S = s!("nCopBeforeA3pl_S", M::Cop);
    let nNeg_S = s!("nNeg_S", M::Neg);

    let pronPers_S = r!("pronPers_S", M::Pron);
    let pronDemons_S = r!("pronDemons_S", M::Pron);
    let pronQuant_S = r!("pronQuant_S", M::Pron);
    let pronQuantModified_S = r!("pronQuantModified_S", M::Pron);
    let pronQues_S = r!("pronQues_S", M::Pron);
    let pronReflex_S = r!("pronReflex_S", M::Pron);
    let pronPers_Mod_S = r!("pronPers_Mod_S", M::Pron);
    let pronAfterRel_S = r!("pronAfterRel_S", M::Pron);
    let pA1sg_S = s!("pA1sg_S", M::A1sg);
    let pA2sg_S = s!("pA2sg_S", M::A2sg);
    let pA1sgMod_S = s!("pA1sgMod_S", M::A1sg);
    let pA2sgMod_S = s!("pA2sgMod_S", M::A2sg);
    let pA3sg_S = s!("pA3sg_S", M::A3sg);
    let pA3sgRel_S = s!("pA3sgRel_S", M::A3sg);
    let pA1pl_S = s!("pA1pl_S", M::A1pl);
    let pA2pl_S = s!("pA2pl_S", M::A2pl);
    let pA3pl_S = s!("pA3pl_S", M::A3pl);
    let pA3plRel_S = s!("pA3plRel_S", M::A3pl);
    let pQuantA3sg_S = s!("pQuantA3sg_S", M::A3sg);
    let pQuantA3pl_S = s!("pQuantA3pl_S", M::A3pl);
    let pQuantModA3pl_S = s!("pQuantModA3pl_S", M::A3pl);
    let pQuantA1pl_S = s!("pQuantA1pl_S", M::A1pl);
    let pQuantA2pl_S = s!("pQuantA2pl_S", M::A2pl);
    let pQuesA3sg_S = s!("pQuesA3sg_S", M::A3sg);
    let pQuesA3pl_S = s!("pQuesA3pl_S", M::A3pl);
    let pReflexA3sg_S = s!("pReflexA3sg_S", M::A3sg);
    let pReflexA3pl_S = s!("pReflexA3pl_S", M::A3pl);
    let pReflexA1sg_S = s!("pReflexA1sg_S", M::A1sg);
    let pReflexA2sg_S = s!("pReflexA2sg_S", M::A2sg);
    let pReflexA1pl_S = s!("pReflexA1pl_S", M::A1pl);
    let pReflexA2pl_S = s!("pReflexA2pl_S", M::A2pl);
    let pPnon_S = s!("pPnon_S", M::Pnon);
    let pPnonRel_S = s!("pPnonRel_S", M::Pnon);
    let pPnonMod_S = s!("pPnonMod_S", M::Pnon);
    let pP1sg_S = s!("pP1sg_S", M::P1sg);
    let pP2sg_S = s!("pP2sg_S", M::P2sg);
    let pP3sg_S = s!("pP3sg_S", M::P3sg);
    let pP1pl_S = s!("pP1pl_S", M::P1pl);
    let pP2pl_S = s!("pP2pl_S", M::P2pl);
    let pP3pl_S = s!("pP3pl_S", M::P3pl);
    let pNom_ST = st!("pNom_ST", M::Nom);
    let pDat_ST = st!("pDat_ST", M::Dat);
    let pAcc_ST = st!("pAcc_ST", M::Acc);
    let pAbl_ST = st!("pAbl_ST", M::Abl);
    let pLoc_ST = st!("pLoc_ST", M::Loc);
    let pGen_ST = st!("pGen_ST", M::Gen);
    let pIns_ST = st!("pIns_ST", M::Ins);
    let pEqu_ST = st!("pEqu_ST", M::Equ);
    let pronZeroDeriv_S = d!("pronZeroDeriv_S", M::Zero);

    let pvPresent_S = s!("pvPresent_S", M::Pres);
    let pvPast_S = s!("pvPast_S", M::Past);
    let pvNarr_S = s!("pvNarr_S", M::Narr);
    let pvCond_S = s!("pvCond_S", M::Cond);
    let pvA1sg_ST = st!("pvA1sg_ST", M::A1sg);
    let pvA2sg_ST = st!("pvA2sg_ST", M::A2sg);
    let pvA3sg_ST = st!("pvA3sg_ST", M::A3sg);
    let pvA3sg_S = s!("pvA3sg_S", M::A3sg);
    let pvA1pl_ST = st!("pvA1pl_ST", M::A1pl);
    let pvA2pl_ST = st!("pvA2pl_ST", M::A2pl);
    let pvA3pl_ST = st!("pvA3pl_ST", M::A3pl);
    let pvCopBeforeA3pl_S = s!("pvCopBeforeA3pl_S", M::Cop);
    let pvCop_ST = st!("pvCop_ST", M::Cop);
    let pvVerbRoot_S = r!("pvVerbRoot_S", M::Verb);

    let advRoot_ST = rt!("advRoot_ST", M::Adv);
    let advNounRoot_ST = rt!("advNounRoot_ST", M::Adv);
    let advForVerbDeriv_ST = rt!("advForVerbDeriv_ST", M::Adv);
    let avNounAfterAdvRoot_ST = r!("advToNounRoot_ST", M::Noun);
    let avA3sg_S = s!("avA3sg_S", M::A3sg);
    let avPnon_S = s!("avPnon_S", M::Pnon);
    let avDat_ST = st!("avDat_ST", M::Dat);
    let avZero_S = d!("avZero_S", M::Zero);
    let avZeroToVerb_S = d!("avZeroToVerb_S", M::Zero);

    let conjRoot_ST = rt!("conjRoot_ST", M::Conj);
    let interjRoot_ST = rt!("interjRoot_ST", M::Interj);
    let detRoot_ST = rt!("detRoot_ST", M::Det);
    let dupRoot_ST = rt!("dupRoot_ST", M::Dup);

    let postpRoot_ST = rt!("postpRoot_ST", M::Postp);
    let postpZero_S = d!("postpZero_S", M::Zero);
    let po2nRoot_S = s!("po2nRoot_S", M::Noun);
    let po2nA3sg_S = s!("po2nA3sg_S", M::A3sg);
    let po2nA3pl_S = s!("po2nA3pl_S", M::A3pl);
    let po2nP3sg_S = s!("po2nP3sg_S", M::P3sg);
    let po2nP1sg_S = s!("po2nP1sg_S", M::P1sg);
    let po2nP2sg_S = s!("po2nP2sg_S", M::P2sg);
    let po2nP1pl_S = s!("po2nP1pl_S", M::P1pl);
    let po2nP2pl_S = s!("po2nP2pl_S", M::P2pl);
    let po2nPnon_S = s!("po2nPnon_S", M::Pnon);
    let po2nNom_ST = st!("po2nNom_ST", M::Nom);
    let po2nDat_ST = st!("po2nDat_ST", M::Dat);
    let po2nAbl_ST = st!("po2nAbl_ST", M::Abl);
    let po2nLoc_ST = st!("po2nLoc_ST", M::Loc);
    let po2nIns_ST = st!("po2nIns_ST", M::Ins);
    let po2nAcc_ST = st!("po2nAcc_ST", M::Acc);
    let po2nGen_ST = st!("po2nGen_ST", M::Gen);
    let po2nEqu_ST = st!("po2nEqu_ST", M::Equ);

    let verbRoot_S = r!("verbRoot_S", M::Verb);
    let verbLastVowelDropModRoot_S = r!("verbLastVowelDropModRoot_S", M::Verb);
    let verbLastVowelDropUnmodRoot_S = r!("verbLastVowelDropUnmodRoot_S", M::Verb);
    let vA1sg_ST = st!("vA1sg_ST", M::A1sg);
    let vA2sg_ST = st!("vA2sg_ST", M::A2sg);
    let vA3sg_ST = st!("vA3sg_ST", M::A3sg);
    let vA1pl_ST = st!("vA1pl_ST", M::A1pl);
    let vA2pl_ST = st!("vA2pl_ST", M::A2pl);
    let vA3pl_ST = st!("vA3pl_ST", M::A3pl);
    let vPast_S = s!("vPast_S", M::Past);
    let vNarr_S = s!("vNarr_S", M::Narr);
    let vCond_S = s!("vCond_S", M::Cond);
    let vCondAfterPerson_ST = st!("vCondAfterPerson_ST", M::Cond);
    let vPastAfterTense_S = s!("vPastAfterTense_S", M::Past);
    let vNarrAfterTense_S = s!("vNarrAfterTense_S", M::Narr);
    let vPastAfterTense_ST = st!("vPastAfterTense_ST", M::Past);
    let vNarrAfterTense_ST = st!("vNarrAfterTense_ST", M::Narr);
    let vCond_ST = st!("vCond_ST", M::Cond);
    let vProgYor_S = s!("vProgYor_S", M::Prog1);
    let vProgMakta_S = s!("vProgMakta_S", M::Prog2);
    let vFut_S = s!("vFut_S", M::Fut);
    let vCop_ST = st!("vCop_ST", M::Cop);
    let vCopBeforeA3pl_S = s!("vCopBeforeA3pl_S", M::Cop);
    let vNeg_S = s!("vNeg_S", M::Neg);
    let vUnable_S = s!("vUnable_S", M::Unable);
    let vNegProg1_S = s!("vNegProg1_S", M::Neg);
    let vUnableProg1_S = s!("vUnableProg1_S", M::Unable);
    let vImp_S = s!("vImp_S", M::Imp);
    let vImpYemekYi_S = s!("vImpYemekYi_S", M::Imp);
    let vImpYemekYe_S = s!("vImpYemekYe_S", M::Imp);
    let vCausT_S = d!("vCausT_S", M::Caus);
    let vCausTir_S = d!("vCausTır_S", M::Caus);
    let vRecip_S = d!("vRecip_S", M::Recip);
    let vImplicitRecipRoot_S = r!("vImplicitRecipRoot_S", M::Verb);
    let vReflex_S = d!("vReflex_S", M::Reflex);
    let vImplicitReflexRoot_S = r!("vImplicitReflexRoot_S", M::Verb);
    let verbRoot_VowelDrop_S = r!("verbRoot_VowelDrop_S", M::Verb);
    let vAor_S = s!("vAor_S", M::Aor);
    let vAorNeg_S = s!("vAorNeg_S", M::Aor);
    let vAorNegEmpty_S = s!("vAorNegEmpty_S", M::Aor);
    let vAorPartNeg_S = d!("vAorPartNeg_S", M::AorPart);
    let vAorPart_S = d!("vAorPart_S", M::AorPart);
    let vAble_S = d!("vAble_S", M::Able);
    let vAbleNeg_S = d!("vAbleNeg_S", M::Able);
    let vAbleNegDerivRoot_S = r!("vAbleNegDerivRoot_S", M::Verb);
    let vPass_S = d!("vPass_S", M::Pass);
    let vOpt_S = s!("vOpt_S", M::Opt);
    let vDesr_S = s!("vDesr_S", M::Desr);
    let vNeces_S = s!("vNeces_S", M::Neces);
    let vInf1_S = d!("vInf1_S", M::Inf1);
    let vInf2_S = d!("vInf2_S", M::Inf2);
    let vInf3_S = d!("vInf3_S", M::Inf3);
    let vAgt_S = d!("vAgt_S", M::Agt);
    let vActOf_S = d!("vActOf_S", M::ActOf);
    let vPastPart_S = d!("vPastPart_S", M::PastPart);
    let vFutPart_S = d!("vFutPart_S", M::FutPart);
    let vPresPart_S = d!("vPresPart_S", M::PresPart);
    let vNarrPart_S = d!("vNarrPart_S", M::NarrPart);
    let vFeelLike_S = d!("vFeelLike_S", M::FeelLike);
    let vNotState_S = d!("vNotState_S", M::NotState);
    let vEverSince_S = d!("vEverSince_S", M::EverSince);
    let vRepeat_S = d!("vRepeat_S", M::Repeat);
    let vAlmost_S = d!("vAlmost_S", M::Almost);
    let vHastily_S = d!("vHastily_S", M::Hastily);
    let vStay_S = d!("vStay_S", M::Stay);
    let vStart_S = d!("vStart_S", M::Start);
    let vWhile_S = d!("vWhile_S", M::While);
    let vWhen_S = d!("vWhen_S", M::When);
    let vAsIf_S = d!("vAsIf_S", M::AsIf);
    let vSinceDoingSo_S = d!("vSinceDoingSo_S", M::SinceDoingSo);
    let vAsLongAs_S = d!("vAsLongAs_S", M::AsLongAs);
    let vByDoingSo_S = d!("vByDoingSo_S", M::ByDoingSo);
    let vAdamantly_S = d!("vAdamantly_S", M::Adamantly);
    let vAfterDoing_S = d!("vAfterDoing_S", M::AfterDoingSo);
    let vWithoutHavingDoneSo_S = d!("vWithoutHavingDoneSo_S", M::WithoutHavingDoneSo);
    let vWithoutBeingAbleToHaveDoneSo_S = d!(
        "vWithoutBeingAbleToHaveDoneSo_S",
        M::WithoutBeingAbleToHaveDoneSo
    );
    let vDeYeRoot_S = r!("vDeYeRoot_S", M::Verb);

    let qPresent_S = s!("qPresent_S", M::Pres);
    let qPast_S = s!("qPast_S", M::Past);
    let qNarr_S = s!("qNarr_S", M::Narr);
    let qA1sg_ST = st!("qA1sg_ST", M::A1sg);
    let qA2sg_ST = st!("qA2sg_ST", M::A2sg);
    let qA3sg_ST = st!("qA3sg_ST", M::A3sg);
    let qA1pl_ST = st!("qA1pl_ST", M::A1pl);
    let qA2pl_ST = st!("qA2pl_ST", M::A2pl);
    let qA3pl_ST = st!("qA3pl_ST", M::A3pl);
    let qCopBeforeA3pl_S = s!("qCopBeforeA3pl_S", M::Cop);
    let qCop_ST = st!("qCop_ST", M::Cop);
    let questionRoot_S = r!("questionRoot_S", M::Ques);

    let imekRoot_S = r!("imekRoot_S", M::Verb);
    let imekPast_S = s!("imekPast_S", M::Past);
    let imekNarr_S = s!("imekNarr_S", M::Narr);
    let imekCond_S = s!("imekCond_S", M::Cond);
    let imekA1sg_ST = st!("imekA1sg_ST", M::A1sg);
    let imekA2sg_ST = st!("imekA2sg_ST", M::A2sg);
    let imekA3sg_ST = st!("imekA3sg_ST", M::A3sg);
    let imekA1pl_ST = st!("imekA1pl_ST", M::A1pl);
    let imekA2pl_ST = st!("imekA2pl_ST", M::A2pl);
    let imekA3pl_ST = st!("imekA3pl_ST", M::A3pl);
    let imekCop_ST = st!("imekCop_ST", M::Cop);

    let nsad = || Cond::NoSurfaceAfterDerivation;

    // ======================= Nouns (connectNounStates) =======================
    g.add(noun_S, a3sg_S, "", has(R::PLURAL).not());
    g.add(
        noun_S,
        a3pl_S,
        "lAr",
        has(R::PLURAL).not().and_not(has(R::COMPOUND_P3SG)),
    );
    g.add(noun_S, a3pl_S, "", has(R::PLURAL));

    // Compound roots (zeytinyağ-, cumhurbaşkan-): only their plural and derivations.
    g.add(
        nounCompoundRoot_S,
        a3sgCompound_S,
        "",
        has(R::COMPOUND_P3SG_ROOT),
    );
    g.add(a3sgCompound_S, pnonCompound_S, "", T);
    g.add(a3sgCompound_S, p3pl_S, "lArI", T);
    g.add(pnonCompound_S, nom_S, "", T);
    g.add(nom_S, become_S, "lAş", T);
    g.add(nom_S, acquire_S, "lAn", T);
    g.add(nom_S, with_S, "lI", cm(&[M::With, M::Without]).not());
    g.add(nom_S, without_S, "sIz", cm(&[M::With, M::Without]).not());
    g.add(nom_S, ness_S, "lI~k", cm(&[M::Ness]).not());
    g.add(nom_S, ness_S, "lI!ğ", cm(&[M::Ness]).not());
    g.add(nom_S, agt_S, ">cI", cm(&[M::Agt]).not());
    g.add(
        nom_S,
        justLike_S,
        "+msI",
        cm(&[M::JustLike]).not().and(ph(P::LAST_LETTER_VOWEL)),
    );
    g.add(
        nom_S,
        dim_S,
        ">cI~k",
        Cond::HasAnySuffixSurface.not().and_not(cm(&[M::Dim])),
    );
    g.add(
        nom_S,
        dim_S,
        ">cI!ğ",
        Cond::HasAnySuffixSurface.not().and_not(cm(&[M::Dim])),
    );
    g.add(nom_S, dim_S, "cAğIz", Cond::HasAnySuffixSurface.not());
    g.add(
        nounCompoundRoot_S,
        a3plCompound_S,
        "lAr",
        has(R::COMPOUND_P3SG_ROOT),
    );
    g.add(
        nounCompoundRoot_S,
        a3plCompound2_S,
        "lArI",
        has(R::COMPOUND_P3SG_ROOT),
    );
    for (to, t) in [
        (p3sg_S, "I"),
        (p2sg_S, "In"),
        (p1sg_S, "Im"),
        (p1pl_S, "ImIz"),
        (p2pl_S, "InIz"),
        (p3pl_S, "I"),
    ] {
        g.add(a3plCompound_S, to, t, T);
    }
    g.add(a3plCompound2_S, pnonCompound2_S, "", T);
    g.add(pnonCompound2_S, nom_ST, "", T);

    // Possessives; family words (annemler) take none. Zemberek also refuses them, and the
    // derivations below, to abbreviations; akana lets the apostrophe decide instead (an
    // abbreviation takes no suffix without one, and any with one: TL'si, ABD'li, TL'lik).
    let possession = || has(R::FAMILY_MEMBER).not();
    g.add(a3sg_S, pnon_S, "", has(R::FAMILY_MEMBER).not());
    g.add(a3sg_S, p1sg_S, "Im", possession());
    g.add(
        a3sg_S,
        p2sg_S,
        "In",
        possession().and_not(pgm(&[M::JustLike])),
    );
    g.add(a3sg_S, p3sg_S, "+sI", possession());
    g.add(a3sg_S, p3sg_S, "", has(R::COMPOUND_P3SG));
    g.add(a3sg_S, p1pl_S, "ImIz", possession());
    g.add(
        a3sg_S,
        p2pl_S,
        "InIz",
        possession().and_not(pgm(&[M::JustLike])),
    );
    g.add(a3sg_S, p3pl_S, "lArI", possession());
    g.add(a3pl_S, pnon_S, "", has(R::FAMILY_MEMBER).not());
    g.add(a3pl_S, p1sg_S, "Im", possession());
    g.add(a3pl_S, p2sg_S, "In", possession());
    g.add(a3pl_S, p1sg_S, "", has(R::IMPLICIT_P1SG));
    g.add(a3pl_S, p2sg_S, "", has(R::IMPLICIT_P2SG));
    g.add(a3pl_S, p3sg_S, "I", possession());
    g.add(a3pl_S, p1pl_S, "ImIz", possession());
    g.add(a3pl_S, p2pl_S, "InIz", possession());
    g.add(a3pl_S, p3pl_S, "I", possession());

    // su, akarsu: suy- before a vowel.
    g.add(nounSuRoot_S, a3sgSu_S, "", T);
    g.add(nounSuRoot_S, a3pl_S, "lar", T);
    for (to, t) in [
        (pnon_S, ""),
        (p1sg_S, "yum"),
        (p2sg_S, "yun"),
        (p3sg_S, "yu"),
        (p1pl_S, "yumuz"),
        (p2pl_S, "yunuz"),
        (p3pl_S, "lArI"),
    ] {
        g.add(a3sgSu_S, to, t, T);
    }

    // Cases.
    g.add(pnon_S, nom_ST, "", has(R::FAMILY_MEMBER).not());
    let able_verb_pastpart = g.sequence(&[M::Able, M::Verb, M::PastPart]);
    let equ_cond = || {
        pm(&[M::A3pl]).or(
            cm(&[M::Adj, M::FutPart, M::PresPart, M::NarrPart, M::PastPart])
                .not()
                .or(Cond::ContainsSequence(able_verb_pastpart)),
        )
    };
    let not_cp3sg = || has(R::COMPOUND_P3SG).not();
    // A -DIk or -(y)AcAk participle takes a possessive before a case (ol-duğ-u-nu,
    // yap-acağ-ın-a); a bare case after it (ol-duğ-u as Acc) is a Zemberek reading akana
    // leaves out, except the ablative and locative (gel-dik-ten sonra, bak-tık-ta).
    let not_participle = || ld(&[vPastPart_S, vFutPart_S]).not();
    let case = |c: Cond| c.and(not_participle());
    g.add(pnon_S, dat_ST, "+yA", case(not_cp3sg()));
    g.add(pnon_S, abl_ST, ">dAn", not_cp3sg());
    g.add(pnon_S, loc_ST, ">dA", not_cp3sg());
    g.add(pnon_S, acc_ST, "+yI", case(not_cp3sg()));
    g.add(pnon_S, gen_ST, "+nIn", case(Cond::PrevStateIsNot(a3sgSu_S)));
    g.add(pnon_S, gen_ST, "yIn", ps(&[a3sgSu_S]));
    g.add(pnon_S, equ_ST, ">cA", case(not_cp3sg().and(equ_cond())));
    g.add(pnon_S, ins_ST, "+ylA", case(T));
    g.add(pnon_S, dat_ST, "+nA", has(R::COMPOUND_P3SG));
    g.add(pnon_S, abl_ST, "+ndAn", has(R::COMPOUND_P3SG));
    g.add(pnon_S, loc_ST, "+ndA", has(R::COMPOUND_P3SG));
    g.add(
        pnon_S,
        equ_ST,
        "+ncA",
        has(R::COMPOUND_P3SG).and(equ_cond()),
    );
    g.add(pnon_S, acc_ST, "+nI", has(R::COMPOUND_P3SG));
    g.add(pnon_S, dat_ST, "", has(R::IMPLICIT_DATIVE));
    let equ_poss = || equ_cond().or(cm(&[M::PastPart]));
    for p in [p1sg_S, p2sg_S, p1pl_S, p2pl_S] {
        g.add(p, nom_ST, "", T);
        g.add(p, dat_ST, "A", T);
        g.add(p, loc_ST, "dA", T);
        g.add(p, abl_ST, "dAn", T);
        g.add(p, ins_ST, "lA", T);
        g.add(p, gen_ST, "In", T);
        g.add(p, equ_ST, "cA", equ_poss());
        g.add(p, acc_ST, "I", T);
    }
    g.add(p3sg_S, nom_ST, "", T);
    g.add(p3sg_S, dat_ST, "nA", T);
    g.add(p3sg_S, loc_ST, "ndA", T);
    g.add(p3sg_S, abl_ST, "ndAn", T);
    g.add(p3sg_S, ins_ST, "ylA", T);
    g.add(p3sg_S, gen_ST, "nIn", T);
    g.add(p3sg_S, equ_ST, "ncA", equ_poss());
    g.add(p3sg_S, acc_ST, "nI", T);
    g.add(p3pl_S, nom_ST, "", T);
    g.add(p3pl_S, dat_ST, "nA", T);
    g.add(p3pl_S, loc_ST, "ndA", T);
    g.add(p3pl_S, abl_ST, "ndAn", T);
    g.add(p3pl_S, ins_ST, "ylA", T);
    g.add(p3pl_S, gen_ST, "nIn", T);
    g.add(p3pl_S, equ_ST, "+ncA", T);
    g.add(p3pl_S, acc_ST, "nI", T);

    // Derivations from the nominative.
    g.add(nom_ST, dim_S, ">cI~k", Cond::HasAnySuffixSurface.not());
    g.add(nom_ST, dim_S, ">cI!ğ", Cond::HasAnySuffixSurface.not());
    g.add(nom_ST, dim_S, "cAğIz", Cond::HasAnySuffixSurface.not());
    g.add(dim_S, noun_S, "", T);
    let empty_adj_noun = g.sequence(&[M::Adj, M::Zero, M::Noun, M::A3sg, M::Pnon, M::Nom]);
    let ness_cond = || {
        nsad()
            .and_not(cm(&[M::Ness]))
            .and_not(Cond::ContainsSequence(empty_adj_noun))
    };
    g.add(nom_ST, ness_S, "lI~k", ness_cond());
    g.add(nom_ST, ness_S, "lI!ğ", ness_cond());
    g.add(ness_S, noun_S, "", T);
    g.add(nom_ST, agt_S, ">cI", nsad().and_not(cm(&[M::Adj, M::Agt])));
    g.add(agt_S, noun_S, "", T);
    let noun_to_verb = || Cond::HasTail.and_not(nsad().and(ld(&[adjZeroDeriv_S])));
    for c in [nom_ST, dat_ST, abl_ST, loc_ST, ins_ST, gen_ST] {
        g.add(c, nounZeroDeriv_S, "", noun_to_verb());
    }
    g.add(nounZeroDeriv_S, nVerb_S, "", T);
    g.add(
        nom_ST,
        with_S,
        "lI",
        nsad().and_not(cm(&[M::With, M::Without])),
    );
    g.add(
        nom_ST,
        without_S,
        "sIz",
        nsad().and_not(cm(&[M::With, M::Without, M::Inf1])),
    );
    let just_like_block = || cm(&[M::JustLike, M::FutPart, M::PastPart, M::PresPart, M::Adj]);
    g.add(
        nom_ST,
        justLike_S,
        "+msI",
        nsad()
            .and_not(just_like_block())
            .and(ph(P::LAST_LETTER_VOWEL)),
    );
    g.add(
        nom_ST,
        justLike_S,
        "ImsI",
        ph(P::LAST_LETTER_VOWEL)
            .not()
            .and(nsad())
            .and_not(just_like_block()),
    );
    g.add(
        nom_ST,
        related_S,
        "sAl",
        nsad().and_not(cm(&[M::With, M::Without, M::Related])),
    );
    for s in [with_S, without_S, related_S, justLike_S] {
        g.add(s, adjectiveRoot_ST, "", T);
    }

    // -ki: after the locative, on time nouns, and after the genitive (masanınki).
    let rel_repetition = g.sequence(&[M::Rel, M::Adj, M::Zero, M::Noun, M::A3sg, M::Pnon, M::Loc]);
    g.add(
        loc_ST,
        rel_S,
        "ki",
        Cond::TailSequence(rel_repetition).not(),
    );
    g.add(rel_S, adjectiveRoot_ST, "", T);
    let time = || nsad().and(Cond::SecondaryPosIs(SecondaryPos::Time));
    let time2 = || items(&[dun.clone(), gun.clone(), bugun.clone()]);
    g.add(nom_ST, rel_S, "ki", time().and_not(time2()));
    g.add(
        nom_ST,
        rel_S,
        "ki",
        items(&[ileri.clone(), geri.clone(), ote.clone(), beri.clone()]),
    );
    g.add(nom_ST, rel_S, "kü", time2().and(time()));
    g.add(gen_ST, relToPron_S, "ki", T);
    g.add(relToPron_S, pronAfterRel_S, "", T);

    let verb_deriv = || cm(&[M::Inf1, M::Inf2, M::Inf3, M::PastPart, M::FutPart]);
    g.add(
        nom_ST,
        become_S,
        "lAş",
        nsad().and_not(cm(&[M::Adj])).and_not(verb_deriv()),
    );
    g.add(become_S, verbRoot_S, "", T);
    g.add(
        nom_ST,
        acquire_S,
        "lAn",
        nsad().and_not(cm(&[M::Adj])).and_not(verb_deriv()),
    );
    g.add(acquire_S, verbRoot_S, "", T);

    // Infinitive -mAk and -mAcA nouns.
    g.add(nounInf1Root_S, a3sgInf1_S, "", T);
    g.add(a3sgInf1_S, pnonInf1_S, "", T);
    g.add(pnonInf1_S, nom_ST, "", T);
    g.add(pnonInf1_S, abl_ST, "tAn", T);
    g.add(pnonInf1_S, loc_ST, "tA", T);
    g.add(pnonInf1_S, ins_ST, "lA", T);
    g.add(nounActOfRoot_S, a3sgActOf_S, "", T);
    g.add(nounActOfRoot_S, a3plActOf_S, "lar", T);
    g.add(a3sgActOf_S, pnonActOf_S, "", T);
    g.add(a3plActOf_S, pnonActOf_S, "", T);
    g.add(pnonActOf_S, nom_ST, "", T);

    // ======================= Proper nouns and abbreviations =======================
    g.add(nounProper_S, a3sg_S, "", T);
    g.add(nounProper_S, a3pl_S, "lAr", T);
    g.add(nounAbbrv_S, a3sg_S, "", T);
    g.add(nounAbbrv_S, a3pl_S, "lAr", T);
    g.add(nounNoSuffix_S, nounA3sgNoSuffix_S, "", T);
    g.add(nounA3sgNoSuffix_S, nounPnonNoSuffix_S, "", T);
    g.add(nounPnonNoSuffix_S, nounNomNoSuffix_ST, "", T);

    // ======================= Adjectives =======================
    g.add(adjectiveRoot_ST, adjZeroDeriv_S, "", Cond::HasTail);
    g.add(adjZeroDeriv_S, noun_S, "", T);
    g.add(adjZeroDeriv_S, nVerb_S, "", T);
    g.add(adjectiveRoot_ST, aLy_S, ">cA", T);
    g.add(aLy_S, advRoot_ST, "", T);
    let adj_blocks = || cm(&[M::AsIf, M::Ly, M::Agt, M::With, M::JustLike]).not();
    g.add(adjectiveRoot_ST, aAsIf_S, ">cA", adj_blocks());
    g.add(aAsIf_S, adjectiveRoot_ST, "", T);
    g.add(adjectiveRoot_ST, aAgt_S, ">cI", adj_blocks());
    g.add(aAgt_S, noun_S, "", T);
    g.add(
        adjectiveRoot_ST,
        justLike_S,
        "+msI",
        nsad()
            .and(cm(&[M::JustLike]).not())
            .and(ph(P::LAST_LETTER_VOWEL)),
    );
    g.add(
        adjectiveRoot_ST,
        justLike_S,
        "ImsI",
        ph(P::LAST_LETTER_VOWEL)
            .not()
            .and(nsad())
            .and(cm(&[M::JustLike]).not()),
    );
    g.add(adjectiveRoot_ST, become_S, "lAş", nsad());
    g.add(adjectiveRoot_ST, acquire_S, "lAn", nsad());
    let c1 = || pm(&[M::FutPart, M::PastPart]);
    g.add(adjAfterVerb_S, aPnon_ST, "", c1());
    g.add(adjAfterVerb_S, aP1sg_ST, "Im", c1());
    g.add(adjAfterVerb_S, aP2sg_ST, "In", c1());
    g.add(adjAfterVerb_S, aP3sg_ST, "I", c1());
    g.add(adjAfterVerb_S, aP1pl_ST, "ImIz", c1());
    g.add(adjAfterVerb_S, aP2pl_ST, "InIz", c1());
    g.add(adjAfterVerb_S, aP3pl_ST, "lArI", c1());
    g.add(adjectiveRoot_ST, ness_S, "lI~k", T);
    g.add(adjectiveRoot_ST, ness_S, "lI!ğ", T);
    g.add(adjAfterVerb_ST, ness_S, "lI~k", pm(&[M::AorPart]));
    g.add(adjAfterVerb_ST, ness_S, "lI!ğ", pm(&[M::AorPart]));

    // ======================= Numerals =======================
    g.add(numeralRoot_ST, ness_S, "lI~k", T);
    g.add(numeralRoot_ST, ness_S, "lI!ğ", T);
    g.add(numeralRoot_ST, numZeroDeriv_S, "", Cond::HasTail);
    g.add(numZeroDeriv_S, noun_S, "", T);
    g.add(numZeroDeriv_S, nVerb_S, "", T);
    g.add(
        numeralRoot_ST,
        justLike_S,
        "+msI",
        nsad()
            .and(cm(&[M::JustLike]).not())
            .and(ph(P::LAST_LETTER_VOWEL)),
    );
    g.add(
        numeralRoot_ST,
        justLike_S,
        "ImsI",
        ph(P::LAST_LETTER_VOWEL)
            .not()
            .and(nsad())
            .and(cm(&[M::JustLike]).not()),
    );

    // ======================= Nominal predicates (connectVerbAfterNounAdjStates) ============
    g.add(nVerb_S, nPresent_S, "", T);
    g.add(nVerb_S, nPast_S, "+y>dI", T);
    g.add(nVerb_S, nNarr_S, "+ymIş", T);
    g.add(nVerb_S, nCond_S, "+ysA", T);
    g.add(nVerb_S, vWhile_S, "+yken", T);
    g.add(
        nVerbDegil_S,
        nNeg_S,
        "",
        items(std::slice::from_ref(&degil)),
    );
    let p1sg_nom = g.sequence(&[M::P1sg, M::Nom]);
    let p2sg_nom = g.sequence(&[M::P2sg, M::Nom]);
    let p3pl_nom = g.sequence(&[M::P3pl, M::Nom]);
    let p2pl_nom = g.sequence(&[M::P2pl, M::Nom]);
    let p1pl_nom = g.sequence(&[M::P1pl, M::Nom]);
    let no_family = || has(R::FAMILY_MEMBER).not();
    let allow_a1sg = || {
        no_family()
            .and_not(Cond::ContainsSequence(p1sg_nom))
            .and_not(verb_deriv())
    };
    let allow_a2sg = || {
        no_family()
            .and_not(Cond::ContainsSequence(p2sg_nom))
            .and_not(verb_deriv())
    };
    let allow_a3pl = || {
        no_family()
            .and_not(pg(&[a3pl_S]))
            .and_not(Cond::ContainsSequence(p3pl_nom))
            .and_not(verb_deriv())
    };
    let allow_a2pl = || {
        no_family()
            .and_not(Cond::ContainsSequence(p2pl_nom))
            .and_not(verb_deriv())
    };
    let allow_a1pl = || {
        no_family()
            .and_not(Cond::ContainsSequence(p1sg_nom))
            .and_not(Cond::ContainsSequence(p1pl_nom))
            .and_not(verb_deriv())
    };
    g.add(nPresent_S, nA1sg_ST, "+yIm", allow_a1sg());
    g.add(nPresent_S, nA2sg_ST, "sIn", allow_a2sg());
    g.add(nPresent_S, nA3sg_S, "", T);
    g.add(
        nPresent_S,
        nA3sg_ST,
        "",
        items(std::slice::from_ref(&degil)),
    );
    g.add(
        nPresent_S,
        nA3pl_ST,
        "lAr",
        has(R::COMPOUND_P3SG)
            .not()
            .and_not(pgm(&[M::Inf1]))
            .and(allow_a3pl()),
    );
    g.add(nPast_S, nA1sg_ST, "m", allow_a1sg());
    g.add(nNarr_S, nA1sg_ST, "Im", allow_a1sg());
    g.add(nPast_S, nA2sg_ST, "n", allow_a2sg());
    g.add(nNarr_S, nA2sg_ST, "sIn", allow_a2sg());
    g.add(nPast_S, nA1pl_ST, "k", allow_a1pl());
    g.add(nNarr_S, nA1pl_ST, "Iz", allow_a1pl());
    g.add(nPresent_S, nA1pl_ST, "+yIz", allow_a1pl());
    g.add(nPast_S, nA2pl_ST, "InIz", allow_a2pl());
    g.add(nNarr_S, nA2pl_ST, "sInIz", allow_a2pl());
    g.add(nPresent_S, nA2pl_ST, "sInIz", allow_a2pl());
    g.add(
        nPast_S,
        nA3pl_ST,
        "lAr",
        has(R::COMPOUND_P3SG).not().and(allow_a3pl()),
    );
    g.add(
        nNarr_S,
        nA3pl_ST,
        "lAr",
        has(R::COMPOUND_P3SG).not().and(allow_a3pl()),
    );
    g.add(nPast_S, nA3sg_ST, "", T);
    g.add(nNarr_S, nA3sg_ST, "", T);
    g.add(nNarr_S, nCond_S, "sA", T);
    g.add(nCond_S, nA1sg_ST, "m", allow_a1sg());
    g.add(nCond_S, nA2sg_ST, "n", allow_a2sg());
    g.add(nCond_S, nA1pl_ST, "k", allow_a1pl());
    g.add(nCond_S, nA2pl_ST, "nIz", allow_a2pl());
    g.add(nCond_S, nA3sg_ST, "", T);
    g.add(nCond_S, nA3pl_ST, "lAr", T);
    let reject_no_copula = || cg(&[nPast_S, nCond_S, nCopBeforeA3pl_S]).not();
    for p in [nA1sg_ST, nA2sg_ST, nA1pl_ST, nA2pl_ST, nA3pl_ST] {
        g.add(p, nCop_ST, "dIr", reject_no_copula());
    }
    g.add(nA3sg_S, nCop_ST, ">dIr", reject_no_copula());
    for p in [nA3sg_ST, nA1sg_ST, nA2sg_ST, nA1pl_ST, nA2pl_ST, nA3pl_ST] {
        g.add(p, vAsIf_S, ">cAsInA", pm(&[M::Narr]));
    }
    g.add(nPresent_S, nCopBeforeA3pl_S, ">dIr", T);
    g.add(nCopBeforeA3pl_S, nA3pl_ST, "lAr", T);

    // ======================= Pronouns =======================
    g.add(pronPers_S, pA1sg_S, "", items(std::slice::from_ref(&ben)));
    g.add(pronPers_S, pA2sg_S, "", items(std::slice::from_ref(&sen)));
    g.add(
        pronPers_S,
        pA3sg_S,
        "",
        items(&[o.clone(), falan.clone(), falanca.clone()]),
    );
    g.add(pronPers_S, pA3pl_S, "nlAr", items(std::slice::from_ref(&o)));
    g.add(
        pronPers_S,
        pA3pl_S,
        "lAr",
        items(&[falan.clone(), falanca.clone()]),
    );
    g.add(pronPers_S, pA1pl_S, "", items(std::slice::from_ref(&biz)));
    g.add(
        pronPers_S,
        pA1pl_S,
        "lAr",
        items(std::slice::from_ref(&biz)),
    );
    g.add(pronPers_S, pA2pl_S, "", items(std::slice::from_ref(&siz)));
    g.add(
        pronPers_S,
        pA2pl_S,
        "lAr",
        items(std::slice::from_ref(&siz)),
    );
    // ban-, san- (bana, sana).
    g.add(
        pronPers_Mod_S,
        pA1sgMod_S,
        "",
        items(std::slice::from_ref(&ben)),
    );
    g.add(
        pronPers_Mod_S,
        pA2sgMod_S,
        "",
        items(std::slice::from_ref(&sen)),
    );
    g.add(pA1sgMod_S, pPnonMod_S, "", T);
    g.add(pA2sgMod_S, pPnonMod_S, "", T);
    g.add(pPnonMod_S, pDat_ST, "A", T);
    for a in [pA1sg_S, pA2sg_S, pA3sg_S, pA1pl_S, pA2pl_S, pA3pl_S] {
        g.add(a, pPnon_S, "", T);
    }
    // masanınki: -ki makes a pronoun.
    g.add(pronAfterRel_S, pA3sgRel_S, "", T);
    g.add(pronAfterRel_S, pA3plRel_S, "lAr", T);
    g.add(pA3sgRel_S, pPnonRel_S, "", T);
    g.add(pA3plRel_S, pPnonRel_S, "", T);
    g.add(pPnonRel_S, pNom_ST, "", T);
    g.add(pPnonRel_S, pDat_ST, "+nA", T);
    g.add(pPnonRel_S, pAcc_ST, "+nI", T);
    g.add(pPnonRel_S, pAbl_ST, "+ndAn", T);
    g.add(pPnonRel_S, pLoc_ST, "+ndA", T);
    g.add(pPnonRel_S, pIns_ST, "+ylA", T);
    g.add(pPnonRel_S, pGen_ST, "+nIn", T);
    // bu, şu, o: bunlar, şunlar, onlar.
    g.add(pronDemons_S, pA3sg_S, "", T);
    g.add(pronDemons_S, pA3pl_S, "nlAr", T);
    // Quantitative pronouns.
    g.add(
        pronQuant_S,
        pQuantA3sg_S,
        "",
        items(&[
            herkes.clone(),
            umum.clone(),
            hepsi.clone(),
            cumlesi.clone(),
            hep.clone(),
            tumu.clone(),
            birkaci.clone(),
            topu.clone(),
        ])
        .not(),
    );
    g.add(
        pronQuant_S,
        pQuantA3pl_S,
        "lAr",
        items(&[
            hep.clone(),
            hepsi.clone(),
            birkaci.clone(),
            umum.clone(),
            cumlesi.clone(),
            cogu.clone(),
            bircogu.clone(),
            herbiri.clone(),
            tumu.clone(),
            hicbiri.clone(),
            topu.clone(),
            oburu.clone(),
        ])
        .not(),
    );
    g.add(
        pronQuant_S,
        pQuantA1pl_S,
        "lAr",
        items(std::slice::from_ref(&bazi)),
    );
    g.add(
        pronQuant_S,
        pQuantA2pl_S,
        "lAr",
        items(std::slice::from_ref(&bazi)),
    );
    g.add(
        pronQuant_S,
        pQuantA3pl_S,
        "",
        items(&[
            herkes.clone(),
            umum.clone(),
            birkaci.clone(),
            hepsi.clone(),
            cumlesi.clone(),
            cogu.clone(),
            bircogu.clone(),
            tumu.clone(),
            topu.clone(),
        ]),
    );
    g.add(pronQuant_S, a3sg_S, "", items(std::slice::from_ref(&kimse)));
    g.add(
        pronQuant_S,
        a3pl_S,
        "lAr",
        items(std::slice::from_ref(&kimse)),
    );
    let quant_1_2_pl = || {
        items(&[
            biri.clone(),
            bazi.clone(),
            birbiri.clone(),
            birkaci.clone(),
            herbiri.clone(),
            hep.clone(),
            kimi.clone(),
            cogu.clone(),
            bircogu.clone(),
            tumu.clone(),
            topu.clone(),
            hicbiri.clone(),
        ])
    };
    g.add(pronQuant_S, pQuantA1pl_S, "", quant_1_2_pl());
    g.add(pronQuant_S, pQuantA2pl_S, "", quant_1_2_pl());
    g.add(pronQuantModified_S, pQuantModA3pl_S, "", T);
    g.add(pQuantModA3pl_S, pP3pl_S, "lArI", T);
    g.add(
        pQuantA3sg_S,
        pP3sg_S,
        "",
        items(&[
            biri.clone(),
            birbiri.clone(),
            kimi.clone(),
            herbiri.clone(),
            hicbiri.clone(),
            oburu.clone(),
            oburku.clone(),
            beriki.clone(),
        ])
        .and(ph(P::MODIFIED_PRONOUN).not()),
    );
    g.add(
        pQuantA3sg_S,
        pP3sg_S,
        "sI",
        items(&[
            biri.clone(),
            bazi.clone(),
            kimi.clone(),
            birbiri.clone(),
            herbiri.clone(),
            hicbiri.clone(),
            oburku.clone(),
        ])
        .and(ph(P::MODIFIED_PRONOUN).not()),
    );
    g.add(
        pQuantA3pl_S,
        pP3pl_S,
        "I",
        items(&[
            biri.clone(),
            bazi.clone(),
            birbiri.clone(),
            kimi.clone(),
            oburku.clone(),
            beriki.clone(),
        ]),
    );
    g.add(
        pQuantA3pl_S,
        pP3pl_S,
        "",
        items(&[
            hepsi.clone(),
            birkaci.clone(),
            cumlesi.clone(),
            cogu.clone(),
            tumu.clone(),
            topu.clone(),
            bircogu.clone(),
        ]),
    );
    g.add(
        pQuantA3pl_S,
        pPnon_S,
        "",
        items(&[herkes.clone(), umum.clone(), oburku.clone(), beriki.clone()]),
    );
    g.add(pQuantA1pl_S, pP1pl_S, "ImIz", T);
    g.add(pQuantA2pl_S, pP2pl_S, "InIz", T);
    // kim, ne, nere.
    g.add(pronQues_S, pQuesA3sg_S, "", T);
    g.add(pronQues_S, pQuesA3pl_S, "lAr", T);
    let is_ne = || items(std::slice::from_ref(&ne));
    g.add(pQuesA3sg_S, pPnon_S, "", T);
    g.add(pQuesA3sg_S, pP3sg_S, "+sI", T);
    g.add(pQuesA3sg_S, pP1sg_S, "Im", is_ne().not());
    g.add(pQuesA3sg_S, pP1sg_S, "yIm", is_ne());
    g.add(pQuesA3sg_S, pP2sg_S, "In", is_ne().not());
    g.add(pQuesA3sg_S, pP2sg_S, "yIn", is_ne());
    g.add(pQuesA3sg_S, pP1pl_S, "ImIz", is_ne().not());
    g.add(pQuesA3sg_S, pP1pl_S, "yImIz", is_ne());
    g.add(pQuesA3pl_S, pPnon_S, "", T);
    g.add(pQuesA3pl_S, pP3sg_S, "I", T);
    g.add(pQuesA3pl_S, pP1sg_S, "Im", T);
    g.add(pQuesA3pl_S, pP1pl_S, "ImIz", T);
    // kendi.
    for a in [
        pReflexA1sg_S,
        pReflexA2sg_S,
        pReflexA3sg_S,
        pReflexA1pl_S,
        pReflexA2pl_S,
        pReflexA3pl_S,
    ] {
        g.add(pronReflex_S, a, "", T);
    }
    g.add(pReflexA1sg_S, pP1sg_S, "Im", T);
    g.add(pReflexA2sg_S, pP2sg_S, "In", T);
    g.add(pReflexA3sg_S, pP3sg_S, "+sI", T);
    g.add(pReflexA3sg_S, pP3sg_S, "", T);
    g.add(pReflexA1pl_S, pP1pl_S, "ImIz", T);
    g.add(pReflexA2pl_S, pP2pl_S, "InIz", T);
    g.add(pReflexA3pl_S, pP3pl_S, "lArI", T);
    // Pronoun cases: o-n-a, bu-n-u; ne-y-e, nere-y-e; bana, benim, bizim.
    let y_group = || {
        items(&[
            ne.clone(),
            nere.clone(),
            falan.clone(),
            falanca.clone(),
            hep.clone(),
            herkes.clone(),
        ])
    };
    let n_group = || y_group().not();
    g.add(pPnon_S, pNom_ST, "", T);
    g.add(
        pPnon_S,
        pDat_ST,
        "+nA",
        items(&[
            ben.clone(),
            sen.clone(),
            ne.clone(),
            nere.clone(),
            falan.clone(),
            falanca.clone(),
            herkes.clone(),
        ])
        .not(),
    );
    g.add(pPnon_S, pDat_ST, "+yA", y_group());
    g.add(pPnon_S, pAcc_ST, "+nI", n_group());
    g.add(pPnon_S, pAcc_ST, "+yI", y_group());
    g.add(pPnon_S, pLoc_ST, "+ndA", n_group());
    g.add(pPnon_S, pLoc_ST, ">dA", y_group());
    g.add(pPnon_S, pAbl_ST, "+ndAn", n_group());
    g.add(pPnon_S, pAbl_ST, ">dAn", y_group());
    g.add(
        pPnon_S,
        pGen_ST,
        "+nIn",
        n_group().and(items(&[biz.clone(), ben.clone(), sen.clone()]).not()),
    );
    g.add(pPnon_S, pGen_ST, "im", items(&[ben.clone(), biz.clone()]));
    g.add(pPnon_S, pGen_ST, "in", items(std::slice::from_ref(&sen)));
    g.add(
        pPnon_S,
        pGen_ST,
        "+yIn",
        y_group().and(items(std::slice::from_ref(&biz)).not()),
    );
    g.add(pPnon_S, pEqu_ST, ">cA", y_group());
    g.add(pPnon_S, pEqu_ST, ">cA", n_group());
    g.add(pPnon_S, pIns_ST, "+ylA", y_group());
    g.add(pPnon_S, pIns_ST, "+nlA", n_group());
    g.add(
        pPnon_S,
        pIns_ST,
        "+nInlA",
        n_group().and(items(&[bu.clone(), su.clone(), o.clone(), sen.clone()])),
    );
    g.add(pPnon_S, pIns_ST, "inle", items(std::slice::from_ref(&siz)));
    g.add(pPnon_S, pIns_ST, "imle", items(&[biz.clone(), ben.clone()]));
    let p1sg_cond = || {
        items(&[
            kim.clone(),
            ben.clone(),
            ne.clone(),
            nere.clone(),
            kendi.clone(),
        ])
    };
    let p2sg_cond = || {
        items(&[
            kim.clone(),
            sen.clone(),
            ne.clone(),
            nere.clone(),
            kendi.clone(),
        ])
    };
    for (p, cond) in [(pP1sg_S, p1sg_cond()), (pP2sg_S, p2sg_cond())] {
        g.add(p, pNom_ST, "", T);
        g.add(p, pDat_ST, "+nA", n_group());
        g.add(p, pAcc_ST, "+nI", n_group());
        g.add(p, pDat_ST, "+yA", y_group());
        g.add(p, pAcc_ST, "+yI", y_group());
        g.add(p, pLoc_ST, "+ndA", items(std::slice::from_ref(&kendi)));
        g.add(p, pAbl_ST, "+ndAn", items(std::slice::from_ref(&kendi)));
        g.add(p, pEqu_ST, "+ncA", items(std::slice::from_ref(&kendi)));
        g.add(p, pIns_ST, "+nlA", cond.clone());
        g.add(p, pGen_ST, "+nIn", cond);
    }
    let p3sg_cond = || {
        items(&[
            kendi.clone(),
            kim.clone(),
            ne.clone(),
            nere.clone(),
            o.clone(),
            bazi.clone(),
            biri.clone(),
            birbiri.clone(),
            herbiri.clone(),
            hep.clone(),
            kimi.clone(),
            hicbiri.clone(),
        ])
    };
    g.add(pP3sg_S, pNom_ST, "", T);
    g.add(pP3sg_S, pDat_ST, "+nA", n_group());
    g.add(pP3sg_S, pAcc_ST, "+nI", n_group());
    g.add(pP3sg_S, pDat_ST, "+yA", y_group());
    g.add(pP3sg_S, pAcc_ST, "+yI", y_group());
    g.add(pP3sg_S, pLoc_ST, "+ndA", p3sg_cond());
    g.add(pP3sg_S, pAbl_ST, "+ndAn", p3sg_cond());
    g.add(pP3sg_S, pGen_ST, "+nIn", p3sg_cond());
    g.add(pP3sg_S, pEqu_ST, "ncA", p3sg_cond());
    g.add(pP3sg_S, pIns_ST, "+ylA", p3sg_cond());
    let hep_cond = || {
        items(&[
            kendi.clone(),
            kim.clone(),
            ne.clone(),
            nere.clone(),
            biz.clone(),
            siz.clone(),
            biri.clone(),
            birbiri.clone(),
            birkaci.clone(),
            herbiri.clone(),
            hep.clone(),
            kimi.clone(),
            cogu.clone(),
            bircogu.clone(),
            tumu.clone(),
            topu.clone(),
            bazi.clone(),
            hicbiri.clone(),
        ])
    };
    for p in [pP1pl_S, pP2pl_S] {
        g.add(p, pNom_ST, "", T);
        g.add(p, pDat_ST, "+nA", n_group());
        g.add(p, pAcc_ST, "+nI", n_group());
        g.add(p, pDat_ST, "+yA", y_group());
        g.add(p, pAcc_ST, "+yI", y_group());
        g.add(p, pLoc_ST, "+ndA", hep_cond());
        g.add(p, pAbl_ST, "+ndAn", hep_cond());
        g.add(p, pGen_ST, "+nIn", hep_cond());
        g.add(p, pEqu_ST, "+ncA", hep_cond());
        g.add(p, pIns_ST, "+nlA", hep_cond());
    }
    let hepsi_cond = || {
        items(&[
            kendi.clone(),
            kim.clone(),
            ne.clone(),
            nere.clone(),
            o.clone(),
            bazi.clone(),
            biri.clone(),
            herkes.clone(),
            umum.clone(),
            birkaci.clone(),
            hepsi.clone(),
            cumlesi.clone(),
            cogu.clone(),
            bircogu.clone(),
            birbiri.clone(),
            tumu.clone(),
            kimi.clone(),
            topu.clone(),
        ])
    };
    g.add(pP3pl_S, pNom_ST, "", T);
    g.add(pP3pl_S, pDat_ST, "+nA", n_group());
    g.add(pP3pl_S, pAcc_ST, "+nI", n_group());
    g.add(pP3pl_S, pDat_ST, "+yA", y_group());
    g.add(pP3pl_S, pAcc_ST, "+yI", y_group());
    g.add(pP3pl_S, pLoc_ST, "+ndA", hepsi_cond());
    g.add(pP3pl_S, pAbl_ST, "+ndAn", hepsi_cond());
    g.add(
        pP3pl_S,
        pGen_ST,
        "+nIn",
        hepsi_cond().or(items(&[sen.clone(), siz.clone()])),
    );
    g.add(pP3pl_S, pEqu_ST, "+ncA", hepsi_cond());
    g.add(pP3pl_S, pIns_ST, "+ylA", hepsi_cond());
    let personal_and_demons = || {
        vec![
            bu.clone(),
            su.clone(),
            o_demons.clone(),
            ben.clone(),
            sen.clone(),
            o.clone(),
            biz.clone(),
            siz.clone(),
        ]
    };
    g.add(pNom_ST, with_S, "+nlI", items(&personal_and_demons()));
    g.add(pNom_ST, with_S, "lI", items(std::slice::from_ref(&nere)));
    g.add(pNom_ST, with_S, "+ylI", items(std::slice::from_ref(&ne)));
    let mut without_items = personal_and_demons();
    without_items.push(nere.clone());
    g.add(pNom_ST, without_S, "+nsIz", items(&without_items));
    g.add(
        pNom_ST,
        without_S,
        "+ysIz",
        items(std::slice::from_ref(&ne)),
    );
    g.add(
        pGen_ST,
        rel_S,
        "ki",
        items(&[
            nere.clone(),
            bu.clone(),
            su.clone(),
            o_demons.clone(),
            ne.clone(),
            sen.clone(),
            o.clone(),
            biz.clone(),
            siz.clone(),
        ]),
    );
    g.add(
        pLoc_ST,
        rel_S,
        "ki",
        Cond::TailSequence(rel_repetition).not(),
    );
    g.add(pIns_ST, vWhile_S, "+yken", T);
    for c in [pNom_ST, pDat_ST, pLoc_ST, pAbl_ST, pGen_ST, pIns_ST] {
        g.add(c, pronZeroDeriv_S, "", Cond::HasTail);
    }
    g.add(pronZeroDeriv_S, pvVerbRoot_S, "", T);

    // ======================= Pronoun predicates (connectVerbAfterPronoun) =======================
    g.add(pvVerbRoot_S, pvPresent_S, "", T);
    g.add(pvVerbRoot_S, vWhile_S, "+yken", T);
    g.add(pvVerbRoot_S, pvPast_S, "+ydI", T);
    g.add(pvVerbRoot_S, pvNarr_S, "+ymIş", T);
    g.add(pvVerbRoot_S, pvCond_S, "+ysA", T);
    let pv_a1sg = || pg(&[pA1pl_S, pP1sg_S]).not();
    let pv_a1pl = || pg(&[pA1sg_S, pA2sg_S, pP1sg_S, pP2sg_S]).not();
    let pv_a2sg = || pg(&[pA2pl_S, pP2sg_S]).not();
    let pv_a2pl = || pg(&[pA2sg_S, pP2pl_S]).not();
    g.add(pvPresent_S, pvA1sg_ST, "+yIm", pv_a1sg());
    g.add(pvPresent_S, pvA2sg_ST, "sIn", pv_a2sg());
    g.add(pvPresent_S, pvA3sg_S, "", T);
    g.add(pvPresent_S, pvA1pl_ST, "+yIz", pv_a1pl());
    g.add(pvPresent_S, pvA2pl_ST, "sInIz", T);
    g.add(pvPresent_S, pvA3pl_ST, "lAr", pg(&[pLoc_ST]));
    g.add(pvPast_S, pvA1sg_ST, "m", pv_a1sg());
    g.add(pvPast_S, pvA2sg_ST, "n", pv_a2sg());
    g.add(pvPast_S, pvA1pl_ST, "k", pv_a1pl());
    g.add(pvPast_S, pvA2pl_ST, "InIz", T);
    g.add(pvPast_S, pvA3pl_ST, "lAr", T);
    g.add(pvPast_S, pvA3sg_ST, "", T);
    g.add(pvNarr_S, pvA1sg_ST, "Im", pv_a1sg());
    g.add(pvNarr_S, pvA2sg_ST, "sIn", pv_a2sg());
    g.add(pvNarr_S, pvA1pl_ST, "Iz", pv_a1pl());
    g.add(pvNarr_S, pvA2pl_ST, "sInIz", T);
    g.add(pvNarr_S, pvA3pl_ST, "lAr", T);
    g.add(pvNarr_S, pvA3sg_ST, "", T);
    g.add(pvNarr_S, pvCond_S, "sA", T);
    g.add(pvCond_S, pvA1sg_ST, "m", pv_a1sg());
    g.add(pvCond_S, pvA2sg_ST, "n", pv_a2sg());
    g.add(pvCond_S, pvA1pl_ST, "k", pv_a1pl());
    g.add(pvCond_S, pvA2pl_ST, "nIz", pv_a2pl());
    g.add(pvCond_S, pvA3sg_ST, "", T);
    g.add(pvCond_S, pvA3pl_ST, "lAr", T);
    let pv_reject = || cg(&[pvPast_S, pvCond_S, pvCopBeforeA3pl_S]).not();
    for p in [pvA1sg_ST, pvA2sg_ST, pvA1pl_ST, pvA2pl_ST, pvA3pl_ST] {
        g.add(p, pvCop_ST, "dIr", pv_reject());
    }
    g.add(pvA3sg_S, pvCop_ST, ">dIr", pv_reject());
    g.add(pvPresent_S, pvCopBeforeA3pl_S, ">dIr", T);
    g.add(pvCopBeforeA3pl_S, pvA3pl_ST, "lAr", T);

    // ======================= Verbs =======================
    g.add(verbRoot_S, vImp_S, "", T);
    g.add(vImp_S, vA2sg_ST, "", T);
    g.add(vImp_S, vA2sg_ST, "sAnA", T);
    g.add(vImp_S, vA3sg_ST, "sIn", T);
    g.add(vImp_S, vA2pl_ST, "+yIn", T);
    g.add(vImp_S, vA2pl_ST, "+yInIz", T);
    g.add(vImp_S, vA2pl_ST, "sAnIzA", T);
    g.add(vImp_S, vA3pl_ST, "sInlAr", T);
    g.add(
        verbRoot_S,
        vCausT_S,
        "t",
        has(R::CAUSATIVE_T)
            .or(ld(&[vCausTir_S]))
            .and_not(ld(&[vCausT_S, vPass_S, vAble_S])),
    );
    g.add(
        verbRoot_S,
        vCausTir_S,
        ">dIr",
        ph(P::LAST_LETTER_CONSONANT).and_not(ld(&[vCausTir_S, vPass_S, vAble_S])),
    );
    g.add(vCausT_S, verbRoot_S, "", T);
    g.add(vCausTir_S, verbRoot_S, "", T);
    g.add(
        verbRoot_S,
        vProgYor_S,
        "Iyor",
        ph(P::LAST_LETTER_VOWEL).not(),
    );
    g.add(verbRoot_VowelDrop_S, vProgYor_S, "Iyor", T);
    for (to, t) in [
        (vA1sg_ST, "um"),
        (vA2sg_ST, "sun"),
        (vA3sg_ST, ""),
        (vA1pl_ST, "uz"),
        (vA2pl_ST, "sunuz"),
        (vA3pl_ST, "lar"),
        (vCond_S, "sa"),
        (vPastAfterTense_S, "du"),
        (vNarrAfterTense_S, "muş"),
        (vCopBeforeA3pl_S, "dur"),
        (vWhile_S, "ken"),
    ] {
        g.add(vProgYor_S, to, t, T);
    }
    g.add(verbRoot_S, vProgMakta_S, "mAktA", T);
    for (to, t) in [
        (vA1sg_ST, "yIm"),
        (vA2sg_ST, "sIn"),
        (vA3sg_ST, ""),
        (vA1pl_ST, "yIz"),
        (vA2pl_ST, "sInIz"),
        (vA3pl_ST, "lAr"),
        (vCond_S, "ysA"),
        (vPastAfterTense_S, "ydI"),
        (vNarrAfterTense_S, "ymIş"),
        (vCopBeforeA3pl_S, "dIr"),
        (vWhile_S, "yken"),
    ] {
        g.add(vProgMakta_S, to, t, T);
    }
    // Aorist: -Ir after any suffix with letters (yap-tır-ır) or on Aorist_I roots; -Ar on
    // bare Aorist_A roots.
    let aor_i = || has(R::AORIST_I).or(Cond::HasAnySuffixSurface);
    let aor_a = || has(R::AORIST_A).and(Cond::HasAnySuffixSurface.not());
    g.add(verbRoot_S, vAor_S, "Ir", aor_i());
    g.add(verbRoot_S, vAor_S, "Ar", aor_a());
    for (to, t) in [
        (vA1sg_ST, "Im"),
        (vA2sg_ST, "sIn"),
        (vA3sg_ST, ""),
        (vA1pl_ST, "Iz"),
        (vA2pl_ST, "sInIz"),
        (vA3pl_ST, "lAr"),
        (vPastAfterTense_S, "dI"),
        (vNarrAfterTense_S, "mIş"),
        (vCond_S, "sA"),
        (vCopBeforeA3pl_S, "dIr"),
        (vWhile_S, "ken"),
    ] {
        g.add(vAor_S, to, t, T);
    }
    // Negation -mA and what follows it.
    g.add(verbRoot_S, vNeg_S, "mA", pm(&[M::Able]).not());
    for (to, t) in [
        (vImp_S, ""),
        (vPast_S, "dI"),
        (vFut_S, "yAcA~k"),
        (vFut_S, "yAcA!ğ"),
        (vNarr_S, "mIş"),
        (vProgMakta_S, "mAktA"),
        (vOpt_S, "yA"),
        (vDesr_S, "sA"),
        (vNeces_S, "mAlI"),
        (vInf1_S, "mAk"),
        (vInf2_S, "mA"),
        (vInf3_S, "yIş"),
        (vActOf_S, "mAcA"),
        (vPastPart_S, "dI~k"),
        (vPastPart_S, "dI!ğ"),
        (vFutPart_S, "yAcA~k"),
        (vFutPart_S, "yAcA!ğ"),
        (vPresPart_S, "yAn"),
        (vNarrPart_S, "mIş"),
        (vSinceDoingSo_S, "yAlI"),
        (vByDoingSo_S, "yArAk"),
        (vHastily_S, "yIver"),
        (vEverSince_S, "yAgör"),
        (vAfterDoing_S, "yIp"),
        (vWhen_S, "yIncA"),
        (vAsLongAs_S, "dIkçA"),
        (vNotState_S, "mAzlI~k"),
        (vNotState_S, "mAzlI!ğ"),
        (vFeelLike_S, "yAsI"),
    ] {
        g.add(vNeg_S, to, t, T);
    }
    // gel-m-iyor: the narrowed negative before -Iyor only.
    g.add(verbRoot_S, vNegProg1_S, "m", T);
    g.add(vNegProg1_S, vProgYor_S, "Iyor", T);
    // gel-me-z, gel-me-m, gel-me-yiz.
    g.add(vNeg_S, vAorNeg_S, "z", T);
    g.add(vNeg_S, vAorNegEmpty_S, "", T);
    for (to, t) in [
        (vA2sg_ST, "sIn"),
        (vA3sg_ST, ""),
        (vA2pl_ST, "sInIz"),
        (vA3pl_ST, "lAr"),
        (vPastAfterTense_S, "dI"),
        (vNarrAfterTense_S, "mIş"),
        (vCond_S, "sA"),
        (vCopBeforeA3pl_S, "dIr"),
        (vWhile_S, "ken"),
    ] {
        g.add(vAorNeg_S, to, t, T);
    }
    g.add(vAorNegEmpty_S, vA1sg_ST, "m", T);
    g.add(vAorNegEmpty_S, vA1pl_ST, "yIz", T);
    g.add(vNeg_S, vAorPartNeg_S, "z", T);
    g.add(vAorPartNeg_S, adjAfterVerb_ST, "", T);
    // Ability: gel-ebil, gel-me-yebil.
    g.add(verbRoot_S, vAble_S, "+yAbil", ld(&[vAble_S]).not());
    g.add(vAble_S, verbRoot_S, "", T);
    g.add(vAbleNeg_S, vAbleNegDerivRoot_S, "", T);
    g.add(vAbleNegDerivRoot_S, vNeg_S, "mA", T);
    g.add(vAbleNegDerivRoot_S, vNegProg1_S, "m", T);
    g.add(vNeg_S, vAble_S, "yAbil", T);
    // Inability: gel-eme-z, gel-eme-m, gel-em-iyor.
    g.add(verbRoot_S, vUnable_S, "+yAmA", pm(&[M::Able]).not());
    g.copy_outgoing(vUnable_S, vNeg_S);
    g.add(verbRoot_S, vUnableProg1_S, "+yAm", T);
    g.add(vUnableProg1_S, vProgYor_S, "Iyor", T);
    // Nominalisations and participles.
    g.add(verbRoot_S, vInf1_S, "mA~k", T);
    g.add(vInf1_S, nounInf1Root_S, "", T);
    g.add(verbRoot_S, vInf2_S, "mA", T);
    g.add(vInf2_S, noun_S, "", T);
    g.add(verbRoot_S, vInf3_S, "+yIş", T);
    g.add(vInf3_S, noun_S, "", T);
    g.add(verbRoot_S, vAgt_S, "+yIcI", T);
    g.add(vAgt_S, noun_S, "", T);
    g.add(vAgt_S, adjAfterVerb_ST, "", T);
    g.add(verbRoot_S, vActOf_S, "mAcA", T);
    g.add(vActOf_S, nounActOfRoot_S, "", T);
    g.add(verbRoot_S, vPastPart_S, ">dI~k", T);
    g.add(verbRoot_S, vPastPart_S, ">dI!ğ", T);
    g.add(vPastPart_S, noun_S, "", T);
    g.add(vPastPart_S, adjAfterVerb_S, "", T);
    g.add(verbRoot_S, vFutPart_S, "+yAcA~k", T);
    g.add(verbRoot_S, vFutPart_S, "+yAcA!ğ", T);
    g.add(vFutPart_S, noun_S, "", Cond::HasTail);
    g.add(vFutPart_S, adjAfterVerb_S, "", T);
    g.add(verbRoot_S, vNarrPart_S, "mIş", T);
    g.add(vNarrPart_S, adjectiveRoot_ST, "", T);
    g.add(verbRoot_S, vAorPart_S, "Ir", aor_i());
    g.add(verbRoot_S, vAorPart_S, "Ar", aor_a());
    g.add(vAorPart_S, adjAfterVerb_ST, "", T);
    g.add(verbRoot_S, vPresPart_S, "+yAn", T);
    g.add(vPresPart_S, noun_S, "", Cond::HasTail);
    g.add(vPresPart_S, adjAfterVerb_ST, "", T);
    g.add(verbRoot_S, vFeelLike_S, "+yAsI", T);
    g.add(vFeelLike_S, noun_S, "", Cond::HasTail);
    g.add(vFeelLike_S, adjAfterVerb_ST, "", T);
    g.add(verbRoot_S, vNotState_S, "mAzlI~k", T);
    g.add(verbRoot_S, vNotState_S, "mAzlI!ğ", T);
    g.add(vNotState_S, noun_S, "", T);
    // Voice.
    g.add(vRecip_S, verbRoot_S, "", T);
    g.add(vImplicitRecipRoot_S, vRecip_S, "", T);
    g.add(vImplicitReflexRoot_S, vReflex_S, "", T);
    g.add(vReflex_S, verbRoot_S, "", T);
    let no_pass = || cm(&[M::Pass]).not();
    // Passive_In describes the root's last letter, so -In and -InIl follow the root, not a
    // causative ending in r or t: kal-dır-ıl, oku-t-ul, not kal-dır-ın, oku-t-un. Zemberek
    // allows them there (deviation).
    let after_root = || {
        has(R::PASSIVE_IN)
            .and(no_pass())
            .and_not(ps(&[vCausT_S, vCausTir_S]))
    };
    g.add(verbRoot_S, vPass_S, "In", after_root());
    g.add(verbRoot_S, vPass_S, "InIl", after_root());
    g.add(
        verbRoot_S,
        vPass_S,
        "+nIl",
        ps(&[vCausT_S, vCausTir_S])
            .or(has(R::PASSIVE_IN).not())
            .and(no_pass()),
    );
    g.add(vPass_S, verbRoot_S, "", T);
    // Tenses and their persons.
    for (to, t) in [
        (vA1sg_ST, "m"),
        (vA2sg_ST, "n"),
        (vA3sg_ST, ""),
        (vA1pl_ST, "k"),
        (vA2pl_ST, "nIz"),
        (vA3pl_ST, "lAr"),
    ] {
        g.add(vCond_S, to, t, T);
    }
    g.add(verbRoot_S, vPast_S, ">dI", T);
    for (to, t) in [
        (vA1sg_ST, "m"),
        (vA2sg_ST, "n"),
        (vA3sg_ST, ""),
        (vA1pl_ST, "k"),
        (vA2pl_ST, "nIz"),
        (vA3pl_ST, "lAr"),
    ] {
        g.add(vPast_S, to, t, T);
    }
    g.add(vPast_S, vCond_S, "ysA", T);
    g.add(verbRoot_S, vNarr_S, "mIş", T);
    for (to, t) in [
        (vA1sg_ST, "Im"),
        (vA2sg_ST, "sIn"),
        (vA3sg_ST, ""),
        (vA1pl_ST, "Iz"),
        (vA2pl_ST, "sInIz"),
        (vA3pl_ST, "lAr"),
    ] {
        g.add(vNarr_S, to, t, T);
    }
    g.add(vNarr_S, vCond_S, "sA", T);
    g.add(vNarr_S, vPastAfterTense_S, "tI", T);
    g.add(vNarr_S, vCopBeforeA3pl_S, "tIr", T);
    g.add(vNarr_S, vWhile_S, "ken", T);
    g.add(vNarr_S, vNarrAfterTense_S, "mIş", T);
    for (to, t) in [
        (vA1sg_ST, "m"),
        (vA2sg_ST, "n"),
        (vA3sg_ST, ""),
        (vA1pl_ST, "k"),
        (vA2pl_ST, "nIz"),
        (vA3pl_ST, "lAr"),
    ] {
        g.add(vPastAfterTense_S, to, t, T);
    }
    for (to, t) in [
        (vA1sg_ST, "Im"),
        (vA2sg_ST, "sIn"),
        (vA3sg_ST, ""),
        (vA1pl_ST, "Iz"),
        (vA2pl_ST, "sInIz"),
        (vA3pl_ST, "lAr"),
    ] {
        g.add(vNarrAfterTense_S, to, t, T);
    }
    g.add(vNarrAfterTense_S, vWhile_S, "ken", T);
    g.add(vNarrAfterTense_S, vCopBeforeA3pl_S, "tIr", T);
    g.add(verbRoot_S, vFut_S, "+yAcA~k", T);
    g.add(verbRoot_S, vFut_S, "+yAcA!ğ", T);
    for (to, t) in [
        (vA1sg_ST, "Im"),
        (vA2sg_ST, "sIn"),
        (vA3sg_ST, ""),
        (vA1pl_ST, "Iz"),
        (vA2pl_ST, "sInIz"),
        (vA3pl_ST, "lAr"),
    ] {
        g.add(vFut_S, to, t, T);
    }
    g.add(vFut_S, vCond_S, "sA", T);
    g.add(vFut_S, vPastAfterTense_S, "tI", T);
    g.add(vFut_S, vNarrAfterTense_S, "mIş", T);
    g.add(vFut_S, vCopBeforeA3pl_S, "tIr", T);
    g.add(vFut_S, vWhile_S, "ken", T);

    // de-/ye- and their narrowed stems di-/yi- (diyor, yiyecek; but der, dedi).
    let di_yi = || Cond::RootSurfaceIsAny(vec!["di", "yi"]);
    let de_ye = || Cond::RootSurfaceIsAny(vec!["de", "ye"]);
    let multi_verb = || {
        pm(&[
            M::EverSince,
            M::Repeat,
            M::Almost,
            M::Hastily,
            M::Stay,
            M::Start,
        ])
        .not()
    };
    for (to, t, c) in [
        (vFut_S, "yece~k", di_yi()),
        (vFut_S, "yece!ğ", di_yi()),
        (vProgYor_S, "yor", di_yi()),
        (vAble_S, "yebil", di_yi()),
        (vAbleNeg_S, "ye", di_yi()),
        (vInf3_S, "yiş", Cond::RootSurfaceIsAny(vec!["yi"])),
        (vFutPart_S, "yece~k", di_yi()),
        (vFutPart_S, "yece!ğ", di_yi()),
        (vPresPart_S, "yen", di_yi()),
        (vEverSince_S, "yegel", di_yi().and(multi_verb())),
        (vRepeat_S, "yedur", di_yi().and(multi_verb())),
        (vRepeat_S, "yegör", di_yi().and(multi_verb())),
        (vAlmost_S, "yeyaz", di_yi().and(multi_verb())),
        (vStart_S, "yekoy", di_yi().and(multi_verb())),
        (vSinceDoingSo_S, "yeli", di_yi()),
        (vByDoingSo_S, "yerek", di_yi()),
        (vFeelLike_S, "yesi", di_yi()),
        (vAfterDoing_S, "yip", di_yi()),
        (vWithoutBeingAbleToHaveDoneSo_S, "yemeden", di_yi()),
        (vOpt_S, "ye", di_yi()),
        (vCausTir_S, "dir", de_ye()),
        (vPass_S, "n", de_ye()),
        (vPass_S, "nil", de_ye()),
        (vPast_S, "di", de_ye()),
        (vNarr_S, "miş", de_ye()),
        (vAor_S, "r", de_ye()),
        (vNeg_S, "me", de_ye()),
        (vNegProg1_S, "m", de_ye()),
        (vProgMakta_S, "mekte", de_ye()),
        (vDesr_S, "se", de_ye()),
        (vInf1_S, "mek", de_ye()),
        (vInf2_S, "me", de_ye()),
        (vInf3_S, "yiş", Cond::RootSurfaceIsAny(vec!["de"])),
        (vPastPart_S, "di~k", de_ye()),
        (vPastPart_S, "di!ğ", de_ye()),
        (vNarrPart_S, "miş", de_ye()),
        (vHastily_S, "yiver", di_yi().and(multi_verb())),
        (vAsLongAs_S, "dikçe", de_ye()),
        (vWithoutHavingDoneSo_S, "meden", de_ye()),
        (vWithoutHavingDoneSo_S, "meksizin", de_ye()),
        (vNeces_S, "meli", de_ye()),
        (vNotState_S, "mezli~k", de_ye()),
        (vNotState_S, "mezli!ğ", de_ye()),
        (vImp_S, "", Cond::RootSurfaceIsAny(vec!["de"])),
        (vImpYemekYe_S, "", Cond::RootSurfaceIsAny(vec!["ye"])),
        (vImpYemekYi_S, "", Cond::RootSurfaceIsAny(vec!["yi"])),
    ] {
        g.add(vDeYeRoot_S, to, t, c);
    }
    g.add(vImpYemekYi_S, vA2pl_ST, "yin", T);
    g.add(vImpYemekYi_S, vA2pl_ST, "yiniz", T);
    g.add(vImpYemekYe_S, vA2sg_ST, "", T);
    g.add(vImpYemekYe_S, vA2sg_ST, "sene", T);
    g.add(vImpYemekYe_S, vA3sg_ST, "sin", T);
    g.add(vImpYemekYe_S, vA2pl_ST, "senize", T);
    g.add(vImpYemekYe_S, vA3pl_ST, "sinler", T);

    // Optative, desire, necessity.
    g.add(verbRoot_S, vOpt_S, "+yA", T);
    for (to, t) in [
        (vA1sg_ST, "yIm"),
        (vA2sg_ST, "sIn"),
        (vA3sg_ST, ""),
        (vA1pl_ST, "lIm"),
        (vA2pl_ST, "sInIz"),
        (vA3pl_ST, "lAr"),
        (vPastAfterTense_S, "ydI"),
        (vNarrAfterTense_S, "ymIş"),
    ] {
        g.add(vOpt_S, to, t, T);
    }
    g.add(verbRoot_S, vDesr_S, "sA", T);
    for (to, t) in [
        (vA1sg_ST, "m"),
        (vA2sg_ST, "n"),
        (vA3sg_ST, ""),
        (vA1pl_ST, "k"),
        (vA2pl_ST, "nIz"),
        (vA3pl_ST, "lAr"),
        (vPastAfterTense_S, "ydI"),
        (vNarrAfterTense_S, "ymIş"),
    ] {
        g.add(vDesr_S, to, t, T);
    }
    g.add(verbRoot_S, vNeces_S, "mAlI", T);
    for (to, t) in [
        (vA1sg_ST, "yIm"),
        (vA2sg_ST, "sIn"),
        (vA3sg_ST, ""),
        (vA1pl_ST, "yIz"),
        (vA2pl_ST, "sInIz"),
        (vA3pl_ST, "lAr"),
        (vPastAfterTense_S, "ydI"),
        (vCond_S, "ysA"),
        (vNarrAfterTense_S, "ymIş"),
        (vCopBeforeA3pl_S, "dIr"),
        (vWhile_S, "yken"),
    ] {
        g.add(vNeces_S, to, t, T);
    }

    // After the person: yaparlardı, gelmiş-im-dir, geldin-se.
    let not_past_narr_cond = || ps(&[vPastAfterTense_S, vNarrAfterTense_S, vCond_S]).not();
    // Unlike Zemberek, not after the simple past either (geldilerdi, geldilermiş); geldilerse
    // stays.
    let not_after_past = || not_past_narr_cond().and_not(ps(&[vPast_S]));
    g.add(vA3pl_ST, vPastAfterTense_ST, "dI", not_after_past());
    g.add(vA3pl_ST, vNarrAfterTense_ST, "mIş", not_after_past());
    g.add(vA3pl_ST, vCond_ST, "sA", not_past_narr_cond());
    let cop_while = || pm(&[M::Prog1, M::Prog2, M::Neces, M::Fut, M::Narr, M::Aor]);
    g.add(vA3pl_ST, vCop_ST, "dIr", cop_while());
    g.add(vA3pl_ST, vWhile_S, "ken", cop_while());
    for p in [vA1sg_ST, vA2sg_ST, vA1pl_ST, vA2pl_ST] {
        g.add(p, vCop_ST, "dIr", cop_while());
    }
    g.add(vA3sg_ST, vCop_ST, ">dIr", cop_while());
    g.add(vCopBeforeA3pl_S, vA3pl_ST, "lAr", T);
    let previous_past = || pm(&[M::Past]).and_not(cm(&[M::Cond, M::Desr]));
    for p in [vA2pl_ST, vA2sg_ST, vA1sg_ST, vA1pl_ST] {
        g.add(p, vCondAfterPerson_ST, "sA", previous_past());
    }

    // Compound verbs -(y)Iver, -(y)Agel, ...
    g.add(verbRoot_S, vEverSince_S, "+yAgel", multi_verb());
    g.add(verbRoot_S, vRepeat_S, "+yAdur", multi_verb());
    g.add(verbRoot_S, vRepeat_S, "+yAgör", multi_verb());
    g.add(verbRoot_S, vAlmost_S, "+yAyaz", multi_verb());
    g.add(verbRoot_S, vHastily_S, "+yIver", multi_verb());
    g.add(verbRoot_S, vStay_S, "+yAkal", multi_verb());
    g.add(verbRoot_S, vStart_S, "+yAkoy", multi_verb());
    for s in [
        vEverSince_S,
        vRepeat_S,
        vAlmost_S,
        vHastily_S,
        vStay_S,
        vStart_S,
    ] {
        g.add(s, verbRoot_S, "", T);
    }
    // Converbs.
    g.add(vA3sg_ST, vAsIf_S, ">cAsInA", pm(&[M::Aor, M::Narr]));
    g.add(verbRoot_S, vWhen_S, "+yIncA", T);
    g.add(verbRoot_S, vSinceDoingSo_S, "+yAlI", T);
    g.add(verbRoot_S, vByDoingSo_S, "+yArAk", T);
    g.add(verbRoot_S, vAdamantly_S, "+yAsIyA", T);
    g.add(verbRoot_S, vAfterDoing_S, "+yIp", T);
    g.add(verbRoot_S, vWithoutBeingAbleToHaveDoneSo_S, "+yAmAdAn", T);
    g.add(verbRoot_S, vAsLongAs_S, ">dIkçA", T);
    g.add(verbRoot_S, vWithoutHavingDoneSo_S, "mAdAn", T);
    g.add(verbRoot_S, vWithoutHavingDoneSo_S, "mAksIzIn", T);
    for s in [
        vAsIf_S,
        vSinceDoingSo_S,
        vByDoingSo_S,
        vAdamantly_S,
        vAfterDoing_S,
        vWithoutBeingAbleToHaveDoneSo_S,
        vAsLongAs_S,
        vWithoutHavingDoneSo_S,
        vWhile_S,
    ] {
        g.add(s, advRoot_ST, "", T);
    }
    // gel-ince-ye: the -(y)IncA converb takes the dative (until).
    g.add(vWhen_S, advNounRoot_ST, "", T);

    // ======================= Question particle =======================
    g.add(questionRoot_S, qPresent_S, "", T);
    g.add(questionRoot_S, qPast_S, "ydI", T);
    g.add(questionRoot_S, qNarr_S, "ymIş", T);
    g.add(qPresent_S, qA1sg_ST, "yIm", T);
    g.add(qPresent_S, qA2sg_ST, "sIn", T);
    g.add(qPresent_S, qA3sg_ST, "", T);
    g.add(qPast_S, qA1sg_ST, "m", T);
    g.add(qNarr_S, qA1sg_ST, "Im", T);
    g.add(qPast_S, qA2sg_ST, "n", T);
    g.add(qNarr_S, qA2sg_ST, "sIn", T);
    g.add(qPast_S, qA1pl_ST, "k", T);
    g.add(qNarr_S, qA1pl_ST, "Iz", T);
    g.add(qPresent_S, qA1pl_ST, "+yIz", T);
    g.add(qPast_S, qA2pl_ST, "InIz", T);
    g.add(qNarr_S, qA2pl_ST, "sInIz", T);
    g.add(qPresent_S, qA2pl_ST, "sInIz", T);
    g.add(qPast_S, qA3pl_ST, "lAr", T);
    g.add(qNarr_S, qA3pl_ST, "lAr", T);
    g.add(qPast_S, qA3sg_ST, "", T);
    g.add(qNarr_S, qA3sg_ST, "", T);
    let q_reject = || cg(&[qPast_S]).not();
    for p in [qA1sg_ST, qA2sg_ST, qA1pl_ST, qA2pl_ST] {
        g.add(p, qCop_ST, "dIr", q_reject());
    }
    g.add(qA3sg_ST, qCop_ST, ">dIr", q_reject());
    g.add(qPresent_S, qCopBeforeA3pl_S, "dIr", T);
    g.add(qCopBeforeA3pl_S, qA3pl_ST, "lAr", T);

    // ======================= Adverbs, last-vowel-drop words, postpositions, imek ==========
    g.add(advNounRoot_ST, avZero_S, "", T);
    g.add(avZero_S, avNounAfterAdvRoot_ST, "", T);
    g.add(avNounAfterAdvRoot_ST, avA3sg_S, "", T);
    g.add(avA3sg_S, avPnon_S, "", T);
    g.add(avPnon_S, avDat_ST, "+yA", T);
    g.add(advForVerbDeriv_ST, avZeroToVerb_S, "", T);
    g.add(avZeroToVerb_S, nVerb_S, "", T);

    g.add(nounLastVowelDropRoot_S, a3sgLastVowelDrop_S, "", T);
    g.add(nounLastVowelDropRoot_S, a3PlLastVowelDrop_S, "lAr", T);
    g.add(a3sgLastVowelDrop_S, pNonLastVowelDrop_S, "", T);
    g.add(a3PlLastVowelDrop_S, pNonLastVowelDrop_S, "", T);
    g.add(pNonLastVowelDrop_S, loc_ST, ">dA", T);
    g.add(pNonLastVowelDrop_S, abl_ST, ">dAn", T);
    g.add(adjLastVowelDropRoot_S, zeroLastVowelDrop_S, "", T);
    g.add(postpLastVowelDropRoot_S, zeroLastVowelDrop_S, "", T);
    g.add(zeroLastVowelDrop_S, nounLastVowelDropRoot_S, "", T);

    g.add(postpRoot_ST, postpZero_S, "", T);
    g.add(postpZero_S, nVerb_S, "", T);
    g.add(
        postpZero_S,
        po2nRoot_S,
        "",
        items(&[gibi.clone(), sonra.clone()]),
    );
    g.add(po2nRoot_S, po2nA3sg_S, "", T);
    g.add(po2nRoot_S, po2nA3pl_S, "lAr", T);
    g.add(po2nA3sg_S, po2nP3sg_S, "+sI", T);
    for (to, t) in [
        (po2nP1sg_S, "m"),
        (po2nP2sg_S, "n"),
        (po2nP1pl_S, "miz"),
        (po2nP2pl_S, "niz"),
    ] {
        g.add(po2nA3sg_S, to, t, items(std::slice::from_ref(&gibi)));
    }
    g.add(po2nA3pl_S, po2nP3sg_S, "+sI", T);
    g.add(po2nA3pl_S, po2nPnon_S, "", T);
    for (to, t) in [
        (po2nNom_ST, ""),
        (po2nDat_ST, "nA"),
        (po2nLoc_ST, "ndA"),
        (po2nAbl_ST, "ndAn"),
        (po2nIns_ST, "ylA"),
        (po2nGen_ST, "nIn"),
        (po2nAcc_ST, "nI"),
    ] {
        g.add(po2nP3sg_S, to, t, T);
    }
    for (to, t) in [
        (po2nNom_ST, ""),
        (po2nDat_ST, "A"),
        (po2nLoc_ST, "dA"),
        (po2nAbl_ST, "dAn"),
        (po2nIns_ST, "lA"),
        (po2nGen_ST, "In"),
        (po2nEqu_ST, "cA"),
        (po2nAcc_ST, "I"),
    ] {
        g.add(po2nPnon_S, to, t, T);
    }
    for p in [po2nP1sg_S, po2nP2sg_S, po2nP1pl_S, po2nP2pl_S] {
        g.add(p, po2nDat_ST, "e", T);
    }

    g.add(imekRoot_S, imekPast_S, "di", T);
    g.add(imekRoot_S, imekNarr_S, "miş", T);
    g.add(imekRoot_S, imekCond_S, "se", T);
    for (to, t) in [
        (imekA1sg_ST, "m"),
        (imekA2sg_ST, "n"),
        (imekA3sg_ST, ""),
        (imekA1pl_ST, "k"),
        (imekA2pl_ST, "niz"),
        (imekA3pl_ST, "ler"),
    ] {
        g.add(imekPast_S, to, t, T);
        g.add(imekCond_S, to, t, T);
    }
    for (to, t) in [
        (imekA1sg_ST, "im"),
        (imekA2sg_ST, "sin"),
        (imekA3sg_ST, ""),
        (imekA1pl_ST, "iz"),
        (imekA2pl_ST, "siniz"),
        (imekA3pl_ST, "ler"),
    ] {
        g.add(imekNarr_S, to, t, T);
    }
    g.add(imekPast_S, imekCond_S, "yse", T);
    g.add(imekNarr_S, imekCond_S, "se", T);
    let imek_reject = || cg(&[imekPast_S]).not();
    for p in [
        imekA1sg_ST,
        imekA2sg_ST,
        imekA1pl_ST,
        imekA2pl_ST,
        imekA3pl_ST,
    ] {
        g.add(p, imekCop_ST, "dir", imek_reject());
    }
    g.add(imekA3sg_ST, imekCop_ST, "tir", imek_reject());

    // ======================= Post-processing =======================
    // değil takes the nominal predicate's persons and tenses (değil-im, değil-di).
    g.copy_outgoing(nNeg_S, nVerb_S);
    // kavur-: kavr-ul (passive on the vowel-dropped stem), kavur-acak (everything else).
    g.add(verbLastVowelDropModRoot_S, vPass_S, "Il", T);
    g.copy_outgoing(verbLastVowelDropUnmodRoot_S, verbRoot_S);
    g.remove_outgoing_to(verbLastVowelDropUnmodRoot_S, M::Pass);

    let roots = RootStates {
        noun: noun_S,
        noun_compound_root: nounCompoundRoot_S,
        noun_su_root: nounSuRoot_S,
        noun_proper: nounProper_S,
        noun_abbrv: nounAbbrv_S,
        noun_no_suffix: nounNoSuffix_S,
        noun_last_vowel_drop: nounLastVowelDropRoot_S,
        adj: adjectiveRoot_ST,
        adj_last_vowel_drop: adjLastVowelDropRoot_S,
        num: numeralRoot_ST,
        pron_pers: pronPers_S,
        pron_pers_mod: pronPers_Mod_S,
        pron_demons: pronDemons_S,
        pron_quant: pronQuant_S,
        pron_quant_modified: pronQuantModified_S,
        pron_ques: pronQues_S,
        pron_reflex: pronReflex_S,
        adv: advRoot_ST,
        adv_for_verb_deriv: advForVerbDeriv_ST,
        conj: conjRoot_ST,
        question: questionRoot_S,
        interj: interjRoot_ST,
        verb: verbRoot_S,
        verb_vowel_drop: verbRoot_VowelDrop_S,
        verb_last_vowel_drop_mod: verbLastVowelDropModRoot_S,
        verb_last_vowel_drop_unmod: verbLastVowelDropUnmodRoot_S,
        verb_implicit_recip: vImplicitRecipRoot_S,
        verb_implicit_reflex: vImplicitReflexRoot_S,
        verb_de_ye: vDeYeRoot_S,
        verb_degil: nVerbDegil_S,
        imek: imekRoot_S,
        punc: puncRoot_ST,
        det: detRoot_ST,
        postp: postpRoot_ST,
        dup: dupRoot_ST,
        nominal_verb: StateSet::of(&[
            nVerb_S,
            nVerbDegil_S,
            pvVerbRoot_S,
            questionRoot_S,
            imekRoot_S,
        ]),
    };
    let _ = (postpLastVowelDropRoot_S, AdvP);
    (g, roots)
}
