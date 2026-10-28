//! The non-finite verb forms: verbal nouns, participles and converbs (UD's `VerbForm` Vnoun,
//! Part and Conv). One inventory, which the dependency parser, the readability metrics and the
//! syntactic analyzer share (review 2 PARSER2-02).

use super::analyzer::{tag_of, MorphParse, ENGINE};
use super::morphotactics::{Morpheme as M, StateId};
use super::pos::PrimaryPos;

/// A non-finite verb form's kind, as UD's `VerbForm` names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VerbForm {
    /// A verbal noun: an infinitive (gel-mek, gel-me, gel-iş) or a participle used as a noun
    /// (gel-diğ-i "that he came").
    Vnoun,
    /// A participle (gel-en, gel-diğ-i gün, gel-miş, gel-ir).
    Part,
    /// A converb (gel-ince, gel-ip, gel-erek, gel-ir-ken, gel-dikçe, gel-ircesine).
    Conv,
}

impl VerbForm {
    pub fn as_str(self) -> &'static str {
        match self {
            VerbForm::Vnoun => "Vnoun",
            VerbForm::Part => "Part",
            VerbForm::Conv => "Conv",
        }
    }
}

/// The non-finite verb form a reading ends in: its tag (akana's: `PastPart`, `AdvInce`,
/// `While`) and its kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NonFinite {
    pub tag: &'static str,
    pub form: VerbForm,
}

/// The kind of verb form a derivation makes from a verb, when it leads to `pos`; `None` for a
/// derivation that makes an ordinary word (okuy-ucu, gel-mez-lik, kovala-maca) or another
/// verb (gel-ebil, yaz-ıver).
pub fn verb_form_of(m: M, pos: PrimaryPos) -> Option<VerbForm> {
    match m {
        M::Inf1 | M::Inf2 | M::Inf3 => Some(VerbForm::Vnoun),
        M::PresPart | M::NarrPart | M::AorPart => Some(VerbForm::Part),
        M::PastPart | M::FutPart | M::FeelLike => match pos {
            PrimaryPos::Adj => Some(VerbForm::Part),
            _ => Some(VerbForm::Vnoun),
        },
        M::ByDoingSo
        | M::When
        | M::AfterDoingSo
        | M::While
        | M::SinceDoingSo
        | M::WithoutHavingDoneSo
        | M::WithoutBeingAbleToHaveDoneSo
        | M::AsLongAs
        | M::AsIf
        | M::Adamantly => Some(VerbForm::Conv),
        _ => None,
    }
}

/// Derivations that make a verb form an ordinary word again (okuyan-lık, bil-diğ-im-siz).
fn lexical(m: M) -> bool {
    matches!(
        m,
        M::Ness | M::Agt | M::With | M::Without | M::Dim | M::Related | M::JustLike
    )
}

/// The non-finite verb form a reading ends in, if it is one: the last derivation from a verb
/// that makes a verbal noun, participle or converb (gel-diğ-i, gel-ince, gel-ir-ken), unless an
/// ordinary derivation follows it (bil-in-mez-lik, okuy-an-lar-sız). -(y)ken after a noun or an
/// adjective (çocuk-ken, hasta-yken) is no verb form, nor is -cA on an adjective (büyük-çe).
pub fn verb_form(parse: &MorphParse) -> Option<NonFinite> {
    let engine = &*ENGINE;
    let graph = &engine.graph;
    let Some(root) = parse.path.root else {
        return verb_form_from_tags(parse);
    };
    // Whether a POS root starts a verb's group (not a noun's or an adjective's copula).
    let verbal = |state: StateId| {
        graph.states[state.0 as usize].morpheme.pos() == Some(PrimaryPos::Verb)
            && !engine.roots.nominal_verb.contains(state)
    };
    let mut verb = verbal(root);
    // The verb form found, waiting for the POS its derivation leads to.
    let mut found: Option<(M, Option<PrimaryPos>)> = None;
    for step in &parse.path.steps {
        let state = &graph.states[step.state.0 as usize];
        let m = state.morpheme;
        if state.pos_root {
            if let Some((_, pos @ None)) = &mut found {
                *pos = m.pos();
            }
            verb = verbal(step.state);
            continue;
        }
        if state.derivative && m != M::Zero {
            // A converb on the copula of a verb form is the word's form: gel-miş-ken,
            // yap-mak-ta-yken (review 3 MORPH3-08); on a noun's (çocuk-ken) it is none.
            let converb_on_form =
                found.is_some() && verb_form_of(m, PrimaryPos::Adv) == Some(VerbForm::Conv);
            if (verb || converb_on_form) && verb_form_of(m, PrimaryPos::Noun).is_some() {
                found = Some((m, None));
            } else if lexical(m) {
                found = None;
            }
            verb = false;
        }
    }
    let (m, pos) = found?;
    Some(NonFinite {
        tag: tag_of(m)?,
        form: verb_form_of(m, pos.unwrap_or(parse.pos))?,
    })
}

/// [`verb_form`] by the tags, for a reading without its path (one deserialized): the last
/// verb-form tag, unless a lexical derivation's follows it; -(y)ken only on a verb.
fn verb_form_from_tags(parse: &MorphParse) -> Option<NonFinite> {
    let tags = &parse.morpheme_tags;
    let morpheme = |tag: &str| {
        M::ALL
            .iter()
            .copied()
            .find(|&m| verb_form_of(m, PrimaryPos::Noun).is_some() && tag_of(m) == Some(tag))
    };
    let k = tags.iter().rposition(|t| morpheme(t).is_some())?;
    let lexical_tags: Vec<&str> = M::ALL
        .iter()
        .copied()
        .filter(|&m| lexical(m))
        .filter_map(tag_of)
        .collect();
    if tags[k + 1..]
        .iter()
        .any(|t| lexical_tags.contains(&t.as_str()))
    {
        return None;
    }
    let m = morpheme(&tags[k])?;
    let verbal = parse.primary_pos == PrimaryPos::Verb
        || tags[..k]
            .iter()
            .any(|t| matches!(t.as_str(), "BecomeVerb" | "Acquire"));
    if !verbal {
        return None;
    }
    Some(NonFinite {
        tag: tag_of(m)?,
        form: verb_form_of(m, parse.pos)?,
    })
}
