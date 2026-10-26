//! Types and structures for Google-style Syntactic Expressive Morphology (FSMNLP 2019).
//! Represents hierarchical Inflectional Groups (IG) and Universal Dependencies (UD) features.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Write;

/// An Inflectional Group (IG): the root, or a derivation, with the inflections that follow it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InflectionalGroup {
    /// Part of speech of the group: NN, VB, ADJ, ADV, PRON, NUM, DET, CONJ, POSTP, INTERJ,
    /// QUES, DUP, PUNC; a noun derived from a verb is NOMP and a derived adjective ADJP.
    pub pos: String,
    /// The derivation that opened the group: Zemberek's morpheme id (Caus, Pass, Able, Inf1,
    /// PresPart, While, Ness, With, Rel, ...), PastNom and FutNom for the -DIk and -(y)AcAk
    /// nominalisations, Cop for a nominal predicate (evde-yim). `None` for the root's group.
    pub derivation: Option<String>,
    /// UD features of the group, as UD Turkish writes them (Case=Loc, Number=Plur, Person=3,
    /// Number[psor]=Sing, Person[psor]=1, Tense=Past, Aspect=Perf, Evident=Nfh, Mood=GenPot,
    /// Polarity=Neg, Voice=Cau, VerbForm=Part).
    pub features: BTreeMap<String, String>,
}

impl InflectionalGroup {
    pub fn new(pos: impl Into<String>) -> Self {
        Self {
            pos: pos.into(),
            derivation: None,
            features: BTreeMap::new(),
        }
    }

    pub fn with_derivation(pos: impl Into<String>, derivation: impl Into<String>) -> Self {
        Self {
            pos: pos.into(),
            derivation: Some(derivation.into()),
            features: BTreeMap::new(),
        }
    }

    pub fn set_feature(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.features.insert(key.into(), value.into());
    }

    /// Formats the inflectional group in Google FSMNLP human-readable format.
    pub fn format_ig(&self, is_root: bool, root_lemma: Option<&str>) -> String {
        let mut out = String::new();
        out.push('(');

        // Writing to a String cannot fail.
        if is_root {
            let lemma = root_lemma.unwrap_or("?");
            let _ = write!(out, "{}[{}]", lemma, self.pos);
        } else {
            let deriv_name = self.derivation.as_deref().unwrap_or("Deriv");
            let _ = write!(out, "[{}]-{}", self.pos, deriv_name);
        }

        for (k, v) in &self.features {
            let _ = write!(out, "+[{k}={v}]");
        }

        out.push(')');
        out
    }
}

/// Full syntactic morphological parse of a surface word.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyntacticParse {
    pub surface: String,
    pub root: String,
    pub root_pos: String,
    pub inflectional_groups: Vec<InflectionalGroup>,
    pub is_proper: bool,
    pub formatted: String,
}

impl SyntacticParse {
    pub fn new(
        surface: impl Into<String>,
        root: impl Into<String>,
        root_pos: impl Into<String>,
        inflectional_groups: Vec<InflectionalGroup>,
        is_proper: bool,
    ) -> Self {
        let surface = surface.into();
        let root = root.into();
        let root_pos = root_pos.into();

        let mut formatted = String::new();
        for (i, ig) in inflectional_groups.iter().enumerate() {
            if i == 0 {
                formatted.push_str(&ig.format_ig(true, Some(&root)));
            } else {
                formatted.push_str(&ig.format_ig(false, None));
            }
        }
        formatted.push_str(if is_proper {
            "+[Proper=True]"
        } else {
            "+[Proper=False]"
        });

        Self {
            surface,
            root,
            root_pos,
            inflectional_groups,
            is_proper,
            formatted,
        }
    }
}
