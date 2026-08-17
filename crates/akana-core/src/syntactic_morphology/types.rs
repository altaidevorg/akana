//! Types and structures for Google-style Syntactic Expressive Morphology (FSMNLP 2019).
//! Represents hierarchical Inflectional Groups (IG) and Universal Dependencies (UD) features.

use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};

/// An Inflectional Group (IG) represents an overt derivational span or root tier in Turkish morphology.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InflectionalGroup {
    /// Part of Speech for this inflectional tier (e.g. "VB", "NOMP", "NN", "ADJP", "ADV")
    pub pos: String,
    /// Overt Derivation tag if this IG was created by a derivational morpheme (e.g. "PastNom", "Agt", "With", "Without", "Caus", "Pass", "Dim")
    pub derivation: Option<String>,
    /// Feature-value dictionary for this group (e.g. {"PersonNumber": "A3sg", "Possessive": "P1sg", "Case": "Loc", "Polarity": "Pos"})
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

        if is_root {
            let lemma = root_lemma.unwrap_or("?");
            out.push_str(&format!("{}[{}]", lemma, self.pos));
        } else {
            let deriv_name = self.derivation.as_deref().unwrap_or("Deriv");
            out.push_str(&format!("[{}]-{}", self.pos, deriv_name));
        }

        for (k, v) in &self.features {
            out.push_str(&format!("+[{k}={v}]"));
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
        formatted.push_str(&format!("+[Proper={}]", if is_proper { "True" } else { "False" }));

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
