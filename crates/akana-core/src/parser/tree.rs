//! Syntactic dependency tree and CoNLL-U format data structures.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DependencyNode {
    /// 1-based index of the token
    pub id: usize,
    /// Word form or punctuation symbol
    pub form: String,
    /// Lemma or stem
    pub lemma: String,
    /// Universal dependency POS tag (e.g. NOUN, VERB, ADJ, ADV, PRON, DET, ADP, CCONJ, PUNCT, etc.)
    pub upos: String,
    /// Language specific POS tag or morphological features
    pub xpos: String,
    /// Morphological features (e.g. Case=Nom|Number=Sing|Person=3)
    pub feats: String,
    /// Head of the current word, which is either a value of ID or zero (0 for root)
    pub head: usize,
    /// Universal dependency relation to the HEAD (e.g. root, nsubj, obj, obl, amod, advmod, punct)
    pub deprel: String,
}

impl DependencyNode {
    pub fn to_conllu_line(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t_\t_",
            self.id, self.form, self.lemma, self.upos, self.xpos, self.feats, self.head, self.deprel
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DependencyTree {
    pub nodes: Vec<DependencyNode>,
}

impl DependencyTree {
    pub fn new(nodes: Vec<DependencyNode>) -> Self {
        Self { nodes }
    }

    pub fn to_conllu(&self) -> String {
        let mut lines = Vec::new();
        for node in &self.nodes {
            lines.push(node.to_conllu_line());
        }
        lines.join("\n")
    }
}
