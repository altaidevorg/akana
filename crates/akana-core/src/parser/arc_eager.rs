//! Turkish Transition-based / Universal Dependencies Parser.

use super::tree::{DependencyNode, DependencyTree};
use crate::morphology::pos::PrimaryPos;
use crate::morphology::{MorphologicalDisambiguator, TurkishMorphology};

pub struct TurkishDependencyParser {
    disambiguator: MorphologicalDisambiguator,
}

impl Default for TurkishDependencyParser {
    fn default() -> Self {
        Self::new()
    }
}

impl TurkishDependencyParser {
    pub fn new() -> Self {
        Self {
            disambiguator: MorphologicalDisambiguator::new(),
        }
    }

    pub fn with_morphology(morphology: TurkishMorphology) -> Self {
        Self {
            disambiguator: MorphologicalDisambiguator::with_morphology(morphology),
        }
    }

    /// Parses a sequence of tokens in a Turkish sentence into a full `DependencyTree`.
    pub fn parse(&self, tokens: &[&str]) -> DependencyTree {
        if tokens.is_empty() {
            return DependencyTree::new(Vec::new());
        }

        let disambiguated = self.disambiguator.disambiguate(tokens);
        let n = tokens.len();

        // 1. Locate the primary root of the sentence (typically the sentence-final verb or copular predicate)
        let mut root_idx = n - 1;
        for (i, p) in disambiguated.iter().enumerate().rev() {
            if p.primary_pos == PrimaryPos::Verb || p.morpheme_tags.iter().any(|t| t.starts_with("Cop")) {
                root_idx = i;
                break;
            }
        }

        let mut nodes = Vec::with_capacity(n);

        for (i, parse) in disambiguated.iter().enumerate() {
            let id = i + 1;
            let form = parse.surface.clone();
            let lemma = parse.lemma.clone();

            let upos = match parse.primary_pos {
                PrimaryPos::Noun => if parse.secondary_pos == crate::morphology::pos::SecondaryPos::ProperNoun { "PROPN" } else { "NOUN" },
                PrimaryPos::Verb => "VERB",
                PrimaryPos::Adj => "ADJ",
                PrimaryPos::Adv => "ADV",
                PrimaryPos::Pron => "PRON",
                PrimaryPos::Num => "NUM",
                PrimaryPos::Conj => "CCONJ",
                PrimaryPos::Postp => "ADP",
                PrimaryPos::Interj => "INTJ",
                PrimaryPos::Punc => "PUNCT",
                _ => "X",
            }.to_string();

            let xpos = parse.primary_pos.as_str().to_string();
            let feats = parse.morpheme_tags.join("|");

            let mut head = root_idx + 1;
            let mut deprel = "dep".to_string();

            if i == root_idx {
                head = 0;
                deprel = "root".to_string();
            } else if parse.primary_pos == PrimaryPos::Punc {
                head = root_idx + 1;
                deprel = "punct".to_string();
            } else if parse.primary_pos == PrimaryPos::Adj {
                // If followed by a Noun, it modifies that noun (amod)
                if i + 1 < n && disambiguated[i + 1].primary_pos == PrimaryPos::Noun {
                    head = i + 2;
                    deprel = "amod".to_string();
                } else {
                    head = root_idx + 1;
                    deprel = "amod".to_string();
                }
            } else if parse.primary_pos == PrimaryPos::Adv {
                head = root_idx + 1;
                deprel = "advmod".to_string();
            } else if parse.primary_pos == PrimaryPos::Conj {
                if i + 1 < n {
                    head = i + 2;
                    deprel = "cc".to_string();
                } else {
                    head = root_idx + 1;
                    deprel = "cc".to_string();
                }
            } else if parse.primary_pos == PrimaryPos::Noun || parse.primary_pos == PrimaryPos::Pron {
                // Check case features
                if parse.morpheme_tags.contains(&"Acc".to_string()) {
                    head = root_idx + 1;
                    deprel = "obj".to_string();
                } else if parse.morpheme_tags.contains(&"Dat".to_string())
                    || parse.morpheme_tags.contains(&"Loc".to_string())
                    || parse.morpheme_tags.contains(&"Abl".to_string())
                    || parse.morpheme_tags.contains(&"Ins".to_string()) {
                    head = root_idx + 1;
                    deprel = "obl".to_string();
                } else if parse.morpheme_tags.contains(&"Gen".to_string()) {
                    // Genitive modifier connects to following noun (nmod:poss)
                    if i + 1 < n && disambiguated[i + 1].primary_pos == PrimaryPos::Noun {
                        head = i + 2;
                        deprel = "nmod:poss".to_string();
                    } else {
                        head = root_idx + 1;
                        deprel = "nmod".to_string();
                    }
                } else {
                    // Nominative subject or object
                    if i == 0 || (i < root_idx && i <= 1) {
                        head = root_idx + 1;
                        deprel = "nsubj".to_string();
                    } else {
                        head = root_idx + 1;
                        deprel = "obj".to_string();
                    }
                }
            }

            nodes.push(DependencyNode {
                id,
                form,
                lemma,
                upos,
                xpos,
                feats,
                head,
                deprel,
            });
        }

        DependencyTree::new(nodes)
    }
}
