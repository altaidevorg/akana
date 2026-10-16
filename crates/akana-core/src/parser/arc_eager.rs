//! Turkish Transition-based / Universal Dependencies Parser.

use super::tree::{DependencyNode, DependencyTree};
use crate::morphology::pos::PrimaryPos;
use crate::morphology::{MorphParse, MorphologicalDisambiguator, TurkishMorphology};

/// Suffixes that turn a verb into a non-finite verb form heading a subordinate clause:
/// infinitives / verbal nouns, participles and converbs.
const VERB_FORM_TAGS: &[&str] = &[
    "Inf",
    "ActN",
    "Manner",
    "PresPart",
    "PastPart",
    "FutPart",
    "AdvErek",
    "AdvInce",
    "AdvIp",
    "While",
    "SinceDoingSo",
    "WithoutHavingDoneSo",
];

/// Lexical derivations: a verb form followed by one of these is an ordinary word again.
const LEXICAL_TAGS: &[&str] = &[
    "Ness", "Agt", "With", "Without", "Dim", "DevN", "DevGi", "DevGin", "DevGen",
];

/// The non-finite verb-form suffix of a parse, if it is a verb form (UD `VerbForm`).
fn verb_form(parse: &MorphParse) -> Option<&str> {
    let tags = &parse.morpheme_tags;
    let k = tags
        .iter()
        .rposition(|t| VERB_FORM_TAGS.contains(&t.as_str()))?;
    if tags[k + 1..]
        .iter()
        .any(|t| LEXICAL_TAGS.contains(&t.as_str()))
    {
        return None;
    }
    // -(y)ken also follows nominals (çocuk-ken), which are not verb forms.
    let verbal = parse.primary_pos == PrimaryPos::Verb
        || tags[..k]
            .iter()
            .any(|t| matches!(t.as_str(), "NounToVerb" | "BecomeVerb" | "CausVerb"));
    if tags[k] == "While" && !verbal {
        return None;
    }
    Some(tags[k].as_str())
}

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
    ///
    /// Categories come from each word's POS after derivation (`MorphParse::pos`), so a
    /// converb or participle is not a noun and a noun-derived verb (baş-la-dı) is a verb.
    pub fn parse(&self, tokens: &[&str]) -> DependencyTree {
        if tokens.is_empty() {
            return DependencyTree::new(Vec::new());
        }

        let disambiguated = self.disambiguator.disambiguate(tokens);
        let n = tokens.len();

        // 1. Locate the primary root of the sentence (typically the sentence-final verb or copular predicate)
        let mut root_idx = n - 1;
        for (i, p) in disambiguated.iter().enumerate().rev() {
            // The question particle carries copula and person (mi-yiz) but is not the predicate.
            if p.pos == PrimaryPos::Q {
                continue;
            }
            if p.pos == PrimaryPos::Verb || p.morpheme_tags.iter().any(|t| t.starts_with("Cop")) {
                root_idx = i;
                break;
            }
        }

        let mut nodes = Vec::with_capacity(n);

        for (i, parse) in disambiguated.iter().enumerate() {
            let id = i + 1;
            let form = parse.surface.clone();
            let lemma = parse.lemma.clone();
            let form_of_verb = verb_form(parse);
            let next_is_noun = i + 1 < n && disambiguated[i + 1].pos == PrimaryPos::Noun;

            // UD: infinitives, participles and converbs are VERB (with a VerbForm).
            let upos = match parse.pos {
                _ if form_of_verb.is_some() => "VERB",
                PrimaryPos::Noun => {
                    if parse.secondary_pos == crate::morphology::pos::SecondaryPos::ProperNoun {
                        "PROPN"
                    } else {
                        "NOUN"
                    }
                }
                PrimaryPos::Verb => "VERB",
                PrimaryPos::Adj => "ADJ",
                PrimaryPos::Adv => "ADV",
                PrimaryPos::Pron => "PRON",
                PrimaryPos::Num => "NUM",
                PrimaryPos::Conj => "CCONJ",
                PrimaryPos::Postp => "ADP",
                PrimaryPos::Interj => "INTJ",
                PrimaryPos::Punc => "PUNCT",
                PrimaryPos::Det => "DET",
                PrimaryPos::Q => "AUX",
                _ => "X",
            }
            .to_string();

            let xpos = parse.primary_pos.as_str().to_string();
            let feats = parse.morpheme_tags.join("|");

            let mut head = root_idx + 1;
            let mut deprel = "dep".to_string();

            if i == root_idx {
                head = 0;
                deprel = "root".to_string();
            } else if parse.pos == PrimaryPos::Punc {
                head = root_idx + 1;
                deprel = "punct".to_string();
            } else if parse.pos == PrimaryPos::Q {
                // The question particle attaches to the word it follows, which is the focus of
                // the question (Ayşe mi okula geldi?); UD Turkish labels every use aux:q.
                if i > 0 {
                    head = i;
                }
                deprel = "aux:q".to_string();
            } else if matches!(
                form_of_verb,
                Some(
                    "AdvErek"
                        | "AdvInce"
                        | "AdvIp"
                        | "While"
                        | "SinceDoingSo"
                        | "WithoutHavingDoneSo"
                )
            ) {
                // A converb heads an adverbial clause of the predicate (advcl)
                deprel = "advcl".to_string();
            } else if matches!(form_of_verb, Some("PresPart" | "PastPart" | "FutPart"))
                && parse.pos == PrimaryPos::Adj
                && next_is_noun
            {
                // A participle before a noun heads a relative clause on it (acl)
                head = i + 2;
                deprel = "acl".to_string();
            } else if parse.pos == PrimaryPos::Adj {
                // If followed by a Noun, it modifies that noun (amod)
                if next_is_noun {
                    head = i + 2;
                }
                deprel = "amod".to_string();
            } else if parse.pos == PrimaryPos::Det {
                // A determiner modifies the following noun (det)
                if next_is_noun {
                    head = i + 2;
                }
                deprel = "det".to_string();
            } else if parse.pos == PrimaryPos::Adv {
                head = root_idx + 1;
                deprel = "advmod".to_string();
            } else if parse.pos == PrimaryPos::Conj {
                if i + 1 < n {
                    head = i + 2;
                    deprel = "cc".to_string();
                } else {
                    head = root_idx + 1;
                    deprel = "cc".to_string();
                }
            } else if parse.pos == PrimaryPos::Noun || parse.pos == PrimaryPos::Pron {
                // Check case features
                if parse.morpheme_tags.contains(&"Acc".to_string()) {
                    head = root_idx + 1;
                    deprel = "obj".to_string();
                } else if parse.morpheme_tags.contains(&"Dat".to_string())
                    || parse.morpheme_tags.contains(&"Loc".to_string())
                    || parse.morpheme_tags.contains(&"Abl".to_string())
                    || parse.morpheme_tags.contains(&"Ins".to_string())
                {
                    head = root_idx + 1;
                    deprel = "obl".to_string();
                } else if parse.morpheme_tags.contains(&"Gen".to_string()) {
                    // Genitive modifier connects to following noun (nmod:poss)
                    if next_is_noun {
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
