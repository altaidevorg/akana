//! Two-level Syntactic Morphological Analyzer with Inflectional Groups (IG) and zero-derivation elimination.

use super::types::{SyntacticParse, InflectionalGroup};
use super::lexicon::{SyntacticLexicon, SyntacticPOS};
use crate::morphology::{TurkishMorphology, PrimaryPos};
use crate::phonology::to_turkish_lower;

pub struct TurkishSyntacticMorphology {
    lexicon: SyntacticLexicon,
    fallback_morphology: TurkishMorphology,
}

impl Default for TurkishSyntacticMorphology {
    fn default() -> Self {
        Self::new()
    }
}

impl TurkishSyntacticMorphology {
    pub fn new() -> Self {
        Self {
            lexicon: SyntacticLexicon::new(),
            fallback_morphology: TurkishMorphology::new(),
        }
    }

    /// Performs two-level syntactic morphological analysis on a Turkish word,
    /// returning hierarchical Inflectional Group (IG) structured parses.
    pub fn analyze(&self, word: &str) -> Vec<SyntacticParse> {
        let lower = to_turkish_lower(word);
        let mut results = Vec::new();

        // 1. Direct dictionary match (Roots without suffixes)
        if let Some(entries) = self.lexicon.lookup(&lower) {
            for entry in entries {
                let is_proper = entry.primary_pos == SyntacticPOS::ProperNoun;
                let mut ig = InflectionalGroup::new(entry.primary_pos.as_str());

                match entry.primary_pos {
                    SyntacticPOS::Noun => {
                        ig.set_feature("PersonNumber", "A3sg");
                        ig.set_feature("Possessive", "Pnon");
                        ig.set_feature("Case", "Nom");
                    }
                    SyntacticPOS::Verb => {
                        ig.set_feature("Polarity", "Pos");
                        ig.set_feature("PersonNumber", "A3sg");
                    }
                    SyntacticPOS::Adj | SyntacticPOS::Adv => {}
                    _ => {}
                }

                results.push(SyntacticParse::new(
                    word,
                    &entry.lemma,
                    entry.primary_pos.as_str(),
                    vec![ig],
                    is_proper,
                ));

                // Add cross-categorized zero-derivation-free parses (e.g. "güzel" -> [ADJ], [ADV], [NN])
                for &sec_pos in &entry.secondary_pos {
                    let mut sec_ig = InflectionalGroup::new(sec_pos.as_str());
                    if sec_pos == SyntacticPOS::Noun {
                        sec_ig.set_feature("PersonNumber", "A3sg");
                        sec_ig.set_feature("Possessive", "Pnon");
                        sec_ig.set_feature("Case", "Nom");
                    }
                    results.push(SyntacticParse::new(
                        word,
                        &entry.lemma,
                        sec_pos.as_str(),
                        vec![sec_ig],
                        false,
                    ));
                }
            }
        }

        // 2. Multi-tier Inflectional Group synthesis from underlying morphotactic engine
        let base_parses = self.fallback_morphology.analyze(word);
        for bp in base_parses {
            let root_lemma = bp.lemma.clone();
            let root_pos = bp.primary_pos;
            let root_pos_str = match root_pos {
                PrimaryPos::Verb => "VB",
                PrimaryPos::Adj => "ADJ",
                PrimaryPos::Adv => "ADV",
                PrimaryPos::Pron => "PRON",
                PrimaryPos::Num => "NUM",
                PrimaryPos::Conj => "CONJ",
                PrimaryPos::Postp => "POSTP",
                PrimaryPos::Interj => "INTERJ",
                _ => "NN",
            };

            let morphemes = bp.morpheme_tags;
            let mut groups = Vec::new();
            let mut current_ig = InflectionalGroup::new(root_pos_str);

            // Set default root features
            if root_pos == PrimaryPos::Verb {
                current_ig.set_feature("Polarity", "Pos");
            } else if root_pos == PrimaryPos::Noun {
                current_ig.set_feature("PersonNumber", "A3sg");
                current_ig.set_feature("Possessive", "Pnon");
                current_ig.set_feature("Case", "Nom");
            }

            for m in &morphemes {
                match m.as_str() {
                    // Derivational Boundaries (Creating new Inflectional Groups)
                    "PastNom" | "PastPart" => {
                        groups.push(current_ig);
                        current_ig = InflectionalGroup::with_derivation("NOMP", "PastNom");
                        current_ig.set_feature("PersonNumber", "A3sg");
                        current_ig.set_feature("Possessive", "Pnon");
                        current_ig.set_feature("Case", "Nom");
                    }
                    "PresPart" => {
                        groups.push(current_ig);
                        current_ig = InflectionalGroup::with_derivation("ADJP", "PresPart");
                    }
                    "FutPart" => {
                        groups.push(current_ig);
                        current_ig = InflectionalGroup::with_derivation("ADJP", "FutPart");
                    }
                    "NarrPart" => {
                        groups.push(current_ig);
                        current_ig = InflectionalGroup::with_derivation("ADJP", "NarrPart");
                    }
                    "Inf" => {
                        groups.push(current_ig);
                        current_ig = InflectionalGroup::with_derivation("NOMP", "Inf");
                        current_ig.set_feature("PersonNumber", "A3sg");
                        current_ig.set_feature("Possessive", "Pnon");
                        current_ig.set_feature("Case", "Nom");
                    }
                    "Agt" => {
                        groups.push(current_ig);
                        current_ig = InflectionalGroup::with_derivation("NN", "Agt");
                        current_ig.set_feature("PersonNumber", "A3sg");
                        current_ig.set_feature("Possessive", "Pnon");
                        current_ig.set_feature("Case", "Nom");
                    }
                    "Dim" => {
                        groups.push(current_ig);
                        current_ig = InflectionalGroup::with_derivation("NN", "Dim");
                    }
                    "With" => {
                        groups.push(current_ig);
                        current_ig = InflectionalGroup::with_derivation("ADJP", "With");
                    }
                    "Without" => {
                        groups.push(current_ig);
                        current_ig = InflectionalGroup::with_derivation("ADJP", "Without");
                    }
                    "Become" => {
                        groups.push(current_ig);
                        current_ig = InflectionalGroup::with_derivation("VB", "Become");
                        current_ig.set_feature("Polarity", "Pos");
                    }
                    "Caus" => {
                        current_ig.set_feature("Voice", "Caus");
                    }
                    "Pass" => {
                        current_ig.set_feature("Voice", "Pass");
                    }
                    "Neg" => {
                        current_ig.set_feature("Polarity", "Neg");
                    }

                    // Nominal Inflections
                    "A1sg" => { current_ig.set_feature("PersonNumber", "A1sg"); }
                    "A2sg" => { current_ig.set_feature("PersonNumber", "A2sg"); }
                    "A3sg" => { current_ig.set_feature("PersonNumber", "A3sg"); }
                    "A1pl" => { current_ig.set_feature("PersonNumber", "A1pl"); }
                    "A2pl" => { current_ig.set_feature("PersonNumber", "A2pl"); }
                    "A3pl" => { current_ig.set_feature("PersonNumber", "A3pl"); }

                    "Pnon" => { current_ig.set_feature("Possessive", "Pnon"); }
                    "P1sg" => { current_ig.set_feature("Possessive", "P1sg"); }
                    "P2sg" => { current_ig.set_feature("Possessive", "P2sg"); }
                    "P3sg" => { current_ig.set_feature("Possessive", "P3sg"); }
                    "P1pl" => { current_ig.set_feature("Possessive", "P1pl"); }
                    "P2pl" => { current_ig.set_feature("Possessive", "P2pl"); }
                    "P3pl" => { current_ig.set_feature("Possessive", "P3pl"); }

                    "Nom" => { current_ig.set_feature("Case", "Nom"); }
                    "Acc" => { current_ig.set_feature("Case", "Acc"); }
                    "Dat" => { current_ig.set_feature("Case", "Dat"); }
                    "Loc" => { current_ig.set_feature("Case", "Loc"); }
                    "Abl" => { current_ig.set_feature("Case", "Abl"); }
                    "Gen" => { current_ig.set_feature("Case", "Gen"); }
                    "Ins" => { current_ig.set_feature("Case", "Ins"); }

                    // Verbal Inflections
                    "Past" => { current_ig.set_feature("Tense", "Past"); }
                    "Narr" => { current_ig.set_feature("Tense", "Narr"); }
                    "Fut" => { current_ig.set_feature("Tense", "Fut"); }
                    "Prog1" | "Prog2" => { current_ig.set_feature("Aspect", "Prog"); }
                    "Aor" => { current_ig.set_feature("Tense", "Aor"); }
                    "Des" => { current_ig.set_feature("Mood", "Des"); }
                    "Nec" => { current_ig.set_feature("Mood", "Nec"); }
                    "Cop" => { current_ig.set_feature("Copula", "PresCop"); }
                    _ => {}
                }
            }

            groups.push(current_ig);

            let is_proper = bp.secondary_pos == crate::morphology::SecondaryPos::ProperNoun;
            let parse = SyntacticParse::new(
                word,
                &root_lemma,
                root_pos_str,
                groups,
                is_proper,
            );

            results.push(parse);
        }

        // Deduplicate parses by formatted representation
        results.sort_by(|a, b| a.formatted.cmp(&b.formatted));
        results.dedup_by(|a, b| a.formatted == b.formatted);

        results
    }
}
