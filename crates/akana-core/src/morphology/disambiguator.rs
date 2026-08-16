//! Contextual Morphological Disambiguator for Turkish sentences.

use super::analyzer::{MorphParse, TurkishMorphology};
use super::pos::PrimaryPos;

pub struct MorphologicalDisambiguator {
    morphology: TurkishMorphology,
}

impl Default for MorphologicalDisambiguator {
    fn default() -> Self {
        Self::new()
    }
}

impl MorphologicalDisambiguator {
    pub fn new() -> Self {
        Self {
            morphology: TurkishMorphology::new(),
        }
    }

    pub fn with_morphology(morphology: TurkishMorphology) -> Self {
        Self { morphology }
    }

    /// Disambiguates a sequence of Turkish tokens in sentence context and returns the best `MorphParse` for each token.
    pub fn disambiguate(&self, tokens: &[&str]) -> Vec<MorphParse> {
        let mut analyses_per_token: Vec<Vec<MorphParse>> = Vec::new();
        for &tok in tokens {
            let parses = self.morphology.analyze(tok);
            analyses_per_token.push(parses);
        }

        let n = tokens.len();
        let mut chosen: Vec<MorphParse> = Vec::with_capacity(n);

        for i in 0..n {
            let parses = &analyses_per_token[i];
            if parses.is_empty() {
                // Fallback unknown parse
                chosen.push(MorphParse {
                    surface: tokens[i].to_string(),
                    lemma: tokens[i].to_string(),
                    root: tokens[i].to_string(),
                    primary_pos: PrimaryPos::Unknown,
                    secondary_pos: super::pos::SecondaryPos::None,
                    morpheme_tags: vec!["Unk".to_string()],
                    formatted: format!("[{}:Unk]", tokens[i]),
                    score: 0.0,
                });
                continue;
            }

            if parses.len() == 1 {
                chosen.push(parses[0].clone());
                continue;
            }

            let is_sentence_final = i == n - 1 || (i == n - 2 && matches!(tokens[n - 1], "." | "!" | "?" | "…"));
            let is_first_char_upper = tokens[i].chars().next().map_or(false, |c| c.is_uppercase());

            let mut best_idx = 0;
            let mut best_score = f32::MIN;

            for (p_idx, parse) in parses.iter().enumerate() {
                let mut score = parse.score;

                // 1. SOV Final Predicate Constraint
                if is_sentence_final {
                    if parse.primary_pos == PrimaryPos::Verb {
                        score += 3.5;
                    } else if parse.morpheme_tags.iter().any(|t| t.starts_with("Cop")) {
                        score += 2.5;
                    }
                }

                // 2. Proper Noun Capitalization Constraint
                if is_first_char_upper && parse.secondary_pos == super::pos::SecondaryPos::ProperNoun {
                    score += 5.0;
                }

                // 3. Pre-nominal modifier constraint (Adj / Demons before Noun)
                if i + 1 < n {
                    let next_parses = &analyses_per_token[i + 1];
                    let next_likely_noun = next_parses.iter().any(|np| np.primary_pos == PrimaryPos::Noun && np.secondary_pos != super::pos::SecondaryPos::ProperNoun);
                    if next_likely_noun && (parse.primary_pos == PrimaryPos::Adj || parse.secondary_pos == super::pos::SecondaryPos::Demonstrative) {
                        score += 2.0;
                    }
                }

                // 4. Post-nominal agreement constraint
                if i > 0 && !chosen.is_empty() {
                    let prev_pos = chosen[i - 1].primary_pos;
                    if prev_pos == PrimaryPos::Adj && parse.primary_pos == PrimaryPos::Noun {
                        score += 1.5;
                    }
                    if prev_pos == PrimaryPos::Pron && parse.primary_pos == PrimaryPos::Verb {
                        score += 1.0;
                    }
                }

                // 5. Simplicity / Occam's razor preference (fewer derivation layers preferred if tied)
                score -= parse.morpheme_tags.len() as f32 * 0.05;

                if score > best_score {
                    best_score = score;
                    best_idx = p_idx;
                }
            }

            chosen.push(parses[best_idx].clone());
        }

        chosen
    }
}
