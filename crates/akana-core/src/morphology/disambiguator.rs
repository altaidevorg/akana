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
                    pos: PrimaryPos::Unknown,
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

            let is_sentence_final =
                i == n - 1 || (i == n - 2 && matches!(tokens[n - 1], "." | "!" | "?" | "…"));
            let is_first_char_upper = tokens[i].chars().next().is_some_and(|c| c.is_uppercase());

            let mut best_idx = 0;
            let mut best_score = f32::MIN;

            for (p_idx, parse) in parses.iter().enumerate() {
                let mut score = parse.score;

                // The rules below look at the word's POS after derivation (`pos`): a converb
                // (baş-la-yınca) is not a noun, a derived verb (an-laş-ıyor-um) is a verb, a
                // nominalised verb (gel-diğ-i) is not a finite predicate.

                // 1. SOV Final Predicate Constraint
                if is_sentence_final {
                    if parse.pos == PrimaryPos::Verb {
                        score += 3.5;
                    } else if parse.morpheme_tags.iter().any(|t| t.starts_with("Cop")) {
                        score += 2.5;
                    }
                }

                // 2. Proper Noun Capitalization Constraint
                if is_first_char_upper
                    && parse.secondary_pos == super::pos::SecondaryPos::ProperNoun
                {
                    score += 5.0;
                }

                // 3. Pre-nominal modifier constraint (Adj / Det before Noun). Determiners (bu, o,
                // bir, her) also have Adj and Pron entries; before a noun the Det reading wins.
                if i + 1 < n && noun_ahead(&analyses_per_token[i + 1..]) {
                    match parse.pos {
                        PrimaryPos::Adj => score += 2.0,
                        PrimaryPos::Det => score += 2.1,
                        _ => {}
                    }
                }

                // 4. Post-nominal agreement constraint
                if i > 0 && !chosen.is_empty() {
                    let prev_pos = chosen[i - 1].pos;
                    if prev_pos == PrimaryPos::Adj && parse.pos == PrimaryPos::Noun {
                        score += 1.5;
                    }
                    if prev_pos == PrimaryPos::Pron && parse.pos == PrimaryPos::Verb {
                        score += 1.0;
                    }
                }

                // 5. Simplicity / Occam's razor preference (fewer derivation layers preferred if tied).
                // Only bare dictionary matches spell out the default A3sg/Nom, so those are not
                // counted: otherwise "Ali" (Noun+A3sg+Nom) loses to "Al-i" (Noun+Acc).
                let marked = parse
                    .morpheme_tags
                    .iter()
                    .filter(|t| !matches!(t.as_str(), "A3sg" | "Nom"))
                    .count();
                score -= marked as f32 * 0.05;

                // 6. Function-word preference: an unsuffixed conjunction, question particle or
                // pronoun outranks an unsuffixed open-class homograph (de, ki, mi, ben), which
                // is far rarer in running text.
                if marked == 1
                    && matches!(
                        parse.primary_pos,
                        PrimaryPos::Conj | PrimaryPos::Q | PrimaryPos::Pron
                    )
                {
                    score += 0.1;
                }

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

/// Whether the following tokens are a noun phrase head, possibly after a run of adjectives
/// ("bu [güzel] kitap"). A token counts as a noun when it has a common-noun reading and no
/// adjective reading; one that could be either continues the run. Ending the run without a
/// noun ("Bu doğru") means the word before is not a pre-nominal modifier.
fn noun_ahead(following: &[Vec<MorphParse>]) -> bool {
    for parses in following {
        let noun = parses.iter().any(|p| {
            p.pos == PrimaryPos::Noun && p.secondary_pos != super::pos::SecondaryPos::ProperNoun
        });
        let adj = parses.iter().any(|p| p.pos == PrimaryPos::Adj);
        if noun && !adj {
            return true;
        }
        if !adj {
            return false;
        }
    }
    false
}
