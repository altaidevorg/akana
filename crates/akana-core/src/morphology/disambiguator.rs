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
            let mut best_not_p2sg = false;

            for (p_idx, parse) in parses.iter().enumerate() {
                let mut score = parse.score;
                // Overt morphemes. Only bare dictionary matches spell out the default A3sg/Nom,
                // so those are not counted.
                let marked = parse
                    .morpheme_tags
                    .iter()
                    .filter(|t| !matches!(t.as_str(), "A3sg" | "Nom"))
                    .count();

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

                // 2. Proper Noun Capitalization Constraint. Only for a bare name or one whose
                // suffixes follow an apostrophe: "Annem" is anne-m, not the name Anne + P1sg.
                if is_first_char_upper
                    && parse.secondary_pos == super::pos::SecondaryPos::ProperNoun
                    && (marked == 1 || tokens[i].contains(['\'', '’']))
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
                    // An attributive adjective before a noun phrase head. Neither may be a
                    // predicate: after "yoğundu" the next word is a new clause (ki), and
                    // "geç kal-dı-m" is a verb, not an adjective + copular noun.
                    let is_predicate =
                        |p: &MorphParse| p.morpheme_tags.iter().any(|t| t.starts_with("Cop"));
                    if prev_pos == PrimaryPos::Adj
                        && parse.pos == PrimaryPos::Noun
                        && !is_predicate(&chosen[i - 1])
                        && !is_predicate(parse)
                        && !is_function_word(parses)
                    {
                        score += 1.5;
                    }
                    // A pronoun subject before an inflected verb (ben geldim); a bare verb stem
                    // is an imperative and gets no help ("ben de" is not "I, say!").
                    if prev_pos == PrimaryPos::Pron && parse.pos == PrimaryPos::Verb && marked > 1 {
                        score += 1.0;
                    }
                }

                // 5. Simplicity / Occam's razor preference (fewer derivation layers preferred if
                // tied), counting overt morphemes: otherwise "Ali" (Noun+A3sg+Nom) loses to
                // "Al-i" (Noun+Acc).
                score -= marked as f32 * 0.05;

                // 6. Function-word preference: an unsuffixed conjunction, question particle or
                // pronoun outranks an unsuffixed open-class homograph (de, ki, mi, ben), which
                // is far rarer in running text. The question particle keeps it with copula and
                // person (mi-yiz, mi-ydi-m), where its homograph is the note "mi".
                if parse.primary_pos == PrimaryPos::Q
                    || (marked == 1
                        && matches!(parse.primary_pos, PrimaryPos::Conj | PrimaryPos::Pron))
                {
                    score += 0.1;
                }

                // An exact tie goes away from the 2nd-person possessive, much rarer than the
                // 3rd in running text (durağ-ı-nda over durağ-ın-da, which used to win as the
                // alphabetically first analysis); other ties keep the analyzer's order.
                let not_p2sg = !parse.morpheme_tags.iter().any(|t| t == "P2sg");
                if score > best_score || (score == best_score && not_p2sg && !best_not_p2sg) {
                    best_score = score;
                    best_not_p2sg = not_p2sg;
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
        if is_function_word(parses) {
            return false;
        }
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

/// Whether a token is read as a bare conjunction, question particle or postposition (hem, ki,
/// mi, kadar). Such a word is not a noun phrase head even when a rare suffixed noun reading
/// exists (he-m "my h"), so it neither attracts a pre-nominal modifier nor gets the
/// noun-after-adjective bonus; "kalacak hem de" kept turning hem into a noun.
fn is_function_word(parses: &[MorphParse]) -> bool {
    parses.iter().any(|p| {
        p.morpheme_tags.len() == 1
            && matches!(
                p.primary_pos,
                PrimaryPos::Conj | PrimaryPos::Q | PrimaryPos::Postp
            )
    })
}
