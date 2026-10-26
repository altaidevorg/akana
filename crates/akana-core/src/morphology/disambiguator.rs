//! Contextual Morphological Disambiguator for Turkish sentences.
//!
//! Each word's analyses come ranked by the analyzer's corpus prior ([`MorphParse::score`]). The
//! disambiguator picks one per word, left to right, by the log of that score plus bonuses for
//! what the neighbouring words say: a genitive wants a possessed noun after it, a postposition
//! the case it governs before it, a light verb (et-, ol-) a bare noun before it, and so on. The
//! bonuses are bounded, so context can overturn a reading the corpus finds a dozen times rarer,
//! not one it finds a hundred times rarer.

use super::analyzer::{MorphParse, TurkishMorphology};
use super::pos::{PrimaryPos, SecondaryPos};

/// The most context can add to one reading's log score (a factor of about 12).
const MAX_BONUS: f32 = 2.5;
/// The lowest score a reading counts with, so that one the prior rules out stays comparable.
const MIN_SCORE: f32 = 1e-4;
/// How much the analyzer's log score counts. EM over a word list makes the prior too sure of
/// itself (kamu is an adjective 250 times more often than a noun), and halving its log lets
/// context decide such cases ("kamu yararı").
const PRIOR_WEIGHT: f32 = 0.5;

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
        let analyses_per_token: Vec<Vec<MorphParse>> = tokens
            .iter()
            .map(|tok| self.morphology.analyze(tok))
            .collect();

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
                    secondary_pos: SecondaryPos::None,
                    morpheme_tags: vec!["Unk".to_string()],
                    formatted: format!("[{}:Unk]", tokens[i]),
                    analysis: format!("[{0}:Unk] {0}:Unk", tokens[i]),
                    score: 0.0,
                    path: Default::default(),
                });
                continue;
            }

            if parses.len() == 1 {
                chosen.push(parses[0].clone());
                continue;
            }

            let is_sentence_final =
                i == n - 1 || (i == n - 2 && matches!(tokens[n - 1], "." | "!" | "?" | "…"));
            // The first word of a sentence or of a quotation is capitalised whatever it is.
            let is_sentence_start = i == 0 || is_opening(tokens[i - 1]);
            let is_first_char_upper = tokens[i].chars().next().is_some_and(|c| c.is_uppercase());
            let previous = if i > 0 { chosen.last() } else { None };
            let following = &analyses_per_token[i + 1..];
            let next_parses = following.first().map(Vec::as_slice).unwrap_or_default();
            let next_is_word = i + 1 < n && !is_punctuation(tokens[i + 1]);
            let modifies_noun = noun_ahead(following);
            let before_light_verb = next_parses.iter().any(|p| {
                p.primary_pos == PrimaryPos::Verb && matches!(p.root.as_str(), "et" | "ol" | "eyle")
            });
            let before_possessed = next_parses.iter().any(|p| {
                p.pos == PrimaryPos::Noun
                    && p.primary_pos == PrimaryPos::Noun
                    && p.morpheme_tags
                        .get(1)
                        .is_some_and(|t| matches!(t.as_str(), "P3sg" | "P3pl" | "A3pl"))
                    && p.morpheme_tags.iter().any(|t| t == "P3sg" || t == "P3pl")
            });
            let has_common_reading = parses
                .iter()
                .any(|p| p.secondary_pos != SecondaryPos::ProperNoun);

            let mut best_idx = 0;
            let mut best_score = f32::MIN;
            for (p_idx, parse) in parses.iter().enumerate() {
                // Overt morphemes. Only bare dictionary matches spell out the default A3sg/Nom,
                // so those are not counted.
                let marked = parse
                    .morpheme_tags
                    .iter()
                    .filter(|t| !matches!(t.as_str(), "A3sg" | "Nom"))
                    .count();
                let bare_noun = parse.primary_pos == PrimaryPos::Noun
                    && parse.secondary_pos != SecondaryPos::ProperNoun
                    && marked == 1;
                let mut bonus = 0.0f32;

                // The rules below look at the word's POS after derivation (`pos`): a converb
                // (baş-la-yınca) is not a noun, a derived verb (an-laş-ıyor-um) is a verb, a
                // nominalised verb (gel-diğ-i) is not a finite predicate.

                // 1. The predicate comes last (SOV).
                if is_sentence_final {
                    if parse.pos == PrimaryPos::Verb {
                        bonus += 2.0;
                    } else if is_predicate(parse) {
                        bonus += 1.0;
                    }
                }

                // 2. A capitalised word inside a sentence is a name. Only for a bare name or
                // one whose suffixes follow an apostrophe: "Annem" is anne-m, not the name
                // Anne + P1sg. At the start of a sentence capitals say nothing ("Yani").
                if is_first_char_upper
                    && parse.secondary_pos == SecondaryPos::ProperNoun
                    && (marked == 1 || tokens[i].contains(['\'', '’']))
                    && (!is_sentence_start || !has_common_reading)
                {
                    bonus += 1.5;
                }

                // 3. A modifier comes before its noun (Adj, Det, Num: "bu kitap", "yüz kişi").
                // Determiners (bu, o, bir, her) also have Pron entries: with no noun after them
                // they stand for one ("Bu doğru", "O gelmedi").
                if !is_predicate(parse) {
                    match parse.pos {
                        PrimaryPos::Adj if modifies_noun => bonus += 0.7,
                        PrimaryPos::Det | PrimaryPos::Num if modifies_noun => bonus += 1.0,
                        PrimaryPos::Det if !modifies_noun => bonus -= 2.0,
                        _ => {}
                    }
                }

                // 4. A bare noun before a possessed one is its first part ("kamu yararı",
                // "gelir vergisi"), and before a light verb its object ("devam etti").
                if bare_noun && (before_possessed || before_light_verb) {
                    bonus += 1.5;
                }

                if let Some(prev) = previous {
                    // 5. An attributive adjective before a noun phrase head. Neither may be a
                    // predicate: after "yoğundu" the next word is a new clause (ki), and
                    // "geç kal-dı-m" is a verb, not an adjective + copular noun.
                    if prev.pos == PrimaryPos::Adj
                        && parse.pos == PrimaryPos::Noun
                        && !is_predicate(prev)
                        && !is_predicate(parse)
                        && !is_function_word(parses)
                    {
                        bonus += 0.5;
                    }
                    // 6. A pronoun subject before an inflected verb (ben geldim); a bare verb
                    // stem is an imperative and gets no help ("ben de" is not "I, say!").
                    if prev.pos == PrimaryPos::Pron && parse.pos == PrimaryPos::Verb && marked > 1 {
                        bonus += 0.5;
                    }
                    // 7. A genitive wants a possessed noun ("mahkemenin kararı").
                    if case_of(prev) == Some("Gen")
                        && parse
                            .morpheme_tags
                            .iter()
                            .any(|t| t == "P3sg" || t == "P3pl")
                    {
                        bonus += 1.5;
                    }
                    // 8. A postposition after the case it governs ("öğleden sonra", "yaşa
                    // kadar").
                    if parse.primary_pos == PrimaryPos::Postp
                        && case_of(prev).is_some_and(|case| {
                            self.morphology.root_items(&parse.root).iter().any(|item| {
                                item.primary_pos == PrimaryPos::Postp
                                    && item.governed_case.as_deref() == Some(case)
                            })
                        })
                    {
                        bonus += 1.0;
                    }
                }

                // 9. An interjection stands alone ("Hayır, ..."); inside a clause the word is
                // its noun or adjective twin ("devam edildi", "ileri sürdü").
                if parse.primary_pos == PrimaryPos::Interj && next_is_word {
                    bonus -= 2.0;
                }

                // 10. Function-word preference: an unsuffixed conjunction, question particle or
                // pronoun outranks an unsuffixed open-class homograph (de, ki, mi, ben), which
                // is far rarer in running text. The question particle keeps it with copula and
                // person (mi-yiz, mi-ydi-m), where its homograph is the note "mi".
                if parse.primary_pos == PrimaryPos::Q
                    || (marked == 1
                        && matches!(parse.primary_pos, PrimaryPos::Conj | PrimaryPos::Pron))
                {
                    bonus += 0.1;
                }

                // Fewer derivation layers on a tie, counting overt morphemes: otherwise "Ali"
                // (Noun+A3sg+Nom) loses to "Al-i" (Noun+Acc).
                let score = PRIOR_WEIGHT * parse.score.max(MIN_SCORE).ln() + bonus.min(MAX_BONUS)
                    - marked as f32 * 0.01;
                // The analyzer's order breaks exact ties.
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

/// A predicate reading: a nominal with the copula (evde-yim, doğru-dur).
fn is_predicate(p: &MorphParse) -> bool {
    p.morpheme_tags.iter().any(|t| t.starts_with("Cop"))
}

fn is_punctuation(token: &str) -> bool {
    token.chars().all(|c| !c.is_alphanumeric())
}

/// A token after which a sentence or quotation starts.
fn is_opening(token: &str) -> bool {
    matches!(
        token,
        "." | "!" | "?" | "…" | ":" | "\"" | "“" | "«" | "(" | "'" | "‘" | "-" | "–" | "—"
    )
}

/// The case of a noun or pronoun reading (its last case tag; a bare one is Nom), else `None`.
fn case_of(p: &MorphParse) -> Option<&str> {
    if !matches!(p.pos, PrimaryPos::Noun | PrimaryPos::Pron) || is_predicate(p) {
        return None;
    }
    let last = p.morpheme_tags.last()?.as_str();
    Some(match last {
        "Nom" | "Acc" | "Dat" | "Loc" | "Abl" | "Gen" | "Ins" | "Equ" => last,
        _ => "Nom",
    })
}

/// Whether the following tokens are a noun phrase head, possibly after a run of adjectives
/// ("bu [güzel] kitap"). A token counts as a noun when it has a common-noun reading and no
/// adjective reading; one that could be either continues the run. Ending the run without a
/// noun ("Bu doğru") or meeting another determiner ("Bu bir kitap") means the word before is
/// not a pre-nominal modifier.
fn noun_ahead(following: &[Vec<MorphParse>]) -> bool {
    for parses in following {
        if is_function_word(parses) || parses.iter().any(|p| p.pos == PrimaryPos::Det) {
            return false;
        }
        // A predicate (mi-yiz "we are the note mi", evde-yim) is not a noun phrase head.
        let noun = parses.iter().any(|p| {
            p.pos == PrimaryPos::Noun
                && p.secondary_pos != SecondaryPos::ProperNoun
                && !is_predicate(p)
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

/// Whether a token is read as a bare conjunction or postposition (hem, ki, kadar) or as the
/// question particle with or without its person (mi, miyiz). Such a word is not a noun phrase
/// head even when a rare suffixed noun reading exists (he-m "my h"), so it neither attracts a
/// pre-nominal modifier nor gets the noun-after-adjective bonus; "kalacak hem de" kept
/// turning hem into a noun.
fn is_function_word(parses: &[MorphParse]) -> bool {
    parses.iter().any(|p| {
        p.primary_pos == PrimaryPos::Q
            || (p.morpheme_tags.len() == 1
                && matches!(p.primary_pos, PrimaryPos::Conj | PrimaryPos::Postp))
    })
}
