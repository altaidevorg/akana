//! Contextual Morphological Disambiguator for Turkish sentences.
//!
//! Each word's analyses come ranked by the analyzer's corpus prior ([`MorphParse::score`]). The
//! disambiguator picks one per word, left to right, by the log of that score plus bonuses for
//! what the neighbouring words say: a genitive wants a possessed noun after it, a postposition
//! the case it governs before it, a light verb (et-, ol-) a bare noun before it, and so on. The
//! bonuses are bounded, so context can overturn a reading the corpus finds a dozen times rarer,
//! not one it finds a hundred times rarer. All of it is in the analyzer's integer rank units
//! ([`rank_units`]), so a sentence disambiguates alike on every platform.

use super::analyzer::{
    co_occurring_extra_tags, prior_capital_units, prior_negative_imperative, prior_register_units,
    rank_units, MorphParse, TurkishMorphology,
};
use super::pos::{PrimaryPos, SecondaryPos};
use crate::phonology::to_turkish_lower;

/// The most context can add to one reading's log score (a factor of about 12).
const MAX_BONUS: i32 = rank_units(2.5);
/// How far below its word's best reading a reading's log score counts (a factor of 10^-4), so
/// that one the prior rules out stays comparable.
const RANK_FLOOR: i32 = rank_units(-9.210_340_371_976_184);
// The analyzer's log score counts at half weight: EM over a word list makes the prior too sure
// of itself (kamu is an adjective 250 times more often than a noun), and halving its log lets
// context decide such cases ("kamu yararı"). Scores are doubled so that this stays an integer.

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
        // The previous word's reading as the rules saw it: a participle's noun reading even
        // where its adjective twin was output, so the twin never changes another word's choice.
        let mut context: Option<usize> = None;
        // A sentence that addresses someone: one that ends in "!", or has a question particle
        // (gelecek misin) or a word whose best reading is a second-person verb (biliyorsun,
        // gel). Its other words' second-person readings (anla "understand!", gel-in, yıka) lose
        // the register factor the corpus prior gives them. Sentences end at . ! ? and …
        let ends = |t: &str| matches!(t, "." | "!" | "?" | "…");
        let mut sentence = vec![0usize; n];
        for i in 1..n {
            sentence[i] = sentence[i - 1] + usize::from(ends(tokens[i - 1]));
        }
        let addressing: Vec<bool> = analyses_per_token
            .iter()
            .map(|parses| {
                parses.first().is_some_and(|p| {
                    p.primary_pos == PrimaryPos::Q
                        || (p.pos == PrimaryPos::Verb
                            && p.morpheme_tags.iter().any(|t| t == "A2sg" || t == "A2pl"))
                })
            })
            .collect();
        let mut addressing_in = vec![0usize; sentence.last().map_or(0, |s| s + 1)];
        for (s, a) in sentence.iter().zip(&addressing) {
            addressing_in[*s] += usize::from(*a);
        }

        for i in 0..n {
            let parses = &analyses_per_token[i];
            let previous = match context {
                Some(j) if i > 0 => analyses_per_token[i - 1].get(j),
                _ if i > 0 => chosen.last(),
                _ => None,
            };
            context = None;
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
                    nonstandard: false,
                    path: Default::default(),
                    yields: false,
                    rank: 0,
                    homograph: 0,
                });
                continue;
            }

            if parses.len() == 1 {
                chosen.push(parses[0].clone());
                context = Some(0);
                continue;
            }

            let is_sentence_final =
                i == n - 1 || (i == n - 2 && matches!(tokens[n - 1], "." | "!" | "?" | "…"));
            // The end of a clause inside the sentence: before a comma, a semicolon or a
            // conjunction that joins clauses.
            let is_clause_final = !is_sentence_final
                && tokens.get(i + 1).is_some_and(|t| {
                    matches!(*t, "," | ";")
                        || matches!(
                            to_turkish_lower(t).as_str(),
                            "ve" | "ama" | "fakat" | "ancak" | "çünkü" | "ki"
                        )
                });
            // The first word of a sentence or of a quotation is capitalised whatever it is.
            let is_sentence_start = i == 0 || is_opening(tokens[i - 1]);
            let is_first_char_upper = tokens[i].chars().next().is_some_and(|c| c.is_uppercase());
            // There, and in a word written in capitals (KARAR, ÜSTÜN) but for one that can be
            // an abbreviation (AT, AB), the capital says nothing about the word: the prior's
            // case factor is divided out ("Kanun , ...", "Ekmek aldım").
            let capital_says_nothing = is_first_char_upper
                && (is_sentence_start
                    || (tokens[i].chars().filter(|c| c.is_alphabetic()).count() > 1
                        && !tokens[i].chars().any(char::is_lowercase)
                        && !parses
                            .iter()
                            .any(|p| p.secondary_pos == SecondaryPos::Abbreviation)));
            // A question particle after the word makes it the predicate ("gelecek misin").
            let before_question = analyses_per_token
                .get(i + 1)
                .and_then(|parses| parses.first())
                .is_some_and(|p| p.primary_pos == PrimaryPos::Q);
            let following = &analyses_per_token[i + 1..];
            let next_parses = following.first().map(Vec::as_slice).unwrap_or_default();
            let next_is_word = i + 1 < n && !is_punctuation(tokens[i + 1]);
            let modifies_noun = noun_ahead(following);
            // An adjective modifies a noun phrase past the indefinite article ("uzun bir
            // süre", "önemli bir karar"); a determiner does not ("Bu bir kitap").
            let modifies_past_article = tokens
                .get(i + 1)
                .is_some_and(|t| to_turkish_lower(t) == "bir")
                && noun_ahead(following.get(1..).unwrap_or_default());
            // Before olmak a word that is also an adjective is its predicate ("yazılı olmayan",
            // "sözlü olarak"), not the first half of a compound verb.
            let before_light_verb = next_parses.iter().any(|p| {
                p.primary_pos == PrimaryPos::Verb
                    && match p.root.as_str() {
                        "et" | "eyle" => true,
                        "ol" => !parses.iter().any(|q| q.pos == PrimaryPos::Adj),
                        _ => false,
                    }
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
            let has_determiner_reading = parses.iter().any(|p| p.pos == PrimaryPos::Det);
            let has_pronoun_reading = parses.iter().any(|p| p.primary_pos == PrimaryPos::Pron);
            let has_adjective_reading = parses.iter().any(|p| p.pos == PrimaryPos::Adj);
            // An adjective with an adverb twin before a verb other than olmak, imek or değil,
            // after which an adjective is the predicate (iyi oldu, az ise, doğru değil). A word
            // with a pronoun twin is a question word or a determiner (ne yiyelim).
            let adverb_before_verb = has_adjective_reading
                && !has_pronoun_reading
                && parses.iter().any(|p| p.pos == PrimaryPos::Adv)
                && next_parses.first().is_some_and(|p| {
                    p.primary_pos == PrimaryPos::Verb
                        && !matches!(p.root.as_str(), "ol" | "i" | "değil")
                });
            // What a postposition could govern: the case of the word before it, if a noun or
            // a pronoun.
            let previous_case = previous.and_then(case_of);
            let exclaims = tokens[i + 1..]
                .iter()
                .find(|t| matches!(**t, "." | "!" | "?"))
                .is_some_and(|t| *t == "!");
            let addresses = exclaims || addressing_in[sentence[i]] > usize::from(addressing[i]);
            // A correlative conjunction comes twice in its clause (ister ... ister(se), ya ...
            // ya, hem ... hem, ne ... ne, gerek ... gerek(se)): the word, or the word with -sA,
            // again before or after it in the sentence.
            let correlative = parses.iter().any(|p| p.primary_pos == PrimaryPos::Conj) && {
                let word = to_turkish_lower(tokens[i]);
                let pairs = |t: &&str| {
                    let t = to_turkish_lower(t);
                    t == word
                        || t.strip_prefix(word.as_str())
                            .is_some_and(|rest| matches!(rest, "se" | "sa"))
                        || word
                            .strip_prefix(t.as_str())
                            .is_some_and(|rest| matches!(rest, "se" | "sa"))
                };
                let end = |t: &&&str| !matches!(**t, "." | "!" | "?" | ";");
                tokens[i + 1..].iter().take_while(end).any(&pairs)
                    || tokens[..i].iter().rev().take_while(end).any(pairs)
            };

            let mut best_idx = 0;
            let mut best_score = i32::MIN;
            let top_rank = parses.iter().map(|p| p.rank).max().unwrap_or(0);
            for (p_idx, parse) in parses.iter().enumerate() {
                if adjective_twin(parse, parses) {
                    continue;
                }
                // Overt morphemes. Only bare dictionary matches spell out the default A3sg/Nom,
                // so those are not counted.
                // A co-occurring set's ambiguous part counts once (A3pl+P3sg as P3pl does), so
                // that the analyzer's order among them stands.
                let marked = parse
                    .morpheme_tags
                    .iter()
                    .filter(|t| !matches!(t.as_str(), "A3sg" | "Nom"))
                    .count()
                    - co_occurring_extra_tags(parse);
                let bare_noun = parse.primary_pos == PrimaryPos::Noun
                    && parse.secondary_pos != SecondaryPos::ProperNoun
                    && marked == 1;
                let mut bonus = 0i32;

                // The rules below look at the word's POS after derivation (`pos`): a converb
                // (baş-la-yınca) is not a noun, a derived verb (an-laş-ıyor-um) is a verb, a
                // nominalised verb (gel-diğ-i) is not a finite predicate.

                // 1. The predicate comes last (SOV), in the sentence and less surely in a
                // clause: a finite verb before a comma or "ve" (gelecek, ...) over its participle.
                // Not a negative imperative there: before "ve" its twin, the verbal noun in -mA,
                // is far likelier ("belgelendirme ve bilgilendirme"). A question particle
                // follows the predicate ("Yarın gelecek misin ?").
                if is_sentence_final || before_question {
                    if parse.pos == PrimaryPos::Verb {
                        bonus += const { rank_units(2.0) };
                    } else if is_predicate(parse) {
                        bonus += const { rank_units(1.0) };
                    }
                } else if is_clause_final
                    && parse.pos == PrimaryPos::Verb
                    && !prior_negative_imperative(parse)
                {
                    bonus += const { rank_units(1.0) };
                }

                // 2. A capitalised word inside a sentence is a name. Only for a bare name or
                // one whose suffixes follow an apostrophe: "Annem" is anne-m, not the name
                // Anne + P1sg. At the start of a sentence capitals say nothing ("Yani").
                if is_first_char_upper
                    && parse.secondary_pos == SecondaryPos::ProperNoun
                    && (marked == 1 || tokens[i].contains(['\'', '’']))
                    && (!is_sentence_start || !has_common_reading)
                {
                    bonus += const { rank_units(1.5) };
                }

                // 3. A modifier comes before its noun (Adj, Det, Num: "bu kitap", "yüz kişi").
                // Determiners with a Pron twin (bu, şu, o) stand for one with no noun after
                // them ("Bu doğru", "O gelmedi"). The adjective twin of a determiner (o, her)
                // is one too, and needs the noun as much. Without a pronoun twin, the determiner
                // stays (bir: "yeni bir kazak ördü", whose noun is also an adjective).
                // A participle modifies its noun phrase past a determiner or a number ("olabilecek
                // bir işlem", "yapılacak bir şey"), as `participle_head` reads it; its finite twin
                // (Fut, Narr, Aor) has no help but at the end of a sentence.
                let participle =
                    parse.pos == PrimaryPos::Adj && parse.analysis.contains("Part→Adj");
                let modifies = modifies_noun
                    || modifies_past_article
                    || participle && participle_head(following, possessive(parse));
                if !is_predicate(parse) {
                    match parse.pos {
                        // Before a verb its adverb twin is meant (12), even with a noun after
                        // the verb's participle ("ileri sürülen temyiz nedenleri").
                        PrimaryPos::Adj if modifies && !adverb_before_verb => {
                            bonus += const { rank_units(0.7) }
                        }
                        PrimaryPos::Det | PrimaryPos::Num if modifies_noun => {
                            bonus += const { rank_units(1.0) }
                        }
                        PrimaryPos::Det if !modifies_noun && has_pronoun_reading => {
                            bonus -= const { rank_units(2.0) }
                        }
                        PrimaryPos::Adj if !modifies_noun && has_determiner_reading => {
                            bonus -= const { rank_units(2.0) }
                        }
                        _ => {}
                    }
                }

                // 4. A bare noun before a possessed one is its first part ("kamu yararı",
                // "gelir vergisi"), and before a light verb its object ("devam etti").
                if bare_noun && (before_possessed || before_light_verb) {
                    bonus += const { rank_units(1.5) };
                }

                if let Some(prev) = previous {
                    // 5. An attributive adjective before a noun phrase head. Neither may be a
                    // predicate: after "yoğundu" the next word is a new clause (ki), and
                    // "geç kal-dı-m" is a verb, not an adjective + copular noun. At the end of
                    // the sentence a word that can be a finite verb is its predicate ("Uzun
                    // yaşa !").
                    if prev.pos == PrimaryPos::Adj
                        && parse.pos == PrimaryPos::Noun
                        && !is_predicate(prev)
                        && !is_predicate(parse)
                        && !is_function_word(parses)
                        && !(is_sentence_final && parses.iter().any(|p| p.pos == PrimaryPos::Verb))
                    {
                        bonus += const { rank_units(0.5) };
                    }
                    // 6. A pronoun subject before an inflected verb (ben geldim); a bare verb
                    // stem is an imperative and gets no help ("ben de" is not "I, say!").
                    if prev.pos == PrimaryPos::Pron && parse.pos == PrimaryPos::Verb && marked > 1 {
                        bonus += const { rank_units(0.5) };
                    }
                    // 7. A genitive wants a possessed noun of its number ("mahkemenin kararı",
                    // "davacının hakları": A3pl+P3sg; "sanıkların hakları": A3pl+P3pl).
                    if case_of(prev) == Some("Gen") {
                        let plural = prev
                            .morpheme_tags
                            .iter()
                            .any(|t| t == "A3pl" || t == "P3pl");
                        let wanted = if plural { "P3pl" } else { "P3sg" };
                        if parse.morpheme_tags.iter().any(|t| t == wanted) {
                            bonus += const { rank_units(1.5) };
                        }
                    }
                }

                // 8. A postposition after the case it governs ("öğleden sonra", "yaşa kadar").
                // A bare noun stands before every word, so the nominative says nothing (Ali ile
                // Veli). One that governs the ablative or the dative has nothing to govern after
                // a word that is no noun or pronoun ("daha sonra", ", sonra da", at the start of
                // a sentence) or after another case ("Bu doğru"), and is its adverb, adjective or
                // noun twin; but a time takes sonra and önce bare ("iki yıl sonra"), and bu, şu,
                // o and ne take kadar.
                if parse.primary_pos == PrimaryPos::Postp {
                    let governed = || {
                        self.morphology
                            .root_items(&parse.root)
                            .iter()
                            .filter(|item| item.primary_pos == PrimaryPos::Postp)
                            .filter_map(|item| item.governed_case.as_deref())
                    };
                    let marked = governed().next().is_some()
                        && governed().all(|c| matches!(c, "Abl" | "Dat"))
                        && parse.root != "kadar";
                    match previous_case {
                        Some(case) if case != "Nom" && governed().any(|c| c == case) => {
                            bonus += const { rank_units(1.0) }
                        }
                        Some("Nom") if marked && governed().any(|c| c == "Abl") => {}
                        _ if marked => bonus -= const { rank_units(1.0) },
                        _ => {}
                    }
                }

                // 9. An interjection stands alone ("Hayır, ..."); inside a clause the word is
                // its noun or adjective twin ("devam edildi", "ileri sürdü").
                if parse.primary_pos == PrimaryPos::Interj && next_is_word {
                    bonus -= const { rank_units(2.0) };
                }

                // 10. The first half of a correlative conjunction ("ister iktidarda ister
                // muhalefette", "hem ... hem de").
                if correlative && parse.primary_pos == PrimaryPos::Conj {
                    bonus += const { rank_units(1.5) };
                }

                // 11. Function-word preference: an unsuffixed conjunction, question particle or
                // pronoun outranks an unsuffixed open-class homograph (de, ki, mi, ben), which
                // is far rarer in running text. The question particle keeps it with copula and
                // person (mi-yiz, mi-ydi-m), where its homograph is the note "mi".
                if parse.primary_pos == PrimaryPos::Q
                    || (marked == 1
                        && matches!(parse.primary_pos, PrimaryPos::Conj | PrimaryPos::Pron))
                {
                    bonus += const { rank_units(0.1) };
                }

                // 12. An adjective's adverb twin before a verb: "yeni yapılan", "ileri sürülen",
                // "uzun sürdü".
                if parse.pos == PrimaryPos::Adv && marked == 1 && adverb_before_verb {
                    bonus += const { rank_units(1.0) };
                }
                // Fewer derivation layers on a tie, counting overt morphemes: otherwise "Ali"
                // (Noun+A3sg+Nom) loses to "Al-i" (Noun+Acc).
                let mut rank = parse.rank;
                if addresses {
                    rank -= prior_register_units(parse);
                }
                if capital_says_nothing {
                    rank -= prior_capital_units(parse);
                }
                let score = rank.max(top_rank + RANK_FLOOR) + 2 * bonus.min(MAX_BONUS)
                    - 2 * marked as i32 * const { rank_units(0.01) };
                // The analyzer's order breaks exact ties.
                if score > best_score {
                    best_score = score;
                    best_idx = p_idx;
                }
            }

            // The noun reading of a participle or -IcI form stood for both twins; its adjective
            // replaces it in the output when the form modifies the words after it.
            context = Some(best_idx);
            let best = &parses[best_idx];
            let twin = parses
                .iter()
                .find(|p| p.formatted == best.formatted && adjective_twin(p, parses));
            match twin {
                Some(adjective) if participle_head(following, possessive(adjective)) => {
                    chosen.push(adjective.clone());
                }
                _ => chosen.push(best.clone()),
            }
        }

        chosen
    }
}

/// The adjective reading of a participle or -IcI form that akana's tags render as its noun
/// reading does (ol-duğ-u: PastPart→Adj "which it is" beside PastPart→Noun "that it is";
/// bağla-yıcı: Agt→Adj beside Agt→Noun; grup-lan-dığ-ı from a noun's verb too). The rules
/// choose among a word's readings with the noun standing for both, so the choice of lemma
/// never depends on the twin, and other words see the noun: the twin only says what the form
/// does in its phrase (`participle_head`).
fn adjective_twin(p: &MorphParse, parses: &[MorphParse]) -> bool {
    p.pos == PrimaryPos::Adj
        && parses
            .iter()
            .any(|q| q.formatted == p.formatted && q.pos == PrimaryPos::Noun)
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
        let adj = parses
            .iter()
            .any(|p| p.pos == PrimaryPos::Adj && !adjective_twin(p, parses));
        if noun && !adj {
            return true;
        }
        if !adj {
            return false;
        }
    }
    false
}

/// Whether a participle or -IcI form modifies the words after it: a relative clause or an
/// attribute ("verdiği karar", "açılacak iptal davası", "olabilecek bir işlem", "bağlayıcı
/// karar") rather than a noun clause ("olduğu anlaşıldı", "edildiği iddiası", "olduğu iddia
/// edilen", "işlendiği takdirde" before the postposition). Its head is the first word after
/// any determiners, number words and bare adjectives; a name or a number in digits after the
/// form starts another phrase ("başvurucu Mecidiye ...", "başvurucu 22/7/2016 tarihli"). The
/// head's best reading must be a common noun that no verb went into (tarih-te-ki counts,
/// tarih-li and an-laş-ıl-dı do not) and that is not the first half of a light-verb compound
/// (iddia et-, sabit ol-), nor possessed: a possessed noun after a clause is the noun the
/// clause complements (iddia-sı, görüş-ü, gerekçe-si), and after a bare -IcI noun the head of
/// a compound (başvurucu vekil-i). A participle with a possessive of its own (oku-duğ-um) also
/// modifies an accusative head (kitab-ı), which looks like a possessed one, so for it
/// (`own_possessive`) the head counts as possessed only when it has no other noun reading.
fn participle_head(following: &[Vec<MorphParse>], own_possessive: bool) -> bool {
    let noun = |p: &MorphParse| {
        p.primary_pos == PrimaryPos::Noun
            && (p.pos == PrimaryPos::Noun || p.morpheme_tags.iter().any(|t| t == "RelClitic"))
            && p.secondary_pos != SecondaryPos::ProperNoun
            && !p.analysis.contains("→Verb")
            && !is_predicate(p)
    };
    for (k, parses) in following.iter().enumerate() {
        let Some(top) = parses.first() else {
            return false;
        };
        if is_function_word(parses)
            || top
                .surface
                .starts_with(|c: char| c.is_uppercase() || c.is_ascii_digit())
        {
            return false;
        }
        if matches!(top.pos, PrimaryPos::Det | PrimaryPos::Num)
            || (top.primary_pos == PrimaryPos::Adj && top.morpheme_tags.len() == 1)
        {
            continue;
        }
        if !noun(top) {
            return false;
        }
        let complemented = if own_possessive {
            parses.iter().filter(|p| noun(p)).all(possessive)
        } else {
            possessive(top)
        };
        let light_verb = following.get(k + 1).is_some_and(|next| {
            next.iter().any(|p| {
                p.primary_pos == PrimaryPos::Verb
                    && matches!(p.root.as_str(), "et" | "ol" | "eyle" | "kıl")
            })
        });
        return !complemented && !light_verb;
    }
    false
}

/// Whether a reading has a possessive (P1sg ... P3pl).
fn possessive(p: &MorphParse) -> bool {
    p.morpheme_tags.iter().any(|t| {
        matches!(
            t.as_str(),
            "P1sg" | "P2sg" | "P3sg" | "P1pl" | "P2pl" | "P3pl"
        )
    })
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
