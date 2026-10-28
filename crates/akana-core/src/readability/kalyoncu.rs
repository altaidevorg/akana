//! Kalyoncu (2025) Turkish Readability Formula implementation.
//! Based on M. Rahman Kalyoncu's master's thesis "Türkçe İçin Yeni Bir Okunabilirlik Formülü" (Ondokuz Mayıs Üniversitesi, 2025).

use super::metrics::{FormulaResult, KalyoncuResults, TextStatistics};
use crate::morphology::{
    verb_form, MorphologicalDisambiguator, PrimaryPos, SecondaryPos, TurkishMorphology,
};
use crate::phonology;
use crate::tokenization::{SentenceSegmenter, TurkishTokenizer};
use std::collections::{HashMap, HashSet};

pub struct KalyoncuAnalyzer {
    word_list: HashSet<String>,
    disambiguator: MorphologicalDisambiguator,
    morphology: TurkishMorphology,
}

/// A word as the familiarity list is looked up: without circumflexes, as the analyzer's roots
/// are (hâl, hikâye, millî are hal, hikaye, milli; review 3 STYLE3-04).
fn fold(word: &str) -> String {
    word.chars()
        .map(|c| match c {
            'â' => 'a',
            'î' => 'i',
            'û' => 'u',
            c => c,
        })
        .collect()
}

impl Default for KalyoncuAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl KalyoncuAnalyzer {
    pub fn new() -> Self {
        let raw_data = include_str!("kalyoncu_words.txt");
        let mut set = HashSet::with_capacity(4650);
        for line in raw_data.lines() {
            let w = line.trim();
            if !w.is_empty() {
                set.insert(fold(w));
            }
        }

        Self {
            word_list: set,
            disambiguator: MorphologicalDisambiguator::new(),
            morphology: TurkishMorphology::new(),
        }
    }

    /// Calculates comprehensive text statistics for Turkish readability.
    pub fn calculate_statistics(&self, text: &str) -> TextStatistics {
        let sentences = SentenceSegmenter::segment(text);
        // Sentences with a word: a paragraph's number or letter alone (1., A.) is none
        // (review 3 STYLE3-03).
        let mut total_sentences = 0;

        let mut total_words = 0;
        let mut total_syllables = 0;
        let mut polysyllabic_words = 0;
        let mut unfamiliar_words = 0;

        let mut lemma_counts: HashMap<String, usize> = HashMap::new();
        let mut total_fiilimsiler = 0;
        let mut complex_sentences = 0;

        let conjunctions: HashSet<&str> = [
            "ve", "veya", "yahut", "ki", "de", "da", "ama", "fakat", "lakin", "ancak", "çünkü",
            "oysa", "oysaki", "halbuki", "ise", "madem", "ile",
        ]
        .iter()
        .cloned()
        .collect();

        for sent in &sentences {
            let tokens = TurkishTokenizer::tokenize_words(sent.text);
            // Each word's reading in its sentence (review 2 PARSER2-02).
            let parses = self.disambiguator.disambiguate(&tokens);
            let mut sent_word_count = 0;
            let mut sent_fiilimsi_count = 0;
            let mut has_conjunction = false;

            for (tok, parse) in tokens.into_iter().zip(&parses) {
                let lower = phonology::to_turkish_lower(tok);
                if lower.chars().all(|c| !c.is_alphabetic()) {
                    continue;
                }

                sent_word_count += 1;
                total_words += 1;

                // Syllables (number of Turkish vowels)
                let syllables = lower
                    .chars()
                    .filter(|&c| phonology::is_turkish_vowel(c))
                    .count();
                total_syllables += syllables;
                if syllables >= 3 {
                    polysyllabic_words += 1;
                }

                // Conjunction check
                if conjunctions.contains(lower.as_str()) {
                    has_conjunction = true;
                }

                // The word's root, whether it is a verbal (fiilimsi: a verbal noun, participle
                // or converb) and whether it is a name or a number.
                let (best_lemma, is_fiilimsi, is_proper_or_num) = if parse.primary_pos
                    == PrimaryPos::Unknown
                {
                    (lower.clone(), false, false)
                } else {
                    let proper_or_num = parse.secondary_pos == SecondaryPos::ProperNoun
                        || parse.primary_pos == PrimaryPos::Num;
                    // A plural verbal noun is a noun (ödemelerin, gelişmelere), no
                    // fiilimsi (review 3 STYLE3-09).
                    let plural_noun = parse
                        .morpheme_tags
                        .iter()
                        .position(|t| matches!(t.as_str(), "ActN" | "Inf"))
                        .is_some_and(|i| parse.morpheme_tags[i + 1..].iter().any(|t| t == "A3pl"));
                    (
                        phonology::to_turkish_lower(&parse.root),
                        verb_form(parse).is_some() && !plural_noun,
                        proper_or_num,
                    )
                };

                if is_fiilimsi {
                    sent_fiilimsi_count += 1;
                    total_fiilimsiler += 1;
                }

                *lemma_counts.entry(best_lemma.clone()).or_insert(0) += 1;

                // Familiarity check:
                // 1. Surface word in list
                // 2. Lemma/Root in list
                // 3. Verb infinitive (-mak / -mek) in list
                // 4. Proper noun or numeral
                // 5. A postposition or adverb the lexicon lists as a word of its own, by the
                //    noun it is spelled from (taraf-ından, iç-inde, üzer-ine; review 3 STYLE3-04)
                let lemma = fold(&best_lemma);
                let lemma_mak = format!("{lemma}mak");
                let lemma_mek = format!("{lemma}mek");
                let known = |w: &str| self.word_list.contains(w);
                let spelled_from_noun = || {
                    matches!(parse.primary_pos, PrimaryPos::Postp | PrimaryPos::Adv)
                        && self
                            .morphology
                            .analyze(&lower)
                            .iter()
                            .any(|p| p.primary_pos == PrimaryPos::Noun && known(&fold(&p.root)))
                };

                let is_familiar = is_proper_or_num
                    || known(&fold(&lower))
                    || known(&lemma)
                    || known(&lemma_mak)
                    || known(&lemma_mek)
                    || spelled_from_noun();

                if !is_familiar {
                    unfamiliar_words += 1;
                }
            }

            if sent_word_count > 0 {
                total_sentences += 1;
            }
            if sent_fiilimsi_count > 0 || has_conjunction || sent_word_count > 10 {
                complex_sentences += 1;
            }
        }
        let total_sentences = total_sentences.max(1);

        let total_words_safe = total_words.max(1);
        let words_per_sentence = total_words as f64 / total_sentences as f64;
        let syllables_per_word = total_syllables as f64 / total_words_safe as f64;
        let unfamiliar_word_ratio = (unfamiliar_words as f64 / total_words_safe as f64) * 100.0;

        let single_occurrence_words = lemma_counts.values().filter(|&&count| count == 1).count();
        let type_token_ratio = (single_occurrence_words as f64 / total_words_safe as f64) * 100.0;

        let complex_sentence_ratio = (complex_sentences as f64 / total_sentences as f64) * 100.0;
        let clause_ratio = (total_fiilimsiler as f64 / total_sentences as f64) * 10.0;
        let polysyllabic_word_ratio = (polysyllabic_words as f64 / total_words_safe as f64) * 100.0;

        TextStatistics {
            total_sentences,
            total_words,
            total_syllables,
            words_per_sentence,
            syllables_per_word,
            unfamiliar_words,
            unfamiliar_word_ratio,
            single_occurrence_words,
            type_token_ratio,
            complex_sentences,
            complex_sentence_ratio,
            total_fiilimsiler,
            clause_ratio,
            polysyllabic_words,
            polysyllabic_word_ratio,
        }
    }

    /// Evaluates text readability according to the 4 Kalyoncu (2025) regression models.
    pub fn evaluate(&self, stats: &TextStatistics) -> KalyoncuResults {
        let kl = stats.unfamiliar_word_ratio;
        let ttr = stats.type_token_ratio;
        let ocu = stats.words_per_sentence;
        let kco = stats.complex_sentence_ratio;
        let yco = stats.clause_ratio;

        // 1. Formül: En Kapsamlı (R² = 0.99)
        // Skor = -29.829 + 0.508*KL + 1.191*TTR + 0.733*KCO - 0.66*OCU
        let f1_score = -29.829 + (0.508 * kl) + (1.191 * ttr) + (0.733 * kco) - (0.660 * ocu);
        let f1_grade = Self::grade_formula_1(f1_score);

        // 2. Formül (R² = 0.82)
        // Skor = 25.457 + 0.877*KL + 0.668*TTR + 0.333*OCU
        let f2_score = 25.457 + (0.877 * kl) + (0.668 * ttr) + (0.333 * ocu);
        let f2_grade = Self::grade_formula_2(f2_score);

        // 3. Formül (R² = 0.83)
        // Skor = 25.153 + 0.834*KL + 0.686*TTR + 0.226*YCO
        let f3_score = 25.153 + (0.834 * kl) + (0.686 * ttr) + (0.226 * yco);
        let f3_grade = Self::grade_formula_3(f3_score);

        // 4. Formül: En Pratik (R² = 0.78)
        // Skor = 43.368 + 1.008*KL + 0.624*OCU
        let f4_score = 43.368 + (1.008 * kl) + (0.624 * ocu);
        let f4_grade = Self::grade_formula_4(f4_score);

        // Without a word every formula grades the empty statistics (review 2 READ-05).
        let no_text = stats.total_words == 0;
        let grade = |g: &'static str| if no_text { super::metrics::NO_TEXT } else { g };
        let score = |s: f64| if no_text { 0.0 } else { s };
        let (f1_grade, f2_grade, f3_grade, f4_grade) = (
            grade(f1_grade),
            grade(f2_grade),
            grade(f3_grade),
            grade(f4_grade),
        );
        let (f1_score, f2_score, f3_score, f4_score) = (
            score(f1_score),
            score(f2_score),
            score(f3_score),
            score(f4_score),
        );
        KalyoncuResults {
            formula1: FormulaResult {
                score: (f1_score * 100.0).round() / 100.0,
                grade_level: f1_grade.to_string(),
                description: "Kalyoncu Formül 1 (En Kapsamlı - R²=0.99)".to_string(),
            },
            formula2: FormulaResult {
                score: (f2_score * 100.0).round() / 100.0,
                grade_level: f2_grade.to_string(),
                description: "Kalyoncu Formül 2 (R²=0.82)".to_string(),
            },
            formula3: FormulaResult {
                score: (f3_score * 100.0).round() / 100.0,
                grade_level: f3_grade.to_string(),
                description: "Kalyoncu Formül 3 (R²=0.83)".to_string(),
            },
            formula4: FormulaResult {
                score: (f4_score * 100.0).round() / 100.0,
                grade_level: f4_grade.to_string(),
                description: "Kalyoncu Formül 4 (En Pratik - R²=0.78)".to_string(),
            },
        }
    }

    fn grade_formula_1(score: f64) -> &'static str {
        if score < 30.0 {
            "3. Sınıf Öncesi"
        } else if score <= 40.0 {
            "3 ve 4. Sınıf"
        } else if score <= 50.0 {
            "5 ve 6. Sınıf"
        } else if score <= 60.0 {
            "7 ve 8. Sınıf"
        } else if score <= 70.0 {
            "9 ve 10. Sınıf"
        } else if score <= 76.0 {
            "11 ve 12. Sınıf"
        } else if score <= 85.0 {
            "Lisans"
        } else {
            "Lisansüstü"
        }
    }

    fn grade_formula_2(score: f64) -> &'static str {
        if score < 35.0 {
            "3. Sınıf Öncesi"
        } else if score <= 45.0 {
            "3 ve 4. Sınıf"
        } else if score <= 55.0 {
            "5 ve 6. Sınıf"
        } else if score <= 60.0 {
            "7 ve 8. Sınıf"
        } else if score <= 64.0 {
            "9 ve 10. Sınıf"
        } else if score <= 70.0 {
            "11 ve 12. Sınıf"
        } else if score <= 85.0 {
            "Lisans"
        } else {
            "Lisansüstü"
        }
    }

    fn grade_formula_3(score: f64) -> &'static str {
        if score < 35.0 {
            "3. Sınıf Öncesi"
        } else if score <= 45.0 {
            "3 ve 4. Sınıf"
        } else if score <= 55.0 {
            "5 ve 6. Sınıf"
        } else if score <= 60.0 {
            "7 ve 8. Sınıf"
        } else if score <= 65.0 {
            "9 ve 10. Sınıf"
        } else if score <= 70.0 {
            "11 ve 12. Sınıf"
        } else if score <= 85.0 {
            "Lisans"
        } else {
            "Lisansüstü"
        }
    }

    fn grade_formula_4(score: f64) -> &'static str {
        if score < 40.0 {
            "3. Sınıf Öncesi"
        } else if score <= 50.0 {
            "3 ve 4. Sınıf"
        } else if score <= 55.0 {
            "5 ve 6. Sınıf"
        } else if score <= 59.0 {
            "7 ve 8. Sınıf"
        } else if score <= 65.0 {
            "9 ve 10. Sınıf"
        } else if score <= 72.0 {
            "11 ve 12. Sınıf"
        } else if score <= 85.0 {
            "Lisans"
        } else {
            "Lisansüstü"
        }
    }
}
