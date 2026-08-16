//! Readability metrics, score results, and grade level classifications.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextStatistics {
    pub total_sentences: usize,
    pub total_words: usize,
    pub total_syllables: usize,
    pub words_per_sentence: f64,    // OCU (Ortalama Cümle Uzunluğu)
    pub syllables_per_word: f64,    // OSU (Ortalama Sözcük Uzunluğu - Hece)
    pub unfamiliar_words: usize,    // Words not in 4600 list
    pub unfamiliar_word_ratio: f64, // KL (Kelime Listesi Puanı - %)
    pub single_occurrence_words: usize, // Unique lemmas occurring only once
    pub type_token_ratio: f64,      // TTR (Sözcük Çeşitliliği - %)
    pub complex_sentences: usize,   // Sentences with fiilimsi, conjunctions or clauses
    pub complex_sentence_ratio: f64,// KCO (Kompleks Cümle Oranı - %)
    pub total_fiilimsiler: usize,   // Total verbal nouns, participles, gerunds
    pub clause_ratio: f64,          // YCO (Yan Cümle Oranı - per 10 sentences)
    pub polysyllabic_words: usize,  // Words with >= 3 syllables
    pub polysyllabic_word_ratio: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FormulaResult {
    pub score: f64,
    pub grade_level: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KalyoncuResults {
    pub formula1: FormulaResult, // En kapsamlı (R²=0.99)
    pub formula2: FormulaResult, // R²=0.82
    pub formula3: FormulaResult, // R²=0.83
    pub formula4: FormulaResult, // En pratik (R²=0.78)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegacyResults {
    pub atesman: FormulaResult,        // Ateşman (1997)
    pub cetinkaya_uzun: FormulaResult, // Çetinkaya-Uzun (2010)
    pub bezirci_yilmaz: FormulaResult, // Bezirci-Yılmaz (2010)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReadabilityReport {
    pub statistics: TextStatistics,
    pub kalyoncu: KalyoncuResults,
    pub legacy: LegacyResults,
}
