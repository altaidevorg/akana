//! Turkish Morphological Analyzer.

use super::dictionary::{DictionaryItem, RootLexicon};
use super::graph::{MorphState, SuffixTransition, TurkishMorphotactics};
use super::pos::{PrimaryPos, RootAttr, SecondaryPos};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MorphemeData {
    pub name: String,
    pub surface: String,
    pub tag: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MorphParse {
    pub surface: String,
    pub lemma: String,
    pub root: String,
    pub primary_pos: PrimaryPos,
    pub secondary_pos: SecondaryPos,
    pub morpheme_tags: Vec<String>,
    pub formatted: String,
    pub score: f32,
}

pub struct TurkishMorphology {
    lexicon: RootLexicon,
    transitions: Vec<SuffixTransition>,
}

impl Default for TurkishMorphology {
    fn default() -> Self {
        Self::new()
    }
}

impl TurkishMorphology {
    pub fn new() -> Self {
        Self {
            lexicon: RootLexicon::new(),
            transitions: TurkishMorphotactics::transitions(),
        }
    }

    pub fn with_lexicon(lexicon: RootLexicon) -> Self {
        Self {
            lexicon,
            transitions: TurkishMorphotactics::transitions(),
        }
    }

    /// Adds a dictionary item to the morphological analyzer at runtime.
    pub fn add_item(&mut self, item: DictionaryItem) {
        self.lexicon.add_item(item);
    }

    /// Loads custom dictionary definitions from a file at runtime.
    pub fn load_dictionary_file<P: AsRef<std::path::Path>>(&mut self, path: P) -> std::io::Result<()> {
        self.lexicon.load_from_file(path)
    }

    /// Loads custom dictionary definitions from a multiline string.
    pub fn load_dictionary_str(&mut self, text: &str) {
        self.lexicon.load_from_str(text);
    }

    /// Analyzes a surface word and returns all valid morphological parses.
    pub fn analyze(&self, word: &str) -> Vec<MorphParse> {
        let clean = word.trim();
        if clean.is_empty() {
            return Vec::new();
        }

        let lower = super::super::phonology::to_turkish_lower(clean);
        let mut results = Vec::new();

        // 1. Direct match in dictionary (uninflected base form)
        if let Some(items) = self.lexicon.get_items(&lower) {
            for item in items {
                let tags = match item.primary_pos {
                    PrimaryPos::Noun => vec!["Noun".to_string(), "A3sg".to_string(), "Nom".to_string()],
                    PrimaryPos::Verb => vec!["Verb".to_string(), "A3sg".to_string()],
                    PrimaryPos::Adj => vec!["Adj".to_string()],
                    PrimaryPos::Adv => vec!["Adv".to_string()],
                    PrimaryPos::Pron => vec!["Pron".to_string(), "A3sg".to_string(), "Nom".to_string()],
                    PrimaryPos::Num => vec!["Num".to_string()],
                    PrimaryPos::Conj => vec!["Conj".to_string()],
                    PrimaryPos::Postp => vec!["Postp".to_string()],
                    PrimaryPos::Interj => vec!["Interj".to_string()],
                    PrimaryPos::Q => vec!["Q".to_string()],
                    PrimaryPos::Punc => vec!["Punc".to_string()],
                    PrimaryPos::Dup => vec!["Dup".to_string()],
                    PrimaryPos::Unknown => vec!["Unk".to_string()],
                };

                let formatted = format!("[{}:{}] {}:{}", item.lemma, item.primary_pos.as_str(), item.root, tags.join("+"));
                results.push(MorphParse {
                    surface: clean.to_string(),
                    lemma: item.lemma.clone(),
                    root: item.root.clone(),
                    primary_pos: item.primary_pos,
                    secondary_pos: item.secondary_pos,
                    morpheme_tags: tags,
                    formatted,
                    score: 1.0,
                });
            }
        }

        // 2. Multi-affix morphological traversal
        let char_count = lower.chars().count();
        for stem_len in 1..=char_count {
            let candidate_stem: String = lower.chars().take(stem_len).collect();
            let suffix_part: String = lower.chars().skip(stem_len).collect();

            if suffix_part.is_empty() {
                continue;
            }

            // a) Direct match
            self.match_and_traverse(&candidate_stem, &suffix_part, clean, &mut results, false, false, false);

            // b) Consonant softening / mutation (e.g. "kitab" from "kitap", "ağac" from "ağaç", "kanad" from "kanat", "ayağ" from "ayak")
            let hardened_stem = self.unsoften_stem(&candidate_stem);
            if hardened_stem != candidate_stem {
                self.match_and_traverse(&hardened_stem, &suffix_part, clean, &mut results, true, false, false);
            }

            // c) Vowel drop (e.g. "burn" from "burun", "akl" from "akıl", "şehr" from "şehir")
            let restored_vowel_stems = self.unvoweldrop_stems(&candidate_stem);
            for restored in restored_vowel_stems {
                self.match_and_traverse(&restored, &suffix_part, clean, &mut results, false, true, false);
            }

            // d) Consonant doubling (e.g. "hakk" from "hak", "hiss" from "his")
            let undoubled = self.undouble_stem(&candidate_stem);
            if let Some(undoubled_root) = undoubled {
                self.match_and_traverse(&undoubled_root, &suffix_part, clean, &mut results, false, false, true);
            }

            // e) Diminutive k-drop (e.g. "küçü" from "küçük", "mini" from "minik", "sıca" from "sıcak")
            let with_k = format!("{}k", candidate_stem);
            self.match_and_traverse(&with_k, &suffix_part, clean, &mut results, false, false, false);
        }

        // Deduplicate results and sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.formatted.cmp(&b.formatted)));
        results.dedup_by(|a, b| a.formatted == b.formatted);

        results
    }

    fn unsoften_stem(&self, stem: &str) -> String {
        if stem.is_empty() {
            return String::new();
        }
        let mut chars: Vec<char> = stem.chars().collect();
        let len = chars.len();
        match chars[len - 1] {
            'b' => chars[len - 1] = 'p',
            'c' => chars[len - 1] = 'ç',
            'd' => chars[len - 1] = 't',
            'ğ' | 'g' => chars[len - 1] = 'k',
            _ => {}
        }
        chars.into_iter().collect()
    }

    fn unvoweldrop_stems(&self, stem: &str) -> Vec<String> {
        let chars: Vec<char> = stem.chars().collect();
        let len = chars.len();
        if len < 2 {
            return Vec::new();
        }
        let mut candidates = Vec::new();
        let last = chars[len - 1];
        let prev = chars[len - 2];
        if !super::super::phonology::is_turkish_vowel(last) && !super::super::phonology::is_turkish_vowel(prev) {
            for &v in &['ı', 'i', 'u', 'ü'] {
                let mut s: Vec<char> = chars[..len - 1].to_vec();
                s.push(v);
                s.push(last);
                candidates.push(s.into_iter().collect());
            }
        }
        candidates
    }

    fn undouble_stem(&self, stem: &str) -> Option<String> {
        let chars: Vec<char> = stem.chars().collect();
        let len = chars.len();
        if len >= 3 && chars[len - 1] == chars[len - 2] && !super::super::phonology::is_turkish_vowel(chars[len - 1]) {
            let s: String = chars[..len - 1].iter().collect();
            Some(s)
        } else {
            None
        }
    }

    fn match_and_traverse(
        &self,
        root_cand: &str,
        suffix_str: &str,
        original_surface: &str,
        results: &mut Vec<MorphParse>,
        was_voiced: bool,
        was_vowel_dropped: bool,
        was_doubled: bool,
    ) {
        if let Some(items) = self.lexicon.get_items(root_cand) {
            for item in items {
                // Verify attributes
                if was_voiced && !item.attributes.contains(RootAttr::VOICING) {
                    continue;
                }
                if was_vowel_dropped && !item.attributes.contains(RootAttr::VOWEL_DROP) {
                    continue;
                }
                if was_doubled && !item.attributes.contains(RootAttr::CONSONANT_DOUBLING) {
                    continue;
                }

                let start_state = match item.primary_pos {
                    PrimaryPos::Noun => MorphState::NounRoot,
                    PrimaryPos::Verb => MorphState::VerbRoot,
                    _ => MorphState::NounRoot,
                };

                let mut current_tags = match item.primary_pos {
                    PrimaryPos::Noun => vec!["Noun".to_string()],
                    PrimaryPos::Verb => vec!["Verb".to_string()],
                    PrimaryPos::Adj => vec!["Adj".to_string()],
                    PrimaryPos::Adv => vec!["Adv".to_string()],
                    PrimaryPos::Pron => vec!["Pron".to_string()],
                    _ => vec![item.primary_pos.as_str().to_string()],
                };

                self.search_suffix_chain(
                    start_state,
                    suffix_str,
                    item,
                    original_surface,
                    &mut current_tags,
                    results,
                    0,
                );
            }
        }
    }

    fn search_suffix_chain(
        &self,
        state: MorphState,
        remaining_suffix: &str,
        item: &DictionaryItem,
        original_surface: &str,
        tags: &mut Vec<String>,
        results: &mut Vec<MorphParse>,
        depth: usize,
    ) {
        if depth > 10 {
            return;
        }

        if remaining_suffix.is_empty() {
            let formatted = format!("[{}:{}] {}:{}", item.lemma, item.primary_pos.as_str(), item.root, tags.join("+"));
            let is_upper = original_surface.chars().next().map_or(false, |c| c.is_uppercase());
            let score = if item.secondary_pos == SecondaryPos::ProperNoun {
                if is_upper { 0.98 } else { 0.6 }
            } else {
                0.95
            };

            results.push(MorphParse {
                surface: original_surface.to_string(),
                lemma: item.lemma.clone(),
                root: item.root.clone(),
                primary_pos: item.primary_pos,
                secondary_pos: item.secondary_pos,
                morpheme_tags: tags.clone(),
                formatted,
                score,
            });
            return;
        }

        for tr in &self.transitions {
            if tr.from_state == state {
                for &template in tr.surface_templates {
                    if remaining_suffix.starts_with(template) {
                        let next_remaining = &remaining_suffix[template.len()..];
                        let tag_str = tr.suffix_type.tag().to_string();
                        tags.push(tag_str);

                        self.search_suffix_chain(
                            tr.to_state,
                            next_remaining,
                            item,
                            original_surface,
                            tags,
                            results,
                            depth + 1,
                        );

                        tags.pop();
                    }
                }
            }
        }
    }
}
