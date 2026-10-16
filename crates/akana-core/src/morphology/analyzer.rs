//! Turkish Morphological Analyzer.

use super::dictionary::{DictionaryItem, RootLexicon};
use super::graph::{MorphState, SuffixTransition, TurkishMorphotactics};
use super::pos::{PrimaryPos, RootAttr, SecondaryPos};
use super::suffixes::SuffixType;
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
    /// POS of the dictionary root (Zemberek's `DictionaryItem.primaryPos`).
    pub primary_pos: PrimaryPos,
    /// POS of the word after its last derivation (Zemberek's `SingleAnalysis.getPos`):
    /// baş-la-yınca is `Adv`, oku-duğ-u is `Noun`, an-laş-ıyor-um is `Verb`. Unlike
    /// Zemberek, the copula does not change it: mutlu-y-duk stays `Adj`.
    #[serde(default = "unknown_pos")]
    pub pos: PrimaryPos,
    pub secondary_pos: SecondaryPos,
    pub morpheme_tags: Vec<String>,
    pub formatted: String,
    pub score: f32,
}

fn unknown_pos() -> PrimaryPos {
    PrimaryPos::Unknown
}

/// The passive allomorph after a stem ending in `prev`, as Zemberek's Passive_In / +nIl / Il
/// conditions: -n after a vowel (oku-n), -In after l (bil-in), -Il after other consonants
/// (yap-ıl), -nIl after a vowel (de-nil). A bare -l is never a passive.
fn passive_allomorph_fits(template: &str, prev: char) -> bool {
    let after_vowel = super::super::phonology::is_turkish_vowel(prev);
    match template {
        "n" | "nil" | "nıl" => after_vowel,
        "in" | "ın" | "un" | "ün" => prev == 'l',
        "il" | "ıl" | "ul" | "ül" => !after_vowel && prev != 'l',
        _ => false,
    }
}

/// How a candidate stem was recovered from the surface; the root must allow it.
#[derive(Clone, Copy)]
enum StemChange {
    None,
    /// kitab- from kitap (RootAttr::VOICING)
    Voiced,
    /// burn- from burun (RootAttr::VOWEL_DROP)
    VowelDropped,
    /// hakk- from hak (RootAttr::CONSONANT_DOUBLING)
    Doubled,
    /// bekl-iyor from bekle (any vowel-final verb)
    ProgressiveVowelDrop,
    /// di-yecek from de, yi-yen from ye
    DeYeNarrowing,
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
    pub fn load_dictionary_file<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> std::io::Result<()> {
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

        // Lexicon roots carry no circumflex (nikâh -> nikah), as in Zemberek's
        // normalizeForAnalysis.
        let lower: String = super::super::phonology::to_turkish_lower(clean)
            .chars()
            .map(|c| match c {
                'â' => 'a',
                'î' => 'i',
                'û' => 'u',
                '’' => '\'',
                _ => c,
            })
            .collect();

        match lower.split_once('\'') {
            Some((stem, ending)) if !stem.is_empty() && !ending.is_empty() => {
                self.analyze_with_apostrophe(clean, stem, ending)
            }
            _ => self.analyze_normalized(clean, &lower),
        }
    }

    /// Zemberek's analyzeWordsWithApostrophe: analyse the word without the quote and keep the
    /// noun parses whose root is the part before it (İstanbul'un, Antalya'da). A name that is
    /// not in the lexicon gets a proper-noun root, as Zemberek's unidentified-token analyzer
    /// does for words with an apostrophe.
    fn analyze_with_apostrophe(&self, clean: &str, stem: &str, ending: &str) -> Vec<MorphParse> {
        let joined = format!("{stem}{ending}");
        let parses: Vec<MorphParse> = self
            .analyze_normalized(clean, &joined)
            .into_iter()
            .filter(|p| p.primary_pos == PrimaryPos::Noun && p.root == stem)
            .collect();
        if !parses.is_empty() {
            return parses;
        }
        let guessed = DictionaryItem::new(
            stem,
            stem,
            PrimaryPos::Noun,
            SecondaryPos::ProperNoun,
            RootAttr::PROPER_NOUN,
        );
        let mut results = Vec::new();
        let mut tags = vec!["Noun".to_string()];
        self.search_suffix_chain(
            MorphState::NounRoot,
            PrimaryPos::Noun,
            stem.chars().last().unwrap_or(' '),
            ending,
            &guessed,
            clean,
            &mut tags,
            &mut results,
            0,
        );
        results
    }

    fn analyze_normalized(&self, clean: &str, lower: &str) -> Vec<MorphParse> {
        let mut results = Vec::new();

        // 1. Direct match in dictionary (uninflected base form)
        if let Some(items) = self.lexicon.get_items(lower) {
            for item in items {
                let tags = match item.primary_pos {
                    PrimaryPos::Noun => {
                        vec!["Noun".to_string(), "A3sg".to_string(), "Nom".to_string()]
                    }
                    // A bare verb stem is the 2nd-person imperative (gel!), as in Zemberek.
                    PrimaryPos::Verb => {
                        vec!["Verb".to_string(), "Imp".to_string(), "A2sg".to_string()]
                    }
                    PrimaryPos::Adj => vec!["Adj".to_string()],
                    PrimaryPos::Adv => vec!["Adv".to_string()],
                    PrimaryPos::Pron => {
                        vec!["Pron".to_string(), "A3sg".to_string(), "Nom".to_string()]
                    }
                    PrimaryPos::Num => vec!["Num".to_string()],
                    PrimaryPos::Conj => vec!["Conj".to_string()],
                    PrimaryPos::Postp => vec!["Postp".to_string()],
                    PrimaryPos::Interj => vec!["Interj".to_string()],
                    PrimaryPos::Q => vec!["Q".to_string()],
                    PrimaryPos::Punc => vec!["Punc".to_string()],
                    PrimaryPos::Dup => vec!["Dup".to_string()],
                    PrimaryPos::Unknown => vec!["Unk".to_string()],
                    PrimaryPos::Det => vec!["Det".to_string()],
                };

                let formatted = format!(
                    "[{}:{}] {}:{}",
                    item.lemma,
                    item.primary_pos.as_str(),
                    item.root,
                    tags.join("+")
                );
                // As for inflected readings below: a proper noun or abbreviation written
                // lower-case (the corpus "proper noun" beklerken) is an unlikely reading.
                let is_upper = clean.chars().next().is_some_and(|c| c.is_uppercase());
                let lower_case_proper = !is_upper
                    && matches!(
                        item.secondary_pos,
                        SecondaryPos::ProperNoun | SecondaryPos::Abbreviation
                    );
                results.push(MorphParse {
                    surface: clean.to_string(),
                    lemma: item.lemma.clone(),
                    root: item.root.clone(),
                    primary_pos: item.primary_pos,
                    pos: item.primary_pos,
                    secondary_pos: item.secondary_pos,
                    morpheme_tags: tags,
                    formatted,
                    // The imperative is a (zero) suffix like any other, so it is scored as an
                    // inflected reading: işe is iş-e before it is the imperative of işemek.
                    score: if lower_case_proper {
                        0.65
                    } else if item.primary_pos == PrimaryPos::Verb {
                        0.95
                    } else {
                        1.0
                    },
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
            self.match_and_traverse(
                &candidate_stem,
                &suffix_part,
                clean,
                &mut results,
                StemChange::None,
            );

            // b) Consonant softening / mutation (e.g. "kitab" from "kitap", "ağac" from "ağaç", "kanad" from "kanat", "ayağ" from "ayak")
            let hardened_stem = self.unsoften_stem(&candidate_stem);
            if hardened_stem != candidate_stem {
                self.match_and_traverse(
                    &hardened_stem,
                    &suffix_part,
                    clean,
                    &mut results,
                    StemChange::Voiced,
                );
            }

            // c) Vowel drop (e.g. "burn" from "burun", "akl" from "akıl", "şehr" from "şehir")
            let restored_vowel_stems = self.unvoweldrop_stems(&candidate_stem);
            for restored in restored_vowel_stems {
                self.match_and_traverse(
                    &restored,
                    &suffix_part,
                    clean,
                    &mut results,
                    StemChange::VowelDropped,
                );
            }

            // d) Consonant doubling (e.g. "hakk" from "hak", "hiss" from "his")
            let undoubled = self.undouble_stem(&candidate_stem);
            if let Some(undoubled_root) = undoubled {
                self.match_and_traverse(
                    &undoubled_root,
                    &suffix_part,
                    clean,
                    &mut results,
                    StemChange::Doubled,
                );
            }

            // e) Diminutive k-drop (e.g. "küçü" from "küçük", "sıca" from "sıcak"), only before
            // the diminutive -cIk itself: anywhere else it invented roots (ben -> bek-n,
            // diyor -> dik-yor, konuda -> konuk-da). Zemberek has no such rule at all.
            if suffix_part.starts_with(['c', 'ç']) {
                let with_k = format!("{candidate_stem}k");
                self.match_and_traverse(
                    &with_k,
                    &suffix_part,
                    clean,
                    &mut results,
                    StemChange::None,
                );
            }

            // g) de/ye narrow to di/yi before a y-initial suffix (di-yecek, di-yerek, yi-yen),
            // Zemberek's deYeCondition.
            if matches!(candidate_stem.as_str(), "di" | "yi") && suffix_part.starts_with('y') {
                let root = if candidate_stem == "di" { "de" } else { "ye" };
                self.match_and_traverse(
                    root,
                    &suffix_part,
                    clean,
                    &mut results,
                    StemChange::DeYeNarrowing,
                );
            }

            // f) Progressive vowel drop: a vowel-final verb stem loses its vowel before -Iyor
            // (bekle-iyor -> bekliyor, başla-ıyor -> başlıyor, de-iyor -> diyor), Zemberek's
            // RootAttribute.ProgressiveVowelDrop.
            if ["iyor", "ıyor", "uyor", "üyor"]
                .iter()
                .any(|p| suffix_part.starts_with(p))
            {
                for vowel in ['a', 'e'] {
                    self.match_and_traverse(
                        &format!("{candidate_stem}{vowel}"),
                        &suffix_part,
                        clean,
                        &mut results,
                        StemChange::ProgressiveVowelDrop,
                    );
                }
            }
        }

        // Sort by score descending, then drop repeats of the same reading wherever they sit,
        // keeping the best-scored one. A reading is its analysis string plus its secondary POS:
        // the string does not show it, so deduplicating on the string alone (adjacent only)
        // dropped Ekmek's proper-noun reading when it tied with the common noun and kept such
        // pairs when they did not tie.
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.formatted.cmp(&b.formatted))
        });
        let mut seen = std::collections::HashSet::new();
        results.retain(|p| seen.insert((p.formatted.clone(), p.secondary_pos)));

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
        if !super::super::phonology::is_turkish_vowel(last)
            && !super::super::phonology::is_turkish_vowel(prev)
        {
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
        if len >= 3
            && chars[len - 1] == chars[len - 2]
            && !super::super::phonology::is_turkish_vowel(chars[len - 1])
        {
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
        change: StemChange,
    ) {
        if let Some(items) = self.lexicon.get_items(root_cand) {
            for item in items {
                // Verify attributes
                let allowed = match change {
                    StemChange::None => true,
                    StemChange::Voiced => item.attributes.contains(RootAttr::VOICING),
                    StemChange::VowelDropped => item.attributes.contains(RootAttr::VOWEL_DROP),
                    StemChange::Doubled => item.attributes.contains(RootAttr::CONSONANT_DOUBLING),
                    StemChange::ProgressiveVowelDrop | StemChange::DeYeNarrowing => {
                        item.primary_pos == PrimaryPos::Verb
                    }
                };
                if !allowed {
                    continue;
                }
                // Closed classes take no suffixes (Zemberek's terminal adverb, conjunction,
                // interjection, determiner, duplicator and postposition root states); they
                // used to inflect like nouns (yo-la, he-m, fır-ın-ın).
                if matches!(
                    item.primary_pos,
                    PrimaryPos::Adv
                        | PrimaryPos::Conj
                        | PrimaryPos::Interj
                        | PrimaryPos::Det
                        | PrimaryPos::Dup
                        | PrimaryPos::Postp
                        | PrimaryPos::Punc
                ) {
                    continue;
                }

                let start_state = match item.primary_pos {
                    // "değil" carries the negation and behaves like a nominal predicate:
                    // değil-im, değil-di, değil-se (Zemberek's nVerbDegil_S).
                    PrimaryPos::Verb if item.root == "değil" => MorphState::NounCase,
                    PrimaryPos::Noun => MorphState::NounRoot,
                    PrimaryPos::Verb => MorphState::VerbRoot,
                    PrimaryPos::Q => MorphState::QuesRoot,
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
                    item.primary_pos,
                    root_cand.chars().last().unwrap_or(' '),
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

    /// `prev` is the last letter matched so far (root or previous suffix), which conditions
    /// allomorphs such as the passive.
    #[allow(clippy::too_many_arguments)]
    fn search_suffix_chain(
        &self,
        state: MorphState,
        pos: PrimaryPos,
        prev: char,
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
            let formatted = format!(
                "[{}:{}] {}:{}",
                item.lemma,
                item.primary_pos.as_str(),
                item.root,
                tags.join("+")
            );
            // Turkish spelling separates the suffixes of a proper noun or abbreviation with an
            // apostrophe (Ayşe'nin, TBMM'de), so an inflected reading of one written without it
            // (k-ı "ki", Anne-m "Annem") is unlikely.
            let has_apostrophe = original_surface.contains(['\'', '’']);
            let score = if matches!(
                item.secondary_pos,
                SecondaryPos::ProperNoun | SecondaryPos::Abbreviation
            ) {
                if has_apostrophe {
                    0.98
                } else {
                    0.6
                }
            } else {
                0.95
            };

            results.push(MorphParse {
                surface: original_surface.to_string(),
                lemma: item.lemma.clone(),
                root: item.root.clone(),
                primary_pos: item.primary_pos,
                pos,
                secondary_pos: item.secondary_pos,
                morpheme_tags: tags.clone(),
                formatted,
                score,
            });
            return;
        }

        for tr in &self.transitions {
            if tr.from_state == state {
                // -CA is the adverbial Ly after an adjective (açık-ça) and the equative case
                // otherwise (aile-ce, sen-ce), as in Zemberek.
                let skip = match tr.suffix_type {
                    SuffixType::DerivLy => pos != PrimaryPos::Adj,
                    SuffixType::CaseEqu => pos == PrimaryPos::Adj,
                    // Lexical deverbal nouns and adjectives (seç-im, sev-gi, yor-gun,
                    // çalış-kan) attach to the bare root, not after voice or other suffixes
                    // (kork-t-um, değ-il-im).
                    SuffixType::DerivIm
                    | SuffixType::DerivGi
                    | SuffixType::DerivGin
                    | SuffixType::DerivGen => tags.len() > 1,
                    _ => false,
                };
                if skip {
                    continue;
                }
                // A derivation opens a new inflectional group; nominal inflection on an
                // adjective is Zemberek's zero derivation to a noun.
                let next_pos = match tr.suffix_type.derived_pos() {
                    Some(derived) => derived,
                    None if pos == PrimaryPos::Adj && tr.suffix_type.is_nominal_inflection() => {
                        PrimaryPos::Noun
                    }
                    None => pos,
                };
                for &template in tr.surface_templates {
                    if tr.suffix_type == SuffixType::DerivPassive
                        && !passive_allomorph_fits(template, prev)
                    {
                        continue;
                    }
                    if let Some(next_remaining) = remaining_suffix.strip_prefix(template) {
                        let tag_str = tr.suffix_type.tag().to_string();
                        tags.push(tag_str);

                        self.search_suffix_chain(
                            tr.to_state,
                            next_pos,
                            template.chars().last().unwrap_or(prev),
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
