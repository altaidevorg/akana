//! Morphological generator for Turkish words given root/lemma and inflectional tags.

use super::dictionary::RootLexicon;
use super::pos::RootAttr;

pub struct TurkishGenerator {
    lexicon: RootLexicon,
}

impl Default for TurkishGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl TurkishGenerator {
    pub fn new() -> Self {
        Self {
            lexicon: RootLexicon::new(),
        }
    }

    /// Generates the surface form for a lemma and a sequence of morpheme tags.
    /// Example: `generate("kitap", &["Noun", "A3sg", "P1sg", "Dat"]) -> "kitabıma"`
    pub fn generate(&self, lemma: &str, tags: &[&str]) -> Option<String> {
        let items = self.lexicon.get_items(lemma)?;
        let item = items.first()?;

        let mut surface = item.root.clone();
        let is_voicing = item.attributes.contains(RootAttr::VOICING);
        let is_vowel_drop = item.attributes.contains(RootAttr::VOWEL_DROP);
        let is_doubling = item.attributes.contains(RootAttr::CONSONANT_DOUBLING);
        let is_inverse_harmony = item.attributes.contains(RootAttr::INVERSE_HARMONY);

        let mut has_applied_stem_mutation = false;

        for &tag in tags {
            match tag {
                // Nominal Plural
                "A3pl" => {
                    let a_vowel = if is_inverse_harmony { 'e' } else { super::super::phonology::get_a_type_harmonic_vowel(&surface) };
                    surface.push_str(if a_vowel == 'e' { "ler" } else { "lar" });
                }
                // Possessive 1sg
                "P1sg" => {
                    if !has_applied_stem_mutation {
                        surface = self.apply_stem_mutation(&surface, is_voicing, is_vowel_drop, is_doubling);
                        has_applied_stem_mutation = true;
                    }
                    let last_c = surface.chars().last().unwrap_or(' ');
                    if super::super::phonology::is_turkish_vowel(last_c) {
                        surface.push('m');
                    } else {
                        let i_vowel = if is_inverse_harmony { 'i' } else { super::super::phonology::get_i_type_harmonic_vowel(&surface) };
                        surface.push(i_vowel);
                        surface.push('m');
                    }
                }
                // Possessive 2sg
                "P2sg" => {
                    if !has_applied_stem_mutation {
                        surface = self.apply_stem_mutation(&surface, is_voicing, is_vowel_drop, is_doubling);
                        has_applied_stem_mutation = true;
                    }
                    let last_c = surface.chars().last().unwrap_or(' ');
                    if super::super::phonology::is_turkish_vowel(last_c) {
                        surface.push('n');
                    } else {
                        let i_vowel = if is_inverse_harmony { 'i' } else { super::super::phonology::get_i_type_harmonic_vowel(&surface) };
                        surface.push(i_vowel);
                        surface.push('n');
                    }
                }
                // Possessive 3sg
                "P3sg" => {
                    if !has_applied_stem_mutation {
                        surface = self.apply_stem_mutation(&surface, is_voicing, is_vowel_drop, is_doubling);
                        has_applied_stem_mutation = true;
                    }
                    let last_c = surface.chars().last().unwrap_or(' ');
                    let i_vowel = if is_inverse_harmony { 'i' } else { super::super::phonology::get_i_type_harmonic_vowel(&surface) };
                    if super::super::phonology::is_turkish_vowel(last_c) {
                        surface.push('s');
                    }
                    surface.push(i_vowel);
                }
                // Dative Case
                "Dat" => {
                    if !has_applied_stem_mutation {
                        surface = self.apply_stem_mutation(&surface, is_voicing, is_vowel_drop, is_doubling);
                        has_applied_stem_mutation = true;
                    }
                    let last_c = surface.chars().last().unwrap_or(' ');
                    let a_vowel = if is_inverse_harmony { 'e' } else { super::super::phonology::get_a_type_harmonic_vowel(&surface) };
                    if super::super::phonology::is_turkish_vowel(last_c) {
                        surface.push('y');
                    }
                    surface.push(a_vowel);
                }
                // Accusative Case
                "Acc" => {
                    if !has_applied_stem_mutation {
                        surface = self.apply_stem_mutation(&surface, is_voicing, is_vowel_drop, is_doubling);
                        has_applied_stem_mutation = true;
                    }
                    let last_c = surface.chars().last().unwrap_or(' ');
                    let i_vowel = if is_inverse_harmony { 'i' } else { super::super::phonology::get_i_type_harmonic_vowel(&surface) };
                    if super::super::phonology::is_turkish_vowel(last_c) {
                        surface.push('y');
                    }
                    surface.push(i_vowel);
                }
                // Locative Case
                "Loc" => {
                    let last_c = surface.chars().last().unwrap_or(' ');
                    let is_hard = super::super::phonology::is_hard_consonant(last_c);
                    let a_vowel = if is_inverse_harmony { 'e' } else { super::super::phonology::get_a_type_harmonic_vowel(&surface) };
                    surface.push(if is_hard { 't' } else { 'd' });
                    surface.push(a_vowel);
                }
                // Ablative Case
                "Abl" => {
                    let last_c = surface.chars().last().unwrap_or(' ');
                    let is_hard = super::super::phonology::is_hard_consonant(last_c);
                    let a_vowel = if is_inverse_harmony { 'e' } else { super::super::phonology::get_a_type_harmonic_vowel(&surface) };
                    surface.push(if is_hard { 't' } else { 'd' });
                    surface.push(a_vowel);
                    surface.push('n');
                }
                // Genitive Case
                "Gen" => {
                    if !has_applied_stem_mutation {
                        surface = self.apply_stem_mutation(&surface, is_voicing, is_vowel_drop, is_doubling);
                        has_applied_stem_mutation = true;
                    }
                    let last_c = surface.chars().last().unwrap_or(' ');
                    let i_vowel = if is_inverse_harmony { 'i' } else { super::super::phonology::get_i_type_harmonic_vowel(&surface) };
                    if super::super::phonology::is_turkish_vowel(last_c) {
                        surface.push('n');
                    }
                    surface.push(i_vowel);
                    surface.push('n');
                }
                // Past Tense (Verbal)
                "Past" => {
                    let last_c = surface.chars().last().unwrap_or(' ');
                    let is_hard = super::super::phonology::is_hard_consonant(last_c);
                    let i_vowel = super::super::phonology::get_i_type_harmonic_vowel(&surface);
                    surface.push(if is_hard { 't' } else { 'd' });
                    surface.push(i_vowel);
                }
                // Progressive Tense (Verbal)
                "Prog" => {
                    if is_voicing {
                        surface = super::super::phonology::apply_stem_softening(&surface);
                    }
                    let last_c = surface.chars().last().unwrap_or(' ');
                    if super::super::phonology::is_turkish_vowel(last_c) {
                        surface.pop();
                    }
                    let i_vowel = super::super::phonology::get_i_type_harmonic_vowel(&surface);
                    surface.push(i_vowel);
                    surface.push_str("yor");
                }
                // Future Tense (Verbal)
                "Fut" => {
                    if is_voicing {
                        surface = super::super::phonology::apply_stem_softening(&surface);
                    }
                    let last_c = surface.chars().last().unwrap_or(' ');
                    let a_vowel = super::super::phonology::get_a_type_harmonic_vowel(&surface);
                    if super::super::phonology::is_turkish_vowel(last_c) {
                        surface.push('y');
                    }
                    surface.push(a_vowel);
                    surface.push_str(if a_vowel == 'e' { "cek" } else { "cak" });
                }
                // Verbal Agreement 1sg
                "A1sg" => {
                    let last_c = surface.chars().last().unwrap_or(' ');
                    if last_c == 'k' {
                        // Soften k -> ğ (e.g. gelecek -> geleceğim)
                        surface.pop();
                        surface.push('ğ');
                        let i_vowel = super::super::phonology::get_i_type_harmonic_vowel(&surface);
                        surface.push(i_vowel);
                        surface.push('m');
                    } else if surface.ends_with("yor") {
                        surface.push_str("um");
                    } else if last_c == 'ı' || last_c == 'i' || last_c == 'u' || last_c == 'ü' {
                        surface.push('m');
                    }
                }
                _ => {}
            }
        }

        Some(surface)
    }

    fn apply_stem_mutation(&self, stem: &str, is_voicing: bool, is_vowel_drop: bool, is_doubling: bool) -> String {
        if is_vowel_drop {
            if let Some(dropped) = super::super::phonology::apply_vowel_drop(stem) {
                if is_voicing {
                    return super::super::phonology::apply_stem_softening(&dropped);
                }
                return dropped;
            }
        }
        if is_doubling {
            return super::super::phonology::apply_consonant_doubling(stem);
        }
        if is_voicing {
            return super::super::phonology::apply_stem_softening(stem);
        }
        stem.to_string()
    }
}
