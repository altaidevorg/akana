//! Plain Turkish lexicons shared by several modules: the place gazetteer (the 81 provinces,
//! major districts, places abroad) and the person-name lexicon (given names, surnames,
//! demonyms), with their lookups.
//!
//! The module is built in every configuration, without `stringzilla`: the phonemiser
//! (`speech`), the dependency parser and NER read it. `pii` re-exports the sets from
//! `pii::gazetteer`, where they used to live, and builds its name detector on the same lists.
//!
//! The lookups take a lower-cased (Turkish rules) word without a suffix. Every set also holds
//! the entries' ASCII-folded variants (Şule and Sule).

mod gazetteer;
#[rustfmt::skip]
pub(crate) mod names_data;

pub use gazetteer::{DISTRICTS, GIVEN_NAMES, POLYSEMOUS_NAMES, PROVINCES, SURNAMES};

use crate::normalization::TurkishAsciifier;
use lazy_static::lazy_static;
use names_data::{
    AMBIGUOUS_NAMES, COMMON_GIVEN_NAMES, COMMON_SURNAMES, DEMONYMS, FOREIGN_PLACES,
    INTERNATIONAL_GIVEN_NAMES, RARE_NAMES,
};
use std::collections::HashSet;

/// Zemberek's person-name dictionary, one lower-case name a line.
pub(crate) const PERSON_NAMES_TXT: &str = include_str!("person_names.txt");

/// Inserts each item and, when it differs, its ASCII-folded form.
pub(crate) fn insert_with_ascii(set: &mut HashSet<String>, items: &[&str]) {
    for &item in items {
        set.insert(item.to_string());
        let asc = TurkishAsciifier::asciify(item);
        if asc != item {
            set.insert(asc);
        }
    }
}

/// Each item followed, when it differs, by its ASCII-folded form.
pub(crate) fn with_ascii<'a>(items: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let mut out = Vec::new();
    for item in items {
        out.push(item.to_string());
        let ascii = TurkishAsciifier::asciify(item);
        if ascii != item {
            out.push(ascii);
        }
    }
    out
}

lazy_static! {
    /// The curated Turkish given names, without places and the names more often something
    /// else (RARE_NAMES): no analysis is needed to build it, as the PII lexicon's is.
    static ref PERSON_GIVEN_NAMES: HashSet<String> = {
        let rare: HashSet<String> = with_ascii(RARE_NAMES.iter().copied()).into_iter().collect();
        GIVEN_NAMES
            .iter()
            .cloned()
            .chain(with_ascii(COMMON_GIVEN_NAMES.iter().copied()))
            .filter(|n| !rare.contains(n) && !PROVINCES.contains(n) && !DISTRICTS.contains(n))
            .collect()
    };

    /// Every given-name form of the person-name lexicon: the curated Turkish and the
    /// international given names (single words), the names that are also everyday words, and
    /// Zemberek's person names. These are the keys of the PII name detector's lexicon, which
    /// adds to each how much evidence it needs (`pii::names`; a test there checks that the two
    /// agree).
    pub(crate) static ref GIVEN_NAME_FORMS: HashSet<String> = {
        let mut forms: HashSet<String> = GIVEN_NAMES
            .iter()
            .cloned()
            .chain(with_ascii(COMMON_GIVEN_NAMES.iter().copied()))
            .chain(with_ascii(INTERNATIONAL_GIVEN_NAMES.iter().copied()))
            .filter(|n| !n.contains(' '))
            .chain(POLYSEMOUS_NAMES.iter().cloned())
            .chain(with_ascii(AMBIGUOUS_NAMES.iter().copied()))
            .chain(with_ascii(RARE_NAMES.iter().copied()))
            .collect();
        for line in PERSON_NAMES_TXT.lines() {
            let name = line.trim();
            if name.is_empty() || name.starts_with('#') || forms.contains(name) {
                continue;
            }
            forms.extend(with_ascii([name]));
        }
        forms
    };

    /// Every surname form of the person-name lexicon.
    pub(crate) static ref SURNAME_FORMS: HashSet<String> = SURNAMES
        .iter()
        .cloned()
        .chain(with_ascii(COMMON_SURNAMES.iter().copied()))
        .collect();
}

/// Whether `lower` (lower-cased) is a Turkish given name that is a person's more than a
/// place's (Ahmet, Mustafa, Deniz; not Ankara or Güneş): the dependency parser's test for a
/// person's name (review 4 DOWN4-07).
pub fn is_given_name(lower: &str) -> bool {
    PERSON_GIVEN_NAMES.contains(lower)
}

/// Whether `lower` (lower-cased, without a suffix) is a person's given name or surname in the
/// person-name lexicon, and no nationality: NER's person names (the lexicon's other proper
/// nouns, places among them, are none).
pub fn is_person_name(lower: &str) -> bool {
    (GIVEN_NAME_FORMS.contains(lower) || SURNAME_FORMS.contains(lower))
        && !DEMONYMS.contains(&lower)
}

/// Whether `lower` (lower-cased, without a suffix) is a surname of the person-name lexicon.
pub fn is_surname(lower: &str) -> bool {
    SURNAME_FORMS.contains(lower)
}

/// Whether `lower` (lower-cased, without a suffix) is a nationality or demonym (Türk, Alman).
pub fn is_demonym(lower: &str) -> bool {
    DEMONYMS.contains(&lower)
}

/// Whether `lower` (lower-cased, without a suffix) is a place: a province, a district, or a
/// country, region or city abroad.
pub fn is_place_name(lower: &str) -> bool {
    PROVINCES.contains(lower) || DISTRICTS.contains(lower) || FOREIGN_PLACES.contains(&lower)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookups_read_the_lists_and_their_ascii_forms() {
        assert!(is_place_name("ankara") && is_place_name("kadıköy") && is_place_name("almanya"));
        assert!(is_place_name("eskisehir"), "ASCII-folded province");
        assert!(!is_place_name("masa"));
        assert!(is_given_name("mehmet") && is_given_name("sule"));
        assert!(!is_given_name("ankara") && !is_given_name("güneş"));
        assert!(is_person_name("yılmaz") && is_person_name("deniz"));
        assert!(!is_person_name("türk"), "a demonym is no person's name");
        assert!(is_surname("öztürk") && !is_surname("ahmet"));
        assert!(is_demonym("alman") && !is_demonym("almanya"));
    }
}
