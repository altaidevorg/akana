//! Person names (AD): given names and surnames from word lists, and names after a title, a role,
//! a kinship word, a self-introduction or a form label, in up to four parts (Mehmet Ali Erbil),
//! hyphenated (Aydın-Özkan), with an initial (A. Yılmaz, John F. Kennedy), in capitals
//! (AHMET YILMAZ) or in lower case (ahmet yılmaz).
//!
//! Given names that are also everyday words (Deniz, Barış, Metin, Emin, Alev) are names only
//! with evidence: a title or role around them, a surname after them, an apostrophe suffix
//! (Deniz'e), or a capital in mid-sentence that nothing else explains (dün Deniz geldi). A
//! surname part may be a word the lexicon does not know (Gürbüzoğlu, Smith) or, last, a bare
//! noun (Orhan Pamuk); an inflected word (Kuvvetleri), a place, a month or an institution word
//! ends the name.
//!
//! Word lists: gazetteer.rs, names_data.rs, and person_names.txt (Zemberek's person-name
//! dictionary). Of the latter, names the morphology lexicon also has as common words (ahu,
//! alev, beste) count as ambiguous.

use super::gazetteer::{
    DISTRICTS, GIVEN_NAMES, POLYSEMOUS_NAMES, PROVINCES, SURNAMES, TITLE_TRIGGERS,
};
use super::morphology::split_stem_suffix;
use super::names_data::*;
use super::passwords::MORPHOLOGY;
use super::{ceil_char_boundary, floor_char_boundary, PiiEmbeddingScorer, PiiEntity, PiiType};
use crate::morphology::{MorphParse, SecondaryPos};
use crate::normalization::TurkishAsciifier;
use crate::phonology::to_turkish_lower;
use crate::tokenization::{Token, TurkishTokenizer};
use lazy_static::lazy_static;
use regex::Regex;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

const PERSON_NAMES_TXT: &str = include_str!("person_names.txt");

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// Only a name (Mehmet, Zeynep, Ecrin).
    Clear,
    /// Also an everyday word (Deniz, Metin, Barış): a name with evidence, which may be an
    /// apostrophe (Deniz'e) or a capital in mid-sentence (dün Deniz geldi).
    Ambiguous,
    /// More often something else (Güneş, Kuzey, Ege, Aydın, Fatih, Kanun): a name only next to
    /// a title, a role or another known name.
    Rare,
}

struct Lexicon {
    given: HashMap<String, Kind>,
    /// Turkish given names from the curated lists, the only ones taken in lower case.
    curated: HashSet<String>,
    surnames: HashSet<String>,
    honorifics: HashSet<String>,
    roles: HashSet<String>,
    org_roles: HashSet<String>,
    post_titles: HashSet<String>,
    stoppers: HashSet<String>,
    colon_triggers: HashSet<String>,
    self_intro: HashSet<String>,
    known_name_intro: HashSet<String>,
    particles: HashSet<String>,
    collisions: HashSet<String>,
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
}

/// Whether `lower` (lower-cased) is a Turkish given name that is a person's more than a
/// place's (Ahmet, Mustafa, Deniz; not Ankara or Güneş): the dependency parser's test for a
/// person's name (review 4 DOWN4-07).
pub(crate) fn is_given_name(lower: &str) -> bool {
    PERSON_GIVEN_NAMES.contains(lower)
}

fn with_ascii<'a>(items: impl IntoIterator<Item = &'a str>) -> Vec<String> {
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

/// The readings of `lower` as a common word: not as a proper noun or an abbreviation (the
/// lexicon has "alat" as both a surname and a+t).
fn common_readings(all: Vec<MorphParse>) -> Vec<MorphParse> {
    all.into_iter()
        .filter(|p| {
            !matches!(
                p.secondary_pos,
                SecondaryPos::ProperNoun | SecondaryPos::Abbreviation
            )
        })
        .collect()
}

/// What the lexicon makes of a word, from one analysis.
#[derive(Clone, Copy)]
struct Readings {
    /// It reads as a common word in some form (kuvvetleri, geldi, pamuk).
    common: bool,
    /// Every common reading carries a case, possessive or plural suffix (Kuvvetleri,
    /// Cumhuriyeti, Polikliniğine): an inflected noun, not a surname. Surnames that are common
    /// words are bare (Pamuk, Koçak), derived (Sarıca, Tütüncü) or verb forms (Solmaz, Yüksel).
    only_inflected: bool,
    /// The lexicon has it as an abbreviation as it stands (AİHM, OHAL, UYAP).
    abbreviation: bool,
    /// The lexicon has it as an adverb, postposition or conjunction as it stands (tarafından,
    /// hakkında, sonra).
    function_word: bool,
}

/// Whether `lower` can name a thing as it stands: a word the lexicon does not know (Hashimoto,
/// endometriozis), or one it has as a bare noun or adjective (migren, çiçek, sedef). Not a verb
/// form (bulunarak, çıkan, yitiren) or an inflected word (konusu, sebeple, başlangıçta), which
/// is what stands before "hastalığın" or "rahatsızlığı" in prose.
pub(super) fn is_bare_nominal_or_unknown(lower: &str) -> bool {
    use crate::morphology::PrimaryPos;
    let fold = |s: &str| -> String {
        s.chars()
            .map(|c| match c {
                'â' => 'a',
                'î' => 'i',
                'û' => 'u',
                _ => c,
            })
            .collect()
    };
    let parses = MORPHOLOGY.analyze(lower);
    let bare = fold(lower);
    parses.is_empty()
        || parses.iter().any(|p| {
            fold(&p.root) == bare
                && matches!(p.primary_pos, PrimaryPos::Noun | PrimaryPos::Adj)
                && matches!(
                    p.pos,
                    PrimaryPos::Noun | PrimaryPos::Adj | PrimaryPos::Unknown
                )
        })
}

/// Whether every reading of `lower` as a common word is an inflected form (kupası,
/// gerekçesiyle): see `Readings::only_inflected`.
pub(super) fn is_inflected_word(lower: &str) -> bool {
    readings(lower).only_inflected
}

fn readings(lower: &str) -> Readings {
    const INFLECTIONS: &[&str] = &[
        "A3pl", "P1sg", "P2sg", "P3sg", "P1pl", "P2pl", "P3pl", "Acc", "Dat", "Loc", "Abl", "Gen",
        "Ins",
    ];
    // A reading this far below the word's best says nothing about the word: tanıkla-r (Aor)
    // beside tanık-lar at one in ten thousand made Tanıklar a first name mid-sentence. Integer
    // rank units, so that ties cannot flip it (review 3 PII3-01).
    const NEGLIGIBLE: i32 = crate::morphology::rank_units(5.0);
    let all = MORPHOLOGY.analyze(lower);
    let abbreviation = all.iter().any(|p| {
        p.secondary_pos == SecondaryPos::Abbreviation
            && p.morpheme_tags
                .iter()
                .skip(1)
                .all(|t| matches!(t.as_str(), "A3sg" | "Pnon" | "Nom"))
    });
    let common = common_readings(all);
    let function_word = common.iter().any(|p| {
        p.morpheme_tags.len() == 1
            && matches!(
                p.primary_pos,
                crate::morphology::PrimaryPos::Adv
                    | crate::morphology::PrimaryPos::Postp
                    | crate::morphology::PrimaryPos::Conj
            )
    });
    let best = common.iter().map(|p| p.rank).max().unwrap_or(0);
    let plural = |p: &MorphParse| {
        common
            .iter()
            .any(|q| q.root == p.root && q.morpheme_tags.iter().any(|t| t == "A3pl"))
    };
    // The bare plural's copula counts as an inflection: every plural noun also reads as "they
    // are Ns" (davacı-lar CopPres3pl), which made Davacılar, Başvurucular a first name before the
    // name after them (review 2 PII2-03). Any other copula form does not: İzzet-ti-n "you were
    // honour", cin-di "it was a genie" are names (review 3 PII3-01).
    let inflected = |p: &MorphParse| {
        p.morpheme_tags
            .iter()
            .any(|t| INFLECTIONS.contains(&t.as_str()) || (t == "CopPres3pl" && plural(p)))
    };
    Readings {
        common: !common.is_empty(),
        only_inflected: !common.is_empty()
            && common
                .iter()
                .filter(|p| p.rank >= best - NEGLIGIBLE)
                .all(inflected),
        abbreviation,
        function_word,
    }
}

lazy_static! {
    static ref LEXICON: Lexicon = {
        let ambiguous: HashSet<String> = POLYSEMOUS_NAMES
            .iter()
            .cloned()
            .chain(with_ascii(AMBIGUOUS_NAMES.iter().copied()))
            .collect();
        let rare: HashSet<String> = with_ascii(RARE_NAMES.iter().copied()).into_iter().collect();
        let places = |w: &str| PROVINCES.contains(w) || DISTRICTS.contains(w);
        let kind_of = |w: &str, common: bool| {
            if rare.contains(w) || places(w) {
                Kind::Rare
            } else if ambiguous.contains(w) {
                Kind::Ambiguous
            } else if common {
                Kind::Rare
            } else {
                Kind::Clear
            }
        };
        let mut given = HashMap::new();
        let mut curated = HashSet::new();
        let turkish = GIVEN_NAMES
            .iter()
            .cloned()
            .chain(with_ascii(COMMON_GIVEN_NAMES.iter().copied()));
        let international = with_ascii(INTERNATIONAL_GIVEN_NAMES.iter().copied());
        for (name, is_turkish) in turkish
            .map(|n| (n, true))
            .chain(international.into_iter().map(|n| (n, false)))
            .filter(|(n, _)| !n.contains(' '))
        {
            let kind = kind_of(&name, false);
            if is_turkish {
                curated.insert(name.clone());
            }
            given.entry(name).or_insert(kind);
        }
        for name in ambiguous.iter().chain(&rare) {
            given.insert(name.clone(), kind_of(name, false));
        }
        for line in PERSON_NAMES_TXT.lines() {
            let name = line.trim();
            if name.is_empty() || name.starts_with('#') || given.contains_key(name) {
                continue;
            }
            let kind = kind_of(name, readings(name).common);
            for form in with_ascii([name]) {
                given.entry(form).or_insert(kind);
            }
        }
        let set = |items: &[&str]| -> HashSet<String> { with_ascii(items.iter().copied()).into_iter().collect() };
        let honorifics = set(HONORIFICS);
        let post_titles = set(POST_TITLES);
        let mut roles = set(NAME_TRIGGERS);
        for t in TITLE_TRIGGERS.iter() {
            let t = t.trim_end_matches('.');
            if !post_titles.contains(t) && !honorifics.contains(t) {
                roles.insert(t.to_string());
            }
        }
        // An ASCII-folded stopper that is a name stays a name: kürt folds to Kurt.
        let stoppers: HashSet<String> = NAME_STOPPERS
            .iter()
            .flat_map(|w| {
                let ascii = TurkishAsciifier::asciify(w);
                let named = given.contains_key(&ascii)
                    || SURNAMES.contains(&ascii)
                    || COMMON_SURNAMES.contains(&ascii.as_str());
                [Some(w.to_string()), (ascii != *w && !named).then_some(ascii)]
            })
            .flatten()
            .collect();
        Lexicon {
            given,
            curated,
            surnames: SURNAMES
                .iter()
                .cloned()
                .chain(with_ascii(COMMON_SURNAMES.iter().copied()))
                .collect(),
            honorifics,
            roles,
            org_roles: set(ORG_ROLES),
            post_titles,
            stoppers,
            colon_triggers: set(COLON_TRIGGERS),
            self_intro: set(SELF_INTRO),
            known_name_intro: set(KNOWN_NAME_INTRO),
            particles: NAME_PARTICLES.iter().map(|w| w.to_string()).collect(),
            collisions: set(LOWERCASE_COLLISIONS),
        }
    };

    /// Public-figure phrases, longest first, over Turkish lower-cased text. Word boundaries are
    /// checked in `public_figure_spans`: a Unicode `\b` keeps the regex off its fast engines,
    /// and with some 300 alternatives that cost 0.8 ms a text.
    static ref PUBLIC_FIGURE_REGEX: Regex = {
        let mut phrases: Vec<String> = with_ascii(PUBLIC_FIGURES.iter().copied())
            .into_iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        phrases.sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(b)));
        let alternatives: Vec<String> = phrases
            .iter()
            .map(|p| p.split_whitespace().map(regex::escape).collect::<Vec<_>>().join(r"\s+"))
            .collect();
        Regex::new(&format!(r"(?:{})", alternatives.join("|"))).unwrap()
    };
}

/// The letters of a token of initials with their periods (A.Ş., A.B., H.M.K.): AŞ, AB, HMK.
fn dotted_initials(text: &str) -> Option<String> {
    let body = text.strip_suffix('.')?;
    let letters: Vec<&str> = body.split('.').collect();
    (letters.len() > 1
        && letters
            .iter()
            .all(|l| l.chars().count() == 1 && l.chars().all(char::is_uppercase)))
    .then(|| letters.concat())
}

/// Whether a word in capitals is shaped like a surname rather than an acronym: four letters
/// or more and two vowels (TOPAL, AKINCI; not GGM, KYO, HAGB, UWW).
fn word_shaped_caps(w: &Word<'_>) -> bool {
    w.letters >= 4
        && w.lower
            .chars()
            .filter(|&c| crate::phonology::is_turkish_vowel(c))
            .count()
            >= 2
}

/// A word token: its stem before any apostrophe suffix, lower-cased, and its case.
struct Word<'a> {
    lower: String,
    suffix: Option<&'a str>,
    has_apostrophe: bool,
    start: usize,
    stem_end: usize,
    cap: bool,
    all_caps: bool,
    letters: usize,
}

impl<'a> Word<'a> {
    fn of(tok: &Token<'a>) -> Option<Self> {
        let clean = tok
            .text
            .trim_matches(|c: char| !c.is_alphabetic() && c != '\'' && c != '’');
        if clean.is_empty() {
            return None;
        }
        let offset = tok.text.find(clean).unwrap_or(0);
        let stemmed = split_stem_suffix(clean, tok.start + offset);
        let stem = stemmed.stem;
        if stem.is_empty() || !stem.chars().all(char::is_alphabetic) {
            return None;
        }
        let letters = stem.chars().count();
        Some(Self {
            lower: to_turkish_lower(stem),
            suffix: stemmed.suffix,
            has_apostrophe: stemmed.apostrophe.is_some(),
            start: stemmed.stem_start,
            stem_end: stemmed.stem_end,
            cap: stem.chars().next().is_some_and(char::is_uppercase),
            all_caps: letters > 1 && stem.chars().all(char::is_uppercase),
            letters,
        })
    }
}

enum Trigger {
    /// Sayın, Dr., Prof., Mr.: any capitalised word after it.
    Honorific,
    /// A form label before a colon (Ad Soyad:, İmza:, Name:).
    Colon,
    /// A role or kinship word (davacı, kiracı, kızım) or a self-introduction (adım, my name
    /// is); `true` for a role an organisation may hold (davacı, müşteri, üye).
    Role(bool),
    /// "ben", "adı", "this is": only a known given name after it.
    Ben,
}

struct Scan<'t, 'a> {
    text: &'t str,
    tokens: &'t [Token<'a>],
    words: Vec<Option<Word<'a>>>,
    lex: &'static Lexicon,
    /// `readings` of the words asked about, so each is analysed once per text.
    readings: RefCell<HashMap<String, Readings>>,
}

impl Scan<'_, '_> {
    fn readings(&self, lower: &str) -> Readings {
        if let Some(r) = self.readings.borrow().get(lower) {
            return *r;
        }
        let r = readings(lower);
        self.readings.borrow_mut().insert(lower.to_string(), r);
        r
    }

    fn is_common_word(&self, lower: &str) -> bool {
        self.readings(lower).common
    }

    fn only_inflected(&self, lower: &str) -> bool {
        self.readings(lower).only_inflected
    }

    fn is_abbreviation(&self, lower: &str) -> bool {
        self.readings(lower).abbreviation
    }

    fn word(&self, i: usize) -> Option<&Word<'_>> {
        self.words.get(i).and_then(Option::as_ref)
    }

    fn same_line(&self, a: usize, b: usize) -> bool {
        !self.text[self.tokens[a].end..self.tokens[b].start].contains('\n')
    }

    fn sentence_initial(&self, i: usize) -> bool {
        if i == 0 || !self.same_line(i - 1, i) {
            return true;
        }
        // after the letter or number of a heading or list item at the start of a line (A.
        // Adil yargılanma hakkı, 3. Başvurucu): review 4 DOWN4-03
        let marker = |t: &str| {
            let body = t.strip_suffix(['.', ')']).unwrap_or("");
            !body.is_empty()
                && (body.chars().all(|c| c.is_ascii_digit())
                    || (body.chars().count() == 1 && body.chars().all(char::is_alphabetic)))
        };
        if marker(self.tokens[i - 1].text) && (i == 1 || !self.same_line(i - 2, i - 1)) {
            return true;
        }
        matches!(
            self.tokens[i - 1].text,
            "." | "!"
                | "?"
                | "…"
                | ":"
                | "\""
                | "“"
                | "”"
                | "«"
                | "»"
                | "-"
                | "–"
                | "—"
                | "•"
                | "*"
                | "#"
                | ">"
                | "'"
        )
    }

    /// The last token of an initial at `i`, a single capital letter glued to a following
    /// period (the A of "A. Yılmaz"): `i` itself where the tokenizer keeps them together as an
    /// abbreviation ("A."), or the period's token after the letter.
    fn initial_end(&self, i: usize) -> Option<usize> {
        // Initials in one token (A.B. Yılmaz), but not a company's or the state's (A.Ş., T.C.);
        // the word after them must be a surname (`surname_like`): review 4 DOWN4-09.
        if let Some(letters) = dotted_initials(self.tokens[i].text) {
            return (!matches!(letters.as_str(), "AŞ" | "TC")).then_some(i);
        }
        let w = self.word(i)?;
        if w.letters != 1 || !w.cap || w.has_apostrophe {
            return None;
        }
        let token = &self.tokens[i];
        let end = if token.text.ends_with('.') {
            i
        } else if self
            .tokens
            .get(i + 1)
            .is_some_and(|t| t.text == "." && t.start == token.end)
        {
            i + 1
        } else {
            return None;
        };
        // Not a letter of dotted initials, which name a company or a state (A.Ş., T.C.): the
        // A of Hayat Sigorta A.Ş. is no initial (review 3 PII3-05).
        let glued_letter = |k: usize| {
            self.tokens.get(k).is_some_and(|t| {
                let mut chars = t.text.chars();
                chars.next().is_some_and(char::is_uppercase)
                    && matches!(chars.next(), Some('.') | None)
            })
        };
        let before = i > 0 && self.tokens[i - 1].end == token.start && glued_letter(i - 1);
        let after = self
            .tokens
            .get(end + 1)
            .is_some_and(|t| t.start == self.tokens[end].end)
            && glued_letter(end + 1);
        (!before && !after).then_some(end)
    }

    fn is_place(&self, lower: &str) -> bool {
        PROVINCES.contains(lower) || DISTRICTS.contains(lower)
    }

    fn is_named(&self, lower: &str) -> bool {
        self.lex.given.contains_key(lower) || self.lex.surnames.contains(lower)
    }

    /// A surname, or a given name that is only a name.
    fn is_clear_name(&self, lower: &str) -> bool {
        self.lex.surnames.contains(lower) || self.lex.given.get(lower) == Some(&Kind::Clear)
    }

    /// Whether the word at `j` is an institution or place word on the line of the word before
    /// it (the Üniversitesi of Yıldız Teknik Üniversitesi).
    fn institution_at(&self, j: usize) -> bool {
        j < self.tokens.len()
            && self.same_line(j - 1, j)
            && self.word(j).is_some_and(|w| {
                w.cap && self.is_stopper(&w.lower) && !self.lex.surnames.contains(&w.lower)
            })
    }

    /// Whether the word before `i` is a capitalised word of a longer proper name: a word the
    /// lexicon does not know, or a common word capitalised in mid-sentence (Ereğli Demir
    /// Çelik, Türk Hava Yolları); not a sentence start, a word of all-capital text (VE), or an
    /// institution word that ends a name (Anayasa Mahkemesi Metin Evecen).
    fn after_proper(&self, i: usize) -> bool {
        i > 0
            && self.same_line(i - 1, i)
            && self.word(i - 1).is_some_and(|prev| {
                let common = self.is_common_word(&prev.lower);
                // a role or title before the word is no proper name it continues: Bu davada
                // Davacı Kudret Demir (review 3 PII3-04)
                prev.cap
                    && !self.is_title(&prev.lower)
                    && !self.is_stopper(&prev.lower)
                    && (!common || (!prev.all_caps && !self.sentence_initial(i - 1)))
            })
    }

    /// A word that ends a name (names_data.rs's NAME_STOPPERS), also with a case or
    /// demonym ending when it is not itself a name: Müdürlüğünün, Üniversitesinde, Koreli.
    fn is_stopper(&self, lower: &str) -> bool {
        if self.lex.stoppers.contains(lower) {
            return true;
        }
        if self.is_named(lower) {
            return false;
        }
        lower.char_indices().rev().take(4).any(|(cut, _)| {
            lower[..cut].chars().count() >= 4 && self.lex.stoppers.contains(&lower[..cut])
        })
    }

    fn is_title(&self, lower: &str) -> bool {
        self.lex.roles.contains(lower)
            || self.lex.honorifics.contains(lower)
            || self.lex.post_titles.contains(lower)
    }

    fn trigger_before(&self, i: usize) -> Option<Trigger> {
        let mut end = i;
        let mut colon = false;
        if end > 0 && matches!(self.tokens[end - 1].text, ":" | "-" | "–") {
            colon = self.tokens[end - 1].text == ":";
            end -= 1;
        }
        for k in (1..=3).rev() {
            if end < k || !self.same_line(end - k, i) {
                continue;
            }
            let mut parts = Vec::with_capacity(k);
            for t in &self.tokens[end - k..end] {
                let s = t.text.trim_end_matches('.');
                if s.is_empty()
                    || !s
                        .chars()
                        .all(|c| c.is_alphabetic() || c == '\'' || c == '’')
                {
                    break;
                }
                parts.push(to_turkish_lower(s));
            }
            if parts.len() < k {
                continue;
            }
            let phrase = parts.join(" ");
            if colon && self.lex.colon_triggers.contains(&phrase) {
                return Some(Trigger::Colon);
            }
            if k == 1 && self.lex.honorifics.contains(&phrase) {
                return Some(Trigger::Honorific);
            }
            if self.lex.known_name_intro.contains(&phrase) {
                return Some(Trigger::Ben);
            }
            if self.lex.roles.contains(&phrase) || self.lex.self_intro.contains(&phrase) {
                return Some(Trigger::Role(self.lex.org_roles.contains(&phrase)));
            }
        }
        None
    }

    /// Whether the word at `j` continues a name whose last part is at `last`; `Some(true)`
    /// when it is a common word (Orhan Pamuk), after which only known names may follow.
    /// `strong`: the name began with a word that is only a name or with a title, so a word
    /// that reads as an inflected noun is a surname too (General Anton Denikin, Bakanı Fatin
    /// Rüştü); after an ambiguous name it is not (Deniz Kuvvetleri).
    fn continues(
        &self,
        last: usize,
        j: usize,
        cap_mode: bool,
        after_given: bool,
        strong: bool,
    ) -> Option<bool> {
        let w = self.word(j)?;
        let prev = self.word(last)?;
        let stopper = self.is_stopper(&w.lower) && !self.lex.surnames.contains(&w.lower);
        if self.lex.post_titles.contains(&w.lower) || stopper {
            return None;
        }
        let named = self.is_named(&w.lower);
        if !cap_mode {
            let ok = !w.cap
                && (self.lex.surnames.contains(&w.lower) || self.lex.curated.contains(&w.lower))
                && !self.lex.collisions.contains(&w.lower);
            return ok.then_some(false);
        }
        if !w.cap || (!named && (self.is_place(&w.lower) || self.is_title(&w.lower))) {
            return None;
        }
        if named {
            return Some(false);
        }
        // A surname in capitals after a name in mixed case, the official form (Muammer TOPAL,
        // Celal Mümtaz AKINCI), whatever the word; a short one (Mustafa BAL, Selami ER) only
        // right after a given name and closing the name (review 3 PII3-02).
        if w.all_caps && !word_shaped_caps(w) {
            let closes = self.tokens.get(j + 1).is_none_or(|t| {
                !self.same_line(j, j + 1) || !t.text.starts_with(char::is_alphanumeric)
            });
            let short_surname = (2..=3).contains(&w.letters)
                && w.lower.chars().any(crate::phonology::is_turkish_vowel)
                && !prev.all_caps
                && after_given
                && closes;
            // otherwise an acronym: Ahmet Yılmaz TC, AHMET YILMAZ TC
            return short_surname.then_some(true);
        }
        if w.all_caps && !prev.all_caps {
            return Some(true);
        }
        // An unknown word, or a common word that is not an inflected noun, is a surname only
        // right after a given name (Ahmet Gürbüzoğlu, Orhan Pamuk, Onurhan Solmaz, Zeki
        // Sarıca'ya; not Ahmet Kuvvetleri), and not in all-capital text (AHMET GELDİ).
        if !after_given || w.letters < 2 {
            return None;
        }
        if !self.is_common_word(&w.lower) {
            return Some(false);
        }
        let surname_shaped = w.has_apostrophe || strong || !self.only_inflected(&w.lower);
        (!prev.all_caps && !w.all_caps && surname_shaped).then_some(true)
    }

    /// Extends a name from its first part at `first` over further parts; returns the last.
    fn extend(&self, first: usize, cap_mode: bool, strong: bool) -> usize {
        let n = self.tokens.len();
        let mut last = first;
        // the first part is a first name, whether a list or a trigger made it one (General
        // Anton Denikin); after a part that is a common word only known names follow (Orhan
        // Pamuk, not Orhan Pamuk Nobel; Mehmet Reşit Arslan)
        let mut after_given = true;
        let mut only_known = false;
        let mut parts = 1;
        // The official form, given names in mixed case and the surname in capitals: after its
        // first surname a second one in capitals follows (Şebnem NEBİOĞLU ÖNER, a married
        // woman's two surnames), and the name ends at the line's end (review 3 PII3-02).
        let official = self.word(first).is_some_and(|f| f.cap && !f.all_caps);
        let mut surnames = 0;
        while parts < 4 {
            let Some(lw) = self.word(last) else { break };
            if lw.has_apostrophe {
                break;
            }
            let mut j = last + 1;
            if j >= n {
                break;
            }
            let glued_hyphen = self.tokens[j].text == "-"
                && j + 1 < n
                && self.tokens[last].end == self.tokens[j].start
                && self.tokens[j].end == self.tokens[j + 1].start;
            if glued_hyphen {
                j += 1; // Aydın-Özkan
            } else if !self.same_line(last, j) {
                // A line break inside a name, as text from PDFs has (Mehmet / Doğan): crossed
                // only right after a given name and onto a known name or a surname in capitals,
                // so that the names of a list stay apart; never after the official form's
                // surname (Hicabi DURSUN, then Hasan Tahsin GÖKCAN on the next line).
                let onto_name = self.word(j).is_some_and(|w| {
                    w.cap && (self.is_named(&w.lower) || (w.all_caps && !lw.all_caps))
                });
                if !(cap_mode && after_given && onto_name) || (official && lw.all_caps) {
                    break;
                }
            } else if let Some(e) = self.initial_end(j).filter(|_| cap_mode) {
                // a middle initial (John F. Kennedy) or a trailing one (Ahmet K.)
                let k = e + 1;
                if k < n && self.same_line(e, k) && self.surname_like(k, false) {
                    after_given = false; // the word after a middle initial is the surname
                    last = k;
                    parts += 2;
                    continue;
                }
                return j;
            }
            // not an inflected word or a title of the text around the name (YILMAZ
            // TARAFINDAN, KAYA KATİP Ayşe DEMİR; review 4 DOWN4-09)
            let second_surname = official
                && surnames == 1
                && lw.all_caps
                && self.word(j).is_some_and(|w| {
                    w.all_caps
                        && word_shaped_caps(w)
                        && !self.is_stopper(&w.lower)
                        && !self.is_title(&w.lower)
                        && !self.only_inflected(&w.lower)
                        && !self.readings(&w.lower).function_word
                });
            if second_surname {
                last = j;
                parts += 1;
                surnames += 1;
                continue;
            }
            if only_known
                && !glued_hyphen
                && !self.word(j).is_some_and(|w| {
                    w.cap
                        && (self.is_clear_name(&w.lower)
                            || (w.all_caps && !lw.all_caps && word_shaped_caps(w)))
                })
            {
                break;
            }
            // a particle and the surname after it (Karl von Frisch, Leonardo da Vinci)
            let particle = cap_mode
                && j + 1 < n
                && self.same_line(j, j + 1)
                && self.word(j).is_some_and(|w| {
                    !w.cap
                        && self.lex.particles.contains(&w.lower)
                        && (!matches!(w.lower.as_str(), "da" | "de")
                            || self
                                .word(first)
                                .is_some_and(|f| !self.lex.curated.contains(&f.lower)))
                })
                && self
                    .word(j + 1)
                    .is_some_and(|w| w.cap && !self.is_clear_name(&w.lower));
            if particle {
                j += 1;
            }
            let after = after_given || particle || glued_hyphen;
            match self.continues(last, j, cap_mode, after, strong) {
                Some(common_part) => {
                    if self.word(j).is_some_and(|w| w.all_caps) {
                        surnames += 1;
                    }
                    after_given = self
                        .word(j)
                        .is_some_and(|w| self.lex.given.contains_key(&w.lower));
                    only_known |= common_part;
                    last = j;
                    parts += 1;
                }
                None => break,
            }
        }
        last
    }

    /// Whether the word at `i` reads as a name in its own right, with the confidence of that.
    fn confirm(&self, i: usize, scorer: Option<&PiiEmbeddingScorer>) -> Option<f32> {
        let w = self.word(i)?;
        let kind = self.lex.given.get(&w.lower).copied();
        if !w.cap && (kind != Some(Kind::Clear) || !self.lex.curated.contains(&w.lower)) {
            return None; // in lower case only a curated name that is nothing else
        }
        let surname = self.lex.surnames.contains(&w.lower);
        let named = kind.is_some() || surname;
        // months, countries, institution words: a name only before a surname or a title
        // (Asya Yılmaz, Cuma Bey), or after a trigger
        let stopper = self.is_stopper(&w.lower);
        if w.letters < 2
            || (!named && (stopper || (w.all_caps && w.letters <= 3)))
            // an acronym the lexicon has (ZMMS, OHAL): review 4 DOWN4-03
            || (!named && w.all_caps && self.is_abbreviation(&w.lower))
            || ((self.lex.roles.contains(&w.lower) || self.lex.honorifics.contains(&w.lower))
                && kind != Some(Kind::Clear)
                && !surname)
            || (w.letters <= 3
                && self
                    .tokens
                    .get(i + 1)
                    .is_some_and(|t| t.text == "." && t.start == self.tokens[i].end))
        {
            // initials (T.S.), abbreviations (Ad.Ş.), titles (Uzman, Veli), institution
            // words, acronyms
            return None;
        }
        let trigger = self.trigger_before(i);
        let after_number = i > 0
            && self.tokens[i - 1].text.chars().any(|c| c.is_ascii_digit())
            && self.tokens[i - 1]
                .text
                .chars()
                .all(|c| c.is_ascii_digit() || c == '.');
        if after_number && trigger.is_none() && kind != Some(Kind::Clear) {
            return None; // 11 Eylül, 3 Nisan; but a footnote number before Mustafa
        }
        let place = self.is_place(&w.lower);

        if !w.cap {
            if !self.lex.curated.contains(&w.lower)
                || kind != Some(Kind::Clear)
                || self.lex.collisions.contains(&w.lower)
                || w.letters < 3
                || self.is_common_word(&w.lower)
            {
                return None;
            }
            if trigger.is_some() {
                return Some(0.85); // kızım ecrin, adım ahmet
            }
            let followed = i + 1 < self.tokens.len()
                && self.same_line(i, i + 1)
                && !w.has_apostrophe
                && self.continues(i, i + 1, false, true, false).is_some();
            if followed {
                return Some(0.85); // ahmet yılmaz
            }
            return (w.letters >= 4).then_some(0.75);
        }

        // After a role an organisation can hold (Davalı, Davacı), a listed word that is also
        // something else, before a company word on its line, names a company unless a known
        // name follows it: Davalı Hayat Sigorta A.Ş. (review 3 PII3-05). Without a company word
        // it is the person the role names (DAVACI : SEDA ARSLANTÜRK; review 4 DOWN4-01).
        let org_word = matches!(trigger, Some(Trigger::Role(true)))
            && kind.is_some_and(|k| k != Kind::Clear)
            && !surname
            && self.company_on_line(i)
            && !(i + 1 < self.tokens.len()
                && self.same_line(i, i + 1)
                && !w.has_apostrophe
                && self
                    .word(i + 1)
                    .is_some_and(|n| n.cap && self.is_named(&n.lower)));
        match trigger {
            _ if org_word => {}
            Some(Trigger::Honorific | Trigger::Colon) if named || !place => return Some(0.95),
            Some(Trigger::Role(_)) if named || (!place && !self.is_common_word(&w.lower)) => {
                return Some(0.95)
            }
            // after a title or a kinship word, a common word that is not an inflected noun
            // (Cumhurbaşkanı İsmet İnönü, kızım Derin); not after a role an organisation can
            // hold (Davacı Şirket, Müşteri Hizmetleri)
            Some(Trigger::Role(false))
                if !place && !stopper && !self.only_inflected(&w.lower) && !w.all_caps =>
            {
                return Some(0.9)
            }
            Some(Trigger::Ben) if kind.is_some() => return Some(0.9),
            _ => {}
        }

        let next = i + 1;
        let next_on_line = next < self.tokens.len() && self.same_line(i, next) && !w.has_apostrophe;
        let next_post_title = || {
            next_on_line
                && self
                    .word(next)
                    .is_some_and(|n| self.lex.post_titles.contains(&n.lower))
        };
        // the next word continues a name: a known name, a word the lexicon does not know
        // (Barış Manço), or a proper noun by its apostrophe (Zeki Sarıca'ya); not the second
        // word of an institution (Yıldız Teknik Üniversitesi)
        let next_part = || {
            next_on_line
                && !self.institution_at(next + 1)
                && self.word(next).is_some_and(|n| {
                    n.cap
                        && !self.lex.post_titles.contains(&n.lower)
                        && !self.is_stopper(&n.lower)
                        && (self.is_named(&n.lower)
                            || (!self.is_place(&n.lower)
                                && !self.is_title(&n.lower)
                                && n.letters > 1
                                && (n.has_apostrophe || !self.is_common_word(&n.lower))))
                })
        };
        let next_surname = || {
            next_on_line
                && self
                    .word(next)
                    .is_some_and(|n| n.cap && self.lex.surnames.contains(&n.lower))
        };
        let next_named = || {
            next_on_line
                && self.word(next).is_some_and(|n| {
                    n.cap && self.is_clear_name(&n.lower) && !self.is_stopper(&n.lower)
                })
        };
        // the official form, a name in mixed case and the surname in capitals (Muammer TOPAL,
        // Serruh KALELİ); not an acronym after a word (Kumkapı GGM, Savcılığın KYO)
        let next_caps_surname = || {
            next_on_line
                && !w.all_caps
                && w.letters > 1
                && (!place || named)
                && self.word(next).is_some_and(|n| {
                    n.all_caps
                        && !self.is_stopper(&n.lower)
                        && (self.is_clear_name(&n.lower)
                            || (word_shaped_caps(n)
                                && !self.is_place(&n.lower)
                                && !self.is_title(&n.lower)
                                // an acronym the lexicon has is no surname, and a common word
                                // that no list has as a name starts none (Ancak AİHM, Ayrıca
                                // OHAL; review 4 DOWN4-03)
                                && !self.is_abbreviation(&n.lower)
                                && (named || !self.is_common_word(&w.lower))))
                })
        };
        // the same with the surname on the next line, as PDF text breaks a signature block
        // (Engin / YILDIRIM); the surname must then be a known one or a word the lexicon does
        // not know (not Ahmet / KARAR)
        let next_caps_below = || {
            next < self.tokens.len()
                && !next_on_line
                && !w.has_apostrophe
                && !w.all_caps
                && named
                && self.word(next).is_some_and(|n| {
                    n.all_caps
                        && !self.is_stopper(&n.lower)
                        && (self.lex.surnames.contains(&n.lower)
                            || (word_shaped_caps(n)
                                && !self.is_place(&n.lower)
                                && !self.is_title(&n.lower)
                                && !self.is_common_word(&n.lower)))
                })
        };
        // a word the lexicon does not know after a name that is more often a place or a word
        // (Fatih Saraman, Ege Bagatur); not a common word (Fatih Camii, Kanun Hükmünde)
        let next_unknown = || {
            next_on_line
                && self.word(next).is_some_and(|n| {
                    n.cap
                        && !n.all_caps
                        && n.letters > 2
                        && !self.is_stopper(&n.lower)
                        && !self.lex.post_titles.contains(&n.lower)
                        && !self.is_place(&n.lower)
                        && !self.is_title(&n.lower)
                        && !self.is_common_word(&n.lower)
                })
        };

        let next_caps_below_unknown = || {
            next < self.tokens.len()
                && !next_on_line
                && !w.has_apostrophe
                && !w.all_caps
                && self
                    .word(next)
                    .is_some_and(|n| n.all_caps && self.lex.surnames.contains(&n.lower))
        };

        if stopper {
            return (next_surname() || next_post_title()).then_some(0.85);
        }
        match kind.map(|k| if place { Kind::Rare } else { k }) {
            Some(Kind::Clear) => Some(0.95),
            Some(Kind::Rare) => (next_post_title()
                || next_named()
                || next_caps_surname()
                || next_caps_below()
                || next_unknown())
            .then_some(0.85),
            Some(Kind::Ambiguous) => {
                if w.has_apostrophe
                    || next_post_title()
                    || next_caps_surname()
                    || next_caps_below()
                    || next_part()
                    || self.mid_sentence_alone(i)
                    || self.mid_sentence_pair(i)
                {
                    return Some(0.85);
                }
                let scorer = scorer?;
                // About 40 bytes of context on each side, widened to character boundaries,
                // without the word itself so its noun sense does not bias the embedding.
                let ctx_start = floor_char_boundary(self.text, w.start.saturating_sub(40));
                let ctx_end = ceil_char_boundary(self.text, w.stem_end + 40);
                let surrounding = format!(
                    "{} {}",
                    &self.text[ctx_start..w.start],
                    &self.text[w.stem_end..ctx_end]
                );
                scorer.is_person_context(surrounding.trim()).then_some(0.8)
            }
            None => {
                // A first name missing from the lists before a known name: a word the lexicon
                // does not know (Caner Çetin), a surname (Yüksel Şahin, Ergin Aktaş), or in
                // mid-sentence a common word that is not inflected (Devran Duran, Tahir
                // Canan); and a surname or unknown word before Bey / Hanım (Yılmaz Bey). The
                // cheap checks on the next word come first: most capitalised words are none
                // of these, and the lexicon is asked only about the rest.
                let unknown = || !place && w.letters > 1 && !self.is_common_word(&w.lower);
                let first_like = || {
                    !place
                        && w.letters > 1
                        && !self.after_proper(i)
                        && (surname
                            || unknown()
                            || (!self.sentence_initial(i) && !self.only_inflected(&w.lower)))
                };
                let hit = (next_named() && first_like())
                    || (next_post_title() && (surname || unknown()))
                    || next_caps_surname()
                    || (next_caps_below_unknown() && unknown());
                hit.then_some(0.9)
            }
        }
    }

    /// The last word of a run of up to four capitalised words that a decision title closes:
    /// "METİN ÖNCÜ BAŞVURUSU", "RÜSES ÖZER VE DİĞERLERİ BAŞVURUSU", "Ayşe Kaya Başvurusu".
    fn application_title(&self, i: usize) -> Option<usize> {
        let mut last = None;
        for j in i..(i + 4).min(self.tokens.len()) {
            if j > i && !self.same_line(j - 1, j) {
                break;
            }
            let Some(w) = self.word(j) else { break };
            if w.lower == "başvurusu" || w.lower == "basvurusu" {
                break;
            }
            if w.lower == "ve" {
                let next = self.word(j + 1).map(|w| w.lower.as_str());
                if matches!(next, Some("diğerleri" | "digerleri")) {
                    break;
                }
            }
            if !w.cap
                || w.has_apostrophe
                || self.tokens[j].text.ends_with('.')
                || self.is_stopper(&w.lower)
                || self.is_title(&w.lower)
            {
                return None; // also a company's LTD. ŞTİ. (review 3 PII3-06)
            }
            last = Some(j);
        }
        let last = last?;
        let after = |k: usize| self.word(k).map(|w| w.lower.as_str());
        let closes = match after(last + 1) {
            Some("başvurusu" | "basvurusu") => true,
            Some("ve") => {
                matches!(after(last + 2), Some("diğerleri" | "digerleri"))
                    && matches!(after(last + 3), Some("başvurusu" | "basvurusu"))
            }
            _ => false,
        };
        (closes && self.same_line(last, last + 1)).then_some(last)
    }

    /// Whether a company word in capitals or a company's initials (A.Ş.) follow the word at
    /// `last` on its line ([`COMPANY_WORDS`]); a company word in lower case is prose (Umut
    /// Kaya şirketi zarara uğrattı; review 4 DOWN4-02).
    fn company_after(&self, last: usize) -> bool {
        let k = last + 1;
        k < self.tokens.len() && self.same_line(last, k) && self.company_word(k)
    }

    fn company_word(&self, k: usize) -> bool {
        if dotted_initials(self.tokens[k].text).is_some_and(|l| l == "AŞ") {
            return true; // A.Ş. as one token
        }
        self.word(k).is_some_and(|w| {
            (w.cap && COMPANY_WORDS.contains(&w.lower.as_str()))
                || (w.letters == 1
                    && self.tokens[k].text.ends_with('.')
                    && self
                        .tokens
                        .get(k + 1)
                        .is_some_and(|t| t.start == self.tokens[k].end && t.text.ends_with('.')))
        })
    }

    /// Whether a company word follows the word at `i` within the next few words of its line.
    fn company_on_line(&self, i: usize) -> bool {
        (i + 1..self.tokens.len().min(i + 6))
            .take_while(|&k| self.same_line(i, k))
            .any(|k| self.company_word(k))
    }

    /// Whether the words `first..=last` before a company word are a person's name, the mark of
    /// a person's business (Ahmet Yılmaz İnşaat, Umut Demir İnşaat Ltd.): a listed given name
    /// and, last, a listed surname or a word the lexicon does not know. One word alone (Doğan
    /// Holding, Hayat Sigorta) or a given name before a common word (Cihan Medya A.Ş., Savaş
    /// Boya San.) names a company.
    fn person_business(&self, first: usize, last: usize) -> bool {
        last > first
            && (first..last).any(|k| {
                self.word(k)
                    .is_some_and(|w| self.lex.given.contains_key(&w.lower))
            })
            && self.word(last).is_some_and(|w| {
                self.lex.surnames.contains(&w.lower) || !self.is_common_word(&w.lower)
            })
    }

    /// Whether the word at `k` can be a surname after an initial: a known name, or a
    /// capitalised word the lexicon does not know (A. Yılmaz, P. Butyagin, John F. Kennedy);
    /// not "B. No", "A. Ulusal Hukuk", "I. Dünya Savaşı". At the start of a line or sentence,
    /// where list and section letters stand (A. Giriş), only a known name.
    fn surname_like(&self, k: usize, line_start: bool) -> bool {
        self.word(k).is_some_and(|w| {
            w.cap
                && w.letters > 1
                && !self.is_stopper(&w.lower)
                && !self.lex.post_titles.contains(&w.lower)
                && (self.is_clear_name(&w.lower)
                    || (!line_start
                        && w.letters >= 3
                        && !self.is_place(&w.lower)
                        && !self.is_title(&w.lower)
                        && !self.is_common_word(&w.lower)))
        })
    }

    /// A name at the start of two capitalised words in mid-sentence whose second is a common
    /// word but not an inflected noun, as many Turkish surnames are (bkz. Sadık Koçak, Levent
    /// Tütüncü, dün Metin Çiçek); an inflected second word makes a compound name (Deniz
    /// Kuvvetleri, Barış Pınarı).
    fn mid_sentence_pair(&self, i: usize) -> bool {
        if self.sentence_initial(i)
            || self.after_proper(i)
            || i + 1 >= self.tokens.len()
            || !self.same_line(i, i + 1)
            || self.institution_at(i + 2)
        {
            return false;
        }
        self.word(i + 1).is_some_and(|n| {
            n.cap
                && !n.all_caps
                && n.letters > 1
                && !self.is_stopper(&n.lower)
                && !self.is_place(&n.lower)
                && !self.is_title(&n.lower)
                && !self.only_inflected(&n.lower)
        })
    }

    /// A capitalised word in mid-sentence that is not part of a longer proper name: the
    /// capital marks a proper noun, and a given name standing alone is a person (dün Deniz
    /// geldi; not Deniz Kuvvetleri, Ege Denizi, Kırmızı Gül).
    fn mid_sentence_alone(&self, i: usize) -> bool {
        if self.sentence_initial(i) || self.after_proper(i) {
            return false;
        }
        let next = i + 1;
        !(next < self.tokens.len()
            && self.same_line(i, next)
            && self.word(next).is_some_and(|n| n.cap))
    }
}

/// Detects person names; see the module documentation.
pub(super) fn detect_names(
    text: &str,
    scorer: Option<&PiiEmbeddingScorer>,
    out: &mut Vec<PiiEntity>,
) {
    let tokens = TurkishTokenizer::tokenize(text);
    let scan = Scan {
        text,
        words: tokens.iter().map(Word::of).collect(),
        tokens: &tokens,
        lex: &LEXICON,
        readings: RefCell::new(HashMap::new()),
    };
    let n = tokens.len();
    let mut i = 0;
    while i < n {
        if let Some(last) = scan.application_title(i) {
            let start = scan.word(i).map_or(tokens[i].start, |w| w.start);
            let end = scan.word(last).map_or(tokens[last].end, |w| w.stem_end);
            let span = &text[start..end];
            out.push(PiiEntity {
                text: span.to_string(),
                label: PiiType::Name.as_str().to_string(),
                pii_type: PiiType::Name,
                start,
                end,
                confidence: 0.9,
                stem: span.to_string(),
                suffix: None,
            });
            i = last + 1;
            continue;
        }
        let initial = scan.initial_end(i).filter(|&e| {
            e + 1 < n
                && scan.same_line(e, e + 1)
                && scan.surname_like(e + 1, scan.sentence_initial(i))
        });
        let (first, start, confidence) = if let Some(e) = initial {
            // A. Yılmaz: the initial and the surname
            (e + 1, tokens[i].start, 0.9)
        } else if let Some(confidence) = scan.confirm(i, scorer) {
            (
                i,
                scan.word(i).map_or(tokens[i].start, |w| w.start),
                confidence,
            )
        } else {
            i += 1;
            continue;
        };
        let cap_mode = scan.word(first).is_some_and(|w| w.cap);
        let last = scan.extend(first, cap_mode, confidence >= 0.9);
        if scan.company_after(last) && !scan.person_business(first, last) {
            i = last + 1; // a company's name (review 3 PII3-05)
            continue;
        }
        let (end, suffix) = match scan.word(last) {
            Some(w) => (w.stem_end, w.suffix),
            None => (tokens[last].end, None), // a trailing initial
        };
        let span = &text[start..end];
        out.push(PiiEntity {
            text: span.to_string(),
            label: PiiType::Name.as_str().to_string(),
            pii_type: PiiType::Name,
            start,
            end,
            confidence,
            stem: span.to_string(),
            suffix: suffix.map(str::to_string),
        });
        i = last + 1;
        // the title after a name is not a name of its own (Fatih Sultan, Ayşe Hanım)
        if scan
            .word(i)
            .is_some_and(|w| scan.lex.post_titles.contains(&w.lower))
        {
            i += 1;
        }
    }
}

/// Byte ranges of well-known public figures' names in `text` (names_data.rs's PUBLIC_FIGURES).
pub(super) fn public_figure_spans(text: &str) -> Vec<(usize, usize)> {
    let (lower, origin) = super::normalize::turkish_lower_with_origin(text);
    let orig = |i: usize| origin.get(i).copied().unwrap_or(text.len());
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(m) = PUBLIC_FIGURE_REGEX.find_at(&lower, at) {
        let before = lower[..m.start()].chars().next_back();
        let after = lower[m.end()..].chars().next();
        if before.is_some_and(char::is_alphanumeric) || after.is_some_and(char::is_alphanumeric) {
            // inside a longer word: look again from the next character
            at = m.start() + lower[m.start()..].chars().next().map_or(1, char::len_utf8);
            continue;
        }
        out.push((orig(m.start()), orig(m.end())));
        at = m.end();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        detect_names(text, None, &mut out);
        out.into_iter().map(|e| e.text).collect()
    }

    /// PII's view of the analyzer for a fixed word list, the frequent types of
    /// tests/data/morphology/top_types.txt and every listed name: whether a word reads as a
    /// common word, only as an inflected one, and as a bare nominal or an unknown word. An
    /// analyzer change that moves masks shows here as a diff of words, not as drift in masks
    /// (review 3 PII3-03). `AKANA_GOLD_WRITE=pii_view` rewrites
    /// tests/data/downstream/pii_view.tsv.
    #[test]
    fn pii_view_of_the_analyzer_stays_as_recorded() {
        const PATH: &str = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/downstream/pii_view.tsv"
        );
        let words: std::collections::BTreeSet<String> =
            include_str!("../../tests/data/morphology/top_types.txt")
                .lines()
                .map(|w| w.trim().to_string())
                .filter(|w| !w.is_empty() && !w.starts_with('#'))
                .chain(LEXICON.given.keys().cloned())
                .chain(LEXICON.surnames.iter().cloned())
                .collect();
        let mut now = String::from(
            "# PII's view of the analyzer (pii::names::tests::pii_view_of_the_analyzer_stays_as_\
             recorded): word, reads as a common word, only as an inflected one, as a bare nominal \
             or unknown (1/0).\n",
        );
        for w in &words {
            let r = readings(w);
            now.push_str(&format!(
                "{w}\t{}\t{}\t{}\n",
                u8::from(r.common),
                u8::from(r.only_inflected),
                u8::from(is_bare_nominal_or_unknown(w))
            ));
        }
        if std::env::var("AKANA_GOLD_WRITE")
            .is_ok_and(|v| v.split(',').any(|s| s.trim() == "pii_view"))
        {
            std::fs::write(PATH, &now).expect("writable pii_view.tsv");
            return;
        }
        let recorded = std::fs::read_to_string(PATH).unwrap_or_default();
        let old: std::collections::BTreeSet<&str> = recorded.lines().collect();
        let new: std::collections::BTreeSet<&str> = now.lines().collect();
        let moved: Vec<&str> = new.difference(&old).copied().take(40).collect();
        let gone: Vec<&str> = old.difference(&new).copied().take(40).collect();
        assert!(
            moved.is_empty() && gone.is_empty(),
            "PII's view of the analyzer moved; now {moved:?}, before {gone:?}. If deliberate, \
             rewrite with AKANA_GOLD_WRITE=pii_view cargo test -p akana-core --lib pii_view and \
             list the diff in the commit."
        );
    }

    #[test]
    fn multi_part_and_triggered_names() {
        for (text, expected) in [
            ("Mehmet Ali Erbil geldi.", vec!["Mehmet Ali Erbil"]),
            ("Davacı AHMET YILMAZ vekili.", vec!["AHMET YILMAZ"]),
            ("Bunu A. Yılmaz imzaladı.", vec!["A. Yılmaz"]),
            (
                "Kerem Aydın-Özkan raporu hazırladı.",
                vec!["Kerem Aydın-Özkan"],
            ),
            ("John F. Kennedy spoke.", vec!["John F. Kennedy"]),
            ("Kızım Ecrin okula başladı.", vec!["Ecrin"]),
            ("Merhaba, ben Cansu.", vec!["Cansu"]),
            ("İmza: Serkan Altıntaş", vec!["Serkan Altıntaş"]),
            (
                "Başvurucu Nurettin Çağlayan'ın talebi",
                vec!["Nurettin Çağlayan"],
            ),
            ("ahmet yılmaz dün geldi", vec!["ahmet yılmaz"]),
            ("Fatma Hanım'a selam söyle.", vec!["Fatma"]),
            ("Dün Deniz geldi.", vec!["Deniz"]),
            ("Orhan Pamuk Nobel aldı.", vec!["Orhan Pamuk"]),
            ("Ahmet Ankara'ya gitti.", vec!["Ahmet"]),
        ] {
            assert_eq!(names(text), expected, "{text}");
        }
    }

    #[test]
    fn words_that_are_not_names() {
        for text in [
            "Deniz kenarında yürüdük.",
            "Emin misin?",
            "Metin aşağıda verilmiştir.",
            "Deniz Kuvvetleri Komutanlığı açıklama yaptı.",
            "Ege Denizi'nde fırtına var.",
            "11 Eylül saldırıları",
            "Davacı Şirket itiraz etti.",
            "Müşteri Hizmetleri aradı.",
            "Ben Ankara'dayım.",
            "if x is nil then",
            "Kuzey Kore füze denedi.",
            "Ramazan Bayramı'nda görüşürüz.",
            "Sorumlu: IT Ekibi",
            "Darbenin Adı Balyoz idi.",
            "A. Ulusal Hukuk\nB. Uluslararası Hukuk",
            "B. No: 2014/8875, 7/6/2018",
            "sanık T.S. ve tanık K.Y.nin ifadeleri",
            "İl Özel İdaresinde çalışıyor.",
            "Kumkapı GGM'de tutuldu.",
            "Güney Koreli şarkıcı",
            "Fatih Camii'nde namaz kıldı.",
        ] {
            assert!(names(text).is_empty(), "{text}: {:?}", names(text));
        }
    }

    #[test]
    fn court_and_reference_formats() {
        for (text, expected) in [
            // signature blocks: the surname in capitals, also on the next line
            (
                "Üyeler\n\n:\n\nMuammer TOPAL\n\nCelal Mümtaz AKINCI",
                vec!["Muammer TOPAL", "Celal Mümtaz AKINCI"],
            ),
            (
                "Başkan\n\n:\n\nEngin\n\nYILDIRIM\n\nÜyeler",
                vec!["Engin\n\nYILDIRIM"],
            ),
            // decision titles
            (
                "ANAYASA MAHKEMESİ\nMETİN ÖNCÜ VE DİĞERLERİ BAŞVURUSU",
                vec!["METİN ÖNCÜ"],
            ),
            // case citations
            (
                "(Sadık Koçak ve diğerleri, B. No: 2013/841)",
                vec!["Sadık Koçak"],
            ),
            (
                "(Mehmet Reşit Arslan ve diğerleri [2. B.])",
                vec!["Mehmet Reşit Arslan"],
            ),
            (
                "Anayasa Mahkemesi Metin Evecen (B. No: 2017/744)",
                vec!["Metin Evecen"],
            ),
            (
                "(Fatih Saraman, § 82; Turgut Duman, § 3)",
                vec!["Fatih Saraman", "Turgut Duman"],
            ),
            ("(Devran Duran, § 64)", vec!["Devran Duran"]),
            // titles before a name that is also a word, and foreign names
            ("Cumhurbaşkanı İsmet İnönü konuştu.", vec!["İsmet İnönü"]),
            ("General Anton Denikin komutasında", vec!["Anton Denikin"]),
            (
                "1982 - Karl von Frisch, Avusturyalı",
                vec!["Karl von Frisch"],
            ),
            (
                "Yargıtay Başsavcısı Vural Savaş açıkladı.",
                vec!["Vural Savaş"],
            ),
        ] {
            assert_eq!(names(text), expected, "{text:?}");
        }
    }

    #[test]
    fn public_figure_filter_keeps_only_whole_phrases() {
        let mut engine = super::super::TurkishPiiEngine::new();
        engine.set_preserve_public_figures(true);
        let names_in = |text: &str| -> Vec<String> {
            engine
                .detect(text)
                .into_iter()
                .filter(|e| e.pii_type == PiiType::Name)
                .map(|e| e.text)
                .collect()
        };
        assert!(names_in("Mustafa Kemal Atatürk Cumhuriyeti kurdu.").is_empty());
        assert!(names_in("Nazım Hikmet şiirleri çok sevilir.").is_empty());
        assert!(names_in("Fatih Sultan Mehmet İstanbul'u fethetti.").is_empty());
        assert_eq!(
            names_in("Mustafa Kemal Yılmaz geldi."),
            ["Mustafa Kemal Yılmaz"]
        );
        assert_eq!(
            names_in("Orhan Pamuk ile Ahmet Yılmaz konuştu."),
            ["Ahmet Yılmaz"]
        );
        // off by default
        let plain = super::super::TurkishPiiEngine::new();
        assert!(!plain.detect("Orhan Pamuk Nobel aldı.").is_empty());
    }

    #[test]
    fn public_figures_are_found_as_phrases() {
        let spans = public_figure_spans("Mustafa Kemal Atatürk'ün ve İSMET İNÖNÜ'nün");
        assert_eq!(spans, [(0, 22), (30, 45)]);
    }
}
