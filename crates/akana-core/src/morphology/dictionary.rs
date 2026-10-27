//! Turkish lexicon: dictionary items and the root index the analyzer looks stems up in.

use super::pos::{ascii_lower, PrimaryPos, RootAttr, SecondaryPos, NAME_BUF};
use super::pronunciation::{guess_abbreviation, letter_pronunciation};
use crate::phonology::{is_turkish_vowel, to_turkish_lower, to_turkish_upper_char};
use ahash::AHashMap;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DictionaryItem {
    pub lemma: String,
    pub root: String,
    pub primary_pos: PrimaryPos,
    pub secondary_pos: SecondaryPos,
    pub attributes: RootAttr,
    /// How the root is read when that differs from its spelling (ABD is "abede", gram
    /// "gıram"). Vowel harmony and the voicing of suffixes follow the pronunciation.
    #[serde(default)]
    pub pronunciation: Option<String>,
    /// The two roots of a compound that ends in its own 3rd-person possessive
    /// (cumhurbaşkanı: cumhur, başkan). Its plural and derivations attach to the stem without
    /// the possessive (cumhurbaşkan-lar-ı). A word that is no compound but has such a stem has
    /// an empty first root (üzeri: "", üzer; üzer-ler-i).
    #[serde(default)]
    pub compound_roots: Option<(String, String)>,
    /// The case a postposition governs (gibi: Gen, göre: Dat), Zemberek's PC* secondary POS.
    #[serde(default)]
    pub governed_case: Option<String>,
    /// Which of several items with the same lemma, root and POS this is (the lexicon's `idx=`;
    /// 0 for the first): kalp "heart" and kalp "counterfeit", the three hal. The prior keeps a
    /// separate frequency for each.
    #[serde(default)]
    pub homograph: u8,
    /// The dictionary's own spelling of the lemma where it is not the one
    /// [`DictionaryItem::display_lemma`] makes up (kâğıt, Hakkâri, e-posta; the lexicon's
    /// `display=`).
    #[serde(default)]
    pub display: Option<String>,
}

impl DictionaryItem {
    pub fn new(
        lemma: impl Into<String>,
        root: impl Into<String>,
        primary_pos: PrimaryPos,
        secondary_pos: SecondaryPos,
        attributes: RootAttr,
    ) -> Self {
        Self {
            lemma: lemma.into(),
            root: root.into(),
            primary_pos,
            secondary_pos,
            attributes,
            pronunciation: None,
            compound_roots: None,
            governed_case: None,
            homograph: 0,
            display: None,
        }
    }

    /// The lemma as the dictionary spells it, as Zemberek prints it: `display` where there is
    /// one, else the lemma, capitalised for a proper noun and upper-cased for an abbreviation
    /// (Ankara, TBMM). `lemma` is the folded key that retrieval and the prior use.
    pub fn display_lemma(&self) -> Cow<'_, str> {
        match &self.display {
            Some(display) => Cow::Borrowed(display),
            None => self.display_lemma_default(),
        }
    }

    /// Writes [`DictionaryItem::display_lemma`] to `out` (the analyzer's `analysis`) without
    /// building a string for it.
    pub(crate) fn push_display_lemma(&self, out: &mut String) {
        if let Some(display) = &self.display {
            out.push_str(display);
            return;
        }
        match self.secondary_pos {
            SecondaryPos::ProperNoun => {
                let mut chars = self.lemma.chars();
                if let Some(first) = chars.next() {
                    out.push(to_turkish_upper_char(first));
                }
                out.push_str(chars.as_str());
            }
            SecondaryPos::Abbreviation => {
                out.extend(self.lemma.chars().map(to_turkish_upper_char));
            }
            _ => out.push_str(&self.lemma),
        }
    }

    /// The spelling [`DictionaryItem::display_lemma`] makes up when the dictionary gives none.
    fn display_lemma_default(&self) -> Cow<'_, str> {
        match self.secondary_pos {
            SecondaryPos::ProperNoun => {
                let mut chars = self.lemma.chars();
                match chars.next() {
                    Some(first) => {
                        let mut out = String::with_capacity(self.lemma.len() + 1);
                        out.push(to_turkish_upper_char(first));
                        out.push_str(chars.as_str());
                        Cow::Owned(out)
                    }
                    None => Cow::Borrowed(&self.lemma),
                }
            }
            SecondaryPos::Abbreviation => {
                Cow::Owned(self.lemma.chars().map(to_turkish_upper_char).collect())
            }
            _ => Cow::Borrowed(&self.lemma),
        }
    }

    /// The pronunciation suffixes harmonise with: the explicit one, or else the root.
    pub fn pronunciation(&self) -> &str {
        self.pronunciation.as_deref().unwrap_or(&self.root)
    }
}

/// Zemberek's `TurkishDictionaryLoader.inferMorphemicAttributes`, on the item's pronunciation:
/// a verb's aorist class, progressive vowel drop and passive and causative forms, and a noun's
/// voicing. The lexicon file already carries the result; user dictionaries get it here.
/// Applying it twice changes nothing.
pub fn infer_attributes(item: &mut DictionaryItem) {
    let pron = item.pronunciation();
    let Some(last) = pron.chars().last() else {
        return;
    };
    let vowels = pron.chars().filter(|&c| is_turkish_vowel(c)).count();
    let last_vowel = is_turkish_vowel(last);
    let nk_or_og = pron.ends_with("nk") || pron.ends_with("og");
    let a = &mut item.attributes;
    match item.primary_pos {
        PrimaryPos::Verb => {
            if last_vowel {
                *a |= RootAttr::PROGRESSIVE_VOWEL_DROP | RootAttr::PASSIVE_IN;
            }
            if vowels > 1 && !a.contains(RootAttr::AORIST_A) {
                *a |= RootAttr::AORIST_I;
            }
            if vowels == 1 && !a.contains(RootAttr::AORIST_I) {
                *a |= RootAttr::AORIST_A;
            }
            if last == 'l' {
                *a |= RootAttr::PASSIVE_IN;
            }
            if last_vowel || (matches!(last, 'l' | 'r') && vowels > 1) {
                *a |= RootAttr::CAUSATIVE_T;
            }
        }
        PrimaryPos::Noun | PrimaryPos::Adj | PrimaryPos::Dup => {
            let named = matches!(
                item.secondary_pos,
                SecondaryPos::ProperNoun | SecondaryPos::Abbreviation
            );
            if vowels > 1
                && matches!(last, 'ç' | 'k' | 'p' | 't')
                && !named
                && !a.intersects(RootAttr::NO_VOICING | RootAttr::INVERSE_HARMONY)
            {
                *a |= RootAttr::VOICING;
            }
            if nk_or_og {
                if !a.contains(RootAttr::NO_VOICING)
                    && item.secondary_pos != SecondaryPos::ProperNoun
                {
                    *a |= RootAttr::VOICING;
                }
            } else if vowels < 2 && !a.contains(RootAttr::VOICING) {
                *a |= RootAttr::NO_VOICING;
            }
        }
        _ => {}
    }
}

/// Lower-case without circumflexes, apostrophes or dashes, as Zemberek normalises dictionary
/// words (Nikâh -> nikah, Kur'an -> kuran).
fn normalize_word(word: &str) -> String {
    // The built-in lexicon is already normal; checking is cheaper than rebuilding the string.
    if !word
        .chars()
        .any(|c| c.is_uppercase() || matches!(c, 'â' | 'î' | 'û' | '\'' | '’' | '-'))
    {
        return word.to_string();
    }
    to_turkish_lower(word)
        .chars()
        .filter(|&c| c != '\'' && c != '’' && c != '-')
        .map(|c| match c {
            'â' => 'a',
            'î' => 'i',
            'û' => 'u',
            c => c,
        })
        .collect()
}

/// A root attribute by its akana or Zemberek name (case-insensitive).
fn attribute_by_name(name: &str) -> Option<RootAttr> {
    let mut buf = [0u8; NAME_BUF];
    Some(match ascii_lower(name, &mut buf)? {
        "voicing" => RootAttr::VOICING,
        "novoicing" => RootAttr::NO_VOICING,
        "voweldrop" | "lastvoweldrop" => RootAttr::VOWEL_DROP,
        "doubling" => RootAttr::CONSONANT_DOUBLING,
        "inverseharmony" => RootAttr::INVERSE_HARMONY,
        "vowelmutation" => RootAttr::VOWEL_MUTATION,
        "compoundp3sg" => RootAttr::COMPOUND_P3SG,
        "plural" | "implicitplural" => RootAttr::PLURAL,
        "aorista" | "aorist_a" => RootAttr::AORIST_A,
        "aoristi" | "aorist_i" => RootAttr::AORIST_I,
        "progressivevoweldrop" => RootAttr::PROGRESSIVE_VOWEL_DROP,
        "passivein" | "passive_in" => RootAttr::PASSIVE_IN,
        "causativet" | "causative_t" => RootAttr::CAUSATIVE_T,
        "reflexive" => RootAttr::REFLEXIVE,
        "reciprocal" => RootAttr::RECIPROCAL,
        "nonreciprocal" => RootAttr::NON_RECIPROCAL,
        "implicitp1sg" => RootAttr::IMPLICIT_P1SG,
        "implicitp2sg" => RootAttr::IMPLICIT_P2SG,
        "implicitdative" => RootAttr::IMPLICIT_DATIVE,
        "familymember" => RootAttr::FAMILY_MEMBER,
        "nosuffix" => RootAttr::NO_SUFFIX,
        "lexicalized" => RootAttr::LEXICALIZED,
        "notlexicalized" => RootAttr::NOT_LEXICALIZED,
        "rare" => RootAttr::RARE,
        "nonstandard" => RootAttr::NONSTANDARD,
        "propernoun" => RootAttr::PROPER_NOUN,
        _ => return None,
    })
}

/// The lexicon's items by root, in the order they were added (the file's order for the
/// built-in lexicon), so that everything built from it is deterministic without sorting.
#[derive(Clone)]
pub struct RootLexicon {
    /// Root -> position of its items in `buckets`.
    index: AHashMap<String, usize>,
    buckets: Vec<Vec<DictionaryItem>>,
    /// Length in characters of the longest root, which bounds the stems worth looking up.
    max_root_chars: usize,
}

impl Default for RootLexicon {
    fn default() -> Self {
        Self::new()
    }
}

impl RootLexicon {
    /// The built-in lexicon (`zemberek_lexicon.txt`, built by `scripts/build_full_lexicon.py`).
    pub fn new() -> Self {
        let mut lexicon = Self::empty();
        lexicon.load_from_str(crate::morphology::ZEMBEREK_LEXICON);
        lexicon
    }

    pub fn empty() -> Self {
        Self {
            index: AHashMap::new(),
            buckets: Vec::new(),
            max_root_chars: 0,
        }
    }

    /// Adds an item; an item equal to one already stored is ignored.
    pub fn add_item(&mut self, item: DictionaryItem) {
        self.max_root_chars = self.max_root_chars.max(item.root.chars().count());
        match self.index.get(&item.root) {
            Some(&i) => {
                let bucket = &mut self.buckets[i];
                if !bucket.contains(&item) {
                    bucket.push(item);
                }
            }
            None => {
                self.index.insert(item.root.clone(), self.buckets.len());
                self.buckets.push(vec![item]);
            }
        }
    }

    pub fn get_items(&self, root: &str) -> Option<&Vec<DictionaryItem>> {
        self.index.get(root).map(|&i| &self.buckets[i])
    }

    /// Every item, grouped by root, in the order the roots were first added.
    pub fn items(&self) -> impl Iterator<Item = &DictionaryItem> {
        self.buckets.iter().flatten()
    }

    /// Length in characters of the longest root in the lexicon.
    pub fn max_root_chars(&self) -> usize {
        self.max_root_chars
    }

    pub fn contains_root(&self, root: &str) -> bool {
        self.index.contains_key(root)
    }

    pub fn count(&self) -> usize {
        self.buckets.iter().map(Vec::len).sum()
    }

    /// Loads dictionary lines of the form
    /// `lemma root POS [SECONDARY] [ATTRIBUTES...] [pr=PRONUNCIATION] [roots=A-B] [pc=CASE]`
    /// (`idx=N` marks a homograph after the first; items keep the file's order; `display=`
    /// spells the lemma as the dictionary does, as does a lemma written with capitals or a
    /// circumflex: `hâkim hakim Adj`),
    /// e.g. `kitap kitap Noun Voicing`, `gelmek gel Verb`, `abd abd Noun Abbrv pr=abede`.
    ///
    /// The secondary POS uses the names of [`SecondaryPos::from_name`] (`Prop`/`ProperNoun`,
    /// `Abbrv`, `Time`, `Pers`, ...). Attributes take akana's or Zemberek's names (`VowelDrop`
    /// or `LastVowelDrop`, `AoristA` or `Aorist_A`, `NoVoicing`, ...); unknown tokens are
    /// ignored. An unknown primary POS is read as `Noun`. Lemma and root are lower-cased and
    /// lose circumflexes, apostrophes and dashes, as `analyze` normalises words; the
    /// attributes Zemberek infers (aorist class, voicing) are added when missing, and an
    /// abbreviation without a pronunciation gets a guessed one (TBK -> tebeka).
    pub fn load_from_str(&mut self, text: &str) {
        // About one item per 40 bytes of the built-in lexicon.
        self.index.reserve(text.len() / 40);
        self.buckets.reserve(text.len() / 40);
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let (Some(lemma), Some(root), Some(pos)) = (parts.next(), parts.next(), parts.next())
            else {
                continue;
            };

            let written = lemma;
            let lemma = normalize_word(lemma);
            let root = normalize_word(root);
            let primary_pos = PrimaryPos::from_name(pos).unwrap_or(PrimaryPos::Noun);
            let mut item =
                DictionaryItem::new(lemma, root, primary_pos, SecondaryPos::None, RootAttr::NONE);

            for p in parts {
                if let Some((key, value)) = p.split_once('=') {
                    match key {
                        "pr" => item.pronunciation = Some(normalize_word(value)),
                        // One part is the stem without the possessive (üzeri roots=üzer), as
                        // Zemberek joins the parts: its first part is empty.
                        "roots" => {
                            item.compound_roots = Some(match value.split_once('-') {
                                Some((a, b)) => (normalize_word(a), normalize_word(b)),
                                None => (String::new(), normalize_word(value)),
                            });
                        }
                        "pc" => item.governed_case = Some(value.to_string()),
                        "idx" => item.homograph = value.parse().unwrap_or(0),
                        "display" => item.display = Some(value.to_string()),
                        _ => {}
                    }
                    continue;
                }
                if let Some(sec) = SecondaryPos::from_name(p) {
                    if item.secondary_pos == SecondaryPos::None {
                        item.secondary_pos = sec;
                    }
                    if sec == SecondaryPos::ProperNoun {
                        item.attributes |= RootAttr::PROPER_NOUN;
                    }
                    continue;
                }
                if let Some(attr) = attribute_by_name(p) {
                    item.attributes |= attr;
                }
            }

            if item.pronunciation.is_none() {
                let guessed = if item.secondary_pos == SecondaryPos::Abbreviation {
                    guess_abbreviation(&item.root)
                } else if !item.root.chars().any(is_turkish_vowel) {
                    letter_pronunciation(&item.root)
                } else {
                    item.root.clone()
                };
                if guessed != item.root {
                    item.pronunciation = Some(guessed);
                }
            }
            if item.display.is_none() && written != item.lemma {
                item.display = Some(written.to_string());
            }
            if item.display.as_deref() == Some(&*item.display_lemma_default()) {
                item.display = None;
            }
            infer_attributes(&mut item);
            self.add_item(item);
        }
    }

    /// Loads dictionary items from a text file.
    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> std::io::Result<()> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let l = line?;
            self.load_from_str(&l);
        }
        Ok(())
    }
}
