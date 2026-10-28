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
use unicode_normalization::UnicodeNormalization;

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
pub(crate) fn normalize_word(word: &str) -> String {
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

/// Zemberek root attributes akana has no use for, which its dictionaries name (Ext, NoQuote
/// ...): a line may carry them. `None` is the secondary-POS column of a line without one.
fn ignored_name(name: &str) -> bool {
    let mut buf = [0u8; NAME_BUF];
    matches!(
        ascii_lower(name, &mut buf),
        Some(
            "none"
                | "ext"
                | "runtime"
                | "dummy"
                | "informal"
                | "localeen"
                | "pronunciationguessed"
                | "noquote"
                | "nounconsinsert_n"
                | "compoundp3sgroot"
                | "special"
        )
    )
}

/// The letters a root may have, as the lexicon build allows them (Turkish letters, q, w, x and
/// digits, superscript ones too: m², km²): anything else, such as a ligature, a combining mark
/// or a format character, would store a root no word reaches.
fn root_letters(root: &str) -> bool {
    !root.is_empty()
        && root.chars().all(|c| {
            c.is_ascii_digit()
                || matches!(c, 'a'..='z' | 'ç' | 'ğ' | 'ı' | 'ö' | 'ş' | 'ü')
                || matches!(c, '¹' | '²' | '³' | '⁰'..='⁹' | '₀'..='₉')
        })
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
        // the short names too, which name the voice of a verb as `analyze` tags it (review 4
        // MORPH4-09) and a pronoun's secondary POS (kendi, birbiri)
        "reflexive" | "reflex" => RootAttr::REFLEXIVE,
        "reciprocal" | "recip" => RootAttr::RECIPROCAL,
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

    /// Adds an item; an item equal to one already stored is ignored. Returns where the item is
    /// stored: its root's bucket and its place in it, which stay as they are whatever is added
    /// later ([`item_at`](Self::item_at)).
    pub fn add_item(&mut self, item: DictionaryItem) -> (usize, usize) {
        self.max_root_chars = self.max_root_chars.max(item.root.chars().count());
        match self.index.get(&item.root) {
            Some(&i) => {
                let bucket = &mut self.buckets[i];
                match bucket.iter().position(|stored| *stored == item) {
                    Some(place) => (i, place),
                    None => {
                        bucket.push(item);
                        (i, bucket.len() - 1)
                    }
                }
            }
            None => {
                self.index.insert(item.root.clone(), self.buckets.len());
                self.buckets.push(vec![item]);
                (self.buckets.len() - 1, 0)
            }
        }
    }

    /// The item at a bucket and place [`add_item`](Self::add_item) returned.
    pub fn item_at(&self, bucket: usize, place: usize) -> &DictionaryItem {
        &self.buckets[bucket][place]
    }

    /// Every item with its bucket and place, in the order of [`items`](Self::items).
    pub fn positions(&self) -> impl Iterator<Item = ((usize, usize), &DictionaryItem)> {
        self.buckets.iter().enumerate().flat_map(|(b, bucket)| {
            bucket
                .iter()
                .enumerate()
                .map(move |(place, item)| ((b, place), item))
        })
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
    /// e.g. `kitap kitap Noun Voicing`, `gelmek gel Verb`, `abd abd Noun Abbrv pr=abede`;
    /// or in Zemberek's own syntax, as its dictionaries and akana's supplements write them:
    /// `word [P:Noun,Prop; A:Voicing; Pr:...; Roots:a-b; Index:1]`, or the word alone (a verb
    /// when it ends in -mAk, else a noun): `blokzincir [A:Voicing]`, `kriptopara [P:Noun;
    /// A:CompoundP3sg; Roots:kripto-para]`, `foobarlamak`.
    ///
    /// The secondary POS uses the names of [`SecondaryPos::from_name`] (`Prop`/`ProperNoun`,
    /// `Abbrv`, `Time`, `Pers`, ...); a secondary alone stands for a noun (`P:Abbrv`).
    /// Attributes take akana's or Zemberek's names (`VowelDrop` or `LastVowelDrop`, `AoristA`
    /// or `Aorist_A`, `NoVoicing`, ...), and Zemberek's that akana does not use (`Ext`,
    /// `NoQuote`) are allowed. Lemma and root are lower-cased and lose circumflexes,
    /// apostrophes and dashes, as `analyze` normalises words; the attributes Zemberek infers
    /// (aorist class, voicing) are added when missing, and an abbreviation without a
    /// pronunciation gets a guessed one (TBK -> tebeka).
    ///
    /// A line in Zemberek's syntax is read as the lexicon build reads it
    /// (`scripts/build_full_lexicon.py`): a word with a capital is a proper noun unless `P:`
    /// says otherwise, and never a verb; a word alone is a verb when it is written in lower
    /// case, ends in -mAk, has more than three letters and is no `CompoundP3sg`; the first
    /// secondary POS counts; `PCDat` and the like give a postposition's case; an abbreviation
    /// written with a Latin I has i in its lemma (`IBAN`: `iban`); `Ref:` names the word a
    /// variant spells, which gives the lemma and its spelling while the root keeps the
    /// variant (`şöför [A:Nonstandard; Ref:şoför]`: lemma şoför, root şöför; a Zemberek
    /// item name such as `kilitlemek_Verb` is its word); a `# comment` may follow the `]`.
    /// `Recip` and `Reflex` on a verb are its voice, as `Reciprocal` and `Reflexive` are, and
    /// a pronoun's secondary POS. A line is read as Unicode NFC, and a leading byte-order mark
    /// is skipped.
    ///
    /// Returns the lines it could not read in full, which add no item: an unknown POS, a
    /// missing column, an unclosed `[`, an unknown attribute, secondary POS or field (a
    /// misspelt `Voicng`), a voice attribute on a word that is neither a verb nor a pronoun,
    /// attributes that contradict each other (`Voicing` and `NoVoicing`, `AoristA` and
    /// `AoristI`), and a root with characters no Turkish word has (review 2 MORPH2-12,
    /// review 3 LEX3-05, BIND3-05, review 4 MORPH4-09).
    pub fn load_from_str(&mut self, text: &str) -> Vec<String> {
        // About one item per 40 bytes of the built-in lexicon.
        self.index.reserve(text.len() / 40);
        self.buckets.reserve(text.len() / 40);
        let mut rejected = Vec::new();
        for written in text.lines() {
            let written = written.trim_start_matches('\u{feff}').trim();
            if written.is_empty() || written.starts_with('#') {
                continue;
            }
            let nfc: String;
            let line = if written.is_ascii()
                || unicode_normalization::is_nfc_quick(written.chars())
                    == unicode_normalization::IsNormalized::Yes
            {
                written
            } else {
                nfc = written.nfc().collect();
                &nfc
            };
            let zemberek = line.contains('[') || line.split_whitespace().nth(1).is_none();
            let parsed = if zemberek {
                parse_zemberek_line(line)
            } else {
                parse_akana_line(line).map(|(lemma, item)| (Cow::Borrowed(lemma), item))
            };
            match parsed {
                Some((spelled, mut item)) => {
                    finish_item(&mut item, &spelled);
                    self.add_item(item);
                }
                None => rejected.push(written.to_string()),
            }
        }
        rejected
    }

    /// Loads dictionary items from a text file.
    /// Returns the lines read neither way, as [`load_from_str`](Self::load_from_str) does.
    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> std::io::Result<Vec<String>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut rejected = Vec::new();
        for line in reader.lines() {
            let l = line?;
            rejected.extend(self.load_from_str(&l));
        }
        Ok(rejected)
    }
}

/// A line in akana's column format (`lemma root POS ...`): the lemma as written and the item,
/// or `None` for an unknown POS or a missing column.
fn parse_akana_line(line: &str) -> Option<(&str, DictionaryItem)> {
    let mut parts = line.split_whitespace();
    let (lemma, root, pos) = (parts.next()?, parts.next()?, parts.next()?);
    let primary_pos = PrimaryPos::from_name(pos)?;
    let mut item = DictionaryItem::new(
        normalize_word(lemma),
        normalize_word(root),
        primary_pos,
        SecondaryPos::None,
        RootAttr::NONE,
    );
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
                "idx" => item.homograph = value.parse().ok()?,
                "display" => item.display = Some(value.to_string()),
                _ => return None,
            }
            continue;
        }
        if !add_name(&mut item, p) {
            return None;
        }
    }
    (root_letters(&item.root) && consistent(&item)).then_some((lemma, item))
}

/// Whether an item's attributes can hold together: a root does not both voice and keep its
/// final consonant, nor take both aorist classes (review 4 MORPH4-09).
fn consistent(item: &DictionaryItem) -> bool {
    !item
        .attributes
        .contains(RootAttr::VOICING | RootAttr::NO_VOICING)
        && !item
            .attributes
            .contains(RootAttr::AORIST_A | RootAttr::AORIST_I)
}

/// A line in Zemberek's syntax (`word [P:..; A:..; Pr:..; Roots:..; Index:..]`), read as
/// `pos_info` and `make_entry` of `scripts/build_full_lexicon.py` read it: the word as written
/// (as the item displays it) and the item, or `None` for a line it cannot read in full.
fn parse_zemberek_line(line: &str) -> Option<(Cow<'_, str>, DictionaryItem)> {
    let (word, meta) = match line.split_once('[') {
        Some((word, rest)) => {
            // Text after the `]` may only be a comment.
            let (meta, after) = rest.split_once(']')?;
            let after = after.trim();
            if !after.is_empty() && !after.starts_with('#') {
                return None;
            }
            (word.trim(), Some(meta))
        }
        None => (line, None),
    };
    if word.is_empty() || word.contains(char::is_whitespace) {
        return None;
    }
    let mut primary = None;
    let mut secondary = SecondaryPos::None;
    let mut governed_case = None;
    let mut names: Vec<&str> = Vec::new();
    let mut pronunciation = None;
    let mut roots = None;
    let mut homograph = 0;
    let mut reference = None;
    for field in meta.into_iter().flat_map(|m| m.split(';')) {
        let Some((key, value)) = field.split_once(':') else {
            if field.trim().is_empty() {
                continue;
            }
            return None;
        };
        let values = value.split(',').map(str::trim).filter(|v| !v.is_empty());
        match key.trim() {
            "P" => {
                for v in values {
                    // The first primary POS, then the first secondary, count (Ques is both).
                    if let (None, Some(pos)) = (primary, PrimaryPos::from_name(v)) {
                        primary = Some(pos);
                    } else if let Some(case) = v.strip_prefix("PC").filter(|c| {
                        matches!(*c, "Dat" | "Abl" | "Nom" | "Gen" | "Ins" | "Acc" | "Loc")
                    }) {
                        governed_case = Some(case.to_string());
                    } else {
                        let sec = SecondaryPos::from_name(v)?;
                        if secondary == SecondaryPos::None {
                            secondary = sec;
                        }
                    }
                }
            }
            "A" => names.extend(values),
            "Pr" => pronunciation = Some(normalize_word(value.trim())),
            "Roots" => {
                roots = Some(match value.trim().split_once('-') {
                    Some((a, b)) => (normalize_word(a), normalize_word(b)),
                    None => (String::new(), normalize_word(value.trim())),
                })
            }
            "Index" => homograph = value.trim().parse().ok()?,
            // the word this one spells; a Zemberek item name has its POS after `_`
            "Ref" => reference = value.trim().split('_').next().filter(|r| !r.is_empty()),
            _ => return None,
        }
    }
    // TurkishDictionaryLoader.isVerb: written in lower case, more than three letters, -mAk; a
    // CompoundP3sg -mAk word is a noun (ortaparmak).
    let verb_ending = ["mak", "mek"].into_iter().find(|e| {
        word.starts_with(char::is_lowercase) && word.chars().count() > 3 && word.ends_with(e)
    });
    let compound = names
        .iter()
        .any(|n| attribute_by_name(n) == Some(RootAttr::COMPOUND_P3SG));
    let primary = match primary {
        Some(p) => p,
        None if verb_ending.is_some() && !compound => PrimaryPos::Verb,
        None => PrimaryPos::Noun,
    };
    if secondary == SecondaryPos::None && word.starts_with(char::is_uppercase) {
        secondary = SecondaryPos::ProperNoun;
    }
    // An acronym written with a Latin I has i in its lemma (IBAN, COVID; Fbı is FBI), and its
    // guessed pronunciation reads the I as the letter ı (FBI "febeı") unless it reads as a
    // word (IBAN "iban").
    let lemma = normalize_word(word);
    let latin_i = secondary == SecondaryPos::Abbreviation
        && word.chars().count() > 1
        && (word.contains('I')
            || (word.starts_with(char::is_uppercase) && word.chars().skip(1).any(|c| c == 'ı')));
    let letter_reading = lemma.clone();
    let lemma = if latin_i {
        normalize_word(&word.replace('I', "İ")).replace('ı', "i")
    } else {
        lemma
    };
    let root = match verb_ending {
        Some(e) if primary == PrimaryPos::Verb => lemma[..lemma.len() - e.len()].to_string(),
        _ => lemma.clone(),
    };
    if !root_letters(&root) {
        return None;
    }
    // The word a Ref: names gives the lemma; the root keeps the variant (şöför: şoför).
    let lemma = reference.map_or(lemma, normalize_word);
    let mut item = DictionaryItem::new(lemma, root, primary, SecondaryPos::None, RootAttr::NONE);
    if secondary != SecondaryPos::None && !add_name(&mut item, secondary.as_str()) {
        return None;
    }
    for name in names {
        if !add_name(&mut item, name) {
            return None;
        }
    }
    if !consistent(&item) {
        return None;
    }
    if pronunciation.is_none() && latin_i {
        let guessed = guess_abbreviation(&letter_reading);
        pronunciation = Some(if guessed == letter_reading {
            guess_abbreviation(&item.root)
        } else {
            guessed
        });
    }
    item.pronunciation = pronunciation;
    item.compound_roots = roots;
    item.homograph = homograph;
    item.governed_case = governed_case;
    let written = if let Some(reference) = reference {
        Cow::Owned(reference.to_string())
    } else if latin_i {
        Cow::Owned(
            word.chars()
                .map(|c| {
                    if c == 'I' {
                        c
                    } else {
                        to_turkish_upper_char(c)
                    }
                })
                .collect(),
        )
    } else {
        Cow::Borrowed(word)
    };
    Some((written, item))
}

/// Applies a secondary POS or an attribute named in a dictionary line. Returns false for a
/// name that is neither (nor one of Zemberek's that akana ignores), and for a voice name on a
/// word that is neither a verb nor a pronoun.
fn add_name(item: &mut DictionaryItem, name: &str) -> bool {
    // Reflexive and Reciprocal name a pronoun's secondary POS (kendi, birbiri) but a verb's
    // root attribute: the verb carries the voice implicitly (yıkan-mak, atış-mak), which its
    // own root state gives every reading. Read as a secondary POS, 52 lexicon verbs lost it.
    let attr = attribute_by_name(name);
    let voice_name = attr.is_some_and(|a| {
        a.intersects(RootAttr::REFLEXIVE | RootAttr::RECIPROCAL | RootAttr::NON_RECIPROCAL)
    });
    let verb = item.primary_pos == PrimaryPos::Verb;
    if voice_name && !verb && item.primary_pos != PrimaryPos::Pron {
        return false;
    }
    match SecondaryPos::from_name(name).filter(|_| !(verb && voice_name)) {
        Some(sec) => {
            if item.secondary_pos == SecondaryPos::None {
                item.secondary_pos = sec;
            }
            if sec == SecondaryPos::ProperNoun {
                item.attributes |= RootAttr::PROPER_NOUN;
            }
            true
        }
        None => match attr {
            Some(attr) => {
                item.attributes |= attr;
                true
            }
            None => ignored_name(name),
        },
    }
}

/// What every parsed line gets: a guessed pronunciation, its display spelling and the
/// attributes Zemberek infers.
fn finish_item(item: &mut DictionaryItem, written: &str) {
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
    infer_attributes(item);
}
