//! Surface morphology heuristics of the content features: lexicon and clitic matchers,
//! suffix detectors (`-mış`, `-makta`, `-dık`, passive), the out-of-vocabulary proxy,
//! predicate tense and label colons.

use std::sync::LazyLock;

use ahash::AHashSet;
use smallvec::SmallVec;

use super::features::{lower_text, scan_tokens, Tok};
use super::lexicons::{LexEntry, Match};
use crate::phonology::alphabet::{is_hard_consonant, is_turkish_vowel, to_turkish_lower};
use crate::tokenization::SentenceSegmenter;

// ------------------------------------------------------------------- matchers

pub(super) fn lex_hit(word: &str, entries: &[LexEntry]) -> bool {
    entries.iter().any(|e| match e.mode {
        Match::Exact => word == e.word,
        Match::Prefix => word.starts_with(e.word),
        Match::QuestionClitic => is_question_clitic(word),
    })
}

const Q_TAILS: &[&str] = &[
    "", "sın", "sin", "sun", "sün", "sınız", "siniz", "sunuz", "sünüz", "yım", "yim", "yum", "yüm",
    "yız", "yiz", "yuz", "yüz", "ydı", "ydi", "ydu", "ydü", "ydım", "ydim", "ydum", "ydüm", "ydın",
    "ydin", "ydun", "ydün", "ydık", "ydik", "yduk", "ydük", "ymış", "ymiş", "ymuş", "ymüş", "yken",
];

pub(super) fn is_question_clitic(word: &str) -> bool {
    let Some(rest) = word.strip_prefix('m') else {
        return false;
    };
    let mut it = rest.chars();
    match it.next() {
        Some('ı' | 'i' | 'u' | 'ü') => Q_TAILS.contains(&it.as_str()),
        _ => false,
    }
}

const HIGH_VOWELS: [char; 4] = ['ı', 'i', 'u', 'ü'];

const MIS: [&str; 4] = ["mış", "miş", "muş", "müş"];
const MIS_TAILS: &[&str] = &[
    "", "tır", "tir", "tur", "tür", "tı", "ti", "tu", "tü", "sın", "sin", "sun", "sün", "ım", "im",
    "um", "üm", "ız", "iz", "uz", "üz", "sınız", "siniz", "sunuz", "sünüz", "lar", "ler", "lardır",
    "lerdir", "sa", "se", "ken", "çasına", "çesine",
];
/// Words ending in `mIş` that are not the suffix.
const MIS_EXCEPTIONS: &[&str] = &["yetmiş", "altmış", "kamış", "yemiş", "gümüş", "dümüş"];

/// `-mIş` (evidential / perfect participle) with an optional person, copula, past,
/// conditional or `-ken` ending, after a stem of at least 2 characters.
///
/// Precision limits: lexicalised `-mIş` words not in [`MIS_EXCEPTIONS`] count
/// (`kıymış`), participles followed by case endings (`gelmişi`) do not.
pub(super) fn is_mis(w: &str) -> bool {
    if !w.contains('ş') {
        return false;
    }
    MIS_TAILS.iter().any(|tail| {
        w.strip_suffix(tail).is_some_and(|base| {
            MIS.iter().any(|m| {
                base.strip_suffix(m).is_some_and(|stem| {
                    stem.chars().count() >= 2 && !MIS_EXCEPTIONS.contains(&base)
                })
            })
        })
    })
}

const MAKTA: [&str; 2] = ["makta", "mekte"];
const MAKTA_TAILS: &[&str] = &[
    "", "dır", "dir", "yım", "yim", "sın", "sin", "yız", "yiz", "sınız", "siniz", "lar", "ler",
    "dırlar", "dirler", "ydı", "ydi", "ymış", "ymiş", "yken", "ysa", "yse",
];

/// `-mAktA` progressive, usually `-mAktAdIr` (`görülmektedir`).
///
/// Precision limits: practically none known in running text; recall misses `-mAktAn`
/// ("from …ing", which is an ablative, not the progressive) on purpose.
pub(super) fn is_makta(w: &str) -> bool {
    if !(w.contains("makta") || w.contains("mekte")) {
        return false;
    }
    MAKTA_TAILS.iter().any(|tail| {
        w.strip_suffix(tail).is_some_and(|base| {
            MAKTA.iter().any(|m| {
                base.strip_suffix(m)
                    .is_some_and(|stem| stem.chars().count() >= 2)
            })
        })
    })
}

pub(super) type Chars = SmallVec<[char; 32]>;

fn starts_with_at(chars: &[char], at: usize, s: &str) -> bool {
    (at..).zip(s.chars()).all(|(k, c)| chars.get(k) == Some(&c))
}

/// `-DIk` participle / nominaliser: `D I ğ` + vowel (`geldiği`, `olduğunu`) or
/// `D I k` + `çA / tAn / lAr` (`geldikçe`, `geldikten`, `yaptıkları`), with at least two
/// characters of stem before it.
///
/// Precision limits: a bare `-DIk` (`geldik`) is the 1st person plural past and is not
/// counted; stems that happen to contain `dığ`/`tuk` + vowel (`tutukla-`) count; the
/// future participle `-AcAğ-` is not included.
pub(super) fn is_dik(chars: &[char]) -> bool {
    let n = chars.len();
    (2..n.saturating_sub(2)).any(|p| {
        if !matches!(chars[p], 'd' | 't') || !HIGH_VOWELS.contains(&chars[p + 1]) {
            return false;
        }
        match chars[p + 2] {
            'ğ' => chars.get(p + 3).is_some_and(|&c| is_turkish_vowel(c)),
            'k' => ["ça", "çe", "tan", "ten", "lar", "ler"]
                .iter()
                .any(|t| starts_with_at(chars, p + 3, t)),
            _ => false,
        }
    })
}

/// Verbal suffixes that may follow a passive marker.
const PASSIVE_TAILS: &[&str] = &[
    "mış", "miş", "muş", "müş", "dı", "di", "du", "dü", "ır", "ir", "ur", "ür", "acak", "ecek",
    "acağ", "eceğ", "makta", "mekte", "mak", "mek", "ma", "me", "abil", "ebil", "dığ", "diğ",
    "duğ", "düğ", "dık", "dik", "duk", "dük", "ıp", "ip", "up", "üp", "arak", "erek", "ınca",
    "ince", "unca", "ünce", "malı", "meli", "sa", "se",
];
/// Endings allowed after the participle `-An` (`yapılan`, `yapılanlar`, `verilene`).
const AN_TAILS: &[&str] = &[
    "", "lar", "ler", "ı", "i", "a", "e", "da", "de", "dan", "den", "ın", "in", "la", "le",
];
/// Verb stems ending in `-lA` whose following `n` is part of the root (`kullan-`).
const N_STEM_EXCEPTIONS: &[&str] = &["kulla"];

fn passive_tail_at(chars: &[char], at: usize) -> bool {
    if PASSIVE_TAILS.iter().any(|t| starts_with_at(chars, at, t)) {
        return true;
    }
    for an in ["an", "en"] {
        if starts_with_at(chars, at, an) {
            let rest: String = chars[at + 2..].iter().collect();
            if AN_TAILS.contains(&rest.as_str()) {
                return true;
            }
        }
    }
    false
}

/// Passive verb form, surface heuristic: stem (≥ 2 chars) + passive marker + verbal
/// suffix from [`PASSIVE_TAILS`] or the participle `-An` (+ plural/case). Markers:
/// `-Il-` after a consonant (`yap-ıl-mış`), `-In-` after `l` (`bil-in-en`), and `-n-`
/// after a `-lA` stem (`söyle-n-di`, `düzenle-n-en`).
///
/// Precision limits: reflexive `-lAn-`/`-In-` verbs (`dinlenen`, `bulunan`,
/// `yararlanan`) are counted as passive; roots ending in consonant + `Il` (`çekil-`,
/// `yayıl-`) also count; impersonal passives are not distinguished from personal ones;
/// passives of vowel-final stems other than `-lA` (`oku-n-an`) are missed.
pub(super) fn is_passive(chars: &[char]) -> bool {
    let n = chars.len();
    for p in 2..n {
        let c = chars[p];
        let next = chars.get(p + 1).copied();
        // -Il- after a consonant
        if HIGH_VOWELS.contains(&c)
            && next == Some('l')
            && !is_turkish_vowel(chars[p - 1])
            && passive_tail_at(chars, p + 2)
        {
            return true;
        }
        // -In- after l
        if HIGH_VOWELS.contains(&c)
            && next == Some('n')
            && chars[p - 1] == 'l'
            && passive_tail_at(chars, p + 2)
        {
            return true;
        }
        // -n- after a -lA stem
        if c == 'n'
            && p >= 3
            && matches!(chars[p - 1], 'a' | 'e')
            && chars[p - 2] == 'l'
            && passive_tail_at(chars, p + 1)
        {
            let stem: String = chars[..p].iter().collect();
            if !N_STEM_EXCEPTIONS.iter().any(|s| stem.ends_with(s)) {
                return true;
            }
        }
    }
    false
}

// ------------------------------------------------------------------- OOV proxy

/// zlib-compressed copy of the root lexicon the detector was trained with (Akana 0.6.0's
/// `morphology/zemberek_lexicon.txt`, byte for byte). The detector keeps its own copy so that
/// the morphology lexicon can be corrected without shifting the model's inputs; drop it when
/// the model is retrained on the current lexicon.
static TRAINING_LEXICON_ZZ: &[u8] = include_bytes!("../../../data/ai_detector/lexicon.txt.zz");

static TRAINING_LEXICON: LazyLock<String> = LazyLock::new(|| {
    use std::io::Read;
    let mut text = String::new();
    flate2::read::ZlibDecoder::new(TRAINING_LEXICON_ZZ)
        .read_to_string(&mut text)
        .expect("lexicon.txt.zz is valid zlib-compressed UTF-8");
    text
});

/// Root lexicon: column 1 (surface lemma) and column 2 (stem) of the training lexicon,
/// lower-cased with Turkish rules. Loaded once.
pub(super) static ROOTS: LazyLock<AHashSet<String>> = LazyLock::new(|| {
    let lex = TRAINING_LEXICON.as_str();
    let mut set = AHashSet::with_capacity(120_000);
    for line in lex.lines() {
        let mut cols = line.split_whitespace();
        for w in [cols.next(), cols.next()].into_iter().flatten() {
            set.insert(to_turkish_lower(w));
        }
    }
    set
});

/// Verb stems of the training lexicon (column 2 of the `Verb` rows), lower-cased with
/// Turkish rules. Only used to tell `-lAr` plurals from aorists (see [`is_aorist`]).
pub(super) static VERBS: LazyLock<AHashSet<String>> = LazyLock::new(|| {
    let lex = TRAINING_LEXICON.as_str();
    let mut set = AHashSet::with_capacity(8_000);
    for line in lex.lines() {
        let mut cols = line.split_whitespace();
        if let (Some(_), Some(stem), Some("Verb")) = (cols.next(), cols.next(), cols.next()) {
            set.insert(to_turkish_lower(stem));
        }
    }
    set
});

/// Surface suffix templates. Archiphonemes: `I` = ı/i/u/ü, `A` = a/e, `D` = d/t,
/// `C` = c/ç, `K` = k/ğ. Harmony between morphs is not enforced.
const SUFFIX_TEMPLATES: &[&str] = &[
    // number, possession, case
    "lAr", "lArI", "Im", "m", "In", "n", "I", "sI", "ImIz", "mIz", "InIz", "nIz", "yI", "nI", "A",
    "yA", "nA", "DA", "nDA", "DAn", "nDAn", "nIn", "lA", "ylA", "CA", "ki", "ken",
    "yken", // copula and person
    "DIr", "yIm", "sIn", "Iz", "yIz", "sInIz", "k", "lAr",
    // tense, aspect, mood, negation
    "DI", "mIş", "Ar", "Ir", "r", "AcAK", "yAcAK", "Iyor", "yor", "mAktA", "mAlI", "sA", "ysA",
    "mA", "mAz", "mAm", "mI", "yAbil", "Abil", "yA", // voice
    "Il", "l", "Iş", "t", "It", "DIr", "Ar", // verbal nouns, participles, converbs
    "mAK", "yIş", "An", "yAn", "DIK", "Ip", "yIp", "ArAk", "yArAk", "IncA", "yIncA", "mAdAn",
    "DIkçA", "CAsInA", // derivation
    "lIK", "lI", "sIz", "CI", "sAl", "lAş", "lAn", "CIK", "gI", "gIn", "IcI", "yIcI", "IntI", "Im",
    "ImsI", "DAş", "lAt",
];

fn expand_template(t: &str, out: &mut AHashSet<String>) {
    let mut acc: Vec<String> = vec![String::new()];
    for c in t.chars() {
        let options: &[char] = match c {
            'I' => &['ı', 'i', 'u', 'ü'],
            'A' => &['a', 'e'],
            'D' => &['d', 't'],
            'C' => &['c', 'ç'],
            'K' => &['k', 'ğ'],
            _ => &[],
        };
        if options.is_empty() {
            for s in &mut acc {
                s.push(c);
            }
        } else {
            acc = acc
                .iter()
                .flat_map(|s| {
                    options.iter().map(move |&o| {
                        let mut s2 = s.clone();
                        s2.push(o);
                        s2
                    })
                })
                .collect();
        }
    }
    out.extend(acc);
}

pub(super) static SUFFIXES: LazyLock<(AHashSet<String>, usize)> = LazyLock::new(|| {
    let mut set = AHashSet::new();
    for t in SUFFIX_TEMPLATES {
        expand_template(t, &mut set);
    }
    let max_chars = set.iter().map(|s| s.chars().count()).max().unwrap_or(0);
    (set, max_chars)
});

/// Reverses final-consonant softening of a stem: `kitab` → `kitap`, `ağac` → `ağaç`,
/// `renğ`/`çocuğ` → `-k`.
fn unsoften(stem: &str) -> Option<String> {
    let last = stem.chars().next_back()?;
    let hard = match last {
        'b' => 'p',
        'c' => 'ç',
        'd' => 't',
        'ğ' | 'g' => 'k',
        _ => return None,
    };
    let mut s = stem[..stem.len() - last.len_utf8()].to_string();
    s.push(hard);
    Some(s)
}

/// Whether a lower-cased word is explained by the root lexicon: either it is a root, or a
/// prefix of at least 2 characters is a root (directly or after reversing consonant
/// softening) and the rest splits into suffixes from [`SUFFIX_TEMPLATES`].
///
/// Precision limits: vowel dropping (`ağzı`, `burnu`), consonant doubling (`hakkı`) and
/// unlisted derivations count as OOV; the suffix chain ignores ordering and harmony, so
/// many misspellings that look like root + suffixes (`tabiki` = `tabi` + `ki`) pass.
pub(super) fn is_known(word: &str) -> bool {
    let roots = &*ROOTS;
    if roots.contains(word) {
        return true;
    }
    let (suffixes, max_chars) = &*SUFFIXES;
    // Char boundaries of `word`, including the end.
    let bounds: SmallVec<[usize; 32]> = word
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(word.len()))
        .collect();
    let n = bounds.len() - 1;
    if n < 3 {
        return false;
    }
    // ok[k]: word[bounds[k]..] is a (possibly empty) chain of suffixes.
    let mut ok: SmallVec<[bool; 32]> = SmallVec::from_elem(false, n + 1);
    ok[n] = true;
    for k in (2..n).rev() {
        ok[k] = (1..=*max_chars).any(|len| {
            k + len <= n && ok[k + len] && suffixes.contains(&word[bounds[k]..bounds[k + len]])
        });
    }
    (2..n).rev().any(|k| {
        if !ok[k] {
            return false;
        }
        let stem = &word[..bounds[k]];
        roots.contains(stem) || unsoften(stem).is_some_and(|s| roots.contains(&s))
    })
}

// ------------------------------------------------------------------- tense

/// Person/number endings after a positive aorist (`gelir-im`, `gelir-sin`, `gelir-iz`,
/// `gelir-siniz`, `gelir-ler`).
const AORIST_TAILS: &[&str] = &[
    "", "ım", "im", "um", "üm", "sın", "sin", "sun", "sün", "ız", "iz", "uz", "üz", "sınız",
    "siniz", "sunuz", "sünüz", "lar", "ler",
];
/// Endings after the negative aorist `-mAz` (`gelmez`, `gelmezsin`, `gelmezsiniz`,
/// `gelmezler`).
const NEG_AORIST_TAILS: &[&str] = &["", "sın", "sin", "sınız", "siniz", "lar", "ler"];
/// Negative aorist persons without `-mAz`: 1sg `-mAm` (`gelmem`), 1pl `-mAyIz` (`gelmeyiz`).
const NEG_AORIST_FORMS: &[&str] = &["mam", "mem", "mayız", "meyiz"];
/// Person endings after the progressive `-Iyor`.
const IYOR_TAILS: &[&str] = &["", "um", "sun", "uz", "sunuz", "lar"];
/// The potential `-(y)AbIl`, removed before the verb-stem lookup (`ol-abil-ir`).
const POTENTIAL: &[&str] = &["yabil", "yebil", "abil", "ebil"];
/// Common nouns, adjectives, interjections and postpositions that end like an aorist.
/// Compared with the whole word and with the base before a person ending (`hazır-ım`).
/// Caveat: the entries were picked by looking at false positives of the surface
/// rule among sentence-final words of the pooled train split (both labels together, not
/// ranked by label); they are linguistic non-verbs, not label-associated words.
const AORIST_EXCEPTIONS: &[&str] = &[
    "ağır",
    "anahtar",
    "bahar",
    "beraber",
    "biber",
    "diğer",
    "fakir",
    "fikir",
    "haber",
    "hamam",
    "hayir",
    "hayır",
    "hazır",
    "huzur",
    "kadar",
    "karar",
    "misafir",
    "miktar",
    "naber",
    "namaz",
    "pazar",
    "pekmez",
    "peynir",
    "rehber",
    "sabır",
    "sefer",
    "selamlar",
    "sıfır",
    "şair",
    "şehir",
    "şeker",
    "şiir",
    "şükür",
    "tamam",
    "tasar",
    "tekrar",
    "tesekkur",
    "teşekkür",
    "zafer",
    "zarar",
];

/// Tense of a sentence-final predicate, for `aorist_ratio`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Tense {
    Aorist,
    Progressive,
}

/// Present progressive `-Iyor` with an optional person ending (`bakıyor`, `diyor`,
/// `geliyorum`, `gelmiyorlar`): at least two characters ending in a high vowel before
/// `yor`, so `yorum` does not count.
///
/// Precision limits: practically none known; the past/evidential/conditional compounds
/// (`geliyordu`, `geliyorsa`) and `-Iyor` + copula (`geliyordur`) are not counted.
fn is_iyor(w: &str) -> bool {
    if !w.contains("yor") {
        return false;
    }
    IYOR_TAILS.iter().any(|tail| {
        w.strip_suffix(tail)
            .and_then(|b| b.strip_suffix("yor"))
            .is_some_and(|stem| {
                stem.chars().count() >= 2
                    && stem
                        .chars()
                        .next_back()
                        .is_some_and(|c| HIGH_VOWELS.contains(&c))
            })
    })
}

/// Whether the positive-aorist base `b` (… vowel + `r`) has a verb stem in [`VERBS`]: `b`
/// minus `r` (`sağla-r`, `etkile-r`) or minus vowel + `r` (`gel-ir`, `böl-er`), optionally
/// after removing the potential `-(y)AbIl` (`ol-abil-ir`) and reversing final-consonant
/// softening (`ed-er` → `et`).
fn has_verb_stem(b: &str) -> bool {
    let verbs = &*VERBS;
    let known = |s: &str| verbs.contains(s) || unsoften(s).is_some_and(|h| verbs.contains(&h));
    let Some(no_r) = b.strip_suffix('r') else {
        return false;
    };
    let no_vr = no_r
        .char_indices()
        .next_back()
        .map_or("", |(i, _)| &no_r[..i]);
    [no_r, no_vr].into_iter().any(|stem| {
        known(stem)
            || POTENTIAL
                .iter()
                .any(|p| stem.strip_suffix(p).is_some_and(known))
    })
}

/// Positive aorist base (the word without its person ending): at least 4 characters ending
/// in `a/e/ı/i/u/ü` + `r`, not the copula `-DIr`, not in [`AORIST_EXCEPTIONS`]. A base that
/// ends in `-lAr` (plural or a `-lA` verb: `kitaplar` / `sağlar`) or carries the 3pl ending
/// (`gelir-ler` / `fikir-ler`) must have a verb stem ([`has_verb_stem`]).
fn is_positive_aorist_base(b: &str, tail: &str) -> bool {
    let c: Chars = b.chars().collect();
    let n = c.len();
    if n < 4 || c[n - 1] != 'r' || !matches!(c[n - 2], 'a' | 'e' | 'ı' | 'i' | 'u' | 'ü') {
        return false;
    }
    if AORIST_EXCEPTIONS.contains(&b) {
        return false;
    }
    // Copula -DIr: d + high vowel (önemli-dir, var-dır), or t after a voiceless consonant
    // (sahip-tir, yok-tur, gelmiş-tir). -tIr after a vowel or a voiced consonant is the
    // aorist of a t-final stem (anlat-ır, belirt-ir).
    if HIGH_VOWELS.contains(&c[n - 2])
        && (c[n - 3] == 'd' || (c[n - 3] == 't' && is_hard_consonant(c[n - 4])))
    {
        return false;
    }
    let plural_like = c[n - 3] == 'l' && matches!(c[n - 2], 'a' | 'e');
    if plural_like || tail == "lar" || tail == "ler" {
        return has_verb_stem(b);
    }
    true
}

/// Aorist predicate, surface heuristic: positive `-Ir/-Ar/-r` with an optional person
/// ending (`gelir`, `bakar`, `okur`, `olabilir`, `ederim`, `gelirler`, `sağlar`), negative
/// `-mAz` (`gelmez`, `alamazsın`, `gelmezler`), `-mAm` (`gelmem`) and `-mAyIz`.
///
/// Precision limits: nouns and adjectives ending in vowel + `r` count unless they are in
/// [`AORIST_EXCEPTIONS`] (`gelir` "income" and `gider` "expense" count); a bare verb stem
/// ending in `r` plus a possessive (`oturum` "session") counts; plurals whose stem + `lA`
/// is a verb (`işler`, `taşlar`) count; `-tIr` after a voiceless consonant is read as the
/// copula, so `çıkart-ır` is missed; 3pl aorists whose stem is not in the lexicon are
/// missed; bases shorter than 4 characters (`der`, `yer`, `var`, `bir`) and the
/// past/conditional compounds (`gelirdi`, `gelirse`) never count.
fn is_aorist(w: &str) -> bool {
    if AORIST_EXCEPTIONS.contains(&w) || is_iyor(w) {
        return false;
    }
    let stem_ok = |stem: &str| stem.chars().count() >= 2;
    let negative = NEG_AORIST_TAILS.iter().any(|tail| {
        w.strip_suffix(tail).is_some_and(|b| {
            ["maz", "mez"]
                .iter()
                .any(|m| b.strip_suffix(m).is_some_and(stem_ok))
        })
    }) || NEG_AORIST_FORMS
        .iter()
        .any(|m| w.strip_suffix(m).is_some_and(stem_ok));
    negative
        || AORIST_TAILS.iter().any(|tail| {
            w.strip_suffix(tail)
                .is_some_and(|b| is_positive_aorist_base(b, tail))
        })
}

/// Tense of a sentence's final predicate, given its last two tokens: the last token, or
/// the word before a final
/// question clitic (`Gelir misin?` → `gelir`). `None` when that token is a mask or neither
/// an aorist nor a progressive form (past, future, `-mIş`, copula `-DIr`, `-mAktA`, nominal
/// sentences, headings, …).
pub(super) fn predicate_tense(last: Option<Tok>, prev: Option<Tok>) -> Option<Tense> {
    let word = |t: Option<Tok>| match t {
        Some(Tok::Word(w)) => Some(lower_text(w)),
        _ => None,
    };
    let mut w = word(last)?;
    if is_question_clitic(&w) {
        w = word(prev)?;
    }
    if is_aorist(&w) {
        Some(Tense::Aorist)
    } else if is_iyor(&w) {
        Some(Tense::Progressive)
    } else {
        None
    }
}

// ------------------------------------------------------------------- label colons

/// Most words a label may have (`Sonuç`, `Adım 1`, `Gereç ve yöntem`).
const MAX_LABEL_WORDS: usize = 4;

/// Cataphoric words: a colon after them introduces the clause or list they point to
/// (`Şunu söyledi:`, `Sorun şu:`, `Dedi ki:`, `İşte bazı öneriler:`), so it stays content
/// punctuation even after a short span.
const CATAPHORIC: &[&str] = &[
    "aşağıdaki",
    "aşağıdakiler",
    "aşağıdakileri",
    "aşağıdakilerdir",
    "işte",
    "ki",
    "şu",
    "şudur",
    "şuna",
    "şunda",
    "şundan",
    "şunlar",
    "şunlara",
    "şunları",
    "şunlardır",
    "şunu",
    "şunun",
    "şuydu",
    "şöyle",
    "şöyledir",
    "şöyleydi",
];

/// Person endings after the past `-DI` (`geldi-m`, `geldi-niz`, `geldi-ler`).
const PAST_TAILS: &[&str] = &["", "m", "n", "k", "nız", "niz", "nuz", "nüz", "lar", "ler"];

/// Past-tense verb `-DI` (+ person), told apart from a noun with a possessive by the root
/// lexicon: the word is not a root, and the word without its final vowel is not a non-verb
/// root, directly or after reversing softening (`fiyat-ı`, `ad-ı`, `adım` are nouns;
/// `dedi`, `söyledi`, `düşünüldü` are verbs, even though `söylet` is a verb root).
///
/// Precision limits: possessive nouns whose stem is not a root count as verbs (`kaydı`,
/// from `kayıt`), and so do nouns whose stem is also a verb (`tadı`, from `tat`); verbs
/// whose form minus the vowel is a noun root do not (`oldu`: `old`, `aldı`: `alt`).
fn is_past(w: &str) -> bool {
    let roots = &*ROOTS;
    if roots.contains(w) {
        return false;
    }
    let verbs = &*VERBS;
    let noun_root = |s: &str| roots.contains(s) && !verbs.contains(s);
    PAST_TAILS.iter().any(|tail| {
        w.strip_suffix(tail).is_some_and(|b| {
            let mut rev = b.chars().rev();
            let (Some(v), Some(d)) = (rev.next(), rev.next()) else {
                return false;
            };
            if !HIGH_VOWELS.contains(&v) || !matches!(d, 'd' | 't') || b.chars().count() < 4 {
                return false;
            }
            let noun = &b[..b.len() - v.len_utf8()];
            !(noun_root(noun) || unsoften(noun).is_some_and(|s| noun_root(&s)))
        })
    })
}

/// Whether the text before a colon, from the start of its line or sentence, is a label:
/// 1 to [`MAX_LABEL_WORDS`] whitespace-separated tokens with at least one letter or digit,
/// no [`CATAPHORIC`] word, and a last word that is not a finite past (`-DI`), progressive
/// (`-Iyor`) or `-mAktA` verb (`Ali dedi:`, `Gereği düşünüldü:`). A mask as the last token
/// (`Adım <SAYI>`) is fine.
fn is_label(span: &str) -> bool {
    let n = span.split_whitespace().count();
    if n == 0 || n > MAX_LABEL_WORDS || !span.chars().any(char::is_alphanumeric) {
        return false;
    }
    let lower = lower_text(span);
    let mut cataphoric = false;
    let mut last_word: Option<&str> = None;
    scan_tokens(&lower, |tok| match tok {
        Tok::Word(w) => {
            cataphoric |= CATAPHORIC.contains(&w);
            last_word = Some(w);
        }
        Tok::Mask => last_word = None,
    });
    !cataphoric && !last_word.is_some_and(|w| is_past(w) || is_iyor(w) || is_makta(w))
}

/// Counts the colons that `colons_per_100w` looks at (`:` followed by whitespace or the end
/// of the text) and how many of them end a line- or sentence-leading label
/// (`Adım 1: …`, `Örnek: …`, `… bitti. Sonuç: …`). Returns `(colons, label_colons)`.
///
/// A colon is a label colon when it is the first such colon of its sentence
/// ([`SentenceSegmenter`] run on each line; the line start counts as a sentence start) and
/// the text from the sentence start to the colon passes [`is_label`]. Features run on
/// preprocessed text, where list markers and emphasis are already stripped, so
/// `- **Adım 1:** …` arrives here as `Adım <SAYI>: …`.
///
/// Precision limits: short imperative or aorist clauses (`Bakın:`, `Hoca der:`) and short
/// noun-phrase lead-ins (`Cevap basit:`) count as labels; see [`is_past`] for the past-tense
/// check; a label glued to the previous sentence without a space (`bitti.Not:`) counts only
/// if the whole span is at most [`MAX_LABEL_WORDS`] tokens.
pub(super) fn label_colon_counts(text: &str) -> (usize, usize) {
    let (mut colons, mut labels) = (0usize, 0usize);
    for line in text.split('\n') {
        let found: SmallVec<[usize; 8]> = line
            .match_indices(':')
            .map(|(i, _)| i)
            .filter(|&i| line[i + 1..].chars().next().is_none_or(char::is_whitespace))
            .collect();
        if found.is_empty() {
            continue;
        }
        colons += found.len();
        let starts: SmallVec<[usize; 8]> = SentenceSegmenter::segment(line)
            .iter()
            .map(|s| s.start)
            .filter(|&s| line.is_char_boundary(s))
            .collect();
        let mut prev_unit = None;
        for i in found {
            let unit = starts
                .iter()
                .copied()
                .filter(|&s| s <= i)
                .max()
                .unwrap_or(0);
            if prev_unit == Some(unit) {
                continue;
            }
            prev_unit = Some(unit);
            labels += usize::from(is_label(&line[unit..i]));
        }
    }
    (colons, labels)
}
