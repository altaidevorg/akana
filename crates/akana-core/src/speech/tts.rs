//! The `tts-v1` and `tts-v2` profiles: Turkish phonemes and a prominence hierarchy for a TTS
//! model, from the phonology literature and Akana's own analyses, checked against the teacher
//! TTS's pitch. No eSpeak-derived data. `tts-v2`'s additions are marked `v2` here and listed in
//! `docs/phonemizer-tts-v2.md`; `tts-v1`'s output on text of letters is frozen.
//!
//! **Word stress** (the syllable that carries a word's stress):
//! - the last syllable by default (Lewis 2001; Kabak & Vogel 2001);
//! - a lexical exception keeps its own stress, and keeps it under suffixes (`şimdi`, `sonra`,
//!   `burada`, `anne`, `masa`; [`STRESS_LEXICON`] gives each entry's source);
//! - a place name is stressed by Sezer's (1981) rule: the antepenult when it is heavy and the
//!   penult light, else the penult (`Ankara`, `İstanbul`, `Antalya`);
//! - a compound the lexicon splits is stressed on its first part (`cumhurbaşkanı`);
//! - the first pre-accenting suffix ends the stress domain, and stress falls on the syllable
//!   before it: negative `-mA` (not the aorist `-mAz`), the copula `-DIr`, `-(y)DI`, `-(y)mIş`,
//!   `-(y)sA` after a nominal, conditional `-sA`, comitative `-(y)lA`, `-cA`, `-(y)ken`,
//!   `-mAdAn`;
//! - `-Iyor`, `-ArAk` and `-IncA` carry the stress on their first syllable.
//! - The teacher's F0 (EchoCraft's index of 14,549 training sentences; tuned on `st*` only)
//!   confirms the exceptions and the pre-accenting suffixes it has enough tokens for. It does not
//!   confirm eSpeak's misfires (`başladı`, `ettim`, `baktım` stay final) or pre-accenting `-ki`,
//!   so this profile has neither.
//!
//! **Prominence levels** (`stress`, per syllable; consonants take their syllable's level):
//! - 7: the nucleus, the clause's main prominence: the host of a question particle; else, before
//!   a clause-final finite verb, the nearest accented word in the clause's last intermediate
//!   phrase (after its last `_!`; the verb itself when there is none); else the clause's last
//!   accented word;
//! - 6: a phrase accent: the stressed syllable of an accented word before the nucleus;
//! - 5: unused (kept for emphasis);
//! - 4: the stressed syllable of a noun inside a phrase whose accent is on its first word: after
//!   an adjective, a numeral other than `bir` or a genitive (`kırmızı çanta`), and a possessed
//!   noun after a bare noun (`kapı kolu`);
//! - 3: the stressed syllable of a function word (pronoun, postposition, conjunction, determiner,
//!   `bir`, `şey`);
//! - 2: the stressed syllable of a word after the nucleus (post-nuclear deaccenting);
//! - 1: an unstressed syllable;
//! - 0: a clitic (`de`, `da`, every form of `mi`, `ki`, `ve`, `ya`).
//!
//! **Pauses**, **segments** and the teacher evidence for each rule: `docs/phonemizer-tts-v1.md`.

use super::letters::{has_vowel, key_to_phonemes, spell, tts_v2_segments, Seg};
use super::text::{segment, TextWord};
use super::{
    Pause, Phone, Phoneme, PhonemizeError, Profile, SegmentSource, StressSource, Token, Utterance,
    Warning, Word, MAX_TEXT_BYTES,
};
use crate::morphology::pos::{PrimaryPos, SecondaryPos};
use crate::morphology::{MorphParse, MorphologicalDisambiguator, TurkishMorphology};
use lazy_static::lazy_static;
use std::ops::Range;
use unicode_normalization::UnicodeNormalization;

pub(crate) struct Nlp {
    morphology: TurkishMorphology,
    disambiguator: MorphologicalDisambiguator,
}

lazy_static! {
    pub(crate) static ref NLP: Nlp = {
        let morphology = TurkishMorphology::new();
        let disambiguator = MorphologicalDisambiguator::with_morphology(morphology.clone());
        Nlp {
            morphology,
            disambiguator,
        }
    };
}

/// Words and roots with lexical (non-final) stress: (word or root, stressed syllable counted
/// from 0, source). Sources: L = Lewis 2001, GK = Göksel & Kerslake 2005, KV = Kabak & Vogel
/// 2001, T = the teacher's F0 (a peak on that syllable well above the control rate in at least
/// six mid-clause tokens), A = EchoCraft's audit (adverbs 67% and kinship terms 79% initial
/// peaks against 38% for controls).
pub(crate) const STRESS_LEXICON: &[(&str, u8, &str)] = &[
    // adverbs and discourse words
    ("şimdi", 0, "adverb:L,A,T"),
    ("sonra", 0, "adverb:L,A,T"),
    ("önce", 0, "adverb:L,T"),
    ("yarın", 0, "adverb:L,A"),
    ("nasıl", 0, "adverb:L,A"),
    ("hangi", 0, "adverb:L,A"),
    ("yine", 0, "adverb:L,A"),
    ("gene", 0, "adverb:L"),
    ("belki", 0, "adverb:L"),
    ("yalnız", 0, "adverb:L"),
    ("sadece", 0, "adverb:L,T"),
    ("bile", 0, "adverb:L,T"),
    ("lütfen", 0, "adverb:L,T"),
    ("hemen", 0, "adverb:L"),
    ("henüz", 0, "adverb:L"),
    ("zaten", 0, "adverb:L"),
    ("daha", 0, "adverb:L"),
    ("artık", 0, "adverb:L"),
    ("ancak", 0, "conjunction:L"),
    ("fakat", 0, "conjunction:L"),
    ("ama", 0, "conjunction:L"),
    ("yani", 0, "conjunction:L,T"),
    ("çünkü", 0, "conjunction:L"),
    ("sanki", 0, "adverb:L"),
    ("evet", 0, "interjection:L"),
    ("hayır", 0, "interjection:L"),
    ("bazen", 0, "adverb:L"),
    ("galiba", 0, "adverb:L"),
    ("tekrar", 0, "adverb:T"),
    ("hadi", 0, "interjection:L,T"),
    ("haydi", 0, "interjection:L,T"),
    ("peki", 0, "interjection:L"),
    ("hani", 0, "interjection:L"),
    ("işte", 0, "adverb:L"),
    ("elbette", 0, "adverb:T"),
    ("değil", 0, "copula:L,GK"),
    ("böyle", 0, "adverb:L,T"),
    ("öyle", 0, "adverb:L,T"),
    ("şöyle", 0, "adverb:L,T"),
    ("kimse", 0, "pronoun:L,T"),
    ("herkes", 0, "pronoun:L,T"),
    ("hepsi", 0, "pronoun:T"),
    ("hâlâ", 0, "adverb:L,T"),
    ("fazla", 0, "adverb:T"),
    ("acaba", 1, "adverb:T"),
    ("aşağı", 1, "adverb:T"),
    ("yukarı", 1, "adverb:L"),
    ("dışarı", 1, "adverb:T"),
    ("içeri", 1, "adverb:T"),
    ("yeniden", 1, "adverb:T"),
    ("tamamen", 1, "adverb:L,T"),
    ("aslında", 1, "adverb:T"),
    ("neredeyse", 1, "adverb:T"),
    ("birdenbire", 1, "adverb:T"),
    ("rastgele", 1, "adverb:T"),
    ("herhangi", 1, "adjective:T"),
    // postpositions (deaccented after their complement; lexically initial)
    ("gibi", 0, "postposition:L,T"),
    ("kadar", 0, "postposition:L,T"),
    ("diye", 0, "postposition:T"),
    ("beri", 0, "postposition:T"),
    // spatial and interrogative stems: stress stays on the stem's first syllable
    ("bura", 0, "pronoun:L,T"),
    ("şura", 0, "pronoun:L"),
    ("ora", 0, "pronoun:L,T"),
    ("nere", 0, "pronoun:L,T"),
    ("niçin", 0, "adverb:L"),
    ("niye", 0, "adverb:L"),
    // kinship terms
    ("anne", 0, "kinship:L,A"),
    ("amca", 0, "kinship:L,A"),
    ("teyze", 0, "kinship:L,A"),
    ("dayı", 0, "kinship:L,A"),
    ("hala", 0, "kinship:L,A"),
    ("abla", 0, "kinship:L"),
    ("ağabey", 0, "kinship:L"),
    // loans with non-final stress
    ("masa", 0, "loan:L"),
    ("lokanta", 1, "loan:L"),
    ("gazete", 1, "loan:L"),
    ("pencere", 0, "loan:L"),
    ("sinema", 0, "loan:L"),
    ("fabrika", 0, "loan:L"),
    ("banka", 0, "loan:L"),
    ("lamba", 0, "loan:L"),
    ("tiyatro", 1, "loan:L"),
    ("radyo", 0, "loan:L"),
    ("sandalye", 1, "loan:L"),
    ("politika", 1, "loan:L"),
    ("efendi", 1, "loan:L"),
    ("merhaba", 0, "interjection:L"),
];

/// Suffixes that end the stress domain (stress on the syllable before them).
const PRE_ACCENTING: &[&str] = &[
    "Neg",
    "Cop",
    "Ins",
    "Equ",
    "AsIf",
    "Ly",
    "While",
    "Cond",
    "WithoutHavingDoneSo",
];
/// Copular suffixes, pre-accenting after a nominal (`|Zero→Verb`).
const COPULAR: &[&str] = &["Past", "Narr", "Cond"];
/// Suffixes stressed on their first syllable (`-(y)AmA` too: söyleYEmem, yaPAmaz).
const STRESSED_SUFFIXES: &[&str] = &["Prog1", "ByDoingSo", "When", "Unable"];
/// Person endings of the z-paradigm: pre-accenting after these (gelmİŞsin, geleCEğim,
/// geLİrim, yorGUNum; Göksel & Kerslake 2005).
const PERSON: &[&str] = &["A1sg", "A2sg", "A1pl", "A2pl"];
const Z_PARADIGM_HOSTS: &[&str] = &["Fut", "Aor", "Narr", "Neces", "Prog1", "Prog2"];
/// Converbs and the conditional: an adverbial clause ends with them, and a speaker may pause
/// after it (Kan 2009; the teacher paused after 79 of them where eSpeak had planned no pause).
const CLAUSE_FINAL_FORMS: &[&str] = &[
    ":ByDoingSo",
    ":When",
    ":AfterDoingSo",
    ":While",
    ":AsLongAs",
    ":WithoutHavingDoneSo",
    ":SinceDoingSo",
    ":Cond",
];
/// Postpositions and nouns that close a participle or verbal-noun clause (`geldiği için`,
/// `gittikten sonra`, `-DIğI halde`): a pause may follow (the teacher paused after 97 of them).
const CLAUSE_CLOSERS: &[&str] = &[
    "için",
    "gibi",
    "rağmen",
    "beraber",
    "sonra",
    "önce",
    "halde",
    "zaman",
    "kadar",
    "dolayı",
    "karşın",
    "üzere",
    "sırada",
    "yerde",
    "anda",
    "esnada",
    "sıralarda",
];
const SUBORDINATE: &[&str] = &[
    "PastPart", "FutPart", "PresPart", "NarrPart", "AorPart", "Inf1", "Inf2", "Inf3",
];
/// Nouns too light to carry a phrase accent (the teacher deaccents them).
const WEAK_NOUNS: &[&str] = &["şey"];
/// Clitics: unstressed, prosodically leaning on their neighbour.
const CLITICS: &[&str] = &["de", "da", "mi", "mı", "mu", "mü", "ki", "ve", "ya"];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    Clitic,
    Weak,
    Content,
}

struct WordInfo {
    segs: Vec<Seg>,
    /// Index of the stressed vowel among the word's vowels.
    stress: Option<usize>,
    class: Class,
    stress_source: StressSource,
    segment_source: SegmentSource,
    spelled: bool,
    finite_verb: bool,
    /// Can open a phrase whose following noun is deaccented.
    modifier: bool,
    noun: bool,
    /// A possessed noun (P3sg/P3pl): the head of a compound or possessive phrase.
    possessed: bool,
    /// A bare common noun: opens a phrase only before a possessed head.
    bare_noun: bool,
    question: bool,
    level: u8,
}

fn is_vowel_letter(c: char) -> bool {
    matches!(
        c,
        'a' | 'e' | 'ı' | 'i' | 'o' | 'ö' | 'u' | 'ü' | 'â' | 'î' | 'û'
    )
}

/// `tts-v2`'s additions to [`STRESS_LEXICON`], the literature consensus of
/// `docs/phonemizer-tts-v2.md` (its source ids L1–L23): D = Dursunoğlu 2006 (*nerede*); GK =
/// Göksel & Kerslake 2025 (L2: unstressable *-rA*, pre-stressing *-en* and *-CA*, compounds); T =
/// Tekin 1991 (L17, quoting Banguoğlu and Gencan); KV = Kabak & Vogel 2001: 328 (*benimki*, a
/// single source); Ö = Özçelik 2014: 251 (L18, *-leyin*).
pub(crate) const STRESS_LEXICON_V2: &[(&str, u8, &str)] = &[
    ("nerede", 0, "pronoun:D,GK"),
    ("nereden", 0, "pronoun:D,GK"),
    ("neresi", 0, "pronoun:D,GK"),
    ("benimki", 1, "pronoun:KV(-ki)"),
    ("bizimki", 1, "pronoun:KV(-ki)"),
    ("sabahleyin", 1, "adverb:Ö(-leyin)"),
    ("akşamleyin", 1, "adverb:Ö(-leyin)"),
    ("geceleyin", 1, "adverb:Ö(-leyin)"),
    // adverbs in Arabic -en/-an are pre-stressing (kísmen, tamámen; L2, L17); only words whose
    // long vowel, if any, falls on the same syllable (not hakikaten)
    ("kısmen", 0, "adverb:GK,T(-en)"),
    ("şahsen", 0, "adverb:GK,T(-en)"),
    ("aslen", 0, "adverb:GK,T(-en)"),
    ("resmen", 0, "adverb:GK,T(-en)"),
    ("nispeten", 1, "adverb:GK,T(-en)"),
    ("fiilen", 0, "adverb:GK,T(-en)"),
    ("aynen", 0, "adverb:GK,T(-en)"),
    ("esasen", 1, "adverb:GK,T(-en)"),
    ("mecburen", 1, "adverb:GK,T(-en)"),
    ("takriben", 1, "adverb:GK,T(-en)"),
    ("tahminen", 1, "adverb:GK,T(-en)"),
    // equative and adverbial -CA is pre-stressing; the morphology has these as one word
    ("çocukça", 1, "adverb:GK,T(-CA)"),
    ("ayrıca", 1, "adverb:GK,T(-CA)"),
    // closed compounds take the first element's stress (L2; Kabak & Vogel 2001)
    ("bugün", 0, "compound:GK"),
    ("başbakan", 0, "compound:GK,KV"),
    ("başkent", 0, "compound:KV"),
    // a place name the morphology has as a common noun (Hárbiye; L17)
    ("harbiye", 0, "place:T(-iye)"),
];

fn lexicon_stress(key: &str, v2: bool) -> Option<(u8, &'static str)> {
    let v2_entry = if v2 {
        STRESS_LEXICON_V2.iter().find(|(w, _, _)| *w == key)
    } else {
        None
    };
    v2_entry
        .or_else(|| STRESS_LEXICON.iter().find(|(w, _, _)| *w == key))
        .map(|(_, s, src)| (*s, *src))
}

/// The standard intensive (emphatic reduplication) adjectives and adverbs: the base's first
/// (C)V, then p, m, r or s, and sometimes a linking a or e, before the base (kap-kara,
/// bem-beyaz, ter-temiz, ap-açık, yapa-yalnız, güpe-gündüz; L2, L17). A closed class: a list,
/// because the pattern alone also matches ordinary words (*araba*, *durdurdu*, *serseri*).
const INTENSIVES: &[&str] = &[
    "apaçık",
    "apak",
    "apaydın",
    "bambaşka",
    "besbelli",
    "bembeyaz",
    "bomboş",
    "büsbütün",
    "capcanlı",
    "çarçabuk",
    "çepeçevre",
    "dapdar",
    "dimdik",
    "dipdiri",
    "dopdolu",
    "dosdoğru",
    "dupduru",
    "dümdüz",
    "düpedüz",
    "gepgenç",
    "gömgök",
    "güpegündüz",
    "ıpıssız",
    "ipince",
    "kapkara",
    "kaskatı",
    "kıpkırmızı",
    "kıpkızıl",
    "kopkoyu",
    "koskoca",
    "kupkuru",
    "masmavi",
    "mosmor",
    "pespembe",
    "sapasağlam",
    "sapsağlam",
    "sapsarı",
    "simsiyah",
    "sımsıcak",
    "sımsıkı",
    "sipsivri",
    "sırsıklam",
    "taptaze",
    "tastamam",
    "tertemiz",
    "upuzun",
    "yamyassı",
    "yapayalnız",
    "yapyalnız",
    "yemyeşil",
    "yepyeni",
    "yusyuvarlak",
];

/// An intensive (one of [`INTENSIVES`], or a word built on one: *kapkaranlık*): the prefix's
/// length and the index of its vowel.
pub(crate) fn intensive(letters: &[char]) -> Option<(usize, usize)> {
    let word: String = letters.iter().collect();
    let entry = INTENSIVES
        .iter()
        .filter(|w| word.starts_with(**w))
        .max_by_key(|w| w.len())?;
    let e: Vec<char> = entry.chars().collect();
    let fv = e.iter().position(|&c| is_vowel_letter(c))?;
    let x = &e[..=fv];
    let k = fv + 1;
    // the prefix: X + consonant, or X + consonant + a linking vowel
    if e.get(k + 1..).is_some_and(|b| b.starts_with(x)) {
        Some((k + 1, fv))
    } else if e.get(k + 2..).is_some_and(|b| b.starts_with(x)) {
        Some((k + 2, fv))
    } else {
        Some((k + 1, fv))
    }
}

/// `tts-v2`'s stress for intensives and names (a vowel letter's index and the source):
/// - an intensive's prefix (kápkara; L2, L17);
/// - a place name (in the gazetteer, or a province written with a capital) by Sezer's rule
///   (Denízli, Ánkara; L20), but final in *-istan* (Hindistán; L2, L17);
/// - a name in *-iye* on the syllable before it (Hárbiye, Süleymániye, Türkiye; L17);
/// - a name in *-CIlAr* by Sezer's rule (Doğáncılar; L17, L20).
///
/// A name is a proper noun, or a word with a capital not at the start of a sentence or with an
/// apostrophe; only its letters before the apostrophe count.
fn v2_word_stress(
    key: &str,
    letters: &[char],
    parse: Option<&MorphParse>,
    capital: bool,
    initial: bool,
) -> Option<(usize, StressSource)> {
    if let Some((_, fv)) = intensive(letters) {
        return Some((
            fv,
            StressSource::Morphology {
                rule: "intensive prefix",
            },
        ));
    }
    let n = key
        .split('\'')
        .next()
        .map_or(0, |w| w.chars().filter(|&c| c != '\u{0302}').count())
        .min(letters.len());
    let name = &letters[..n];
    let vowels: Vec<usize> = (0..n).filter(|&i| is_vowel_letter(name[i])).collect();
    if vowels.len() < 2 {
        return None;
    }
    let name_str: String = name.iter().collect();
    let proper = parse.is_some_and(|p| p.secondary_pos == SecondaryPos::ProperNoun);
    let apostrophe = key.contains('\'');
    let named = proper || (capital && (!initial || apostrophe));
    // a province with a capital, unless it starts a sentence and the lexicon also has it as a
    // common word (Ordu "army"; Denizli has no common entry)
    let common = NLP
        .morphology
        .lexicon()
        .get_items(&name_str)
        .is_some_and(|items| {
            items.iter().any(|it| {
                !matches!(
                    it.secondary_pos,
                    SecondaryPos::ProperNoun | SecondaryPos::Abbreviation
                )
            })
        });
    let province =
        capital && crate::pii::gazetteer::PROVINCES.contains(&name_str) && (named || !common);
    let place = province || (named && crate::pii::is_place_name(&name_str));
    if !named && !place {
        return None;
    }
    if place && name_str.ends_with("stan") {
        return Some((
            *vowels.last()?,
            StressSource::Morphology {
                rule: "final(place name in -istan)",
            },
        ));
    }
    // -iye and -CIlAr only in a name the morphology or an apostrophe marks (not *Bu
    // Belediye*, *Satıcılar* in a heading)
    let marked = proper || place || apostrophe;
    if marked && name_str.ends_with("iye") && n >= 5 {
        if let Some(&v) = vowels.iter().rev().find(|&&v| v < n - 3) {
            return Some((
                v,
                StressSource::Morphology {
                    rule: "pre-accenting(-iye, name)",
                },
            ));
        }
    }
    let cilar = n >= 6
        && matches!(name[n - 5], 'c' | 'ç')
        && matches!(name[n - 4], 'ı' | 'i' | 'u' | 'ü')
        && name[n - 3] == 'l'
        && matches!(name[n - 2], 'a' | 'e')
        && name[n - 1] == 'r';
    if place || (marked && cilar) {
        return Some((vowels[sezer(name)], StressSource::Sezer));
    }
    None
}

/// Roots whose /e/ is not lowered before a coda sonorant in the modern standard (*hem*,
/// *elbise*, *Belçika*; L6, L1). *kendi* and *bel* vary and are left to the rule.
const NON_LOWERING: &[&str] = &["hem", "elbise", "belçika"];

/// `tts-v2`'s segment rules that need the word's analysis:
/// - the /e/ of the negative aorist *-mAz* and of *pekmez* is [æ] before the coda z (all nine
///   speakers of L6; L7); *merkez* varies and is left;
/// - [`NON_LOWERING`] roots keep [e];
/// - an intensive's prefix copies the base's first vowel (bem-beyaz [bem], pes-pembe [pæs]; L8,
///   L6).
fn v2_word_segments(segs: &mut [Seg], letters: &[char], parse: Option<&MorphParse>) {
    let e_at = |segs: &mut [Seg], at: usize, ph: Phoneme| {
        for s in segs.iter_mut() {
            if s.at as usize == at && !s.long && matches!(s.ph, Phoneme::E | Phoneme::Ae) {
                s.ph = ph;
            }
        }
    };
    // the z closes its syllable: no vowel follows it
    let coda = |i: usize| !letters.get(i + 1).is_some_and(|&c| is_vowel_letter(c));
    if let Some(p) = parse {
        let ms = morphemes(&p.analysis);
        let mut offsets = Vec::with_capacity(ms.len());
        let mut acc = 0usize;
        for m in &ms {
            offsets.push(acc);
            acc += m.surface.chars().count();
        }
        if acc == letters.len() {
            for k in 1..ms.len().saturating_sub(1) {
                if ms[k].id == "Neg" && ms[k + 1].id == "Aor" && ms[k + 1].surface.starts_with('z')
                {
                    let e = offsets[k] + 1;
                    if letters.get(e) == Some(&'e') && coda(offsets[k + 1]) {
                        e_at(segs, e, Phoneme::Ae);
                    }
                }
            }
        }
    }
    let root: String = match parse {
        Some(p)
            if letters
                .iter()
                .collect::<String>()
                .starts_with(p.root.as_str()) =>
        {
            p.root.clone()
        }
        _ => letters.iter().collect(),
    };
    let root_len = root.chars().count();
    if root == "pekmez" && coda(5) {
        e_at(segs, 4, Phoneme::Ae);
    }
    if NON_LOWERING.contains(&root.as_str()) {
        for s in segs.iter_mut() {
            // the e of el-bise, not one a suffix closes (elbise-m)
            if (s.at as usize) + 1 < root_len && s.ph == Phoneme::Ae {
                s.ph = Phoneme::E;
            }
        }
    }
    if let Some((plen, fv)) = intensive(letters) {
        let base = segs
            .iter()
            .find(|s| s.at as usize == plen + fv && s.ph.is_vowel())
            .map(|s| s.ph);
        if let Some(b @ (Phoneme::E | Phoneme::Ae)) = base {
            e_at(segs, fv, b);
        }
    }
}

/// One morpheme of an analysis: its surface, its id, and whether it follows a copular
/// derivation.
struct Morph<'a> {
    surface: &'a str,
    id: &'a str,
    after_copula: bool,
}

/// The morphemes of an analysis in Zemberek's long format
/// (`[gelmek:Verb] gel:Verb+me:Neg+di:Past+A3sg`).
fn morphemes(analysis: &str) -> Vec<Morph<'_>> {
    let body = match analysis.find("] ") {
        Some(i) => &analysis[i + 2..],
        None => analysis,
    };
    let mut out = Vec::new();
    let mut after_copula = false;
    for part in body.split(['+', '|']) {
        if part.starts_with("Zero→Verb") {
            after_copula = true;
            continue;
        }
        let part = part.split('→').next().unwrap_or(part);
        let (surface, id) = match part.split_once(':') {
            Some((s, i)) => (s, i),
            None => ("", part),
        };
        out.push(Morph {
            surface,
            id,
            after_copula,
        });
    }
    out
}

/// Sezer's rule on a root: the antepenult if it is heavy and the penult light, else the
/// penult (a syllable is heavy when closed or long).
fn sezer(root: &[char]) -> usize {
    let vowels: Vec<usize> = (0..root.len())
        .filter(|&i| is_vowel_letter(root[i]))
        .collect();
    let n = vowels.len();
    if n < 2 {
        return n.saturating_sub(1);
    }
    // a syllable is closed when two consonants follow its vowel before the next vowel, or one
    // or more at the end
    let heavy = |k: usize| -> bool {
        let v = vowels[k];
        if matches!(root[v], 'â' | 'î' | 'û') {
            return true;
        }
        let next = vowels.get(k + 1).copied().unwrap_or(root.len());
        let cons = next - v - 1;
        if k + 1 == n {
            cons >= 1
        } else {
            cons >= 2
        }
    };
    if n >= 3 && heavy(n - 3) && !heavy(n - 2) {
        n - 3
    } else {
        n - 2
    }
}

/// The word's stressed vowel letter (index among its letters), class and source.
fn word_stress(
    key: &str,
    letters: &[char],
    parse: Option<&MorphParse>,
    v2: bool,
    capital: bool,
    initial: bool,
) -> (Option<usize>, Class, StressSource) {
    let vowels: Vec<usize> = (0..letters.len())
        .filter(|&i| is_vowel_letter(letters[i]))
        .collect();
    let last = vowels.last().copied();
    let pos = parse.map(|p| p.pos).unwrap_or(PrimaryPos::Unknown);
    // the question particle in every form (mi, mısın, miydi, misiniz) is a clitic
    if pos == PrimaryPos::Q {
        return (None, Class::Clitic, StressSource::Clitic);
    }
    if CLITICS.contains(&key)
        && matches!(
            pos,
            PrimaryPos::Conj | PrimaryPos::Q | PrimaryPos::Unknown | PrimaryPos::Interj
        )
    {
        return (None, Class::Clitic, StressSource::Clitic);
    }
    let class = match pos {
        PrimaryPos::Pron
        | PrimaryPos::Postp
        | PrimaryPos::Conj
        | PrimaryPos::Det
        | PrimaryPos::Q
        | PrimaryPos::Interj => Class::Weak,
        _ if key == "bir" => Class::Weak,
        _ if parse.is_some_and(|p| WEAK_NOUNS.contains(&p.lemma.as_str())) => Class::Weak,
        _ => Class::Content,
    };
    let nth = |k: u8| vowels.get(k as usize).copied();
    if let Some((k, src)) = lexicon_stress(key, v2) {
        if let Some(v) = nth(k) {
            return (Some(v), class, StressSource::Lexical { source: src });
        }
    }
    if v2 {
        if let Some((v, src)) = v2_word_stress(key, letters, parse, capital, initial) {
            return (Some(v), class, src);
        }
    }
    let Some(p) = parse else {
        return (last, class, StressSource::DefaultFinal);
    };
    let ms = morphemes(&p.analysis);
    // morpheme letter offsets; the analysis must cover the word's letters
    let mut offsets = Vec::with_capacity(ms.len());
    let mut acc = 0usize;
    for m in &ms {
        offsets.push(acc);
        acc += m.surface.chars().count();
    }
    if acc != letters.len() || ms.is_empty() {
        return (last, class, StressSource::DefaultFinal);
    }
    // the end of the stress domain: the first pre-accenting suffix with a surface
    let mut domain_end = letters.len();
    let mut pre_id = "";
    for (k, m) in ms.iter().enumerate().skip(1) {
        if m.surface.is_empty() {
            continue;
        }
        let person = PERSON.contains(&m.id)
            && (m.after_copula || k >= 1 && Z_PARADIGM_HOSTS.contains(&ms[k - 1].id));
        // tts-v2: the pronominal -ki after a genitive (seNİNki; Kabak & Vogel 2001: 328)
        let rel_gen = v2 && m.id == "Rel" && k >= 1 && ms[k - 1].id == "Gen";
        let pre = (PRE_ACCENTING.contains(&m.id)
            || (m.after_copula && COPULAR.contains(&m.id))
            || person
            || rel_gen)
            // the aorist negative -mAz is stressed (gelmez)
            && !(m.id == "Neg"
                && ms
                    .get(k + 1)
                    .is_some_and(|n| n.surface.starts_with('z') && n.id.contains("Aor")));
        if pre {
            // the comitative's own part is its last two letters (-lA): the analyses `imle`,
            // `unla`, `yle` carry a genitive or a glide before it (beNİMle, oNUNla)
            let len = m.surface.chars().count();
            domain_end = if m.id == "Ins" && len > 2 {
                offsets[k] + len - 2
            } else {
                offsets[k]
            };
            pre_id = if person {
                "Person"
            } else if rel_gen {
                "RelGen"
            } else {
                m.id
            };
            break;
        }
    }
    let before = |limit: usize| vowels.iter().copied().rfind(|&v| v < limit);
    // lexical stress of the root: the stem lexicon, a place name, a compound
    let root_len = ms[0].surface.chars().count();
    let root: Vec<char> = letters[..root_len.min(letters.len())].to_vec();
    let root_key: String = root.iter().collect();
    let root_vowels: Vec<usize> = (0..root.len())
        .filter(|&i| is_vowel_letter(root[i]))
        .collect();
    let mut lexical: Option<(usize, StressSource)> = None;
    let lemma_key = p.lemma.as_str();
    if let Some((k, src)) = lexicon_stress(&root_key, v2).or_else(|| lexicon_stress(lemma_key, v2))
    {
        if let Some(&v) = root_vowels.get(k as usize) {
            lexical = Some((v, StressSource::Lexical { source: src }));
        }
    }
    if lexical.is_none()
        && p.secondary_pos == SecondaryPos::ProperNoun
        && crate::pii::is_place_name(lemma_key)
        && !root_vowels.is_empty()
    {
        let k = sezer(&root);
        lexical = Some((root_vowels[k], StressSource::Sezer));
    }
    if lexical.is_none()
        && p.secondary_pos == SecondaryPos::ProperNoun
        && root_vowels.len() >= 2
        && ["ca", "ce", "ça", "çe"]
            .iter()
            .any(|e| root_key.ends_with(e))
    {
        lexical = Some((
            root_vowels[root_vowels.len() - 2],
            StressSource::Morphology {
                rule: "pre-accenting(-cA, language name)",
            },
        ));
    }
    if lexical.is_none() {
        if let Some(items) = NLP.morphology.lexicon().get_items(&p.root) {
            if let Some((first, _)) = items
                .iter()
                .find(|i| i.lemma == p.lemma)
                .and_then(|i| i.compound_roots.as_ref())
            {
                let n = first.chars().count();
                if n > 0 && n < root.len() {
                    if let Some(v) = before(n) {
                        lexical = Some((v, StressSource::CompoundFirst));
                    }
                }
            }
        }
    }
    if let Some((v, src)) = lexical {
        if v < domain_end {
            return (Some(v), class, src);
        }
    }
    // a stressed suffix before the domain's end
    for (k, m) in ms.iter().enumerate().rev() {
        let stressed = STRESSED_SUFFIXES.contains(&m.id) || (v2 && m.id == "Hastily");
        if stressed && offsets[k] < domain_end {
            let first = vowels.iter().copied().find(|&v| v >= offsets[k]);
            if let Some(v) = first.filter(|&v| v < domain_end) {
                return (
                    Some(v),
                    class,
                    StressSource::Morphology {
                        rule: stressed_rule(m.id),
                    },
                );
            }
        }
    }
    if domain_end < letters.len() {
        if let Some(v) = before(domain_end) {
            return (
                Some(v),
                class,
                StressSource::Morphology {
                    rule: pre_rule(pre_id),
                },
            );
        }
    }
    (last, class, StressSource::DefaultFinal)
}

fn pre_rule(id: &str) -> &'static str {
    match id {
        "Neg" => "pre-accenting(-mA)",
        "Cop" => "pre-accenting(-DIr)",
        "Ins" => "pre-accenting(-(y)lA)",
        "Equ" | "AsIf" | "Ly" => "pre-accenting(-cA)",
        "While" => "pre-accenting(-(y)ken)",
        "Cond" => "pre-accenting(-(y)sA)",
        "Past" => "pre-accenting(copula -(y)DI)",
        "Narr" => "pre-accenting(copula -(y)mIş)",
        "WithoutHavingDoneSo" => "pre-accenting(-mAdAn)",
        "Person" => "pre-accenting(z-paradigm person)",
        "RelGen" => "pre-accenting(-ki after a genitive)",
        _ => "pre-accenting",
    }
}

fn stressed_rule(id: &str) -> &'static str {
    match id {
        "Prog1" => "stressed-suffix(-Iyor)",
        "ByDoingSo" => "stressed-suffix(-ArAk)",
        "When" => "stressed-suffix(-IncA)",
        "Unable" => "stressed-suffix(-(y)AmA)",
        "Hastily" => "stressed-suffix(-(y)Iver)",
        _ => "stressed-suffix",
    }
}

/// Lexicon attributes that change segments: the clear l of a loan with front-vowel suffixes
/// (Zemberek's InverseHarmony: hal, rol, kabul, meşgul), on the root's l.
fn lexical_segments(segs: &mut [Seg], parse: Option<&MorphParse>) -> bool {
    let Some(p) = parse else { return false };
    let Some(items) = NLP.morphology.lexicon().get_items(&p.root) else {
        return false;
    };
    let inverse = items.iter().any(|i| {
        i.lemma == p.lemma
            && i.attributes
                .contains(crate::morphology::pos::RootAttr::INVERSE_HARMONY)
    });
    if !inverse {
        return false;
    }
    let root_len = p.root.chars().count() as u32;
    let mut changed = false;
    for s in segs.iter_mut() {
        if s.at < root_len {
            let ph = match s.ph {
                Phoneme::LDark => Phoneme::L,
                Phoneme::LDarkCoda => Phoneme::LCoda,
                other => other,
            };
            if ph != s.ph {
                s.ph = ph;
                changed = true;
            }
        }
    }
    changed
}

fn finite_verb(p: &MorphParse) -> bool {
    p.pos == PrimaryPos::Verb && !p.analysis.contains("Zero→Verb")
}

/// An adjective, a numeral other than `bir`, or a genitive noun: opens a phrase whose following
/// noun is deaccented.
fn modifier(p: &MorphParse) -> bool {
    if p.analysis.contains("Zero→Verb") {
        return false;
    }
    match p.pos {
        PrimaryPos::Adj => true,
        PrimaryPos::Num => p.lemma != "bir",
        PrimaryPos::Noun => p.analysis.contains(":Gen"),
        _ => false,
    }
}

/// A common noun without case or possessive (the first part of a compound: `kapı kolu`).
fn bare_noun(p: &MorphParse) -> bool {
    p.pos == PrimaryPos::Noun
        && p.secondary_pos != SecondaryPos::ProperNoun
        && !p.analysis.contains("Zero→Verb")
        && !p.morpheme_tags.iter().any(|t| {
            matches!(
                t.as_str(),
                "Dat"
                    | "Loc"
                    | "Abl"
                    | "Acc"
                    | "Ins"
                    | "Equ"
                    | "Gen"
                    | "P1sg"
                    | "P2sg"
                    | "P3sg"
                    | "P1pl"
                    | "P2pl"
                    | "P3pl"
                    | "A3pl"
            )
        })
}

fn possessed(p: &MorphParse) -> bool {
    p.pos == PrimaryPos::Noun && p.morpheme_tags.iter().any(|t| t == "P3sg" || t == "P3pl")
}

/// The lexicon's pronunciation of an abbreviation (`tbmm` -> `tebememe`), for the verbaliser.
pub(crate) fn abbreviation_pronunciation(lower: &str) -> Option<String> {
    NLP.morphology
        .lexicon()
        .get_items(lower)?
        .iter()
        .find(|it| it.secondary_pos == SecondaryPos::Abbreviation && it.homograph == 0)
        .and_then(|it| it.pronunciation.clone())
}

/// Whether Akana's morphology reads `lower` as a word other than an abbreviation (`bugün`,
/// `ankara`), for words written in capitals.
pub(crate) fn is_known_word(lower: &str) -> bool {
    NLP.morphology
        .analyze(lower)
        .iter()
        .any(|p| p.secondary_pos != SecondaryPos::Abbreviation)
}

/// Phonemises `text` with a profile: verbalises it, phonemises the spoken text and maps every
/// word and warning back to the span of the text it reads.
pub(crate) fn phonemize_tts(
    text: &str,
    out: &mut Utterance,
    profile: Profile,
) -> Result<(), PhonemizeError> {
    out.words.clear();
    out.tokens.clear();
    out.warnings.clear();
    if text.len() > MAX_TEXT_BYTES {
        return Err(PhonemizeError::TooLong {
            len: text.len(),
            max: MAX_TEXT_BYTES,
        });
    }
    // tts-v1 keeps words written in letters as written: its output on such text is frozen
    let spoken = super::verbalize::verbalize_with(text, profile == Profile::TtsV2);
    if spoken.segments.is_empty() {
        return phonemize_spoken(text, out, profile);
    }
    phonemize_spoken(&spoken.text, out, profile)?;
    for w in &mut out.words {
        let (span, kind) = spoken.source(w.span.clone(), text.len());
        w.span = span;
        if kind.is_some() {
            w.normalized = kind;
        }
    }
    for w in &mut out.warnings {
        w.span = spoken.source(w.span.clone(), text.len()).0;
    }
    Ok(())
}

/// Phonemises text already verbalised.
fn phonemize_spoken(
    text: &str,
    out: &mut Utterance,
    profile: Profile,
) -> Result<(), PhonemizeError> {
    let v2 = profile == Profile::TtsV2;
    let mut tw: Vec<TextWord> = Vec::new();
    let final_gap = segment(text, &mut tw, v2);
    // analyses of the sentence's words, disambiguated in context
    // tts-v2 analyses the composed form of a decomposed word (NFD), without the dot of i̇
    let composed: Vec<String> = if v2 {
        tw.iter()
            .map(|w| {
                text[w.span.clone()]
                    .nfc()
                    .filter(|&c| c != '\u{0307}')
                    .collect()
            })
            .collect()
    } else {
        Vec::new()
    };
    let surfaces: Vec<&str> = tw
        .iter()
        .enumerate()
        .map(|(k, w)| {
            if w.number {
                w.key.trim_start_matches('\'')
            } else if v2 {
                composed[k].as_str()
            } else {
                &text[w.span.clone()]
            }
        })
        .collect();
    let parses: Vec<MorphParse> = if surfaces.is_empty() {
        Vec::new()
    } else {
        NLP.disambiguator.disambiguate(&surfaces)
    };
    let n_words = tw.len();
    let parse_of = |i: usize| -> Option<&MorphParse> {
        if parses.len() == n_words {
            parses.get(i).filter(|p| !p.analysis.is_empty())
        } else {
            None
        }
    };
    // optional pauses where an adverbial clause ends
    for i in 0..tw.len().saturating_sub(1) {
        if !tw[i + 1].pauses.is_empty() {
            continue;
        }
        let Some(p) = parse_of(i) else { continue };
        let converb = CLAUSE_FINAL_FORMS.iter().any(|f| p.analysis.contains(f));
        let closer = CLAUSE_CLOSERS.contains(&tw[i].key.as_str())
            && i > 0
            && parse_of(i - 1).is_some_and(|q| SUBORDINATE.iter().any(|s| q.analysis.contains(s)));
        if converb || closer {
            tw[i + 1].pauses.push(Pause::Short);
        }
    }
    let mut infos: Vec<WordInfo> = Vec::with_capacity(tw.len());
    for (i, w) in tw.iter().enumerate() {
        let letters: Vec<char> = w
            .key
            .chars()
            .filter(|&c| c != '\'' && c != '\u{0302}')
            .collect();
        let letter_str: String = letters.iter().collect();
        let parse = parse_of(i);
        let mut segs = Vec::new();
        if !has_vowel(&letter_str) {
            spell(&letter_str, &mut segs);
            let nv = segs.iter().filter(|s| s.ph.is_vowel()).count();
            infos.push(WordInfo {
                segs,
                stress: nv.checked_sub(1),
                class: Class::Content,
                stress_source: StressSource::Spelled,
                segment_source: SegmentSource::Spelled,
                spelled: true,
                finite_verb: false,
                modifier: false,
                noun: true,
                possessed: false,
                bare_noun: false,
                question: false,
                level: 0,
            });
            continue;
        }
        key_to_phonemes(&w.key, &mut segs);
        let attribute = lexical_segments(&mut segs, parse);
        if v2 {
            tts_v2_segments(&mut segs, &letters);
            v2_word_segments(&mut segs, &letters, parse);
        }
        let segment_source = if attribute {
            SegmentSource::LexiconAttribute
        } else if v2 {
            SegmentSource::TtsV2Rules
        } else {
            SegmentSource::TtsRules
        };
        // a capital, and the start of a sentence (for tts-v2's name rules)
        let word_text = &text[w.span.clone()];
        // a word in capitals throughout (a heading, shouting) is not a capitalised name
        let capital = word_text.chars().next().is_some_and(char::is_uppercase)
            && word_text.chars().any(char::is_lowercase);
        // the start of a sentence, a line, a list item or a quotation
        let initial = i == 0
            || text[w.gap.clone()].contains([
                '.', '!', '?', '…', ':', '\n', '\r', '"', '“', '«', '(', '–', '—',
            ]);
        let (stressed_letter, class, stress_source) =
            word_stress(&w.key, &letters, parse, v2, capital, initial);
        // the stressed letter's vowel segment (a merged long vowel starts at its first letter)
        let vowel_segs: Vec<u32> = segs
            .iter()
            .filter(|s| s.ph.is_vowel())
            .map(|s| s.at)
            .collect();
        let stress = stressed_letter.and_then(|l| {
            vowel_segs
                .iter()
                .rposition(|&at| at as usize <= l)
                .or(if vowel_segs.is_empty() { None } else { Some(0) })
        });
        let pos = parse.map(|p| p.pos).unwrap_or(PrimaryPos::Unknown);
        infos.push(WordInfo {
            segs,
            stress,
            class,
            stress_source,
            segment_source,
            spelled: false,
            finite_verb: parse.is_some_and(finite_verb),
            modifier: parse.is_some_and(modifier),
            noun: pos == PrimaryPos::Noun,
            possessed: parse.is_some_and(possessed),
            bare_noun: parse.is_some_and(bare_noun),
            question: pos == PrimaryPos::Q || matches!(w.key.as_str(), "mi" | "mı" | "mu" | "mü"),
            level: 0,
        });
    }
    // prominence, clause by clause
    let mut nucleus_of: Vec<bool> = vec![false; tw.len()];
    let mut start = 0;
    while start < tw.len() {
        let mut end = start + 1;
        while end < tw.len() && !tw[end].boundary {
            end += 1;
        }
        let nuc = prominence(&tw[start..end], &mut infos[start..end]);
        if let Some(n) = nuc {
            nucleus_of[start + n] = true;
        }
        start = end;
    }
    // marks: the punctuation that ends each clause
    let mut clause_marks: Vec<char> = Vec::new();
    for w in tw.iter().skip(1) {
        if w.boundary {
            clause_marks.push(mark_of(&text[w.gap.clone()]));
        }
    }
    clause_marks.push(mark_of(&text[final_gap.clone()]));
    // warnings
    for range in tw
        .iter()
        .map(|w| w.gap.clone())
        .chain(std::iter::once(final_gap.clone()))
    {
        warn(text, range, profile, &mut out.warnings);
    }
    // tokens
    let mut clause = 0usize;
    let mark = |c: usize| clause_marks.get(c).copied().unwrap_or('.');
    out.tokens.push(Token {
        profile,
        ph: Phoneme::Pause,
        stress: 0,
        word: false,
        lengthen: false,
        mark: mark(0),
    });
    for (wi, (w, info)) in tw.iter().zip(&infos).enumerate() {
        for p in &w.pauses {
            // a clause-ending pause belongs to the clause it ends
            out.tokens.push(Token {
                profile,
                ph: p.phoneme(),
                stress: 0,
                word: false,
                lengthen: false,
                mark: mark(clause),
            });
        }
        if w.boundary {
            clause += 1;
        }
        let link = tw.get(wi + 1).is_some_and(|n| n.pauses.is_empty())
            && infos
                .get(wi + 1)
                .and_then(|n| n.segs.first())
                .is_some_and(|s| s.ph.is_vowel());
        let levels = syllable_levels(info);
        // tts-v2, at the word's end: an open /e/ is [ɛ] in every word (Göksel & Kerslake, L1);
        // /i u/ are lowered only phrase-finally (Zimmer & Orgun 1999, L3; /y/ stays [y], the
        // aligner having no [ʏ]); /ɾ/ is devoiced and fricated phrase-finally (L13). Phrase-final
        // is before a clause pause `_:` or at the end of the text, not before the optional `_!`
        // a conjunction may take
        let phrase_final = v2
            && tw
                .get(wi + 1)
                .is_none_or(|n| n.pauses.contains(&Pause::ClauseEnd));
        let mut phones = Vec::with_capacity(info.segs.len());
        for (pi, s) in info.segs.iter().enumerate() {
            let mut ph = s.ph;
            if v2 && pi + 1 == info.segs.len() && !s.long {
                ph = match ph {
                    Phoneme::E => Phoneme::ELax,
                    Phoneme::I if phrase_final => Phoneme::ILax,
                    Phoneme::U if phrase_final => Phoneme::ULax,
                    Phoneme::Tap if phrase_final && s.letter == 'r' => Phoneme::RFricVoiceless,
                    p => p,
                };
            }
            if pi + 1 == info.segs.len() && link {
                ph = match ph {
                    Phoneme::LCoda => Phoneme::L,
                    Phoneme::LDarkCoda => Phoneme::LDark,
                    p => p,
                };
            }
            let stress = levels[pi];
            out.tokens.push(Token {
                profile,
                ph,
                stress,
                word: pi == 0,
                lengthen: s.long,
                mark: mark(clause),
            });
            phones.push(Phone {
                phoneme: ph,
                stress,
                lengthen: s.long,
            });
        }
        out.words.push(Word {
            span: w.span.clone(),
            reading: w.key.clone(),
            pauses_before: w.pauses.to_vec(),
            phones,
            clause,
            stress_source: info.stress_source.clone(),
            segment_source: info.segment_source,
            promoted: false,
            tonic: nucleus_of[wi],
            normalized: if w.number { Some("number") } else { None },
            prominence: info.level,
        });
    }
    out.tokens.push(Token {
        profile,
        ph: Phoneme::Pause,
        stress: 0,
        word: false,
        lengthen: false,
        mark: mark(clause),
    });
    Ok(())
}

/// The mark of a clause from the punctuation after it: its last `,` `.` `;` `:` `!` `?`; an
/// ellipsis alone is `.`; none is `.`.
fn mark_of(gap: &str) -> char {
    gap.chars()
        .rfind(|c| matches!(c, ',' | '.' | ';' | ':' | '!' | '?' | '…'))
        .map(|c| if c == '…' { '.' } else { c })
        .unwrap_or('.')
}

fn warn(text: &str, range: Range<usize>, profile: Profile, warnings: &mut Vec<Warning>) {
    for (off, c) in text[range.clone()].char_indices() {
        if c.is_whitespace()
            || matches!(
                c,
                ',' | '.'
                    | ';'
                    | ':'
                    | '!'
                    | '?'
                    | '…'
                    | '"'
                    | '“'
                    | '”'
                    | '«'
                    | '»'
                    | '\''
                    | '’'
                    | '-'
                    | '('
                    | ')'
            )
        {
            continue;
        }
        let s = range.start + off;
        warnings.push(Warning {
            span: s..s + c.len_utf8(),
            message: format!("{c:?} not read by {}", profile.name()),
        });
    }
}

/// Word levels in one clause; returns the nucleus's index in the clause.
fn prominence(tw: &[TextWord], infos: &mut [WordInfo]) -> Option<usize> {
    let n = infos.len();
    for i in 0..n {
        let level = match infos[i].class {
            Class::Clitic => 0,
            Class::Weak => 3,
            Class::Content => {
                let head = i > 0
                    && infos[i].noun
                    && !infos[i].finite_verb
                    && tw[i].pauses.is_empty()
                    && infos[i - 1].class == Class::Content
                    && !infos[i - 1].spelled
                    && (infos[i - 1].modifier || (infos[i - 1].bare_noun && infos[i].possessed));
                if head {
                    4
                } else {
                    6
                }
            }
        };
        infos[i].level = level;
    }
    // the phrase accent a word belongs to: itself, or for a phrase-internal head the word that
    // opens its phrase
    let accent_of = |infos: &[WordInfo], mut i: usize| -> usize {
        while i > 0 && infos[i].level == 4 {
            i -= 1;
        }
        i
    };
    let last_non_clitic = (0..n).rev().find(|&i| infos[i].class != Class::Clitic);
    let host = (1..n)
        .rev()
        .find(|&i| infos[i].question && infos[i - 1].class != Class::Clitic)
        .map(|i| i - 1);
    let nucleus = if let Some(h) = host {
        Some(accent_of(infos, h))
    } else if let Some(p) = last_non_clitic {
        if infos[p].finite_verb {
            // the accented word before the verb, within the verb's intermediate phrase (after the
            // last optional pause: coordinated verbs each keep their own accent)
            let phrase_start = (1..=p)
                .rev()
                .find(|&i| tw[i].pauses.contains(&Pause::Short))
                .unwrap_or(0);
            let before = (phrase_start..p)
                .rev()
                .find(|&i| infos[i].class == Class::Content && infos[i].stress.is_some());
            match before {
                Some(b) => Some(accent_of(infos, b)),
                None => Some(p),
            }
        } else {
            (0..n)
                .rev()
                .find(|&i| infos[i].level == 6)
                .or_else(|| (0..n).rev().find(|&i| infos[i].level == 4))
                .or(Some(p))
        }
    } else {
        None
    };
    if let Some(k) = nucleus {
        infos[k].level = 7;
        for info in infos.iter_mut().skip(k + 1) {
            if info.class != Class::Clitic {
                info.level = 2;
            }
        }
    }
    nucleus
}

/// The level of every segment of a word: its stressed syllable at the word's level, its other
/// syllables 1 (0 for a clitic), letter names of a spelled word at 3; a consonant takes its
/// syllable's level (one consonant before a vowel is that vowel's onset; the rest close the
/// syllable before).
fn syllable_levels(info: &WordInfo) -> Vec<u8> {
    let segs = &info.segs;
    let vowels: Vec<usize> = (0..segs.len()).filter(|&i| segs[i].ph.is_vowel()).collect();
    let nv = vowels.len();
    let vowel_level = |k: usize| -> u8 {
        if info.class == Class::Clitic {
            0
        } else if Some(k) == info.stress {
            info.level
        } else if info.spelled {
            info.level.min(3)
        } else {
            1
        }
    };
    let mut out = vec![0u8; segs.len()];
    if nv == 0 {
        return out;
    }
    for (pi, s) in segs.iter().enumerate() {
        // `vowels` is sorted: binary searches keep a long word linear-logarithmic
        let syl = if s.ph.is_vowel() {
            vowels.binary_search(&pi).unwrap_or(0)
        } else {
            // the vowel after it, when it is that vowel's single onset
            let k = vowels.partition_point(|&v| v <= pi);
            let next = (k < nv).then_some(k);
            match next {
                Some(k) if vowels[k] == pi + 1 => k,
                Some(0) => 0,
                Some(k) => k - 1,
                None => nv - 1,
            }
        };
        out[pi] = vowel_level(syl);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::{Phonemizer, Profile, Utterance};

    fn p() -> Phonemizer {
        Phonemizer::new(Profile::TtsV1)
    }

    /// The index (among its vowels) of each word's most prominent syllable.
    fn stressed(text: &str) -> Vec<usize> {
        let u = p().phonemize(text);
        u.words
            .iter()
            .map(|w| {
                let v: Vec<u8> = w
                    .phones
                    .iter()
                    .filter(|p| p.phoneme.is_vowel())
                    .map(|p| p.stress)
                    .collect();
                (0..v.len()).rev().max_by_key(|&k| v[k]).unwrap_or(0)
            })
            .collect()
    }

    fn prominence(text: &str) -> Vec<u8> {
        p().phonemize(text)
            .words
            .iter()
            .map(|w| w.prominence)
            .collect()
    }

    #[test]
    fn word_stress_follows_the_literature() {
        assert_eq!(stressed("gelmedi"), [0]); // pre-accenting negative
        assert_eq!(stressed("gelmez"), [1]); // the aorist negative is stressed
        assert_eq!(stressed("şimdi"), [0]); // lexical adverb
        assert_eq!(stressed("geliyorum"), [1]); // -Iyor
        assert_eq!(stressed("benimle"), [1]); // comitative -(y)lA
        assert_eq!(stressed("gelince"), [1]); // -IncA
        assert_eq!(stressed("öğretmendir"), [2]); // copula -DIr
        assert_eq!(stressed("söyleyemem"), [2]); // -(y)AmA
        assert_eq!(stressed("başladı"), [2]); // no eSpeak misfire
        assert_eq!(stressed("ettim"), [1]);
        assert_eq!(stressed("Ankara'da"), [0]); // Sezer, kept under suffixes
        assert_eq!(stressed("burada"), [0]); // spatial stem
    }

    #[test]
    fn prominence_marks_phrases_and_the_nucleus() {
        // adjective + noun: one phrase accent; the place name before the verb is the nucleus
        assert_eq!(
            prominence("Kırmızı çantayı Ankara'dan getirdi."),
            [6, 4, 7, 2]
        );
        // the question particle's host is the nucleus; the particle is a clitic
        assert_eq!(prominence("Serkan mı bavulu hazırladı?"), [7, 0, 2, 2]);
        // every clause has exactly one nucleus
        let u = p().phonemize("Eve gelince yemeğini yedi ve hemen uyudu, sonra çıktı.");
        for c in 0..=u.words.iter().map(|w| w.clause).max().unwrap_or(0) {
            let n = u
                .words
                .iter()
                .filter(|w| w.clause == c && w.prominence == 7)
                .count();
            assert_eq!(n, 1, "clause {c}");
        }
    }

    #[test]
    fn pauses_where_the_teacher_pauses() {
        let syms = |t: &str| -> Vec<&'static str> {
            p().phonemize(t)
                .tokens
                .iter()
                .filter(|t| t.ph.is_pause())
                .map(|t| t.ph.symbol())
                .collect()
        };
        // one `_:` at a clause boundary; none after a word with an apostrophe
        assert_eq!(syms("Ali'nin kitabı geldi, gitti."), ["_", "_:", "_"]);
        // optional `_!` before a conjunction and after a converb
        assert_eq!(syms("Eve gelince yedi ve uyudu."), ["_", "_!", "_!", "_"]);
        // a quote
        assert_eq!(syms("O \"evet\" dedi."), ["_", "_!", "_!", "_"]);
    }

    /// `tts-v2` as EchoCraft's probe prints it: IPA, words separated by spaces, `ˈ` before the
    /// utterance's most prominent vowel; pauses left out.
    fn v2(text: &str) -> String {
        let u = Phonemizer::new(Profile::TtsV2).phonemize(text);
        let toks: Vec<_> = u.tokens.iter().filter(|t| !t.ph.is_pause()).collect();
        let top = toks
            .iter()
            .filter(|t| t.ph.is_vowel())
            .map(|t| t.stress)
            .max()
            .unwrap_or(0);
        let mut out = String::new();
        let mut marked = false;
        for t in toks {
            if t.word && !out.is_empty() {
                out.push(' ');
            }
            if t.ph.is_vowel() && t.stress == top && !marked {
                out.push('ˈ');
                marked = true;
            }
            out.push_str(t.ipa().unwrap_or("?"));
        }
        out
    }

    /// The literature-consensus update of 2026-11-08 (EchoCraft's brief, items S1–S5, T1–T5),
    /// one assertion per expected output.
    #[test]
    fn tts_v2_follows_the_literature_consensus() {
        // S1: /ɾ/ a tap, devoiced only before a pause
        assert_eq!(v2("gelir."), "ɟelˈir̝̊");
        assert_eq!(v2("kar."), "kˈar̝̊");
        assert_eq!(v2("bir de"), "bˈiɾ dɛ");
        assert!(v2("bir gün geldi").starts_with("biɾ ɟˈyn"));
        assert!(v2("kar yağdı").starts_with("kˈaɾ "));
        assert_eq!(v2("var mı?"), "vˈaɾ mɯ");
        assert_eq!(v2("bir elma"), "biɾ ælmˈa");
        assert_eq!(v2("her şey"), "hæɾ ʃˈej");
        assert!(v2("kar topu").starts_with("kˈaɾ "));
        assert_eq!(v2("ruh"), "ɾˈuh");
        assert_eq!(v2("sonra"), "sˈonɾa");
        assert_eq!(v2("evlerde"), "evlæɾdˈɛ");
        // S2: -mAz and pekmez lowered before the coda z; not before a vowel
        assert_eq!(v2("gelmez"), "ɟælmˈæz");
        assert_eq!(v2("gelmezler"), "ɟælmæzlˈær̝̊");
        assert_eq!(v2("gelmez misin?"), "ɟælmˈæz misin");
        assert_eq!(v2("pekmez"), "pecmˈæz");
        assert_eq!(v2("gelmeziz"), "ɟælmezˈiz");
        // S3: hem, elbise, Belçika not lowered; kendi left to the rule
        assert_eq!(v2("hem"), "hˈem");
        assert_eq!(v2("elbise"), "elbisˈɛ");
        assert_eq!(v2("Belçika"), "bˈeltʃika");
        assert_eq!(v2("kendi"), "cændˈɪ");
        // S4: a word-final open /e/ is [ɛ] in every word; /i/ only phrase-finally
        assert_eq!(v2("dede geldi"), "dedˈɛ ɟældɪ");
        assert!(v2("bende kalsın").starts_with("bændˈɛ "));
        // S5: k palatal when tautosyllabic with a front vowel, velar before a back vowel
        assert_eq!(v2("ekonomi"), "ekonomˈɪ");
        assert_eq!(v2("dekan"), "dekˈan");
        assert_eq!(v2("teklif"), "teclˈif");
        assert_eq!(v2("ekmek"), "ecmˈec");
        assert_eq!(v2("gök"), "ɟˈœc");
        // T1: Arabic -en adverbs pre-stressing
        assert_eq!(v2("kısmen"), "kˈɯsmæn");
        assert_eq!(v2("şahsen"), "ʃˈahsæn");
        assert_eq!(v2("aslen"), "ˈaslæn");
        assert_eq!(v2("resmen"), "ɾˈesmæn");
        assert_eq!(v2("nispeten"), "nispˈetæn");
        assert_eq!(v2("fiilen"), "fˈiːlæn");
        assert_eq!(v2("tamamen"), "tamˈamæn");
        // T2: -CA pre-stressing
        assert_eq!(v2("çocukça"), "tʃodʒˈuktʃa");
        assert_eq!(v2("ayrıca"), "ajɾˈɯdʒa");
        assert_eq!(v2("kadınca"), "kadˈɯndʒa");
        // T3: place names by Sezer's rule, -iye pre-stressing, -istan final, -lAr not final
        assert_eq!(v2("Denizli"), "denˈizlɪ");
        assert!(v2("Biz Doğancılar'a gittik.").contains("doˈandʒɯɫaɾa"));
        assert_eq!(v2("Harbiye"), "hˈaɾbijɛ");
        assert_eq!(v2("Süleymaniye"), "sylejmˈanijɛ");
        assert_eq!(v2("Hindistan"), "hindistˈan");
        // a common word at the start of a sentence is not a name
        assert!(v2("Satıcılar indirim yaptı.").starts_with("satɯdʒɯɫaɾ"));
        // T4: compounds stressed on the first element; listed exceptions final
        assert_eq!(v2("bugün"), "bˈuɟyn");
        assert_eq!(v2("başbakan"), "bˈaʃbakan");
        assert_eq!(v2("bilgisayar"), "bilɟisajˈar̝̊");
        // T5: intensives stress the prefix, whose vowel copies the base's (L8)
        assert_eq!(v2("kapkara"), "kˈapkaɾa");
        assert_eq!(v2("bembeyaz"), "bˈembejaz");
        assert_eq!(v2("sapsarı"), "sˈapsaɾɯ");
        assert_eq!(v2("tertemiz"), "tˈeɾtemiz");
        assert_eq!(v2("masmavi"), "mˈasmavɪ");
        assert_eq!(v2("yapayalnız"), "jˈapajaɫnɯz");
        assert_eq!(v2("pespembe"), "pˈæspæmbɛ");
        assert_eq!(v2("mermer"), "mæɾmˈær̝̊");
        // the review of the update: no false intensives, names or devoicing
        assert_eq!(v2("araba"), "aɾabˈa");
        assert_eq!(v2("durdurdu"), "duɾduɾdˈʊ");
        assert_eq!(v2("serseri"), "sæɾseɾˈɪ");
        assert_eq!(v2("berbere"), "bæɾbeɾˈɛ");
        assert!(v2("Amasya").starts_with("amˈas"));
        assert!(v2("kapkaranlık").starts_with("kˈap"));
        assert!(v2("Bu Belediye").ends_with("beledijˈɛ"));
        assert!(v2("Başlık: Satıcılar indirim yaptı.")
            .replace('ˈ', "")
            .contains("satɯdʒɯɫaɾ"));
        assert!(v2("BU SATICILAR GELDİ.").contains("satɯdʒɯɫˈaɾ"));
        assert!(v2("Kar ve yağmur yağdı.").starts_with("kaɾ vɛ"));
        assert_eq!(v2("elbisem"), "elbisˈæm");
        // T6: unchanged single-source items
        assert_eq!(v2("yeniden"), "jenˈidæn");
        assert_eq!(v2("benimki"), "benˈimcɪ");
    }

    #[test]
    fn segments_long_vowels_and_geminates() {
        let ph = |t: &str| -> String {
            p().phonemize(t)
                .tokens
                .iter()
                .filter(|t| !t.ph.is_pause())
                .map(|t| t.ipa().unwrap_or("?"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        assert_eq!(ph("kâr"), "c aː r");
        assert_eq!(ph("saat"), "s aː t");
        assert_eq!(ph("eğlence"), "eː l æ n dʒ ɛ");
        assert_eq!(ph("değil"), "d e j ɪ l");
        assert_eq!(ph("hissetmek"), "h ɪ s s ɛ t m ɛ k");
        assert_eq!(ph("kabul"), "k a b ʊ l"); // clear l of an InverseHarmony loan
        assert_eq!(ph("elli"), "æ lː ɪ");
        // a geminate closes the syllable before it: lax ɪ
        assert_eq!(ph("dikkat"), "d ɪ kː a t");
        // ğ: long before a consonant or the end, one long vowel between equal vowels, silent
        // between back vowels, a glide between front vowels
        assert_eq!(ph("dağ"), "d aː");
        assert_eq!(ph("doğru"), "d oː ɾ ʊ");
        assert_eq!(ph("ağaç"), "aː tʃ");
        assert_eq!(ph("soğuk"), "s o ʊ k");
        assert_eq!(ph("sağır"), "s a ɯ r");
        assert_eq!(ph("düğün"), "d y j ø n");
        assert_eq!(ph("öğretmen"), "œː ɾ ɛ t m æ n");
    }

    /// The entries of the CTC aligner's vocabulary the profile may use
    /// (wav2vec2-xlsr-53-espeak-cv-ft, as EchoCraft's `aligner-vocab.txt` lists them).
    const ALIGNER: &[&str] = &[
        "a", "e", "i", "o", "u", "y", "ɛ", "ɪ", "ʊ", "ɔ", "æ", "ø", "œ", "ɯ", "aː", "eː", "iː",
        "oː", "uː", "yː", "øː", "œː", "ɛː", "ɪː", "ʊː", "ɔː", "æː", "ɯɯ", "b", "c", "d", "dʒ", "f",
        "ɡ", "ɟ", "h", "ʒ", "k", "l", "ɫ", "m", "n", "p", "r", "ɾ", "s", "ʃ", "t", "tʃ", "v", "j",
        "z", "ɣ", "w", "bː", "cː", "dː", "dʒː", "ɡː", "ɟː", "kː", "lː", "pː", "tː", "tʃː",
    ];

    #[test]
    fn every_ipa_is_one_aligner_entry() {
        let mut u = Utterance::default();
        let data = include_str!("../../tests/data/speech/tts_v1_golden.jsonl");
        let mut texts: Vec<String> = data
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .filter_map(|v| v["text"].as_str().map(str::to_string))
            .collect();
        for t in [
            "Hâlâ kâğıt, rüzgâr, dükkân, mahkûm, saat, tabii, Allah, eğri, iğne, hissetmek.",
            "TBMM, sms, Windows, iPhone, 1923'te, %25, 20:00, x, ğ.",
            "Kaç kişi? Yüz kişi! Hiç kimse gelmedi; herkes evdeydi...",
        ] {
            texts.push(t.to_string());
        }
        for t in &texts {
            p().phonemize_into(t, &mut u).expect("phonemize");
            for tok in &u.tokens {
                if tok.ph.is_pause() {
                    assert!(tok.ipa().is_none());
                    continue;
                }
                let ipa = tok.ipa().unwrap_or("");
                assert!(ALIGNER.contains(&ipa), "{ipa:?} in {t:?}");
                assert!(tok.stress <= 7);
            }
        }
    }

    #[test]
    fn deterministic_and_never_panics() {
        let a = p().phonemize("Ömer iki kadehten fazla içmemişti, fakat başı dönüyordu.");
        let b = p().phonemize("Ömer iki kadehten fazla içmemişti, fakat başı dönüyordu.");
        assert_eq!(a.tokens, b.tokens);
        for t in ["", "...", "\u{0302}", "-i", "'", "ğğğ", "😀", "a b c"] {
            let u = p().phonemize(t);
            assert_eq!(u.tokens.first().map(|t| t.ph.symbol()), Some("_"));
            assert_eq!(u.tokens.last().map(|t| t.ph.symbol()), Some("_"));
        }
    }
}
