//! Letters to phonemes. [`key_to_phonemes`] gives `tts-v1`'s segments (frozen: vowel laxing in
//! closed and word-final syllables, `&` before a tautosyllabic m n l, the tap between vowels and
//! the trill elsewhere, clear and dark l by the neighbouring vowel, `J` before a front vowel,
//! long vowels and geminates); [`tts_v2_segments`] turns them into `tts-v2`'s, the allophones of
//! the phonology literature.

use super::Phoneme::{self, *};

/// One phoneme of a word, with the letter it came from when a later rule needs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Seg {
    pub ph: Phoneme,
    pub long: bool,
    /// The source letter of a rule-derived segment (`g` for `g`/`J`); `\0` from the lexicon.
    pub letter: char,
    /// The index of its first letter among the word's letters (apostrophes left out); 0 from the
    /// lexicon.
    pub at: u32,
}

impl Seg {
    pub fn new(ph: Phoneme, long: bool) -> Self {
        Seg {
            ph,
            long,
            letter: '\0',
            at: 0,
        }
    }
}

#[derive(Clone, Copy)]
enum Unit {
    V {
        v: char,
        long: bool,
        circ: bool,
        at: usize,
    },
    C {
        c: char,
        long: bool,
        at: usize,
    },
}

fn is_vowel_letter(c: char) -> bool {
    matches!(
        c,
        'a' | 'e' | 'ı' | 'i' | 'o' | 'ö' | 'u' | 'ü' | 'â' | 'î' | 'û'
    )
}

fn is_front_letter(v: char) -> bool {
    matches!(v, 'e' | 'i' | 'ö' | 'ü')
}

/// Whether a lower-case word has a vowel letter (otherwise it is spelled).
pub(crate) fn has_vowel(letters: &str) -> bool {
    letters.chars().any(is_vowel_letter)
}

/// `tts-v1`'s phonemes of a lexicon key: lower-case letters, circumflex vowels kept, apostrophes where
/// the text had them. An l before an apostrophe takes its quality from the vowel before it and
/// is not a coda (observed: Kemal'i `L`, İsmail'e `l`); the lone letter i is tense (observed:
/// the i of iPhone).
///
/// The profile's long vowels and geminates (Zimmer & Orgun 1999, Göksel & Kerslake 2005, Lewis
/// 2001):
/// - a circumflex vowel is long, and k before it palatal `c` (kâr, dükkân, mahkûm);
/// - two equal vowel letters are one long vowel (saat, kanaat, tabii);
/// - ğ after e or i lengthens the vowel before a consonant or at the end (eğlence, iğne), and is
///   `j` only between vowels (değil);
/// - long vowels are tense;
/// - doubled ç and g are one long consonant, as doubled stops are, and a doubled clear l one
///   long `l`;
/// - no eSpeak `-ken` palatal.
pub(crate) fn key_to_phonemes(key: &str, out: &mut Vec<Seg>) {
    out.clear();
    let mut w: Vec<char> = Vec::with_capacity(key.len());
    let mut before_apostrophe: Vec<bool> = Vec::with_capacity(key.len());
    for c in key.chars() {
        if c == '\'' {
            if let Some(b) = before_apostrophe.last_mut() {
                *b = true;
            }
            continue;
        }
        if c == '\u{0302}' {
            continue;
        }
        w.push(c);
        before_apostrophe.push(false);
    }
    if w.len() == 1 && w[0] == 'i' {
        out.push(Seg {
            ph: I,
            long: false,
            letter: 'i',
            at: 0,
        });
        return;
    }
    let n = w.len();
    let mut units: Vec<Unit> = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let ch = w[i];
        if is_vowel_letter(ch) {
            let base = match ch {
                'â' => 'a',
                'î' => 'i',
                'û' => 'u',
                c => c,
            };
            let circ = matches!(ch, 'â' | 'î' | 'û');
            if i + 1 < n && w[i + 1] == 'ğ' {
                // ğ (Göksel & Kerslake 2005; Ünal-Logacev, Żygis & Fuchs 2019): before a consonant
                // or the end it lengthens the vowel; two equal vowels across it are one long vowel;
                // between back vowels a hiatus, the first vowel short; between front vowels a
                // glide j
                let next = w.get(i + 2).map(|&c| match c {
                    'â' => 'a',
                    'î' => 'i',
                    'û' => 'u',
                    c => c,
                });
                match next {
                    Some(nb) if is_vowel_letter(nb) => {
                        if !matches!(base, 'e' | 'i' | 'ö' | 'ü') {
                            let same = nb == base;
                            units.push(Unit::V {
                                v: base,
                                long: same || circ,
                                circ,
                                at: i,
                            });
                            i += if same { 3 } else { 2 };
                            continue;
                        }
                        // a front vowel: the vowel, then ğ as j
                    }
                    _ => {
                        units.push(Unit::V {
                            v: base,
                            long: true,
                            circ,
                            at: i,
                        });
                        i += 2;
                        continue;
                    }
                }
            }
            // two equal vowel letters: one long vowel
            if i + 1 < n && w[i + 1] == ch {
                units.push(Unit::V {
                    v: base,
                    long: true,
                    circ,
                    at: i,
                });
                i += 2;
                continue;
            }
            units.push(Unit::V {
                v: base,
                long: circ,
                circ,
                at: i,
            });
            i += 1;
            continue;
        }
        if ch == 'ğ' {
            let prev = if i > 0 { w[i - 1] } else { '\0' };
            let c = if matches!(prev, 'e' | 'i' | 'ö' | 'ü') {
                'ğ'
            } else {
                'Q'
            };
            units.push(Unit::C {
                c,
                long: false,
                at: i,
            });
            i += 1;
            continue;
        }
        let doubled_long = matches!(ch, 'b' | 'c' | 'd' | 'k' | 'p' | 't' | 'ç' | 'g');
        if i + 1 < n && w[i + 1] == ch && doubled_long {
            units.push(Unit::C {
                c: ch,
                long: true,
                at: i,
            });
            i += 2;
            continue;
        }
        units.push(Unit::C {
            c: ch,
            long: false,
            at: i,
        });
        i += 1;
    }
    let m = units.len();
    let vowel_at = |k: usize| -> Option<(char, bool)> {
        match units.get(k) {
            Some(Unit::V { v, circ, .. }) => Some((*v, *circ)),
            _ => None,
        }
    };
    // the nearest vowel before each unit (linear in a long word)
    let mut vowel_before: Vec<Option<(char, bool)>> = vec![None; m];
    for k in 1..m {
        vowel_before[k] = vowel_at(k - 1).or(vowel_before[k - 1]);
    }
    for k in 0..m {
        match units[k] {
            Unit::V { v, long, at, .. } => {
                let mut j = k + 1;
                while j < m && matches!(units[j], Unit::C { .. }) {
                    j += 1;
                }
                // a geminate closes the syllable before it (dik.kat), as two letters do
                let ncons = j - k - 1
                    + units[k + 1..j]
                        .iter()
                        .filter(|u| matches!(u, Unit::C { long: true, .. }))
                        .count();
                let fin = j == m;
                let closed = (fin && ncons >= 1) || (!fin && ncons >= 2);
                let lax = (closed || (fin && ncons == 0)) && !long;
                let ph = match v {
                    'a' => A,
                    'ı' => Ih,
                    'ö' => Oe,
                    'i' => {
                        if lax {
                            ILax
                        } else {
                            I
                        }
                    }
                    'u' => {
                        if lax {
                            ULax
                        } else {
                            U
                        }
                    }
                    'ü' => {
                        if lax {
                            UeLax
                        } else {
                            Ue
                        }
                    }
                    'o' => {
                        if lax {
                            OLax
                        } else {
                            O
                        }
                    }
                    _ => {
                        // e
                        if lax {
                            let next = match units.get(k + 1) {
                                Some(Unit::C { c, .. }) => Some(*c),
                                _ => None,
                            };
                            if closed && matches!(next, Some('m' | 'n' | 'l')) {
                                Ae
                            } else {
                                ELax
                            }
                        } else {
                            E
                        }
                    }
                };
                out.push(Seg {
                    ph,
                    long,
                    letter: v,
                    at: at as u32,
                });
            }
            Unit::C { c, long, at } => {
                let prev_v = k > 0 && vowel_at(k - 1).is_some();
                let next_v = vowel_at(k + 1);
                let ph = match c {
                    'r' => {
                        if prev_v && next_v.is_some() {
                            Tap
                        } else {
                            R
                        }
                    }
                    'l' if before_apostrophe[at] && next_v.is_some() => {
                        let pv = vowel_before[k];
                        match pv {
                            Some((v, circ)) if is_front_letter(v) || circ => L,
                            _ => LDark,
                        }
                    }
                    'l' => {
                        if let Some((nv, circ)) = next_v {
                            if is_front_letter(nv) || circ {
                                L
                            } else {
                                LDark
                            }
                        } else {
                            let pv = vowel_before[k];
                            match pv {
                                Some((v, circ)) if is_front_letter(v) || circ => LCoda,
                                _ => LDarkCoda,
                            }
                        }
                    }
                    'g' => match next_v {
                        Some((nv, circ)) if is_front_letter(nv) || circ => GPalatal,
                        _ => G,
                    },
                    'k' => match next_v {
                        Some((_, true)) => KPalatal,
                        _ => K,
                    },
                    'Q' => Gh,
                    'x' => {
                        out.push(Seg {
                            ph: K,
                            long: false,
                            letter: 'x',
                            at: at as u32,
                        });
                        S
                    }
                    'b' => B,
                    'c' => Dzh,
                    'ç' => Tsh,
                    'd' => D,
                    'f' => F,
                    'h' => H,
                    'j' => Zh,
                    'm' => M,
                    'n' => N,
                    'p' => P,
                    's' => S,
                    'ş' => Sh,
                    't' => T,
                    'v' => V,
                    'y' | 'ğ' => J,
                    'z' => Z,
                    'w' => W,
                    'q' => K,
                    _ => continue,
                };
                out.push(Seg {
                    ph,
                    long,
                    letter: c,
                    at: at as u32,
                });
            }
        }
    }
    // a doubled clear l: one long l (the aligner has `lː`, not a long dark l)
    let mut k = 0;
    while k + 1 < out.len() {
        if out[k].ph == LCoda && out[k + 1].ph == L && out[k].letter == 'l' {
            out[k].ph = L;
            out[k].long = true;
            out.remove(k + 1);
        }
        k += 1;
    }
}

/// `tts-v2`'s segments, from `tts-v1`'s (`letters`: the word's letters, as [`Seg::at`] counts
/// them). Source ids refer to `docs/phonemizer-tts-v2.md`:
/// - vowels are tense inside the word: no closed-syllable laxing, which is receding (Gopal &
///   Nichols 2022, L7), and no lax o; the caller lowers a word-final open /e/ and phrase-final
///   /i u/ (Göksel & Kerslake, L1; Zimmer & Orgun 1999, L3);
/// - /e/ is [æ] before a tautosyllabic coda m, n, l or r, syllabified within the word (Dadan,
///   Erken & Canalis 2024, L6; Gopal & Nichols, L7; İleri 2025, L8);
/// - /ɾ/ is a tap; the caller devoices it before a pause (Sanıyar & Tunçer 2022, L13);
/// - /k g/ are palatal [c ɟ] when tautosyllabic with a front vowel: before a front vowel in the
///   onset, after one in the coda (Clements & Sezer 1982, via van der Hulst & van de Weijer
///   1991, L15), and before a circumflex vowel (lexical palatals of loans: *kâr*, *rüzgâr*);
/// - a circumflex vowel is long only in an open syllable: underlying long vowels of loans
///   shorten in closed syllables (van der Hulst & van de Weijer 1991, after Underhill, L15).
pub(crate) fn tts_v2_segments(segs: &mut [Seg], letters: &[char]) {
    let n = segs.len();
    for s in segs.iter_mut() {
        s.ph = match s.ph {
            ILax => I,
            ULax => U,
            UeLax => Ue,
            OLax => O,
            ELax | Ae => E,
            p => p,
        };
    }
    let front_vowel = |s: &Seg| -> bool {
        matches!(s.ph, E | ELax | Ae | I | ILax | Ue | UeLax | Oe)
            || matches!(letters.get(s.at as usize), Some('î'))
    };
    let circumflex = |s: &Seg| matches!(letters.get(s.at as usize), Some('â' | 'î' | 'û'));
    // the nearest vowel before and after each segment (linear in a long word)
    let mut prev_vowel: Vec<Option<usize>> = vec![None; n];
    let mut last = None;
    for k in 0..n {
        prev_vowel[k] = last;
        if segs[k].ph.is_vowel() {
            last = Some(k);
        }
    }
    let mut next_vowel: Vec<Option<usize>> = vec![None; n];
    let mut last = None;
    for k in (0..n).rev() {
        next_vowel[k] = last;
        if segs[k].ph.is_vowel() {
            last = Some(k);
        }
    }
    for k in 0..n {
        let s = segs[k];
        if s.ph.is_vowel() {
            // the consonants before the next vowel (a geminate counts twice)
            let mut j = k + 1;
            while j < n && !segs[j].ph.is_vowel() {
                j += 1;
            }
            let ncons: usize = segs[k + 1..j].iter().map(|c| 1 + usize::from(c.long)).sum();
            let closed = if j == n { ncons >= 1 } else { ncons >= 2 };
            if circumflex(&s) && closed {
                segs[k].long = false;
            }
            if s.ph == E
                && !segs[k].long
                && closed
                && segs
                    .get(k + 1)
                    .is_some_and(|c| matches!(c.letter, 'm' | 'n' | 'l' | 'r'))
            {
                segs[k].ph = Ae;
            }
            continue;
        }
        match s.letter {
            'r' => segs[k].ph = Tap,
            'k' | 'g' => {
                // the next vowel when one follows directly, else the vowel before, else after
                let near = segs
                    .get(k + 1)
                    .filter(|x| x.ph.is_vowel())
                    .or_else(|| prev_vowel[k].map(|v| &segs[v]))
                    .or_else(|| next_vowel[k].map(|v| &segs[v]));
                let palatal = near.is_some_and(|v| front_vowel(v) || circumflex(v));
                segs[k].ph = match (s.letter, palatal) {
                    ('k', true) => KPalatal,
                    ('k', false) => K,
                    (_, true) => GPalatal,
                    _ => G,
                };
            }
            _ => {}
        }
    }
}

/// Letter names for spelling a word without vowels (observed: `hmm` h E m E m E, `sms`, `vb`).
fn letter_name(c: char) -> &'static [Phoneme] {
    match c {
        'b' => &[B, ELax],
        'c' => &[Dzh, ELax],
        'ç' => &[Tsh, ELax],
        'd' => &[D, ELax],
        'f' => &[F, ELax],
        'g' => &[GPalatal, ELax],
        'h' => &[H, ELax],
        'j' => &[Zh, ELax],
        'k' => &[K, ELax],
        'l' => &[L, ELax],
        'm' => &[M, ELax],
        'n' => &[N, ELax],
        'p' => &[P, ELax],
        'r' => &[R, ELax],
        's' => &[S, ELax],
        'ş' => &[Sh, ELax],
        't' => &[T, ELax],
        'v' => &[V, ELax],
        'y' => &[J, ELax],
        'z' => &[Z, ELax],
        'x' => &[I, K, S],
        'w' => &[V, ELax],
        'q' => &[K, ELax],
        'ğ' => &[J, U, M, U, Sh, A, K, GPalatal, ELax],
        _ => &[],
    }
}

/// A word without vowels spelled letter by letter: the names run together, so r between two
/// vowels is a tap (observed: `Dr.` d E * E), and the i of x is lax unless x stands alone
/// (observed: `XX` I k s I k s, `x` i k s).
pub(crate) fn spell(letters: &str, out: &mut Vec<Seg>) {
    out.clear();
    let alone = letters.chars().count() == 1;
    for c in letters.chars() {
        for &ph in letter_name(c) {
            let ph = if c == 'x' && ph == I && !alone {
                ILax
            } else {
                ph
            };
            out.push(Seg::new(ph, false));
        }
    }
    for k in 1..out.len().saturating_sub(1) {
        if out[k].ph == R && out[k - 1].ph.is_vowel() && out[k + 1].ph.is_vowel() {
            out[k].ph = Tap;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g2p(w: &str, v2: bool) -> String {
        let mut out = Vec::new();
        key_to_phonemes(w, &mut out);
        if v2 {
            let letters: Vec<char> = w.chars().filter(|&c| c != '\'').collect();
            tts_v2_segments(&mut out, &letters);
        }
        out.iter()
            .map(|s| format!("{}{}", s.ph.symbol(), if s.long { ":" } else { "" }))
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn tts_v1_segments() {
        // laxing in closed and final syllables, & before m n l, the tap, dark l
        assert_eq!(g2p("kadehten", false), "k a d E h t & n");
        assert_eq!(g2p("burası", false), "b u * a s @");
        assert_eq!(g2p("fazla", false), "f a z L a");
        assert_eq!(g2p("gül", false), "J Y l/");
        // ğ: one long vowel for VğV, j after e/i
        assert_eq!(g2p("ağaçta", false), "a: tS t a");
        assert_eq!(g2p("değil", false), "d e j I l/");
        // doubled stops are one long stop; other doubled consonants stay double
        assert_eq!(g2p("attı", false), "a t: @");
        assert_eq!(g2p("anne", false), "a n n E");
        // no -ken palatal; circumflex: long, clear l and J before â
        assert_eq!(g2p("erken", false), "E R k & n");
        assert_eq!(g2p("hâlâ", false), "h a: l a:");
        assert_eq!(g2p("rüzgâr", false), "R Y z J a: R");
        // before an apostrophe, l follows the vowel before it
        assert_eq!(g2p("kemal'i", false), "k e m a L I");
        assert_eq!(g2p("i", false), "i");
    }

    #[test]
    fn tts_v2_segments_follow_the_literature() {
        // tense vowels inside the word; æ before a coda m n l r
        assert_eq!(g2p("kadehten", true), "k a d e h t & n");
        assert_eq!(g2p("gül", true), "J y l/");
        assert_eq!(g2p("söyle", true), "s W j l e");
        // /ɾ/ is a tap in the word (the phonemiser devoices it before a pause)
        assert_eq!(g2p("burası", true), "b u * a s @");
        assert_eq!(g2p("kardeş", true), "k a * d e S");
        assert_eq!(g2p("bir", true), "b i *");
        assert_eq!(g2p("erken", true), "& * c & n");
        // k and g palatal next to a front vowel
        assert_eq!(g2p("ekmek", true), "e c m e c");
        assert_eq!(g2p("kitap", true), "c i t a p");
        assert_eq!(g2p("türk", true), "t y * c");
        // a circumflex vowel is long only in an open syllable
        assert_eq!(g2p("hâlâ", true), "h a: l a:");
        assert_eq!(g2p("rüzgâr", true), "* y z J a *");
        assert_eq!(g2p("imkânsız", true), "i m c a n s @ z");
    }

    #[test]
    fn spelled_words_run_together() {
        let mut out = Vec::new();
        spell("dr", &mut out);
        let s: Vec<&str> = out.iter().map(|s| s.ph.symbol()).collect();
        assert_eq!(s, ["d", "E", "*", "E"]);
    }
}
