//! Letters to phonemes for words outside the lexicon, as inferred from the reference fixtures:
//! vowel laxing in closed and word-final syllables, `&` before a tautosyllabic m n l, the tap
//! between vowels, clear and dark l by the neighbouring vowel, `J` before a front vowel, `c` in
//! `-ken`, ğ as length or `j`, and doubled stops as one long stop.

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
    pub at: u16,
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

/// Endings after `-ken` with which k is palatal `c` (observed: erken, gereken(i), yelkenler,
/// erkenden, işkence; not kenin, kenlere, kensiz, kenir).
const KEN_TAILS: [&str; 6] = ["", "i", "e", "ler", "den", "ce"];

/// The phonemes of a lower-case word (letters only, circumflex vowels kept).
#[cfg(test)]
pub(crate) fn letters_to_phonemes(word: &str, out: &mut Vec<Seg>) {
    key_to_phonemes(word, out)
}

/// The phonemes of a lexicon key: lower-case letters, circumflex vowels kept, apostrophes where
/// the text had them. An l before an apostrophe takes its quality from the vowel before it and
/// is not a coda (observed: Kemal'i `L`, İsmail'e `l`); the lone letter i is tense (observed:
/// the i of iPhone).
pub(crate) fn key_to_phonemes(key: &str, out: &mut Vec<Seg>) {
    key_to_phonemes_with(key, out, false)
}

/// As [`key_to_phonemes`]; with `tts`, the `tts-v1` profile's segments (Zimmer & Orgun 1999,
/// Göksel & Kerslake 2005, Lewis 2001):
/// - a circumflex vowel is long, and k before it palatal `c` (kâr, dükkân, mahkûm);
/// - two equal vowel letters are one long vowel (saat, kanaat, tabii);
/// - ğ after e or i lengthens the vowel before a consonant or at the end (eğlence, iğne), and is
///   `j` only between vowels (değil);
/// - long vowels are tense;
/// - doubled ç and g are one long consonant, as doubled stops are, and a doubled clear l one
///   long `l`;
/// - no eSpeak `-ken` palatal.
pub(crate) fn key_to_phonemes_with(key: &str, out: &mut Vec<Seg>, tts: bool) {
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
            if tts && i + 1 < n && w[i + 1] == 'ğ' {
                // ğ (Göksel & Kerslake 2005; teacher durations): before a consonant or the end it
                // lengthens the vowel; two equal vowels across it are one long vowel; between back
                // vowels it is silent and the first vowel stays short (soğuk: 7 frames against 6
                // for a short o, 10 before a consonant); between front vowels it is a glide j
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
            // VğV with the same vowel: one long vowel (not for e, i, ı: observed ığı = @: @)
            if !tts
                && i + 2 < n
                && w[i + 1] == 'ğ'
                && w[i + 2] == ch
                && !matches!(ch, 'e' | 'i' | 'ı')
            {
                units.push(Unit::V {
                    v: base,
                    long: true,
                    circ,
                    at: i,
                });
                i += 3;
                continue;
            }
            // ğ after a vowel other than e, i: the vowel's length mark
            if !tts && i + 1 < n && w[i + 1] == 'ğ' && !matches!(base, 'e' | 'i') {
                units.push(Unit::V {
                    v: base,
                    long: true,
                    circ,
                    at: i,
                });
                i += 2;
                continue;
            }
            if tts {
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
            }
            units.push(Unit::V {
                v: base,
                long: tts && circ,
                circ,
                at: i,
            });
            i += 1;
            continue;
        }
        if ch == 'ğ' {
            let prev = if i > 0 { w[i - 1] } else { '\0' };
            let c = if prev == 'e' || prev == 'i' || (tts && (prev == 'ö' || prev == 'ü')) {
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
        let doubled_long = if tts {
            matches!(ch, 'b' | 'c' | 'd' | 'k' | 'p' | 't' | 'ç' | 'g')
        } else {
            matches!(ch, 'b' | 'c' | 'd' | 'k' | 'p' | 't')
        };
        if i + 1 < n && w[i + 1] == ch && doubled_long {
            units.push(Unit::C {
                c: ch,
                long: true,
                at: if tts { i } else { i + 1 },
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
    for k in 0..m {
        match units[k] {
            Unit::V { v, long, at, .. } => {
                let mut j = k + 1;
                while j < m && matches!(units[j], Unit::C { .. }) {
                    j += 1;
                }
                // in tts-v1 a geminate closes the syllable before it (dik.kat), as two letters do
                let ncons = j - k - 1
                    + if tts {
                        units[k + 1..j]
                            .iter()
                            .filter(|u| matches!(u, Unit::C { long: true, .. }))
                            .count()
                    } else {
                        0
                    };
                let fin = j == m;
                let closed = (fin && ncons >= 1) || (!fin && ncons >= 2);
                let lax = (closed || (fin && ncons == 0)) && !(tts && long);
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
                    at: at as u16,
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
                        let pv = (0..k).rev().find_map(vowel_at);
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
                            let pv = (0..k).rev().find_map(vowel_at);
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
                    'k' if tts => match next_v {
                        Some((_, true)) => KPalatal,
                        _ => K,
                    },
                    'k' => {
                        let rest: String = w[(at + 1).min(n)..].iter().collect();
                        if at > 0 && rest.starts_with("en") && KEN_TAILS.contains(&&rest[2..]) {
                            KPalatal
                        } else {
                            K
                        }
                    }
                    'Q' => Gh,
                    'x' => {
                        out.push(Seg {
                            ph: K,
                            long: false,
                            letter: 'x',
                            at: at as u16,
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
                    at: at as u16,
                });
            }
        }
    }
    if tts {
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
}

/// Segment choices that depend on stress, for rule-derived words (observed): a non-initial g
/// before a front vowel stays `g` in a stressed final syllable (bugün, hangi, zengin) and is `J`
/// elsewhere; tense e after c or ç is lax `E` before an unstressed final syllable with a lax
/// vowel after d, y, l or n (bahçede, düşünceli), but not before the j of ğ (geleceğim; train:
/// 122 tense, 10 lax). `levels` are the word's vowel stress levels.
/// Returns whether a segment changed.
pub(crate) fn stress_dependent_segments(segs: &mut [Seg], levels: &[u8]) -> bool {
    let mut vix: Vec<isize> = Vec::with_capacity(segs.len());
    let mut v: isize = -1;
    for s in segs.iter() {
        if s.ph.is_vowel() {
            v += 1;
        }
        vix.push(v);
    }
    let nv = v + 1;
    let level = |syl: isize| -> u8 {
        if syl >= 0 {
            levels.get(syl as usize).copied().unwrap_or(0)
        } else {
            0
        }
    };
    let front = |p: Phoneme| matches!(p, E | ELax | Ae | I | ILax | Ue | UeLax | Oe);
    let lax = |p: Phoneme| matches!(p, ELax | ILax | Ih | ULax | UeLax | OLax | Ae | A | Oe);
    let mut changed = false;
    for k in 0..segs.len() {
        let s = segs[k];
        if !s.ph.is_vowel()
            && s.letter == 'g'
            && k > 0
            && k + 1 < segs.len()
            && segs[k + 1].ph.is_vowel()
            && front(segs[k + 1].ph)
        {
            let syl = vix[k] + 1;
            let stressed = level(syl) >= 4;
            let closed2 =
                k + 3 < segs.len() && !segs[k + 2].ph.is_vowel() && !segs[k + 3].ph.is_vowel();
            let ph = if stressed && syl == nv - 1 && segs[k + 1].ph != Ae && !closed2 {
                G
            } else {
                GPalatal
            };
            if ph != s.ph {
                segs[k].ph = ph;
                changed = true;
            }
        }
        if s.ph == E && k > 0 && matches!(segs[k - 1].ph, Dzh | Tsh) {
            let syl = vix[k];
            if syl == nv - 2
                && k + 2 < segs.len()
                && matches!(segs[k + 1].ph, D | J | L | N)
                && segs[k + 1].letter != 'ğ'
                && segs[k + 2].ph.is_vowel()
                && lax(segs[k + 2].ph)
                && level(syl + 1) < 4
            {
                segs[k].ph = ELax;
                changed = true;
            }
        }
    }
    changed
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

    fn g2p(w: &str) -> String {
        let mut out = Vec::new();
        letters_to_phonemes(w, &mut out);
        out.iter()
            .map(|s| format!("{}{}", s.ph.symbol(), if s.long { ":" } else { "" }))
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn observed_segment_rules() {
        // laxing in closed and final syllables, & before m n l, the tap, dark l
        assert_eq!(g2p("kadehten"), "k a d E h t & n");
        assert_eq!(g2p("burası"), "b u * a s @");
        assert_eq!(g2p("fazla"), "f a z L a");
        assert_eq!(g2p("gül"), "J Y l/");
        // ğ: length after back vowels, j after e/i, one long vowel for VğV
        assert_eq!(g2p("ağaçta"), "a: tS t a");
        assert_eq!(g2p("değil"), "d e j I l/");
        assert_eq!(g2p("olduğu"), "O L/ d U:");
        // doubled stops are one long stop; other doubled consonants stay double
        assert_eq!(g2p("attı"), "a t: @");
        assert_eq!(g2p("anne"), "a n n E");
        // c in -ken; circumflex: clear l and J before â
        assert_eq!(g2p("erken"), "E R c & n");
        assert_eq!(g2p("kendi"), "k & n d I");
        assert_eq!(g2p("hâlâ"), "h a l a");
        assert_eq!(g2p("rüzgâr"), "R Y z J a R");
        // before an apostrophe, l follows the vowel before it
        assert_eq!(g2p("kemal'i"), "k e m a L I");
        assert_eq!(g2p("ismail'e"), "I s m a i l E");
        assert_eq!(g2p("i"), "i");
    }

    #[test]
    fn spelled_words_run_together() {
        let mut out = Vec::new();
        spell("dr", &mut out);
        let s: Vec<&str> = out.iter().map(|s| s.ph.symbol()).collect();
        assert_eq!(s, ["d", "E", "*", "E"]);
    }

    #[test]
    fn stress_moves_g_and_e() {
        let mut segs = Vec::new();
        letters_to_phonemes("bugün", &mut segs);
        stress_dependent_segments(&mut segs, &[1, 6]);
        assert_eq!(segs[2].ph, G);
        letters_to_phonemes("bugünkü", &mut segs);
        stress_dependent_segments(&mut segs, &[1, 0, 6]);
        assert_eq!(segs[2].ph, GPalatal);
        letters_to_phonemes("bahçede", &mut segs);
        stress_dependent_segments(&mut segs, &[6, 1, 1]);
        assert_eq!(segs[4].ph, ELax);
    }
}
