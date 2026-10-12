//! Content channel: the cleaning pipeline shared by detector training and inference.
//!
//! `preprocess` = `strip_format` → `normalize` → `drop_chatbot_opener` → `mask`.
//!
//! Also holds the chatbot-opener list, the markdown / HTML recognisers and the character
//! classes the cleaning uses.

use crate::phonology::alphabet::to_turkish_lower;
use crate::tokenization::SentenceSegmenter;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use unicode_normalization::UnicodeNormalization;

/// Placeholder for a 4-digit year between 1900 and 2099.
pub const MASK_YEAR: &str = "<YIL>";
/// Placeholder for any other number.
pub const MASK_NUMBER: &str = "<SAYI>";
/// Placeholder for a URL.
pub const MASK_URL: &str = "<URL>";
/// Placeholder for an e-mail address.
pub const MASK_EMAIL: &str = "<EPOSTA>";
/// Placeholder for a (heuristically detected) proper noun.
pub const MASK_PROPER_NOUN: &str = "<ÖZEL>";

const PLACEHOLDERS: &[&str] = &[
    MASK_YEAR,
    MASK_NUMBER,
    MASK_URL,
    MASK_EMAIL,
    MASK_PROPER_NOUN,
];

/// Which optional stages of [`preprocess`] run. `normalize` always runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreprocessOptions {
    pub strip: bool,
    pub drop_opener: bool,
    pub mask: bool,
}

impl Default for PreprocessOptions {
    fn default() -> Self {
        Self {
            strip: true,
            drop_opener: true,
            mask: true,
        }
    }
}

/// Removes formatting from raw (often markdown) text while keeping the words:
///
/// - `<think>…</think>` blocks (an orphan `</think>` drops everything before it, an
///   unclosed `<think>` everything after it);
/// - LaTeX math (`$$…$$`, `\[…\]`, `\(…\)`, math-looking `$…$`) and `\commands`;
/// - fenced code blocks with their content;
/// - HTML tags and comments (`<br>` and closing block tags become line breaks);
/// - horizontal rules / setext underlines;
/// - markdown header markers, blockquote markers, list markers (`-`, `*`, `+`, `•`, `1.`,
///   `1)`, task boxes, also `–`/`—` used as bullets);
/// - table pipes and alignment rows; each row's cells are joined into one sentence
///   (`| Ankara | 5,6 milyon |` → `Ankara, 5,6 milyon.`);
/// - links (`[metin](url)` → `metin`), bold/italic/strikethrough and inline-code markers;
/// - emoji (ZWJ sequences, skin tones, flags, keycaps).
///
/// Lines are kept; runs of blank lines collapse to one `\n\n` paragraph break. A line that
/// starts with `- ` is treated as a list item, so dialogue dashes in fiction are removed too.
pub fn strip_format(raw: &str) -> String {
    let unified;
    let text = if raw.contains('\r') {
        unified = raw.replace("\r\n", "\n").replace('\r', "\n");
        unified.as_str()
    } else {
        raw
    };
    let text = remove_think_blocks(text);
    let text = strip_html(&text);
    let (text, _) = strip_latex(&text);

    let mut lines: Vec<&str> = Vec::new();
    let mut in_fence = false;
    for line in text.split('\n') {
        if is_fence_line(line) {
            in_fence = !in_fence;
            lines.push("");
        } else if !in_fence {
            lines.push(line);
        }
    }

    let flags = table_flags(&lines);
    let mut out = String::with_capacity(text.len());
    for (line, flag) in lines.iter().zip(flags) {
        match flag {
            TableLine::Separator => continue,
            TableLine::Row => {
                let row = table_row_to_sentence(line);
                out.push_str(&strip_inline(&row));
            }
            TableLine::None => {
                let no_emoji = remove_emoji(line);
                if !is_hr_line(&no_emoji) {
                    out.push_str(&strip_inline(&no_emoji));
                }
            }
        }
        out.push('\n');
    }
    collapse_blank_lines(&out)
}

/// Block prefixes, links and emphasis of a single line; prefixes are stripped again after
/// the inline pass because `**1.** Madde` only reveals its list marker then.
fn strip_inline(line: &str) -> String {
    let t = strip_line_prefixes(line);
    let linked = strip_links(t);
    let (plain, _, _) = strip_emphasis(&linked);
    let plain = remove_emoji(&plain);
    strip_line_prefixes(&plain).to_string()
}

/// Unicode/typography normalization. Never changes letter case.
///
/// 1. Removes zero-width/invisible characters (ZWSP, ZWNJ, ZWJ, BOM, soft hyphen, word
///    joiner, variation selectors) and control characters; NBSP/NNBSP become spaces;
///    `\r\n`/`\r`/U+2028/U+0085 become `\n`, U+2029 a paragraph break; the acute accent
///    `´` and modifier apostrophe `ʼ` become `'` (NFKC would turn `´` into a space plus a
///    combining accent).
/// 2. Applies NFKC (`ﬁ` → `fi`, full-width → ASCII, `…` → `...`, superscripts → digits).
/// 3. Curly quotes, guillemets and primes → `"` / `'`; `‐ ‑ ‒ – — ― −` → `-`, and any run
///    of `-` (so also `--`) collapses to one `-`.
/// 4. Collapses horizontal whitespace to one space and trims lines; a single line break is
///    kept, two or more (possibly with blank-looking lines between) become exactly `\n\n`.
pub fn normalize(text: &str) -> String {
    let mut pre = String::with_capacity(text.len());
    let mut it = text.chars().peekable();
    while let Some(c) = it.next() {
        match c {
            c if is_removable_invisible(c) => {}
            NBSP | NNBSP | '\u{2007}' => pre.push(' '),
            '\u{00B4}' | '\u{02BC}' => pre.push('\''),
            '\r' => {
                if it.peek() != Some(&'\n') {
                    pre.push('\n');
                }
            }
            '\u{2028}' | '\u{0085}' => pre.push('\n'),
            '\u{2029}' => pre.push_str("\n\n"),
            '\n' | '\t' => pre.push(c),
            c if c.is_control() => {}
            c => pre.push(c),
        }
    }

    let mut out = String::with_capacity(pre.len());
    let mut pending_space = false;
    let mut pending_newlines = 0usize;
    for c in pre.nfkc() {
        let c = match c {
            '‘' | '’' | '‚' | '‛' | '‹' | '›' | '′' | '‵' => '\'',
            '“' | '”' | '„' | '‟' | '«' | '»' | '″' | '‶' => '"',
            '\u{2010}'..='\u{2015}' | '\u{2212}' => '-',
            c => c,
        };
        if c == '\n' {
            pending_newlines += 1;
            continue;
        }
        if c.is_whitespace() {
            pending_space = true;
            continue;
        }
        let separated = pending_newlines > 0 || pending_space;
        if !out.is_empty() {
            if pending_newlines >= 2 {
                out.push_str("\n\n");
            } else if pending_newlines == 1 {
                out.push('\n');
            } else if pending_space {
                out.push(' ');
            }
        }
        pending_newlines = 0;
        pending_space = false;
        if c == '-' && !separated && out.ends_with('-') {
            continue;
        }
        if c == '…' {
            out.push_str("...");
        } else {
            out.push(c);
        }
    }
    out
}

/// Removes the leading sentence(s) when they are a chatbot opener from
/// [`CHATBOT_OPENERS`]. Opener units are found with the crate's
/// [`crate::tokenization::SentenceSegmenter`], cut at the first line break and at a `:`
/// followed by whitespace. Stacked openers (`Elbette! İşte örnekler:`) are all removed,
/// which also makes the function idempotent. Text without an opener is returned unchanged.
pub fn drop_chatbot_opener(text: &str) -> String {
    let mut rest = text;
    let mut dropped = false;
    while let Some(n) = match_chatbot_opener(rest) {
        if n == 0 || n > rest.len() || !rest.is_char_boundary(n) {
            break;
        }
        rest = &rest[n..];
        dropped = true;
    }
    if dropped {
        rest.trim_start().to_string()
    } else {
        text.to_string()
    }
}

fn is_sentence_boundary(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '…' | ':' | '\n')
}

fn is_apostrophe(c: char) -> bool {
    c == '\'' || c == '’'
}

/// If `i` starts an apostrophe followed by at least one letter, returns the end of the
/// suffix (`'da`, `’nin`).
fn apostrophe_suffix_end(text: &str, i: usize) -> Option<usize> {
    let mut it = text[i..].char_indices();
    let (_, a) = it.next()?;
    if !is_apostrophe(a) {
        return None;
    }
    let mut end = None;
    for (k, c) in it {
        if is_letter(c) {
            end = Some(i + k + c.len_utf8());
        } else {
            break;
        }
    }
    end
}

fn starts_with_ci(s: &str, prefix: &str) -> bool {
    s.len() >= prefix.len() && s.as_bytes()[..prefix.len()].eq_ignore_ascii_case(prefix.as_bytes())
}

fn try_url(text: &str, i: usize) -> Option<usize> {
    let rest = &text[i..];
    let prefix_len = ["https://", "http://", "www."]
        .iter()
        .find(|p| starts_with_ci(rest, p))
        .map(|p| p.len())?;
    let mut end = rest
        .char_indices()
        .find(|&(_, c)| {
            c.is_whitespace()
                || matches!(
                    c,
                    '<' | '>' | '"' | '\'' | '’' | '`' | '“' | '”' | '«' | '»'
                )
        })
        .map(|(k, _)| k)
        .unwrap_or(rest.len());
    while let Some(last) = rest[..end].chars().next_back() {
        let trailing = match last {
            '.' | ',' | ';' | ':' | '!' | '?' | ']' | '}' => true,
            ')' => rest[..end].matches('(').count() < rest[..end].matches(')').count(),
            _ => false,
        };
        if !trailing {
            break;
        }
        end -= last.len_utf8();
    }
    (end > prefix_len).then_some(i + end)
}

fn try_email(text: &str, i: usize) -> Option<usize> {
    let rest = &text[i..];
    let is_local = |c: char| c.is_alphanumeric() || matches!(c, '.' | '_' | '%' | '+' | '-');
    let mut at = None;
    for (k, c) in rest.char_indices().take(65) {
        if c == '@' {
            at = Some(k);
            break;
        }
        if !is_local(c) {
            return None;
        }
    }
    let at = at.filter(|&a| a > 0)?;
    let domain_start = at + 1;
    let mut end = domain_start;
    for (k, c) in rest[domain_start..].char_indices().take(255) {
        if c.is_alphanumeric() || c == '.' || c == '-' {
            end = domain_start + k + c.len_utf8();
        } else {
            break;
        }
    }
    let domain = rest[domain_start..end].trim_end_matches(['.', '-']);
    let end = domain_start + domain.len();
    let first_ok = domain.chars().next().is_some_and(char::is_alphanumeric);
    let tld_ok = domain
        .rsplit_once('.')
        .is_some_and(|(_, tld)| tld.chars().count() >= 2 && tld.chars().all(char::is_alphabetic));
    (first_ok && tld_ok).then_some(i + end)
}

fn scan_number(text: &str, i: usize) -> usize {
    let mut end = i;
    let mut it = text[i..].char_indices().peekable();
    while let Some((k, c)) = it.next() {
        if c.is_numeric() {
            end = i + k + c.len_utf8();
        } else if matches!(c, '.' | ',')
            && end == i + k
            && it.peek().is_some_and(|&(_, n)| n.is_numeric())
        {
            continue;
        } else {
            break;
        }
    }
    end
}

fn is_year(num: &str) -> bool {
    num.len() == 4
        && num.bytes().all(|b| b.is_ascii_digit())
        && num.parse::<u32>().is_ok_and(|y| (1900..=2099).contains(&y))
}

/// Masks topic-specific tokens so the detector cannot key on them:
///
/// 1. URLs (`http://`, `https://`, `www.`) → `<URL>`; e-mails → `<EPOSTA>`. These are
///    recognised first even though they come later in the list: masking digits first
///    would break `ali1990@x.com` or `…/2023/…` apart.
/// 2. 4-digit years 1900–2099 → `<YIL>`.
/// 3. Other numbers, including decimals and thousand separators (`3,14`, `1.000.000`,
///    `19.05.2023`) → `<SAYI>`.
/// 4. Proper nouns → `<ÖZEL>`. An apostrophe suffix is kept after any placeholder
///    (`İstanbul'da` → `<ÖZEL>'da`, `1990'lı` → `<YIL>'lı`).
///
/// # Proper-noun heuristic
///
/// A word is masked when it either carries an apostrophe suffix (`'` or `’` followed by
/// letters), wherever it stands, or starts with an upper-case letter and is not
/// sentence-initial. Sentence-initial means: start of text, start of a line, or after
/// `. ! ? … :`. No lexicon and no `ner` module are used.
///
/// Limits:
/// - sentence-initial proper nouns without a suffix are missed (`Ankara güzel.` keeps
///   `Ankara`);
/// - every capitalised non-initial word is masked: acronyms and units (`NATO`, `TL`),
///   the polite `Siz`, a quoted sentence (`"Merhaba" dedi`), title-cased headings and
///   capitalised table cells;
/// - after an abbreviation (`Dr. Ahmet`) the name is treated as sentence-initial;
/// - informal apostrophes on common words (`n'aber`) and foreign contractions (`don't`)
///   are masked as proper nouns;
/// - multi-word names produce one placeholder per word.
///
/// The function is idempotent: placeholders are never re-masked.
pub fn mask(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + text.len() / 8);
    let n = text.len();
    let mut i = 0;
    let mut sentence_start = true;
    let mut prev: Option<char> = None;

    while i < n {
        let rest = &text[i..];
        if rest.starts_with('<') {
            if let Some(p) = PLACEHOLDERS.iter().find(|p| rest.starts_with(**p)) {
                out.push_str(p);
                i += p.len();
                i = copy_suffix(text, i, &mut out);
                sentence_start = false;
                prev = Some('>');
                continue;
            }
        }
        let Some(c) = rest.chars().next() else {
            break;
        };

        let word_start = prev.is_none_or(|p| {
            !(p.is_alphanumeric()
                || unicode_normalization::char::is_combining_mark(p)
                || matches!(p, '.' | '_' | '%' | '+' | '-' | '@' | '/'))
        });
        if word_start && c.is_alphanumeric() {
            let hit = try_url(text, i)
                .map(|e| (e, MASK_URL))
                .or_else(|| try_email(text, i).map(|e| (e, MASK_EMAIL)));
            if let Some((end, placeholder)) = hit {
                out.push_str(placeholder);
                i = copy_suffix(text, end, &mut out);
                sentence_start = false;
                prev = Some('>');
                continue;
            }
        }

        if c.is_numeric() {
            let end = scan_number(text, i);
            let num = &text[i..end];
            out.push_str(if is_year(num) { MASK_YEAR } else { MASK_NUMBER });
            i = copy_suffix(text, end, &mut out);
            sentence_start = false;
            prev = Some('0');
            continue;
        }

        if is_letter(c) {
            let end = rest
                .char_indices()
                .find(|&(_, ch)| !is_letter(ch))
                .map(|(k, _)| i + k)
                .unwrap_or(n);
            let word = &text[i..end];
            let has_alpha = word.chars().any(char::is_alphabetic);
            if let Some(suffix_end) = apostrophe_suffix_end(text, end).filter(|_| has_alpha) {
                out.push_str(MASK_PROPER_NOUN);
                out.push_str(&text[end..suffix_end]);
                i = suffix_end;
            } else if !sentence_start && word.chars().next().is_some_and(char::is_uppercase) {
                out.push_str(MASK_PROPER_NOUN);
                i = end;
            } else {
                out.push_str(word);
                i = end;
            }
            sentence_start = false;
            prev = text[..i].chars().next_back();
            continue;
        }

        out.push(c);
        i += c.len_utf8();
        if is_sentence_boundary(c) {
            sentence_start = true;
        }
        prev = Some(c);
    }
    out
}

/// Copies an apostrophe suffix starting at `i` verbatim; returns the new position.
fn copy_suffix(text: &str, i: usize, out: &mut String) -> usize {
    match apostrophe_suffix_end(text, i) {
        Some(e) => {
            out.push_str(&text[i..e]);
            e
        }
        None => i,
    }
}

/// Runs the content-channel pipeline: `strip_format` → `normalize` → `drop_chatbot_opener`
/// → `mask`, skipping the stages switched off in `opts`.
///
/// When stripping, `normalize` also runs *before* `strip_format`, so that markup produced by
/// NFKC or dash folding (full-width `＊＊`, `—` bullets that become `-`) is stripped too. This
/// makes `preprocess(preprocess(x)) == preprocess(x)`.
pub fn preprocess(raw: &str, opts: &PreprocessOptions) -> String {
    let mut text = if opts.strip {
        normalize(&strip_format(&normalize(raw)))
    } else {
        normalize(raw)
    };
    if opts.drop_opener {
        text = drop_chatbot_opener(&text);
    }
    if opts.mask {
        text = mask(&text);
    }
    text
}

/// [`preprocess`] over many documents, spread over the available CPU cores.
/// Output order matches input order.
pub fn preprocess_batch<S: AsRef<str> + Sync>(
    texts: &[S],
    opts: &PreprocessOptions,
) -> Vec<String> {
    let run = |chunk: &[S]| -> Vec<String> {
        chunk.iter().map(|t| preprocess(t.as_ref(), opts)).collect()
    };
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(texts.len().div_ceil(16).max(1));
    if threads <= 1 {
        return run(texts);
    }
    let chunk_size = texts.len().div_ceil(threads);
    std::thread::scope(|scope| {
        let handles: Vec<_> = texts
            .chunks(chunk_size)
            .map(|chunk| {
                let handle = std::thread::Builder::new().spawn_scoped(scope, move || run(chunk));
                (chunk, handle)
            })
            .collect();
        let mut out = Vec::with_capacity(texts.len());
        for (chunk, handle) in handles {
            match handle
                .map_err(|_| ())
                .and_then(|h| h.join().map_err(|_| ()))
            {
                Ok(v) => out.extend(v),
                Err(()) => out.extend(run(chunk)),
            }
        }
        out
    })
}

// ======================================================================================
// Chatbot openers
// ======================================================================================
// Chatbot openers ("Elbette!", "Tabii ki!") dropped from the start of a text.

/// Extra condition an opener needs besides its prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenerRequirement {
    /// The prefix alone is enough.
    None,
    /// The prefix must be followed directly by `!` (`Kesinlikle!`, but not
    /// `Kesinlikle katılıyorum.`).
    Exclamation,
    /// The opener unit must end with `:` (`İşte bazı öneriler:`, but not
    /// `İşte bu yüzden geldim.`).
    EndsWithColon,
    /// The opener unit must contain this lower-cased word. If the opener unit is a bare
    /// greeting of at most four words, the following unit may supply it instead, and both
    /// units are then treated as the opener (`Merhaba! Size nasıl yardımcı olabilirim?`).
    Contains(&'static str),
}

/// A Turkish assistant opener.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatbotOpener {
    /// Lower-cased (Turkish rules) prefix; it must be followed by a non-letter.
    pub prefix: &'static str,
    pub requirement: OpenerRequirement,
}

const fn opener(prefix: &'static str, requirement: OpenerRequirement) -> ChatbotOpener {
    ChatbotOpener {
        prefix,
        requirement,
    }
}

/// Turkish chat-assistant openers removed by [`drop_chatbot_opener`].
///
/// Matching is done on the first *opener unit* of the cleaned text: the first sentence
/// according to [`SentenceSegmenter`], additionally cut at the first line break and at the
/// first `:` followed by whitespace. The unit is lower-cased with Turkish rules
/// (`İşte` → `işte`) and leading punctuation is skipped.
///
/// | prefix                   | extra requirement                              |
/// |--------------------------|------------------------------------------------|
/// | `elbette`                | –                                              |
/// | `tabii ki`, `tabi ki`    | –                                              |
/// | `tabii`                  | –                                              |
/// | `harika bir soru`        | –                                              |
/// | `harika soru`            | –                                              |
/// | `güzel bir soru`         | –                                              |
/// | `işte`                   | unit ends with `:`                             |
/// | `memnuniyetle`           | –                                              |
/// | `kesinlikle`             | directly followed by `!`                       |
/// | `sorunuz için teşekkürler`, `sorunuz için teşekkür ederim` | –           |
/// | `merhaba`                | `yardımcı` in the unit or the next one         |
///
/// Known limits: human text that genuinely starts with "Elbette …" or "Tabii …" (e.g. a
/// forum reply) also matches; openers in other phrasings ("Bu harika bir soru",
/// "Size yardımcı olmaktan mutluluk duyarım") are not listed; ASCII-only spellings with a
/// dotless `I` in upper case (`TABII`) do not match.
pub const CHATBOT_OPENERS: &[ChatbotOpener] = &[
    opener("elbette", OpenerRequirement::None),
    opener("tabii ki", OpenerRequirement::None),
    opener("tabi ki", OpenerRequirement::None),
    opener("tabii", OpenerRequirement::None),
    opener("harika bir soru", OpenerRequirement::None),
    opener("harika soru", OpenerRequirement::None),
    opener("güzel bir soru", OpenerRequirement::None),
    opener("işte", OpenerRequirement::EndsWithColon),
    opener("memnuniyetle", OpenerRequirement::None),
    opener("kesinlikle", OpenerRequirement::Exclamation),
    opener("sorunuz için teşekkürler", OpenerRequirement::None),
    opener("sorunuz için teşekkür ederim", OpenerRequirement::None),
    opener("merhaba", OpenerRequirement::Contains("yardımcı")),
];

/// Byte length of the first opener unit of `body` (which must not start with whitespace).
fn first_unit_len(body: &str) -> Option<usize> {
    let line_end = body.find('\n').unwrap_or(body.len());
    let line = &body[..line_end];
    let sentences = SentenceSegmenter::segment(line);
    let mut end = sentences
        .first()
        .map(|s| s.start + s.text.len())
        .filter(|&e| e <= line.len() && line.is_char_boundary(e))
        .unwrap_or(line.len());
    let mut iter = line[..end].char_indices().peekable();
    while let Some((idx, c)) = iter.next() {
        if c == ':' && iter.peek().is_none_or(|&(_, n)| n.is_whitespace()) {
            end = idx + 1;
            break;
        }
    }
    (end > 0).then_some(end)
}

/// Splits leading whitespace and the first opener unit off `text`:
/// returns `(lead_len, unit_len)`.
fn split_unit(text: &str) -> Option<(usize, usize)> {
    let lead = text.len() - text.trim_start().len();
    first_unit_len(&text[lead..]).map(|u| (lead, u))
}

fn starts_with_word(s: &str, prefix: &str) -> bool {
    s.starts_with(prefix)
        && s[prefix.len()..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_alphabetic())
}

/// If `text` starts with a chatbot opener from [`CHATBOT_OPENERS`], returns the byte
/// length of the opener (including the whitespace before it). Intended for cleaned text
/// (after `strip_format` + `normalize`).
pub fn match_chatbot_opener(text: &str) -> Option<usize> {
    let (lead, u1) = split_unit(text)?;
    let unit = &text[lead..lead + u1];
    let lower = to_turkish_lower(unit);
    let core = lower.trim_start_matches(|c: char| !c.is_alphanumeric());
    for op in CHATBOT_OPENERS {
        if !starts_with_word(core, op.prefix) {
            continue;
        }
        let end1 = lead + u1;
        match op.requirement {
            OpenerRequirement::None => return Some(end1),
            OpenerRequirement::Exclamation => {
                if core[op.prefix.len()..].trim_start().starts_with('!') {
                    return Some(end1);
                }
            }
            OpenerRequirement::EndsWithColon => {
                if unit.trim_end().ends_with(':') {
                    return Some(end1);
                }
            }
            OpenerRequirement::Contains(word) => {
                if lower.contains(word) {
                    return Some(end1);
                }
                if unit.split_whitespace().count() <= 4 {
                    let rest = &text[end1..];
                    if let Some((lead2, u2)) = split_unit(rest) {
                        let unit2 = &rest[lead2..lead2 + u2];
                        if to_turkish_lower(unit2).contains(word) {
                            return Some(end1 + lead2 + u2);
                        }
                    }
                }
            }
        }
    }
    None
}

// ======================================================================================
// Markup recognisers
// ======================================================================================
// Line-level markdown/HTML recognisers used by `strip_format`.

/// Longest emphasis span (in chars) searched for a closing delimiter. Bounds the work on
/// pathological lines full of unmatched `*`/`_`.
const MAX_EMPHASIS_SPAN: usize = 1000;
/// Longest HTML tag (in bytes) considered.
const MAX_TAG_LEN: usize = 1000;

/// List bullets. The dashes `–`/`—` are only treated as bullets when stripping (because
/// `normalize` turns them into `-`, which is a bullet).
const BULLETS: &[char] = &[
    '-', '*', '+', '•', '◦', '▪', '▫', '‣', '●', '○', '■', '□', '➤', '►', '▶', '✓', '✔', '→', '⁃',
    '∙',
];
const DASH_BULLETS: &[char] = &['–', '—'];

/// HTML-looking placeholders produced by `mask`; never treated as tags.
const PLACEHOLDER_TAGS: &[&str] = &["YIL", "SAYI", "URL", "EPOSTA"];

pub(crate) fn is_fence_line(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("```") || t.starts_with("~~~")
}

/// Horizontal rules and setext underlines (`---`, `***`, `___`, `===`, also the single
/// `-` that `normalize` makes out of `---`).
pub(crate) fn is_hr_line(line: &str) -> bool {
    let t = line.trim();
    !t.is_empty()
        && t.chars()
            .all(|c| matches!(c, '-' | '*' | '_' | '=' | '~' | '–' | '—' | ' ' | '\t'))
}

fn leading_ws_len(s: &str) -> usize {
    s.len() - s.trim_start_matches([' ', '\t']).len()
}

/// `#`…`######` followed by whitespace (or nothing).
pub(crate) fn header_marker_len(t: &str) -> Option<usize> {
    let hashes = t.len() - t.trim_start_matches('#').len();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = &t[hashes..];
    if rest.is_empty() {
        return Some(hashes);
    }
    let ws = leading_ws_len(rest);
    (ws > 0).then_some(hashes + ws)
}

pub(crate) fn bullet_marker_len(t: &str, include_dashes: bool) -> Option<usize> {
    let c = t.chars().next()?;
    if !(BULLETS.contains(&c) || (include_dashes && DASH_BULLETS.contains(&c))) {
        return None;
    }
    let after = c.len_utf8();
    let ws = leading_ws_len(&t[after..]);
    (ws > 0).then_some(after + ws)
}

/// `1.`, `12)`, … (one to three digits) followed by whitespace.
pub(crate) fn numbered_marker_len(t: &str) -> Option<usize> {
    let digits = t.len() - t.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    if digits == 0 || digits > 3 {
        return None;
    }
    let rest = &t[digits..];
    if !(rest.starts_with('.') || rest.starts_with(')')) {
        return None;
    }
    let ws = leading_ws_len(&rest[1..]);
    (ws > 0).then_some(digits + 1 + ws)
}

fn quote_marker_len(t: &str) -> Option<usize> {
    t.starts_with('>').then_some(1)
}

fn checkbox_len(t: &str) -> Option<usize> {
    for b in ["[ ]", "[x]", "[X]"] {
        if let Some(rest) = t.strip_prefix(b) {
            if rest.is_empty() || rest.starts_with([' ', '\t']) {
                return Some(b.len());
            }
        }
    }
    None
}

/// Strips any stack of block prefixes (`> `, `## `, `- `, `1. `, `[x] `) from a line.
pub(crate) fn strip_line_prefixes(line: &str) -> &str {
    let mut t = line.trim_start();
    loop {
        let n = header_marker_len(t)
            .or_else(|| quote_marker_len(t))
            .or_else(|| bullet_marker_len(t, true))
            .or_else(|| numbered_marker_len(t))
            .or_else(|| checkbox_len(t));
        match n {
            Some(n) if n > 0 && n <= t.len() => t = t[n..].trim_start(),
            _ => return t,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TableLine {
    None,
    Row,
    Separator,
}

/// `|---|:--:|` style alignment row (also the `|-|-|` form left after `normalize`).
pub(crate) fn is_table_separator(line: &str) -> bool {
    let t = line.trim();
    t.contains('|')
        && t.contains('-')
        && t.chars().all(|c| matches!(c, '|' | ':' | '-' | ' ' | '\t'))
}

/// Classifies lines as table rows. A run of consecutive pipe-containing lines is a table
/// when it contains an alignment row; otherwise only lines that start with `|` and hold at
/// least two pipes count.
pub(crate) fn table_flags(lines: &[&str]) -> Vec<TableLine> {
    let mut flags = vec![TableLine::None; lines.len()];
    let mut i = 0;
    while i < lines.len() {
        if !lines[i].contains('|') {
            i += 1;
            continue;
        }
        let start = i;
        while i < lines.len() && lines[i].contains('|') {
            i += 1;
        }
        let block = &lines[start..i];
        let has_sep = block.iter().any(|l| is_table_separator(l));
        for (k, l) in block.iter().enumerate() {
            let t = l.trim();
            flags[start + k] = if is_table_separator(l) {
                TableLine::Separator
            } else if has_sep || (t.starts_with('|') && t.matches('|').count() >= 2) {
                TableLine::Row
            } else {
                TableLine::None
            };
        }
    }
    flags
}

/// Joins a table row's cells into one sentence: `| Ankara | 5,6 milyon |` →
/// `Ankara, 5,6 milyon.`
pub(crate) fn table_row_to_sentence(line: &str) -> String {
    let t = line.trim();
    let t = t.strip_prefix('|').unwrap_or(t);
    let t = t.strip_suffix('|').unwrap_or(t);
    let cells: Vec<&str> = t
        .split('|')
        .map(str::trim)
        .filter(|c| !c.is_empty())
        .collect();
    let mut s = cells.join(", ");
    if let Some(last) = s.chars().next_back() {
        if !matches!(last, '.' | '!' | '?' | '…' | ':' | ';') {
            s.push('.');
        }
    }
    s
}

fn run_len(chars: &[char], i: usize, d: char) -> usize {
    chars[i..].iter().take_while(|&&c| c == d).count()
}

/// Removes matched `d`-delimiter pairs of exactly `len` characters, keeping the content.
/// An opener must not follow a letter/digit and must precede a non-space; a closer must
/// follow a non-space and must not precede a letter/digit. This keeps `snake_case`,
/// `2*3*4` and `İstanbul`da` intact.
fn remove_pairs(chars: &mut Vec<char>, d: char, len: usize) -> usize {
    let n = chars.len();
    if !chars.contains(&d) {
        return 0;
    }
    let mut out = Vec::with_capacity(n);
    let mut count = 0;
    let mut i = 0;
    while i < n {
        if chars[i] != d {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        let run = run_len(chars, i, d);
        if run == len {
            let after = i + len;
            let left_ok = i == 0 || !chars[i - 1].is_alphanumeric();
            if left_ok && after < n && !chars[after].is_whitespace() {
                let limit = n.min(after + MAX_EMPHASIS_SPAN);
                let mut j = after + 1;
                let mut found = None;
                while j < limit {
                    if chars[j] == d {
                        let r = run_len(chars, j, d);
                        if r == len
                            && !chars[j - 1].is_whitespace()
                            && chars.get(j + len).is_none_or(|c| !c.is_alphanumeric())
                        {
                            found = Some(j);
                            break;
                        }
                        j += r;
                    } else {
                        j += 1;
                    }
                }
                if let Some(j) = found {
                    out.extend_from_slice(&chars[after..j]);
                    count += 1;
                    i = j + len;
                    continue;
                }
            }
        }
        out.extend_from_slice(&chars[i..i + run]);
        i += run;
    }
    *chars = out;
    count
}

/// Removes inline emphasis/code markers from one line. Returns the cleaned line plus the
/// number of bold and italic spans found (`***x***` counts as both).
pub(crate) fn strip_emphasis(line: &str) -> (Cow<'_, str>, usize, usize) {
    if !line.contains(['*', '_', '~', '`']) {
        return (Cow::Borrowed(line), 0, 0);
    }
    let mut chars: Vec<char> = line.chars().collect();
    let mut bold = 0;
    let mut italic = 0;
    for d in ['*', '_'] {
        let both = remove_pairs(&mut chars, d, 3);
        bold += both + remove_pairs(&mut chars, d, 2);
        italic += both + remove_pairs(&mut chars, d, 1);
    }
    remove_pairs(&mut chars, '~', 2);
    remove_pairs(&mut chars, '`', 2);
    remove_pairs(&mut chars, '`', 1);
    (Cow::Owned(chars.into_iter().collect()), bold, italic)
}

/// `[text](url)` → `text`, `![alt](url)` → `alt`.
pub(crate) fn strip_links(s: &str) -> Cow<'_, str> {
    if !s.contains("](") {
        return Cow::Borrowed(s);
    }
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut last = 0;
    let mut i = 0;
    while i < b.len() {
        let text_start = if b[i] == b'[' {
            i + 1
        } else if b[i] == b'!' && b.get(i + 1) == Some(&b'[') {
            i + 2
        } else {
            i += 1;
            continue;
        };
        let close = b[text_start..]
            .iter()
            .take(500)
            .position(|&c| matches!(c, b']' | b'[' | b'\n'))
            .map(|p| text_start + p)
            .filter(|&k| b[k] == b']' && b.get(k + 1) == Some(&b'('));
        let Some(k) = close else {
            i += 1;
            continue;
        };
        let paren = b[k + 2..]
            .iter()
            .take(2000)
            .position(|&c| matches!(c, b')' | b'\n'))
            .map(|p| k + 2 + p)
            .filter(|&m| b[m] == b')');
        let Some(m) = paren else {
            i += 1;
            continue;
        };
        out.push_str(&s[last..i]);
        out.push_str(&s[text_start..k]);
        last = m + 1;
        i = m + 1;
    }
    out.push_str(&s[last..]);
    Cow::Owned(out)
}

/// Removes HTML tags and comments. `<br>` and closing block tags become a newline.
/// Mask placeholders such as `<YIL>` are left alone.
pub(crate) fn strip_html(s: &str) -> Cow<'_, str> {
    if !s.contains('<') {
        return Cow::Borrowed(s);
    }
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut last = 0;
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'<' {
            i += 1;
            continue;
        }
        match parse_tag(s, i) {
            Some((end, replacement)) => {
                out.push_str(&s[last..i]);
                out.push_str(replacement);
                last = end;
                i = end;
            }
            None => i += 1,
        }
    }
    out.push_str(&s[last..]);
    Cow::Owned(out)
}

/// Parses an HTML tag starting at byte `p` (which holds `<`). Returns the end byte and
/// the replacement text.
fn parse_tag(s: &str, p: usize) -> Option<(usize, &'static str)> {
    let rest = &s[p..];
    if rest.starts_with("<!--") {
        return rest.find("-->").map(|e| (p + e + 3, ""));
    }
    let b = rest.as_bytes();
    let closing = b.get(1) == Some(&b'/');
    let name_start = if closing { 2 } else { 1 };
    if !b.get(name_start)?.is_ascii_alphabetic() {
        return None;
    }
    let mut k = name_start;
    while k < b.len() && (b[k].is_ascii_alphanumeric() || b[k] == b'-') {
        k += 1;
    }
    let name = &rest[name_start..k];
    let next = *b.get(k)?;
    if !(next == b'>' || next == b'/' || next.is_ascii_whitespace()) {
        return None;
    }
    if !closing && next == b'>' && PLACEHOLDER_TAGS.contains(&name) {
        return None;
    }
    let gt = b[k..]
        .iter()
        .take(MAX_TAG_LEN)
        .position(|&c| c == b'>' || c == b'<')
        .map(|q| k + q)
        .filter(|&q| b[q] == b'>')?;
    let lname = name.to_ascii_lowercase();
    let block = matches!(
        lname.as_str(),
        "p" | "div"
            | "li"
            | "tr"
            | "ul"
            | "ol"
            | "table"
            | "blockquote"
            | "pre"
            | "h1"
            | "h2"
            | "h3"
            | "h4"
            | "h5"
            | "h6"
    );
    let replacement = if lname == "br" || (closing && block) {
        "\n"
    } else {
        ""
    };
    Some((p + gt + 1, replacement))
}

/// ASCII case-insensitive search for an ASCII `needle` starting with `<`.
fn find_ci(hay: &str, needle: &str) -> Option<usize> {
    let h = hay.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() || h.len() < n.len() {
        return None;
    }
    (0..=h.len() - n.len())
        .filter(|&i| h[i] == n[0])
        .find(|&i| h[i..i + n.len()].eq_ignore_ascii_case(n))
}

pub(crate) fn has_think_tag(s: &str) -> bool {
    find_ci(s, "<think>").is_some() || find_ci(s, "</think>").is_some()
}

/// Removes `<think>…</think>` reasoning blocks (case-insensitive). An orphan `</think>`
/// (template stripped the opener) drops everything before it; an unclosed `<think>` drops
/// everything after it.
pub(crate) fn remove_think_blocks(s: &str) -> Cow<'_, str> {
    if !has_think_tag(s) {
        return Cow::Borrowed(s);
    }
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    loop {
        let open = find_ci(rest, OPEN);
        let close = find_ci(rest, CLOSE);
        match (open, close) {
            (None, None) => {
                out.push_str(rest);
                break;
            }
            (o, Some(c)) if o.is_none_or(|o| c < o) => {
                rest = &rest[c + CLOSE.len()..];
                out.push_str("\n\n");
            }
            (Some(o), _) => {
                out.push_str(&rest[..o]);
                out.push_str("\n\n");
                let after = &rest[o + OPEN.len()..];
                match find_ci(after, CLOSE) {
                    Some(c) => rest = &after[c + CLOSE.len()..],
                    None => break,
                }
            }
            (None, Some(_)) => break,
        }
    }
    Cow::Owned(out)
}

/// Longest inline `$…$` span (bytes) still treated as math.
const MAX_INLINE_DOLLAR: usize = 300;

/// Removes LaTeX math and commands, returning the cleaned text and the number of LaTeX
/// constructs found: display `$$…$$` / `\[…\]`, inline `\(…\)`, a `$…$` span on one line
/// whose content looks like math (contains `\ ^ _ { =`), and bare `\command`s (`\frac`,
/// `\cdot`, `\text`). Each removed construct becomes a single space. Braces are dropped
/// only when the text contained LaTeX, so ordinary prose (and `$5 ve $10`) is untouched.
pub(crate) fn strip_latex(s: &str) -> (Cow<'_, str>, usize) {
    if !s.contains('\\') && !s.contains('$') {
        return (Cow::Borrowed(s), 0);
    }
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut count = 0usize;
    let mut last = 0usize;
    let mut i = 0usize;
    // Every cut position below sits on an ASCII byte (`$`, `\`, letters), so slicing is safe.
    let cut = |out: &mut String, start: usize, end: usize, last: &mut usize| {
        out.push_str(&s[*last..start]);
        out.push(' ');
        *last = end;
    };
    while i < b.len() {
        let next = b.get(i + 1).copied();
        let skip_to = match (b[i], next) {
            (b'$', Some(b'$')) => s[i + 2..].find("$$").map(|e| i + 2 + e + 2),
            (b'$', _) => {
                let rest = &s[i + 1..];
                let line = &rest[..rest.find('\n').unwrap_or(rest.len())];
                line.find('$')
                    .filter(|&e| e > 0 && e <= MAX_INLINE_DOLLAR)
                    .filter(|&e| line[..e].contains(['\\', '^', '_', '{', '=']))
                    .map(|e| i + 1 + e + 1)
            }
            (b'\\', Some(b'[')) => s[i + 2..].find("\\]").map(|e| i + 2 + e + 2),
            (b'\\', Some(b'(')) => s[i + 2..].find("\\)").map(|e| i + 2 + e + 2),
            (b'\\', Some(b'\\')) => Some(i + 2),
            (b'\\', Some(c)) if c.is_ascii_alphabetic() => {
                let mut j = i + 1;
                while j < b.len() && b[j].is_ascii_alphabetic() {
                    j += 1;
                }
                if b.get(j) == Some(&b'*') {
                    j += 1;
                }
                Some(j)
            }
            _ => None,
        };
        match skip_to {
            Some(end) => {
                cut(&mut out, i, end, &mut last);
                count += 1;
                i = end;
            }
            None => i += 1,
        }
    }
    if count == 0 {
        return (Cow::Borrowed(s), 0);
    }
    out.push_str(&s[last..]);
    out.retain(|c| c != '{' && c != '}');
    (Cow::Owned(out), count)
}

/// Trims line ends and collapses runs of blank lines into a single `\n\n` paragraph break.
pub(crate) fn collapse_blank_lines(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut blank_pending = false;
    for line in s.split('\n') {
        let l = line.trim_end();
        if l.trim_start().is_empty() {
            blank_pending = !out.is_empty();
            continue;
        }
        if !out.is_empty() {
            out.push_str(if blank_pending { "\n\n" } else { "\n" });
        }
        blank_pending = false;
        out.push_str(l);
    }
    out
}

// ======================================================================================
// Character classes
// ======================================================================================
// Character classes shared by the format and content channels.

pub(crate) const ZWSP: char = '\u{200B}';
pub(crate) const ZWNJ: char = '\u{200C}';
pub(crate) const ZWJ: char = '\u{200D}';
pub(crate) const BOM: char = '\u{FEFF}';
pub(crate) const NBSP: char = '\u{00A0}';
pub(crate) const NNBSP: char = '\u{202F}';
pub(crate) const SOFT_HYPHEN: char = '\u{00AD}';
pub(crate) const WORD_JOINER: char = '\u{2060}';
pub(crate) const VS15: char = '\u{FE0E}';
pub(crate) const VS16: char = '\u{FE0F}';
pub(crate) const KEYCAP: char = '\u{20E3}';

/// Zero-width characters that [`normalize`] deletes outright.
///
/// NBSP and NNBSP are not listed here: they separate words, so they become a plain space.
#[inline]
pub(crate) fn is_removable_invisible(c: char) -> bool {
    matches!(
        c,
        ZWSP | ZWNJ | ZWJ | BOM | SOFT_HYPHEN | WORD_JOINER | VS15 | VS16
    )
}

/// Characters that start (or form) an emoji. Skin-tone modifiers, variation selectors and
/// tag characters are [`is_emoji_component`]s instead, so a modified emoji counts once.
#[inline]
pub(crate) fn is_emoji_base(c: char) -> bool {
    matches!(
        c as u32,
        0x1F000..=0x1F2FF
            | 0x1F300..=0x1F3FA
            | 0x1F400..=0x1FAFF
            | 0x2600..=0x27BF
            | 0x2B05..=0x2B07
            | 0x2B1B
            | 0x2B1C
            | 0x2B50
            | 0x2B55
            | 0x231A
            | 0x231B
            | 0x2328
            | 0x23CF
            | 0x23E9..=0x23F3
            | 0x23F8..=0x23FA
            | 0x24C2
            | 0x25AA
            | 0x25AB
            | 0x25B6
            | 0x25C0
            | 0x25FB..=0x25FE
            | 0x3030
            | 0x303D
            | 0x3297
            | 0x3299
            | 0x20E3
    )
}

/// Modifiers that only make sense attached to an emoji.
#[inline]
pub(crate) fn is_emoji_component(c: char) -> bool {
    matches!(c as u32, 0x1F3FB..=0x1F3FF | 0xFE0E | 0xFE0F | 0xE0020..=0xE007F)
}

/// Removes emoji sequences (including keycaps such as `1️⃣`). When an emoji sat directly
/// between two word characters a space is inserted so the words do not merge; a space left
/// dangling before whitespace or punctuation is dropped (`ve 🚀!` → `ve!`).
pub(crate) fn remove_emoji(s: &str) -> String {
    if !s.chars().any(|c| is_emoji_base(c) || is_emoji_component(c)) {
        return s.to_string();
    }
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < n {
        let c = chars[i];
        let mut j = i;
        if matches!(c, '0'..='9' | '#' | '*') {
            let mut k = i + 1;
            if k < n && chars[k] == VS16 {
                k += 1;
            }
            if k < n && chars[k] == KEYCAP {
                j = k + 1;
            }
        }
        if j == i && (is_emoji_base(c) || is_emoji_component(c)) {
            j = i + 1;
        }
        if j == i {
            out.push(c);
            i += 1;
            continue;
        }
        // Extend over the rest of the sequence.
        while j < n {
            let d = chars[j];
            if is_emoji_base(d) || is_emoji_component(d) {
                j += 1;
            } else if d == ZWJ && j + 1 < n && is_emoji_base(chars[j + 1]) {
                j += 2;
            } else {
                break;
            }
        }
        let next = chars.get(j).copied();
        let before_word = out.chars().next_back().is_some_and(char::is_alphanumeric);
        if before_word && next.is_some_and(char::is_alphanumeric) {
            out.push(' ');
        } else if out.ends_with(' ')
            && next.is_none_or(|c| {
                c.is_whitespace() || matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | ')')
            })
        {
            out.pop();
        }
        i = j;
    }
    out
}

/// Letters and combining marks: the characters a word is made of.
#[inline]
pub(crate) fn is_letter(c: char) -> bool {
    c.is_alphabetic() || unicode_normalization::char::is_combining_mark(c)
}
