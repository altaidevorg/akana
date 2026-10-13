//! Content channel: the cleaning pipeline shared by detector training and inference.
//!
//! `preprocess` = `strip_format` → `normalize` → `drop_chatbot_opener` → `mask`.
//!
//! Also holds the chatbot-opener list, the markdown / HTML recognisers and the character
//! classes the cleaning uses.

use unicode_normalization::UnicodeNormalization;

use super::markup::{
    collapse_blank_lines, is_fence_line, is_hr_line, is_letter, is_removable_invisible,
    remove_emoji, remove_think_blocks, strip_emphasis, strip_html, strip_latex,
    strip_line_prefixes, strip_links, table_flags, table_row_to_sentence, TableLine, NBSP, NNBSP,
};
use crate::phonology::alphabet::to_turkish_lower;
use crate::tokenization::SentenceSegmenter;

/// Placeholder for a 4-digit year between 1900 and 2099.
pub(super) const MASK_YEAR: &str = "<YIL>";
/// Placeholder for any other number.
pub(super) const MASK_NUMBER: &str = "<SAYI>";
/// Placeholder for a URL.
pub(super) const MASK_URL: &str = "<URL>";
/// Placeholder for an e-mail address.
pub(super) const MASK_EMAIL: &str = "<EPOSTA>";
/// Placeholder for a (heuristically detected) proper noun.
pub(super) const MASK_PROPER_NOUN: &str = "<ÖZEL>";

const PLACEHOLDERS: &[&str] = &[
    MASK_YEAR,
    MASK_NUMBER,
    MASK_URL,
    MASK_EMAIL,
    MASK_PROPER_NOUN,
];

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
pub(super) fn strip_format(raw: &str) -> String {
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
pub(super) fn normalize(text: &str) -> String {
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
pub(super) fn drop_chatbot_opener(text: &str) -> String {
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
pub(super) fn mask(text: &str) -> String {
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

/// Cleans raw text for the content features: `strip_format` → `normalize` →
/// `drop_chatbot_opener` → `mask`.
///
/// `normalize` also runs *before* `strip_format`, so that markup produced by NFKC or dash
/// folding (full-width `＊＊`, `—` bullets that become `-`) is stripped too. This makes
/// `preprocess(preprocess(x)) == preprocess(x)`.
pub(super) fn preprocess(raw: &str) -> String {
    let text = normalize(&strip_format(&normalize(raw)));
    mask(&drop_chatbot_opener(&text))
}

// ------------------------------------------------------------------- chatbot openers
// Chatbot openers ("Elbette!", "Tabii ki!") dropped from the start of a text.

/// Extra condition an opener needs besides its prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OpenerRequirement {
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
pub(super) struct ChatbotOpener {
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
pub(super) const CHATBOT_OPENERS: &[ChatbotOpener] = &[
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
pub(super) fn match_chatbot_opener(text: &str) -> Option<usize> {
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

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------- normalize

    #[test]
    fn normalize_keeps_turkish_case_and_multibyte_letters() {
        let s = "İSTANBUL'DA ılık bir gün; Iğdır ve ışık. ŞÇĞÜÖ şçğüö";
        assert_eq!(normalize(s), s);
    }

    #[test]
    fn normalize_typography() {
        assert_eq!(
            normalize("“Merhaba” dedi—sonra ‘tamam’… «evet» 1990–2000 a -- b"),
            "\"Merhaba\" dedi-sonra 'tamam'... \"evet\" 1990-2000 a - b"
        );
        // Curly apostrophe and acute accent used as apostrophes.
        assert_eq!(normalize("Türkiye’nin İzmir´de"), "Türkiye'nin İzmir'de");
    }

    #[test]
    fn normalize_removes_invisibles_and_nfkc() {
        let s =
        "\u{FEFF}Bu\u{200B}gün\u{00A0}hava\u{202F}çok\u{00AD} güzel\u{200D}\u{200C}. ﬁlm ＡＢＣ";
        assert_eq!(normalize(s), "Bugün hava çok güzel. film ABC");
    }

    #[test]
    fn normalize_whitespace_and_paragraphs() {
        let s = "  Birinci   satır\t burada.\r\nİkinci satır.\n\n\n   \n\nYeni paragraf.  \n";
        assert_eq!(
            normalize(s),
            "Birinci satır burada.\nİkinci satır.\n\nYeni paragraf."
        );
    }

    // ---------------------------------------------------------------- strip_format

    #[test]
    fn strip_nested_markdown() {
        let raw = "## **Önemli** Başlık\n\n\
               - **Kalın** ve *eğik* madde\n  \
               - İç içe _vurgulu_ ***ikisi*** madde\n\
               1. Numaralı [bağlantı](https://ornek.com) ve `kod`\n\
               > Alıntı ~~silik~~ satır\n\n\
               ---\n\
               Düz metin snake_case ve 2*3*4 kalır.";
        assert_eq!(
            strip_format(raw),
            "Önemli Başlık\n\n\
         Kalın ve eğik madde\n\
         İç içe vurgulu ikisi madde\n\
         Numaralı bağlantı ve kod\n\
         Alıntı silik satır\n\n\
         Düz metin snake_case ve 2*3*4 kalır."
        );
    }

    #[test]
    fn strip_table_joins_cells_into_sentences() {
        let raw = "Tablo:\n\n| Şehir | Nüfus |\n|:------|------:|\n| **Ankara** | 5,6 milyon |\n| İzmir | 4,4 milyon |\n\nSon.";
        assert_eq!(
            strip_format(raw),
            "Tablo:\n\nŞehir, Nüfus.\nAnkara, 5,6 milyon.\nİzmir, 4,4 milyon.\n\nSon."
        );
    }

    #[test]
    fn strip_think_block_code_fence_html_and_emoji() {
        let raw = "<think>\nKullanıcı ne istiyor? Düşüneyim.\n</think>\n\n\
               Merhaba 🚀 dünya 👩‍💻 ve 🇹🇷!\n\n\
               ```python\nprint('kod')\n```\n\n\
               <p>Bir <b>paragraf</b></p>1️⃣ Son.";
        assert_eq!(strip_format(raw), "Merhaba dünya ve!\n\nBir paragraf\nSon.");
        // Orphan closing tag: everything before it is reasoning.
        assert_eq!(strip_format("düşünce\n</think>\nCevap."), "Cevap.");
        // Case-insensitive, unclosed opener drops the rest.
        assert_eq!(strip_format("Cevap.\n<THINK>yarım kalmış"), "Cevap.");
    }

    #[test]
    fn strip_and_normalize_leave_plain_text_unchanged() {
        let plain = "Dün akşam İstanbul'dan Ankara'ya 450 km yol yaptık.\n\
                 Yolda 3,5 saat kaldık; hava 12 derece idi.\n\n\
                 Ertesi gün \"çok yorgunum\" dedi ve ışıkları kapattı. Ilık bir sabahtı (gerçekten).";
        assert_eq!(strip_format(plain), plain);
        assert_eq!(normalize(&strip_format(plain)), plain);
        // Only whitespace differs.
        let messy = "  Bir   cümle.\t\nİkinci   cümle.  ";
        assert_eq!(normalize(&strip_format(messy)), normalize(messy));
    }

    #[test]
    fn strip_keeps_mask_placeholders() {
        let masked = "<ÖZEL>'da <YIL> yılında <SAYI> kişi <URL> ve <EPOSTA> kullandı.";
        assert_eq!(strip_format(masked), masked);
    }

    // ---------------------------------------------------------------- openers

    #[test]
    fn each_opener_is_dropped() {
        let cases = [
            (
                "Elbette! Türkiye'nin başkenti Ankara'dır.",
                "Türkiye'nin başkenti Ankara'dır.",
            ),
            (
                "Tabii ki, işte cevabınız:\nAnkara başkenttir.",
                "Ankara başkenttir.",
            ),
            (
                "Tabi ki yardımcı olurum. Ankara başkenttir.",
                "Ankara başkenttir.",
            ),
            ("Tabii. Ankara başkenttir.", "Ankara başkenttir."),
            ("Harika bir soru! Ankara başkenttir.", "Ankara başkenttir."),
            ("Harika soru. Ankara başkenttir.", "Ankara başkenttir."),
            ("Güzel bir soru. Ankara başkenttir.", "Ankara başkenttir."),
            ("İşte bazı öneriler:\nSu için.", "Su için."),
            (
                "Memnuniyetle açıklarım. Ankara başkenttir.",
                "Ankara başkenttir.",
            ),
            ("Kesinlikle! Ankara başkenttir.", "Ankara başkenttir."),
            (
                "Sorunuz için teşekkürler. Ankara başkenttir.",
                "Ankara başkenttir.",
            ),
            (
                "Sorunuz için teşekkür ederim! Ankara başkenttir.",
                "Ankara başkenttir.",
            ),
            (
                "Merhaba! Size nasıl yardımcı olabilirim? Ankara başkenttir.",
                "Ankara başkenttir.",
            ),
            (
                "Merhaba, bu konuda yardımcı olabilirim. Ankara başkenttir.",
                "Ankara başkenttir.",
            ),
            ("ELBETTE! Ankara başkenttir.", "Ankara başkenttir."),
            // Stacked openers are all removed.
            (
                "Elbette! İşte açıklama:\n\nAnkara başkenttir.",
                "Ankara başkenttir.",
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(drop_chatbot_opener(input), expected, "input: {input:?}");
            assert!(match_chatbot_opener(input).is_some(), "input: {input:?}");
        }
        // Every opener in the list is exercised above.
        assert_eq!(CHATBOT_OPENERS.len(), 13);
    }

    #[test]
    fn non_openers_are_kept() {
        for s in [
            "İşte bu yüzden geldim.",
            "Kesinlikle katılıyorum.",
            "Merhaba, ben Ali. Bugün hava güzel.",
            "Tabiiyet kanunu değişti.",
            "Ankara başkenttir.",
            "",
        ] {
            assert_eq!(drop_chatbot_opener(s), s, "input: {s:?}");
            assert!(match_chatbot_opener(s).is_none(), "input: {s:?}");
        }
    }

    #[test]
    fn opener_found_through_markdown_and_think() {
        let raw = "<think>plan</think>\n\n**Elbette!** İşte *liste*:\n\n- Bir";
        assert_eq!(preprocess(raw), "Bir");
    }

    // ---------------------------------------------------------------- mask

    #[test]
    fn mask_year_number_decimal() {
        assert_eq!(
            mask("bu 1999 ve 2023'te, 2150 ile 1899 değil; 3,14 ile 1.000.000 ve 12 oldu."),
            "bu <YIL> ve <YIL>'te, <SAYI> ile <SAYI> değil; <SAYI> ile <SAYI> ve <SAYI> oldu."
        );
        assert_eq!(mask("tarih 19.05.2023 idi"), "tarih <SAYI> idi");
        assert_eq!(mask("sene 2023."), "sene <YIL>.");
    }

    #[test]
    fn mask_urls_and_emails() {
        assert_eq!(
        mask("bkz. https://ornek.com/2023/yazi?id=5, ya da www.site.org'a bak; ali1990@posta.com.tr yaz."),
        "bkz. <URL>, ya da <URL>'a bak; <EPOSTA> yaz."
    );
    }

    #[test]
    fn mask_apostrophe_proper_noun_keeps_suffix() {
        assert_eq!(mask("İstanbul'da yaşıyorum."), "<ÖZEL>'da yaşıyorum.");
        assert_eq!(
            mask("dün Türkiye’nin en büyük şehri İzmir'e gittik."),
            "dün <ÖZEL>’nin en büyük şehri <ÖZEL>'e gittik."
        );
    }

    #[test]
    fn mask_sentence_initial_capital_is_not_masked() {
        assert_eq!(
            mask("Ankara güzel. Bugün Ali geldi!\nIşık yandı: Sonra Zeynep gitti."),
            "Ankara güzel. Bugün <ÖZEL> geldi!\nIşık yandı: Sonra <ÖZEL> gitti."
        );
    }

    #[test]
    fn mask_is_idempotent() {
        let s = "Ali 1990'da İstanbul'a geldi ve NATO raporunu https://x.com/a adresinden okudu.";
        let once = mask(s);
        assert_eq!(mask(&once), once);
    }

    // ---------------------------------------------------------------- pipeline

    const SAMPLES: &[&str] = &[
    "Elbette! İşte **Türkiye** hakkında bilgiler:\n\n## Genel\n\n- Başkent: *Ankara*\n- Nüfus: 85.000.000\n\n| Şehir | Yıl |\n|---|---|\n| İstanbul | 1453 |\n\nDaha fazlası için https://tr.wikipedia.org adresine bakın 🚀.",
    "<think>Kullanıcı Türkçe soruyor.</think>Merhaba! Size nasıl yardımcı olabilirim?\n\n1. Birinci — madde\n2. İkinci – madde…",
    "Dün akşam Kadıköy'de “harika” bir konser vardı; 2.500 kişi geldi. Ilık bir geceydi.",
    "— Nereye gidiyorsun? dedi Ahmet.\n— Eve, diye cevap verdi.",
    "＊＊Tam genişlik＊＊ metin ve ＃ başlık\n\n— madde bir\n— madde iki",
    "**1.** Madde ve `İstanbul`da` kod\n\n> > İç içe alıntı",
    "Fiyat 3,5 TL'den 4.250,75 TL'ye çıktı (%21). ali.veli@ornek.com.tr adresine yazın.",
    "",
    "   \n\n  ",
];

    #[test]
    fn preprocess_full_pipeline() {
        let raw = SAMPLES[0];
        assert_eq!(
        preprocess(raw),
        "Genel\n\nBaşkent: Ankara\nNüfus: <SAYI>\n\nŞehir, <ÖZEL>.\nİstanbul, <SAYI>.\n\nDaha fazlası için <URL> adresine bakın."
    );
    }

    #[test]
    fn stages_compose_into_preprocess() {
        let raw = "Elbette! **Ankara** 1923'te başkent oldu.";
        assert_eq!(
            normalize(&strip_format(&normalize(raw))),
            "Elbette! Ankara 1923'te başkent oldu."
        );
        assert_eq!(preprocess(raw), "Ankara <YIL>'te başkent oldu.");
    }

    #[test]
    fn preprocess_is_idempotent() {
        for raw in SAMPLES {
            let once = preprocess(raw);
            let twice = preprocess(&once);
            assert_eq!(twice, once, "raw: {raw:?}");
            let n = normalize(raw);
            assert_eq!(normalize(&n), n, "normalize not idempotent for {raw:?}");
        }
    }

    #[test]
    fn no_panic_on_tricky_unicode() {
        let long_word = "x".repeat(10_000);
        let unmatched_stars = "*a ".repeat(3_000);
        let tricky = [
            "",
            "\u{0301}",
            "\u{0301}\u{0301}x",
            "👨‍👩‍👧‍👦👍🏽🏳️‍🌈🇹🇷1️⃣#️⃣",
            "مرحبا بالعالم 123 ٣٤٥",
            "שלום עולם",
            "\u{202E}tersine\u{202C}",
            "<<<>>>**__``~~||",
            "*",
            "|",
            "- ",
            "#",
            "```",
            "<think>",
            "</think>",
            "İ\u{0307}i\u{0307}",
            "a\u{0000}b\u{0007}c",
            "\u{FEFF}\u{200B}\u{200D}",
            "'''’’’",
            "http://",
            "@@@.com",
            "1.2.3.4.5.6.7.8.9",
            long_word.as_str(),
            unmatched_stars.as_str(),
            "漢字かなカナ한국어",
            "\u{10FFFF}\u{E0020}\u{E007F}",
        ];
        for s in tricky {
            let p = preprocess(s);
            let _ = strip_format(s);
            let _ = mask(s);
            let _ = drop_chatbot_opener(s);
            assert_eq!(preprocess(&p), p, "not idempotent for {s:?}");
        }
    }

    #[test]
    fn strip_removes_latex_math_and_commands() {
        let cases = [
            "Sonuç \\(x^2 + 1\\) olur.",
            "Formül şudur: $$\\frac{a}{b} \\cdot c$$ ve devamı.",
            "Hız $v = \\frac{d}{t}$ formülüyle bulunur.",
            "Toplam \\[ \\sum_{i=1}^{n} i \\] eder.",
            "Burada \\text{Hız} = 5 alınır.",
        ];
        for c in cases {
            let out = strip_format(c);
            assert!(!out.contains('\\'), "backslash left in {out:?}");
            assert!(
                !out.contains("frac") && !out.contains("cdot") && !out.contains("sum"),
                "{out:?}"
            );
            assert!(!out.contains('{') && !out.contains('}'), "{out:?}");
        }
        assert!(strip_format("Burada \\text{Hız} = 5 alınır.").contains("Hız"));
    }

    #[test]
    fn strip_keeps_dollar_prices_and_plain_text() {
        let plain = "Fiyat $5 ve $10 arasında değişiyor, {not} kalsın.";
        assert_eq!(strip_format(plain), plain);
    }
}
