//! Line-level markdown / HTML / LaTeX recognisers and the character classes used by the
//! content cleaning ([`super::preprocess()`](super::preprocess::preprocess)).

use std::borrow::Cow;

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

pub(super) fn is_fence_line(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("```") || t.starts_with("~~~")
}

/// Horizontal rules and setext underlines (`---`, `***`, `___`, `===`, also the single
/// `-` that `normalize` makes out of `---`).
pub(super) fn is_hr_line(line: &str) -> bool {
    let t = line.trim();
    !t.is_empty()
        && t.chars()
            .all(|c| matches!(c, '-' | '*' | '_' | '=' | '~' | '–' | '—' | ' ' | '\t'))
}

fn leading_ws_len(s: &str) -> usize {
    s.len() - s.trim_start_matches([' ', '\t']).len()
}

/// `#`…`######` followed by whitespace (or nothing).
pub(super) fn header_marker_len(t: &str) -> Option<usize> {
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

pub(super) fn bullet_marker_len(t: &str, include_dashes: bool) -> Option<usize> {
    let c = t.chars().next()?;
    if !(BULLETS.contains(&c) || (include_dashes && DASH_BULLETS.contains(&c))) {
        return None;
    }
    let after = c.len_utf8();
    let ws = leading_ws_len(&t[after..]);
    (ws > 0).then_some(after + ws)
}

/// `1.`, `12)`, … (one to three digits) followed by whitespace.
pub(super) fn numbered_marker_len(t: &str) -> Option<usize> {
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
pub(super) fn strip_line_prefixes(line: &str) -> &str {
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
pub(super) enum TableLine {
    None,
    Row,
    Separator,
}

/// `|---|:--:|` style alignment row (also the `|-|-|` form left after `normalize`).
pub(super) fn is_table_separator(line: &str) -> bool {
    let t = line.trim();
    t.contains('|')
        && t.contains('-')
        && t.chars().all(|c| matches!(c, '|' | ':' | '-' | ' ' | '\t'))
}

/// Classifies lines as table rows. A run of consecutive pipe-containing lines is a table
/// when it contains an alignment row; otherwise only lines that start with `|` and hold at
/// least two pipes count.
pub(super) fn table_flags(lines: &[&str]) -> Vec<TableLine> {
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
pub(super) fn table_row_to_sentence(line: &str) -> String {
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
pub(super) fn strip_emphasis(line: &str) -> (Cow<'_, str>, usize, usize) {
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
pub(super) fn strip_links(s: &str) -> Cow<'_, str> {
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
pub(super) fn strip_html(s: &str) -> Cow<'_, str> {
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

pub(super) fn has_think_tag(s: &str) -> bool {
    find_ci(s, "<think>").is_some() || find_ci(s, "</think>").is_some()
}

/// Memo of a forward search: the first match at or after `from` was `at` (`None`: there was
/// none). That answer holds for every later start up to `at`, so a scan whose start only moves
/// forward looks at each byte once instead of searching the rest of the text again per query.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct NextMatch {
    memo: Option<(usize, Option<usize>)>,
}

impl NextMatch {
    /// The first match at or after `from`; `search(from)` finds it (as an absolute offset).
    pub(super) fn find(
        &mut self,
        from: usize,
        search: impl FnOnce(usize) -> Option<usize>,
    ) -> Option<usize> {
        if let Some((f, at)) = self.memo {
            if f <= from && at.is_none_or(|a| from <= a) {
                return at;
            }
        }
        let at = search(from);
        self.memo = Some((from, at));
        at
    }
}

/// Removes `<think>…</think>` reasoning blocks (case-insensitive). An orphan `</think>`
/// (template stripped the opener) drops everything before it; an unclosed `<think>` drops
/// everything after it.
pub(super) fn remove_think_blocks(s: &str) -> Cow<'_, str> {
    if !has_think_tag(s) {
        return Cow::Borrowed(s);
    }
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    // Each tag is found from a forward cursor: searching the rest of the text for `<think>`
    // again after every `</think>` made 200 KB of closing tags take 5 s.
    let (mut opens, mut closes) = (NextMatch::default(), NextMatch::default());
    let next = |memo: &mut NextMatch, from: usize, tag: &str| {
        memo.find(from, |f| find_ci(&s[f..], tag).map(|k| f + k))
    };
    let mut out = String::with_capacity(s.len());
    let mut rest = 0;
    loop {
        let open = next(&mut opens, rest, OPEN);
        let close = next(&mut closes, rest, CLOSE);
        match (open, close) {
            (None, None) => {
                out.push_str(&s[rest..]);
                break;
            }
            (o, Some(c)) if o.is_none_or(|o| c < o) => {
                rest = c + CLOSE.len();
                out.push_str("\n\n");
            }
            (Some(o), _) => {
                out.push_str(&s[rest..o]);
                out.push_str("\n\n");
                match next(&mut closes, o + OPEN.len(), CLOSE) {
                    Some(c) => rest = c + CLOSE.len(),
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
pub(super) fn strip_latex(s: &str) -> (Cow<'_, str>, usize) {
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
    // Closing delimiters are found from forward cursors: searching to the end of the text again
    // for every unclosed `\[` or `\(` made 200 KB of them take 2 s.
    let (mut dollars, mut brackets, mut parens) = Default::default();
    let close = |memo: &mut NextMatch, from: usize, delim: &str| {
        memo.find(from, |f| s[f..].find(delim).map(|k| f + k))
            .map(|k| k + delim.len())
    };
    while i < b.len() {
        let next = b.get(i + 1).copied();
        let skip_to = match (b[i], next) {
            (b'$', Some(b'$')) => close(&mut dollars, i + 2, "$$"),
            (b'$', _) => {
                // The closing `$` must come within MAX_INLINE_DOLLAR bytes on the same line, so
                // only that far is looked at (looking for the line end made a line of `$5`
                // quadratic).
                let rest = &b[i + 1..];
                let window = &rest[..rest.len().min(MAX_INLINE_DOLLAR + 1)];
                window
                    .iter()
                    .position(|&c| c == b'$' || c == b'\n')
                    .filter(|&e| rest[e] == b'$' && e > 0)
                    .filter(|&e| s[i + 1..i + 1 + e].contains(['\\', '^', '_', '{', '=']))
                    .map(|e| i + 1 + e + 1)
            }
            (b'\\', Some(b'[')) => close(&mut brackets, i + 2, "\\]"),
            (b'\\', Some(b'(')) => close(&mut parens, i + 2, "\\)"),
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
pub(super) fn collapse_blank_lines(s: &str) -> String {
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

// ------------------------------------------------------------------- character classes
// Character classes shared by the format and content channels.

pub(super) const ZWSP: char = '\u{200B}';
pub(super) const ZWNJ: char = '\u{200C}';
pub(super) const ZWJ: char = '\u{200D}';
pub(super) const BOM: char = '\u{FEFF}';
pub(super) const NBSP: char = '\u{00A0}';
pub(super) const NNBSP: char = '\u{202F}';
pub(super) const SOFT_HYPHEN: char = '\u{00AD}';
pub(super) const WORD_JOINER: char = '\u{2060}';
pub(super) const VS15: char = '\u{FE0E}';
pub(super) const VS16: char = '\u{FE0F}';
pub(super) const KEYCAP: char = '\u{20E3}';

/// Zero-width characters that [`super::preprocess::normalize`] deletes outright.
///
/// NBSP and NNBSP are not listed here: they separate words, so they become a plain space.
#[inline]
pub(super) fn is_removable_invisible(c: char) -> bool {
    matches!(
        c,
        ZWSP | ZWNJ | ZWJ | BOM | SOFT_HYPHEN | WORD_JOINER | VS15 | VS16
    )
}

/// Characters that start (or form) an emoji. Skin-tone modifiers, variation selectors and
/// tag characters are [`is_emoji_component`]s instead, so a modified emoji counts once.
#[inline]
pub(super) fn is_emoji_base(c: char) -> bool {
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
pub(super) fn is_emoji_component(c: char) -> bool {
    matches!(c as u32, 0x1F3FB..=0x1F3FF | 0xFE0E | 0xFE0F | 0xE0020..=0xE007F)
}

/// Removes emoji sequences (including keycaps such as `1️⃣`). When an emoji sat directly
/// between two word characters a space is inserted so the words do not merge; a space left
/// dangling before whitespace or punctuation is dropped (`ve 🚀!` → `ve!`).
pub(super) fn remove_emoji(s: &str) -> String {
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
pub(super) fn is_letter(c: char) -> bool {
    c.is_alphabetic() || unicode_normalization::char::is_combining_mark(c)
}
