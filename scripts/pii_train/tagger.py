"""A transformer token classifier for PII spans: BIO labels from character spans for training, and
span prediction with BIO-constrained Viterbi decoding over sliding windows for inference.
"""
from __future__ import annotations

import math
import re
import unicodedata

LABELS = ("PERSON", "USERNAME", "ADDRESS", "PHONE", "EMAIL", "URL", "ID", "SECRET", "DATE")
TAGS = ["O"] + [f"{p}-{l}" for l in LABELS for p in ("B", "I")]
TAG_ID = {t: i for i, t in enumerate(TAGS)}


_SYMBOL_CATEGORIES = {"So", "Sk", "Co", "Cn"}
_VARIATION = re.compile("[︀-️󠄀-󠇯]")


def normalize(text: str) -> str:
    """The text the model reads: emoji and other symbols the tokenizer would glue to a word (turning the
    whole word into [UNK], "Yılmaz🎉") become spaces. One character for one, so offsets still hold."""
    if text.isascii():
        return text
    text = _VARIATION.sub(" ", text)
    return "".join(" " if ord(c) > 127 and unicodedata.category(c) in _SYMBOL_CATEGORIES else c for c in text)


def token_tags(offsets, spans) -> list[int]:
    """BIO tag ids of tokens with character `offsets` for gold `spans` [(start, end, label)]; -100
    for special tokens. A token belongs to a span when most of it lies inside."""
    tags = []
    for k, (a, b) in enumerate(offsets):
        if a == b:
            tags.append(-100)
            continue
        tag = "O"
        for s, e, lab in spans:
            inside = min(b, e) - max(a, s)
            if inside > 0 and inside * 2 >= b - a:
                prev = offsets[k - 1] if k else (0, 0)
                starts = a <= s or not (prev[0] != prev[1] and min(prev[1], e) - max(prev[0], s) > 0)
                tag = ("B-" if starts else "I-") + lab
                break
        tags.append(TAG_ID[tag])
    return tags


def _allowed(prev: int, cur: int) -> bool:
    t = TAGS[cur]
    if not t.startswith("I-"):
        return True
    p = TAGS[prev] if prev >= 0 else "O"
    return p != "O" and p[2:] == t[2:]


def _transitions():
    import numpy as np
    k = len(TAGS)
    trans = np.array([[0.0 if _allowed(p, t) else -math.inf for t in range(k)] for p in range(k)])
    start = np.array([0.0 if _allowed(-1, t) else -math.inf for t in range(k)])
    return trans, start


_TRANS = None


def viterbi(logprobs) -> list[int]:
    """Best tag sequence under BIO constraints (an I- tag only after its B- or I-)."""
    import numpy as np
    global _TRANS
    if _TRANS is None:
        _TRANS = _transitions()
    trans, start = _TRANS
    lp = np.asarray(logprobs, dtype=np.float64)
    n = len(lp)
    if n == 0:
        return []
    best = start + lp[0]
    back = np.zeros((n, lp.shape[1]), dtype=np.int64)
    for i in range(1, n):
        scores = best[:, None] + trans  # previous tag x current tag
        back[i] = scores.argmax(0)
        best = scores.max(0) + lp[i]
    t = int(best.argmax())
    out = [t]
    for i in range(n - 1, 0, -1):
        t = int(back[i][t])
        out.append(t)
    return out[::-1]


def spans_from_tags(offsets, tags) -> list[tuple[int, int, str]]:
    spans, cur = [], None
    for (a, b), t in zip(offsets, tags):
        tag = TAGS[t]
        if tag == "O":
            if cur:
                spans.append(cur)
            cur = None
        elif tag.startswith("B-") or cur is None or cur[2] != tag[2:]:
            if cur:
                spans.append(cur)
            cur = [a, b, tag[2:]]
        else:
            cur[1] = b
    if cur:
        spans.append(cur)
    return [tuple(s) for s in spans]


class Tagger:
    """Loads a fine-tuned token classifier and predicts spans for texts."""

    def __init__(self, model_dir: str, device: str | None = None, max_length: int = 256, stride: int = 64, batch_size: int = 8,
                 o_bias: float = 0.0, bias_labels: tuple[str, ...] | None = None):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir).to(self.device).eval()
        self.max_length, self.stride, self.batch_size = max_length, stride, batch_size
        # subtracted from the O tag's log-probability before decoding: above 0 finds more spans
        # (recall), below 0 fewer (precision); with `bias_labels`, added to those labels' tags instead
        self.o_bias, self.bias_labels = o_bias, bias_labels
        self._batches = 0

    def predict(self, texts: list[str], guard_every: int = 50) -> list[list[tuple[int, int, str]]]:
        return self.decode(self.scores(texts, guard_every), len(texts), texts=texts)

    def windows(self, texts: list[str]) -> list[tuple[int, list, list]]:
        """(text index, character offsets, input ids) of every window of every text."""
        out = []
        for i, text in enumerate(texts):
            enc = self.tok(normalize(text), return_offsets_mapping=True, truncation=True, max_length=self.max_length,
                           stride=self.stride, return_overflowing_tokens=True)
            out += [(i, offs, ids) for ids, offs in zip(enc["input_ids"], enc["offset_mapping"])]
        return out

    def logprobs(self, windows, guard_every: int = 50) -> list:
        """Tag log-probabilities of every token of every window (special tokens included)."""
        import guard

        torch = self.torch
        out = []
        for b in range(0, len(windows), self.batch_size):
            self._batches += 1
            if self._batches % guard_every == 0:  # counted across calls: one text per call is common
                guard.wait(min_ram_gb=0.4, min_commit_gb=0.5)
            batch = windows[b:b + self.batch_size]
            width = max(len(w[2]) for w in batch)
            ids = torch.full((len(batch), width), self.tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for j, (_, _, x) in enumerate(batch):
                ids[j, :len(x)] = torch.tensor(x)
                mask[j, :len(x)] = 1
            with torch.no_grad(), torch.autocast(self.device, enabled=self.device == "cuda"):
                logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).logits.float()
            logp = torch.log_softmax(logits, -1).cpu().numpy()
            out += [logp[j][:len(x)] for j, (_, _, x) in enumerate(batch)]
        return out

    def scores(self, texts: list[str], guard_every: int = 50) -> list[tuple[int, list, list]]:
        """(text index, character offsets, tag log-probabilities) of every window's real tokens."""
        wins = self.windows(texts)
        return real_tokens(wins, self.logprobs(wins, guard_every))

    def decode(self, scored, n_texts: int, o_bias: float | None = None, texts: list[str] | None = None) -> list[list[tuple[int, int, str]]]:
        return decode(scored, n_texts, self.o_bias if o_bias is None else o_bias, texts, self.bias_labels)


def real_tokens(windows, logprobs) -> list[tuple[int, list, list]]:
    """Drops special tokens: (text index, character offsets, log-probabilities) per window."""
    out = []
    for (i, offs, _), lp in zip(windows, logprobs):
        keep = [k for k, (a, c) in enumerate(offs) if a != c]
        out.append((i, [offs[k] for k in keep], lp[keep]))
    return out


def uncertainty(lp) -> float:
    """How unsure a window is: the largest probability mass a token leaves outside its best tag."""
    import numpy as np
    return float(1 - np.exp(lp.max(-1)).min()) if len(lp) else 0.0


class Cascade:
    """The small tagger reads every window; windows it is unsure about (uncertainty() above
    `threshold`) are read again by the big one, or by each of several, whose distributions replace
    the small one's or are averaged with it, every tagger with equal weight. A threshold below 0
    sends every window: the taggers' average. All must share a tokenizer, so that their windows are
    the same."""

    def __init__(self, small: Tagger, big: Tagger | list[Tagger], threshold: float = 0.1, mode: str = "average",
                 o_bias: float = 0.0, bias_labels: tuple[str, ...] | None = None):
        self.bigs = list(big) if isinstance(big, (list, tuple)) else [big]
        assert all(small.tok.get_vocab() == b.tok.get_vocab() for b in self.bigs), "the taggers tokenize differently"
        self.small, self.big, self.threshold, self.mode = small, self.bigs[0], threshold, mode
        self.o_bias, self.bias_labels = o_bias, bias_labels
        self.windows_seen = self.windows_escalated = 0

    def scores(self, texts: list[str]):
        import numpy as np
        wins = self.small.windows(texts)
        lps = self.small.logprobs(wins)
        unsure = [k for k, ((_, offs, _), lp) in enumerate(zip(wins, lps))
                  if uncertainty(lp[[j for j, (a, c) in enumerate(offs) if a != c]]) > self.threshold]
        read = [b.logprobs([wins[k] for k in unsure]) for b in self.bigs]
        for i, k in enumerate(unsure):
            members = [r[i] for r in read] + ([] if self.mode == "replace" else [lps[k]])
            lps[k] = np.logaddexp.reduce(np.stack(members), axis=0) - np.log(len(members))
        self.windows_seen += len(wins)
        self.windows_escalated += len(unsure)
        return real_tokens(wins, lps)

    def predict(self, texts: list[str]):
        return decode(self.scores(texts), len(texts), self.o_bias, texts, self.bias_labels)


def decode(scored, n_texts: int, bias: float = 0.0, texts: list[str] | None = None,
           labels: tuple[str, ...] | None = None) -> list[list[tuple[int, int, str]]]:
    """Spans of every text from its windows' log-probabilities; `bias` is Tagger's o_bias, moved
    off the O tag onto the B and I tags of `labels` when given (raising every label's tags equals
    lowering O). Windows overlap: a token's scores come from the window where it sits farthest
    from an edge (a window's last tokens see no right context and may end inside a word), and each
    text is decoded once. With the `texts`, spans are widened to whole words (snap)."""
    import numpy as np
    cols = [TAG_ID[f"{p}-{label}"] for label in labels for p in "BI"] if labels else None
    best = [{} for _ in range(n_texts)]  # character offsets -> (distance from the window's edge, scores)
    for i, offs, lp in scored:
        n = len(offs)
        for k, off in enumerate(offs):
            depth = min(k, n - 1 - k)
            if off not in best[i] or depth > best[i][off][0]:
                best[i][off] = (depth, lp[k])
    out = []
    for tokens in best:
        if not tokens:
            out.append([])
            continue
        offs = sorted(tokens)
        lp = np.stack([tokens[o][1] for o in offs])
        if bias:
            lp = lp.copy()
            if cols:
                lp[:, cols] += bias
            else:
                lp[:, 0] -= bias
        out.append(spans_from_tags(offs, viterbi(lp)))
    if texts is not None:
        out = [snap(text, spans) for text, spans in zip(texts, out)]
    return out


def _merge(spans) -> list[tuple[int, int, str]]:
    spans = sorted(spans, key=lambda s: (s[0], -(s[1] - s[0])))
    kept = []
    for s in spans:
        if kept and s[0] < kept[-1][1]:
            if s[1] > kept[-1][1] and s[2] == kept[-1][2]:
                kept[-1] = (kept[-1][0], s[1], s[2])
            continue
        kept.append(s)
    return kept


# characters inside a code, date, e-mail or link that join its parts (29/9/2016, SRM-2021-K2VM6, ad.soyad@...)
_JOINERS = "-/._@+"


def snap(text: str, spans) -> list[tuple[int, int, str]]:
    """Widens spans that start or end inside a word to the whole word. A tag sequence can stop after
    a word's first sub-token ("Kr" of "Kristof", "pınar" of "pınarla", "29" of "29/9/2016"). Names
    widen over letters and digits only, so an apostrophe's suffix stays out; other values also over
    the joiners inside codes, dates, e-mails and links."""
    n = len(text)
    out = []
    for s, e, label in spans:
        joiners = "" if label == "PERSON" else _JOINERS

        def inside(i):
            c = text[i]
            return c.isalnum() or (c in joiners and 0 < i < n - 1 and text[i - 1].isalnum() and text[i + 1].isalnum())

        while 0 < s < n and inside(s - 1) and inside(s):
            s -= 1
        while 0 < e < n and inside(e) and inside(e - 1):
            e += 1
        out.append((s, e, label))
    return _merge(out)
