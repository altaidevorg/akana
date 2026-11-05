"""A strict scorer for PII masking: entity precision, recall and F1 under exact and overlap
matching, full-cover recall (the value masked completely, the privacy measure), the share of PII
characters left in clear, over-masking, and false alarms on PII-free rows; per dataset, coarse
class and slice, with bootstrap intervals over rows.

A system is a callable `text -> [(start, end, coarse, label)]`. Precision counts only the
predictions `in_scope` accepts (a dataset that does not annotate health data does not count an
SAGLIK detection against the system); every prediction counts for leakage and over-masking. A
row's `ignore` spans, tagged values outside the scheme being scored, count neither way: a
prediction touching one is dropped, and a row holding one is not PII-free.

Spans are compared by their core: without surrounding spaces and punctuation, without a case
suffix after an apostrophe (Kaplan'ün), which akana leaves outside the mask on purpose and which
reveals nothing, and without a title before a name or an honorific after it (Dyt. Gizem Demir,
Prof. Dr. Ali Kaya, Ayşe Hanım): datasets differ on whether titles belong to the span, and a title
is not personal data.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict

# apostrophes as typed, plus U+0092: a cp1252 right quote decoded as Latin-1, common in scraped text
APOSTROPHE_SUFFIX = re.compile("['’`´ʼ\x92][a-zçğıöşüâîû]{1,6}$", re.IGNORECASE)
EDGE = ".,;:!?\"'()[]"

COUNTS = ("n_gold", "tp_exact", "tp_overlap", "typed_overlap", "full_cover", "half_cover",
          "missed", "n_pred", "pred_exact", "pred_overlap", "gold_chars", "gold_chars_masked",
          "other_chars", "other_chars_masked", "neg_rows", "neg_rows_flagged", "neg_pred",
          "neg_chars")


TITLE_PREFIX = re.compile(
    r"(?:(?:Prof|Doç|Dr|Dt|Dyt|Av|Uzm|Op|Yrd|Öğr|Arş|Gör|Psk|Müh|Sn|Hz|Mr|Mrs|Ms)\.\s*"
    r"|(?:Psikolog|Danışman|Hemşire|Avukat|Doktor|Öğretmen|Mühendis|Savcı|Bay|Bayan|Sayın|SMMM|Mali Müşavir)\s+)+",
    re.IGNORECASE)  # no word that is also a name (Hakim, Uzman): trimming it would hide a leak
HONORIFIC_SUFFIX = re.compile(r"\s+(?:Bey|Hanım|Beyefendi|Hanımefendi|Efendi)$", re.IGNORECASE)


def _strip_edges(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and (text[s].isspace() or text[s] in EDGE):
        s += 1
    while e > s and (text[e - 1].isspace() or text[e - 1] in EDGE):
        e -= 1
    return s, e


def core(text: str, s: int, e: int) -> tuple[int, int]:
    s, e = _strip_edges(text, s, e)
    m = APOSTROPHE_SUFFIX.search(text[s:e])
    if m and m.start() > 0:
        e = s + m.start()
    m = TITLE_PREFIX.match(text, s, e)
    if m and m.end() < e:
        s = m.end()
    m = HONORIFIC_SUFFIX.search(text[s:e])
    if m and m.start() > 0:
        e = s + m.start()
    return _strip_edges(text, s, e)


def _new():
    return dict.fromkeys(COUNTS, 0)


def score_row(row, preds, in_scope):
    """Counts of one row: overall, per coarse class of the gold or prediction, and the errors."""
    text, n = row.text, len(row.text)
    ignore = getattr(row, "ignore", None) or []
    preds = [p for p in preds if not any(p[0] < b and a < p[1] for a, b in ignore)]
    masked = bytearray(n)
    for p in preds:
        for i in range(max(0, p[0]), min(n, p[1])):
            masked[i] = 1
    gold_core = [core(text, g[0], g[1]) for g in row.gold]
    kind = bytearray(n)  # 0 other, 1 PII core, 2 trimmed edge of a gold span or an ignored value (neither)
    for a, b in ignore:
        for i in range(a, min(n, b)):
            kind[i] = 2
    for g, (cs, ce) in zip(row.gold, gold_core):
        for i in range(g[0], min(n, g[1])):
            kind[i] = max(kind[i], 2)
        for i in range(cs, min(n, ce)):
            kind[i] = 1
    scoped = [p for p in preds if in_scope(p)]
    pred_core = [core(text, p[0], p[1]) for p in scoped]
    total = _new()
    by_type = defaultdict(_new)
    errors = {"missed": [], "partial": [], "false": []}
    total["n_pred"] = len(scoped)
    if not row.gold and not ignore:
        total["neg_rows"] = 1
        total["neg_rows_flagged"] = int(bool(scoped))
        total["neg_pred"] = len(scoped)
        total["neg_chars"] = n
    for i in range(n):
        if kind[i] == 1:
            total["gold_chars"] += 1
            total["gold_chars_masked"] += masked[i]
        elif kind[i] == 0:
            total["other_chars"] += 1
            total["other_chars_masked"] += masked[i]
    for g, (cs, ce) in zip(row.gold, gold_core):
        cover = sum(masked[cs:ce]) / max(1, ce - cs)
        overlapping = [p for p in scoped if p[0] < g[1] and g[0] < p[1]]
        exact = any(pc == (cs, ce) for pc in pred_core)
        for t in (total, by_type[g[2]]):
            t["n_gold"] += 1
            t["tp_exact"] += exact
            t["tp_overlap"] += bool(overlapping)
            t["typed_overlap"] += any(p[2] == g[2] for p in overlapping)
            t["full_cover"] += cover >= 0.999
            t["half_cover"] += cover >= 0.5
            t["missed"] += cover == 0
        if cover == 0:
            errors["missed"].append((g[3], text[g[0]:g[1]]))
        elif cover < 0.999:
            errors["partial"].append((g[3], text[g[0]:g[1]], [text[p[0]:p[1]] for p in overlapping]))
    for p, pc in zip(scoped, pred_core):
        overlap = any(p[0] < g[1] and g[0] < p[1] for g in row.gold)
        exact = any(pc == gc for gc in gold_core)
        total["pred_exact"] += exact
        total["pred_overlap"] += overlap
        by_type[p[2]]["n_pred"] += 1
        by_type[p[2]]["pred_exact"] += exact
        by_type[p[2]]["pred_overlap"] += overlap
        if not overlap:
            errors["false"].append((p[3], text[p[0]:p[1]]))
    return total, by_type, errors


def evaluate(rows, system, in_scope=lambda p: True, keep_errors=300):
    """Scores every row. Returns per-row totals (for bootstrapping), sums per class and per
    slice, and up to `keep_errors` examples of each error kind with their row."""
    per_row, by_type, by_slice = [], defaultdict(_new), defaultdict(_new)
    examples = {"missed": [], "partial": [], "false": []}
    # a model scores every text in batches when it can
    predictions = system.batch([r.text for r in rows]) if hasattr(system, "batch") else None
    for k, row in enumerate(rows):
        preds = predictions[k] if predictions is not None else system(row.text)
        total, types, errors = score_row(row, preds, in_scope)
        per_row.append(total)
        for k, t in types.items():
            _add(by_type[k], t)
        _add(by_slice[row.slice], total)
        for kind, items in errors.items():
            for item in items:
                if len(examples[kind]) < keep_errors:
                    examples[kind].append((row.rid, *item, row.text))
    return per_row, by_type, by_slice, examples


def _add(into, counts):
    for k in COUNTS:
        into[k] += counts[k]


def total_of(per_row):
    t = _new()
    for r in per_row:
        _add(t, r)
    return t


def _ratio(a, b):
    return a / b if b else None


def _f1(p, r):
    return 2 * p * r / (p + r) if p and r else 0.0


def summary(t: dict) -> dict:
    exact_p, exact_r = _ratio(t["pred_exact"], t["n_pred"]), _ratio(t["tp_exact"], t["n_gold"])
    overlap_p, overlap_r = _ratio(t["pred_overlap"], t["n_pred"]), _ratio(t["tp_overlap"], t["n_gold"])
    return {
        "n_gold": t["n_gold"], "n_pred": t["n_pred"],
        "exact_p": exact_p, "exact_r": exact_r, "exact_f1": _f1(exact_p, exact_r),
        "overlap_p": overlap_p, "overlap_r": overlap_r, "overlap_f1": _f1(overlap_p, overlap_r),
        "typed_r": _ratio(t["typed_overlap"], t["n_gold"]),
        "full_cover_r": _ratio(t["full_cover"], t["n_gold"]),
        "half_cover_r": _ratio(t["half_cover"], t["n_gold"]),
        "missed": _ratio(t["missed"], t["n_gold"]),
        "char_leak": None if not t["gold_chars"] else 1 - t["gold_chars_masked"] / t["gold_chars"],
        "overmask": _ratio(t["other_chars_masked"], t["other_chars"]),
        "neg_rows": t["neg_rows"],
        "neg_flagged": _ratio(t["neg_rows_flagged"], t["neg_rows"]),
        "neg_fp_per_10k": None if not t["neg_chars"] else 1e4 * t["neg_pred"] / t["neg_chars"],
    }


def bootstrap(per_row, metrics=("exact_f1", "full_cover_r", "char_leak", "neg_flagged"), n=1000, seed=0):
    """Percentile 95% intervals of `metrics`, resampling rows with replacement."""
    rng = random.Random(seed)
    values = {m: [] for m in metrics}
    k = len(per_row)
    for _ in range(n):
        t = _new()
        for _ in range(k):
            _add(t, per_row[rng.randrange(k)])
        s = summary(t)
        for m in metrics:
            if s[m] is not None:
                values[m].append(s[m])
    out = {}
    for m, v in values.items():
        if v:
            v.sort()
            out[m] = (v[int(0.025 * (len(v) - 1))], v[int(0.975 * (len(v) - 1))])
    return out


def fmt(d: dict) -> str:
    def f(x):
        return "  -  " if x is None else f"{x:.3f}"
    s = (f"gold {d['n_gold']:5d} pred {d['n_pred']:5d} | exact P/R/F1 {f(d['exact_p'])}/{f(d['exact_r'])}/{f(d['exact_f1'])}"
         f" | overlap P/R/F1 {f(d['overlap_p'])}/{f(d['overlap_r'])}/{f(d['overlap_f1'])}"
         f" | full-cover R {f(d['full_cover_r'])} | leak {f(d['char_leak'])} overmask {f(d['overmask'])}")
    if d["neg_rows"]:
        s += f" | PII-free flagged {f(d['neg_flagged'])} of {d['neg_rows']}"
    return s
