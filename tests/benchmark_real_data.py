#!/usr/bin/env python3
"""
Grammatical error detection (GED) benchmark of Akana's grammar checker on GGLab/GECTurk:
1. In-domain test set: 20,769 sentences (230,571 tokens, 25 synthetic error types)
2. Out-of-domain movie reviews: 300 human-written sentences

    python tests/benchmark_real_data.py                 # the default rules
    python tests/benchmark_real_data.py --experimental  # every rule
    python tests/benchmark_real_data.py --limit 5000

A GECTurk row is a list of tokens with an error label each (0 = correct); the sentence checked
is the tokens joined by single spaces. A token counts as flagged when a finding's span
overlaps it. Findings carry character offsets (the binding converts the core's byte
offsets), so the token spans here are character spans too, and every finding is checked to
slice its own text out of the sentence. Reported:
- token-level precision/recall/F0.5 against the labels;
- sentence-level scores, and the share of error-free rows with any finding;
- each rule's precision: the share of its findings that overlap a labelled token. GECTurk
  only labels the errors it injected, so a rule finding real errors of a type GECTurk does not
  inject (apostrophes on proper nouns, "peşpeşe") scores low here although its fixes are right.
"""

import argparse
import collections
import io
import json
import os
import time
import urllib.request

import akana
import pyarrow.parquet as pq

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache_gecturk")


def download_dataset(split_filename):
    os.makedirs(CACHE_DIR, exist_ok=True)
    local_path = os.path.join(CACHE_DIR, os.path.basename(split_filename))
    if os.path.exists(local_path):
        print(f"Loading cached dataset from: {local_path}")
        return pq.read_table(local_path).to_pylist()

    url = f"https://huggingface.co/datasets/GGLab/GECTurk/resolve/main/{split_filename}"
    print(f"Downloading dataset from: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    with open(local_path, "wb") as f:
        f.write(data)
    return pq.read_table(io.BytesIO(data)).to_pylist()


def f_beta(p, r, beta=0.5):
    if p + r == 0:
        return 0.0
    b2 = beta**2
    return (1 + b2) * (p * r) / (b2 * p + r)


def ratio(a, b):
    return a / b if b else 0.0


def evaluate_split(rows, split_name, checker, limit=None):
    if limit:
        rows = rows[:limit]
    print("\n" + "=" * 80)
    print(f" {split_name}: {len(rows):,} sentences")
    print("=" * 80)

    tok = collections.Counter()  # tp, fp, fn, tn
    sent = collections.Counter()
    gold_by_label = collections.Counter()
    hit_by_label = collections.Counter()
    rule_tp = collections.Counter()
    rule_n = collections.Counter()
    clean_rows = clean_rows_flagged = 0
    offset_mismatches = 0

    start_time = time.perf_counter()
    for row in rows:
        tokens, labels = row["tokens"], row["labels"]
        text = " ".join(tokens)
        findings = json.loads(checker.check_json(text))["findings"]

        # character span of each token in `text`
        spans, pos = [], 0
        for t in tokens:
            spans.append((pos, pos + len(t)))
            pos += len(t) + 1

        for f in findings:
            a, b = f["start_offset"], f["end_offset"]
            if text[a:b] != f["original_text"]:
                offset_mismatches += 1
            covered = [i for i, (s, e) in enumerate(spans) if s < b and a < e]
            rule_n[f["rule"]] += 1
            if any(labels[i] > 0 for i in covered):
                rule_tp[f["rule"]] += 1

        flagged = [any(s < f["end_offset"] and f["start_offset"] < e for f in findings) for s, e in spans]
        for label, pred in zip(labels, flagged):
            gold = label > 0
            tok["tp" if gold and pred else "fp" if pred else "fn" if gold else "tn"] += 1
            if gold:
                gold_by_label[label] += 1
                hit_by_label[label] += pred

        has_gold, has_pred = any(l > 0 for l in labels), bool(findings)
        sent["tp" if has_gold and has_pred else "fp" if has_pred else "fn" if has_gold else "tn"] += 1
        if not has_gold:
            clean_rows += 1
            clean_rows_flagged += has_pred

    elapsed = time.perf_counter() - start_time
    p_tok = ratio(tok["tp"], tok["tp"] + tok["fp"])
    r_tok = ratio(tok["tp"], tok["tp"] + tok["fn"])
    p_sent = ratio(sent["tp"], sent["tp"] + sent["fp"])
    r_sent = ratio(sent["tp"], sent["tp"] + sent["fn"])

    print(f"\nSpeed: {len(rows) / elapsed:,.0f} sentences/s ({elapsed:.1f} s)")
    print(f"Findings whose offsets do not slice their text: {offset_mismatches}")
    print("\nToken-level detection:")
    print(f"  precision {p_tok:.1%}  recall {r_tok:.1%}  F0.5 {f_beta(p_tok, r_tok):.1%}")
    print(f"  ({tok['tp']:,} TP, {tok['fp']:,} FP, {tok['fn']:,} FN)")
    print("Sentence-level detection:")
    print(f"  precision {p_sent:.1%}  recall {r_sent:.1%}  F0.5 {f_beta(p_sent, r_sent):.1%}")
    print(
        f"  error-free rows with a finding: {clean_rows_flagged:,}/{clean_rows:,}"
        f" ({ratio(clean_rows_flagged, clean_rows):.1%})"
    )
    print("\nRecall per GECTurk label:")
    print("  " + "  ".join(f"{l}:{hit_by_label[l]}/{gold_by_label[l]}" for l in sorted(gold_by_label)))
    print("\nPrecision per rule (findings on a labelled token / findings):")
    for rule, n in rule_n.most_common():
        print(f"  {rule:<24} {ratio(rule_tp[rule], n):6.1%} of {n:,}")

    return {
        "split": split_name,
        "sentences": len(rows),
        "throughput": len(rows) / elapsed,
        "p_tok": p_tok,
        "r_tok": r_tok,
        "f05_tok": f_beta(p_tok, r_tok),
        "clean_flagged": ratio(clean_rows_flagged, clean_rows),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GECTurk benchmark of the grammar checker")
    ap.add_argument("--experimental", action="store_true", help="run every rule, not only the default ones")
    ap.add_argument("--limit", type=int, help="only the first N rows of each split")
    args = ap.parse_args()

    checker = akana.GrammarChecker(experimental=args.experimental)
    rules = "all rules" if args.experimental else "default rules"
    results = [
        evaluate_split(
            download_dataset("data/movie_reviews-00000-of-00001-b0322cba85108ee9.parquet"),
            f"GECTurk movie reviews, {rules}",
            checker,
            args.limit,
        ),
        evaluate_split(
            download_dataset("data/test-00000-of-00001-82bf0acf5b396a77.parquet"),
            f"GECTurk test set, {rules}",
            checker,
            args.limit,
        ),
    ]

    print("\n" + "=" * 80)
    print(f"{'Split':<45} {'P':>7} {'R':>7} {'F0.5':>7} {'clean flagged':>14}")
    for r in results:
        print(
            f"{r['split']:<45} {r['p_tok']:7.1%} {r['r_tok']:7.1%} {r['f05_tok']:7.1%}"
            f" {r['clean_flagged']:14.1%}"
        )
