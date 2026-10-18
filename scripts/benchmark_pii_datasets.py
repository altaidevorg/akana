"""PII detection on three public Turkish PII datasets, from the local Hugging Face cache.

- negentropi/belgin-pii-dataset (all splits): span recall per type, and how many of the 300
  public-date rows (campaigns, holidays) have a date masked
- BTX24/turkish-privacy-pii-ner (test): span recall per type
- cagrigungor/turkish-pii-masking-benchmark (benchmark_1000.csv): the input is aligned with
  the expected masked output, and each masked value counts as found when the engine's spans
  cover half of it

A span counts as found when at least half of its characters are masked; "over-masked" is the
share of characters outside the gold spans that the engine masks. Fetch the datasets first
(e.g. `huggingface-cli download --repo-type dataset negentropi/belgin-pii-dataset`).

Usage: python scripts/benchmark_pii_datasets.py [--json out.json]
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import akana

HUB = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
DATE_TYPES = {"DOGUM_TARIHI", "OZEL_TARIH", "TARIH"}
TAG = re.compile(r"\[[A-ZÇĞİÖŞÜ_0-9]+\]")


def snapshot(repo: str) -> Path:
    root = HUB / ("datasets--" + repo.replace("/", "--")) / "snapshots"
    snaps = sorted(root.glob("*"), key=lambda p: p.stat().st_mtime)
    if not snaps:
        sys.exit(f"{repo} is not in {HUB}; download it first")
    return snaps[-1]


ENGINE = akana.TurkishPiiEngine()


def detect(text):
    try:
        return ENGINE.detect(text)
    except BaseException:  # older versions panicked on some rows: count as nothing masked
        return []


def covered(ents, s, e):
    hit = set()
    for x in ents:
        a, b = max(s, x["start"]), min(e, x["end"])
        if a < b:
            hit.update(range(a, b))
    return len(hit) / max(1, e - s)


def masked_chars(ents, n):
    m = set()
    for x in ents:
        m.update(range(x["start"], min(n, x["end"])))
    return m


def span_eval(rows):
    found, total = Counter(), Counter()
    outside = outside_masked = public_rows = public_masked = 0
    for text, spans, case in rows:
        ents = detect(text)
        gold = set()
        for t, s, e in spans:
            total[t] += 1
            found[t] += covered(ents, s, e) >= 0.5
            gold.update(range(s, e))
        outside += len(text) - len(gold)
        outside_masked += len(masked_chars(ents, len(text)) - gold)
        if case == "public_date_negative":
            public_rows += 1
            public_masked += any(x["pii_type"] in DATE_TYPES for x in ents)
    out = {t: {"recall": found[t] / total[t], "n": total[t]} for t in sorted(total)}
    out["_all"] = {"recall": sum(found.values()) / sum(total.values()), "n": sum(total.values())}
    out["_overmasked"] = outside_masked / max(1, outside)
    if public_rows:
        out["_public_dates_masked"] = f"{public_masked}/{public_rows}"
    return out


def belgin():
    rows = []
    for f in sorted((snapshot("negentropi/belgin-pii-dataset") / "data").glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            spans = [(k.split(":")[0], s, e) for k, v in r["spans"].items() for s, e in v]
            rows.append((r["text"], spans, r["info"].get("case")))
    return span_eval(rows)


def btx24():
    rows = []
    for line in open(snapshot("BTX24/turkish-privacy-pii-ner") / "test.jsonl", encoding="utf-8"):
        r = json.loads(line)
        rows.append((r["text"], [(x["category"], x["start"], x["end"]) for x in r["label"]], None))
    return span_eval(rows)


def saturday():
    found, total = Counter(), Counter()
    kept = kept_masked = 0
    path = snapshot("cagrigungor/turkish-pii-masking-benchmark") / "benchmark_1000.csv"
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        text, exp = r["input"], r["expected_output"]
        parts, tags = TAG.split(exp), TAG.findall(exp)
        pos, values, literal = 0, [], set()
        for i, part in enumerate(parts):
            j = text.find(part, pos) if part else pos
            if j < 0:
                break
            if i > 0:
                values.append((tags[i - 1], pos, j))
            literal.update(range(j, j + len(part)))
            pos = j + len(part)
        else:
            if len(parts) > 1 and not parts[-1]:
                values[-1] = (values[-1][0], values[-1][1], len(text))
            ents = detect(text)
            for tag, s, e in values:
                if e > s:
                    total[tag] += 1
                    found[tag] += covered(ents, s, e) >= 0.5
            kept += len(literal)
            kept_masked += len(masked_chars(ents, len(text)) & literal)
    out = {t: {"recall": found[t] / total[t], "n": total[t]} for t in sorted(total)}
    out["_all"] = {"recall": sum(found.values()) / sum(total.values()), "n": sum(total.values())}
    out["_overmasked"] = kept_masked / max(1, kept)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    res = {"version": akana.__version__, "belgin": belgin(), "btx24": btx24(), "saturday_labs": saturday()}
    for name in ("belgin", "btx24", "saturday_labs"):
        r = res[name]
        extra = f", public dates masked {r['_public_dates_masked']}" if "_public_dates_masked" in r else ""
        print(f"\n{name}: recall {r['_all']['recall']:.3f} over {r['_all']['n']} values, "
              f"over-masked {r['_overmasked']:.1%}{extra}")
        for k, v in r.items():
            if not k.startswith("_"):
                print(f"  {k:24s} {v['recall']:.3f}  (n={v['n']})")
    if args.json:
        json.dump(res, open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
