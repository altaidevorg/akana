"""Applies the annotation guideline to court-citation dates in teacher label files. The date of a
decision on a named applicant's individual application (`Ahmet Sağlam [2. B.], B. No: 2013/3351,
18/9/2013`) is personal: the guideline's "a private person's hearing or application". The Sonnet
passes marked 49 of 64 such dates public, and the adjudicated dev gold marked them inconsistently
(corrected in gold_corrections.json), so a model trained on the passes learns "public".

    python scripts/pii_data/relabel_dates.py silver1_A silver2_A
Writes .cache/pii_data/real/labels/<run>_cit.jsonl beside each input (which stays as it is) and
prints what changed.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LABELS = Path(__file__).resolve().parents[2] / ".cache" / "pii_data" / "real" / "labels"
CITATION_DATE = re.compile(r"B\. ?No ?: ?\d{4}/\d+\s?,\s?(\d{1,2}[./]\d{1,2}[./]\d{4})")


def relabel(row: dict) -> tuple[dict, int, int]:
    """The row with every citation date personal; (row, dates made personal, dates added)."""
    spans = [list(sp) for sp in row["spans"]]
    flipped = added = 0
    for m in CITATION_DATE.finditer(row["text"]):
        s, e = m.span(1)
        hit = [sp for sp in spans if sp[0] < e and s < sp[1]]
        dates = [sp for sp in hit if sp[2] == "DATE"]
        for sp in dates:
            attrs = sp[3] if len(sp) > 3 else {}
            if attrs.get("personal") is not True:
                flipped += 1
            sp[3:] = [{**attrs, "personal": True}]
        if not hit:
            spans.append([s, e, "DATE", {"personal": True}])
            added += 1
    spans.sort(key=lambda sp: (sp[0], sp[1]))
    return {**row, "spans": spans}, flipped, added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    for run in args.runs:
        rows = [json.loads(line) for line in open(LABELS / f"{run}.jsonl", encoding="utf-8")]
        out, flipped, added = [], 0, 0
        for r in rows:
            r, f, a = relabel(r)
            out.append(r)
            flipped += f
            added += a
        with open(LABELS / f"{run}_cit.jsonl", "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{run}: {len(rows)} units, {flipped} citation dates made personal, {added} added -> {run}_cit.jsonl")


if __name__ == "__main__":
    main()
