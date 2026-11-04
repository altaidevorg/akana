"""Batches for the gold adjudication: every unit of a pool with the candidate values its teacher
passes proposed (which passes proposed each), for an adjudicator that keeps, fixes, drops or adds
values by the guideline.

    python scripts/pii_data/adjudication_batches.py --pool test --labels pilot_A pilot_B pilot_C test1_A test1_B --run gold_test
Prints the batch paths as a JSON list.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / ".cache" / "pii_data" / "real"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--labels", nargs="+", required=True, help="label files in labels/ (without .jsonl)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()
    units = [json.loads(l) for l in open(REAL / f"{args.pool}.jsonl", encoding="utf-8")]
    # candidates by text: (value, type, attributes) -> passes that proposed it
    proposals = defaultdict(dict)
    labelled = set()
    for name in args.labels:
        tag = name.split("_")[-1]
        for line in open(REAL / "labels" / f"{name}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            labelled.add(r["text"])
            for s, e, t, attrs in r["spans"]:
                key = (r["text"][s:e], t, json.dumps(attrs, sort_keys=True, ensure_ascii=False))
                proposals[r["text"]].setdefault(key, set()).add(tag)
    missing = [u["id"] for u in units if u["text"] not in labelled]
    if missing:
        sys.exit(f"{len(missing)} unit(s) of the pool have no teacher labels, e.g. {missing[:3]}")
    out = REAL / "batches" / args.run
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in range(0, len(units), args.size):
        path = out / f"J_{k // args.size:03d}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for u in units[k:k + args.size]:
                cands = [{"text": v, "type": t, **json.loads(a), "proposed_by": "".join(sorted(p))}
                         for (v, t, a), p in sorted(proposals[u["text"]].items(), key=lambda kv: u["text"].find(kv[0][0]))]
                f.write(json.dumps({"id": u["id"], "text": u["text"], "candidates": cands}, ensure_ascii=False) + "\n")
        paths.append(str(path))
    sys.stdout.write(json.dumps(paths) + "\n")


if __name__ == "__main__":
    main()
