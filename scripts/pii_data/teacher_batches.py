"""Cuts a pool of units (sample_real.py) into batch files for the teacher workflow.

Each pass gets its own batch files: pass A keeps the pool's order, pass B reverses it inside every
batch, so that two independent labellings of the same units see them in different positions.

    python scripts/pii_data/teacher_batches.py --pool test --run pilot --start 0 --count 320 --size 64 --passes A,B
Prints the batch paths as a JSON list (the workflow's `args.batches`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / ".cache" / "pii_data" / "real"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="a pool file in .cache/pii_data/real (test, dev, train, hard1, ...)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--passes", default="A")
    ap.add_argument("--exclude", nargs="*", default=[], help="label files whose texts are already labelled")
    args = ap.parse_args()
    units = [json.loads(l) for l in open(REAL / f"{args.pool}.jsonl", encoding="utf-8")]
    done = {json.loads(l)["text"] for path in args.exclude for l in open(path, encoding="utf-8")}
    units = [u for u in units if u["text"] not in done][args.start:args.start + args.count]
    out = REAL / "batches" / args.run
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for p in args.passes.split(","):
        for k in range(0, len(units), args.size):
            batch = units[k:k + args.size]
            if p == "B":
                batch = batch[::-1]
            path = out / f"{p}_{k // args.size:03d}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for u in batch:
                    f.write(json.dumps({"id": u["id"], "text": u["text"]}, ensure_ascii=False) + "\n")
            paths.append(str(path))
    sys.stdout.write(json.dumps(paths) + "\n")


if __name__ == "__main__":
    main()
