"""Takes units by domain from an existing training-side pool (sample_real.py / sample_more.py format)
into a new pool for the teacher, dropping any unit that shares text with a dev or test unit.

    python scripts/pii_data/select_pool.py --source unl1 --out silver3 --take tweets=3000 social=2400 court=1500
Writes .cache/pii_data/real/<out>.jsonl, the chosen units in a seeded random order.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / ".cache" / "pii_data" / "real"
HELD = ("dev", "test", "gold_dev", "gold_test")


def shingles(text: str, n: int = 8) -> set[str]:
    """Word n-grams of a unit, lower case and without punctuation: a shared one is shared text."""
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--take", nargs="+", required=True, help="domain=count; a count above the pool's takes all")
    ap.add_argument("--seed", type=int, default=20261103)
    args = ap.parse_args()
    held = set()
    for name in HELD:
        if (REAL / f"{name}.jsonl").is_file():
            for line in open(REAL / f"{name}.jsonl", encoding="utf-8"):
                held |= shingles(json.loads(line)["text"])
    units = [json.loads(l) for l in open(REAL / f"{args.source}.jsonl", encoding="utf-8")]
    rng = random.Random(args.seed)
    chosen = []
    for spec in args.take:
        domain, count = spec.split("=")
        pool = [u for u in units if u["domain"] == domain]
        clean = [u for u in pool if not shingles(u["text"]) & held]
        rng.shuffle(clean)
        chosen += clean[:int(count)]
        print(f"{domain}: {len(pool)} in {args.source}, {len(pool) - len(clean)} share text with dev or test, "
              f"{min(int(count), len(clean))} taken")
    rng.shuffle(chosen)
    with open(REAL / f"{args.out}.jsonl", "w", encoding="utf-8") as f:
        for u in chosen:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    print(f"{len(chosen)} units -> {REAL / f'{args.out}.jsonl'}")


if __name__ == "__main__":
    main()
