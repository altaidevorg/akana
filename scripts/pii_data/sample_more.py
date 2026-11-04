"""Samples further training units from the training side of sample_real.py's sources, skipping every
unit already in a pool (test, dev, train), for another round of teacher labels.

    python scripts/pii_data/sample_more.py --out train2 --total 6400
Writes .cache/pii_data/real/<out>.jsonl in sample_real.py's format. Only documents whose hash puts
them in the training pool are read, so test and dev documents stay apart.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sample_real as sr  # noqa: E402

# share of the new units per domain: more where the tagger misses names in informal text
SHARES = {"court": 0.17, "wikipedia": 0.15, "news": 0.15, "tweets": 0.16, "reviews": 0.05, "assistant": 0.12,
          "prompts": 0.05, "social": 0.11, "word_problems": 0.04}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--total", type=int, default=6400)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--exclude", nargs="*", default=[], help="further pools in .cache/pii_data/real whose units to skip")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    real = sr.ROOT / ".cache" / "pii_data" / "real"
    pools = ("test", "dev", "train", *args.exclude)
    used = {json.loads(l)["text"].lower() for p in pools for l in open(real / f"{p}.jsonl", encoding="utf-8")}
    names = sr.given_names()
    rng = random.Random(args.seed)
    chosen = []
    for domain, loader in sr.LOADERS.items():
        rich, plain, seen = [], [], set()
        for doc_id, text in loader():
            if sr.pool_of(f"{domain}:{doc_id}") != "train":
                continue
            for k, unit in enumerate(sr.units_of(text)):
                key = unit.lower()
                if key in seen or key in used:
                    continue
                seen.add(key)
                (rich if sr.is_rich(unit, names) else plain).append({"id": f"{domain}:{doc_id}:{k}", "domain": domain, "text": unit})
        want = round(args.total * SHARES[domain])
        rng.shuffle(rich)
        rng.shuffle(plain)
        n_rich = min(len(rich), round(want * 0.7))
        units = [dict(u, rich=True) for u in rich[:n_rich]] + [dict(u, rich=False) for u in plain[:want - n_rich]]
        if len(units) < want:
            units += [dict(u, rich=True) for u in rich[n_rich:n_rich + want - len(units)]]
        chosen += units
        print(f"{domain:13s} {len(units):5d} units (available {len(rich)} rich, {len(plain)} plain)", flush=True)
    rng.shuffle(chosen)
    with open(real / f"{args.out}.jsonl", "w", encoding="utf-8") as f:
        for u in chosen:
            f.write(json.dumps({"id": u["id"], "domain": u["domain"], "rich": u["rich"], "text": u["text"]}, ensure_ascii=False) + "\n")
    print(f"{len(chosen)} units -> {args.out}.jsonl")


if __name__ == "__main__":
    main()
