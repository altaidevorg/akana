"""Samples a fresh development pool (dev2) from documents that no model was trained or chosen on.

sample_real.py splits every source's documents by hash: 10 % are dev documents, and dev.jsonl (the
gold-dev units) took units from some of them. dev2 takes units only from dev documents that gave
none, so neither the shipped models nor the candidates were selected on any of its text. A unit is
also dropped when it shares an 8-word shingle (select_pool.py) with any unit of a training, dev or
test pool, so a model trained on text that reappears here gains nothing.

The domains keep gold-dev's proportions at twice its size. Within a domain, about a third of the
units hold a date or an @handle (where the tagger's dev evidence is thinnest), a third are other
"rich" units (sample_real.is_rich) and 0.3 are plain, so false alarms on ordinary text are measured
too. Where the dev side has too few fresh documents, a source no PII model has used tops the domain
up: sentences of WikiNER's validation split for Wikipedia (the models trained on its train split,
whose sentences are among the shingles a unit must not share), the AYTAS prompt set for prompts.

    python scripts/pii_data/sample_fresh_dev.py --dry-run
    python scripts/pii_data/sample_fresh_dev.py --out dev2
Writes .cache/pii_data/real/<out>.jsonl: {"id", "domain", "rich", "stratum", "text"}.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sample_real as sr  # noqa: E402
from select_pool import shingles  # noqa: E402

sys.path.insert(0, str(sr.ROOT / "scripts" / "pii_eval"))
import datasets as evsets  # noqa: E402

REAL = sr.ROOT / ".cache" / "pii_data" / "real"
# every pool whose text a model may have trained on or been chosen on
SEEN = ("test", "dev", "train", "train2", "unl1", "silver3", "hard1", "hard2", "hard3", "gold_dev", "gold_test")
TARGETS = {"court": 360, "wikipedia": 240, "news": 90, "tweets": 240, "assistant": 240, "word_problems": 120,
           "prompts": 120, "reviews": 120, "social": 120}
MONTHS = "ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık"
DATE = re.compile(rf"\b\d{{1,2}}[./-]\d{{1,2}}[./-]\d{{2,4}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b\d{{1,2}}\s+(?:{MONTHS})\b", re.IGNORECASE)
HANDLE = re.compile(r"(?<![\w.])@\w{2,}")


def stratum(text: str, names: set[str]) -> str:
    if DATE.search(text) or HANDLE.search(text):
        return "target"
    return "rich" if sr.is_rich(text, names) else "plain"


def aytas_docs():
    for path in sr.hub("datasets--OnerAYTAS--Turkish_prompt_injection_jailbreak_dataset/snapshots/*/AYTAS.csv"):
        for k, row in enumerate(csv.DictReader(open(path, encoding="utf-8"))):
            yield f"aytas-{k}", row["prompt"]


def wikiner_docs():
    for k, row in enumerate(evsets.wikiner_persons("validation")):
        yield f"wikiner-val-{k}", row.text


TOP_UPS = {"wikipedia": wikiner_docs, "prompts": aytas_docs}


def pick(by: dict, want: int) -> list[dict]:
    """About a third targets, 0.3 plain, the rest rich; whatever is left fills a shortfall, targets first."""
    n_target = min(len(by["target"]), round(want / 3))
    n_plain = min(len(by["plain"]), round(want * 0.3))
    n_rich = min(len(by["rich"]), want - n_target - n_plain)
    picked = by["target"][:n_target] + by["rich"][:n_rich] + by["plain"][:n_plain]
    rest = by["target"][n_target:] + by["rich"][n_rich:] + by["plain"][n_plain:]
    return picked + rest[:max(0, want - len(picked))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dev2")
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    seen_text, seen_shingles, used_docs = set(), set(), set()
    for name in SEEN:
        path = REAL / f"{name}.jsonl"
        if not path.is_file():
            continue
        for line in open(path, encoding="utf-8"):
            u = json.loads(line)
            seen_text.add(u["text"].lower())
            seen_shingles |= shingles(u["text"])
            if name in ("dev", "gold_dev") and "id" in u:
                used_docs.add(u["id"].rsplit(":", 1)[0])
    for split in ("train", "test"):
        for row in evsets.wikiner_persons(split):
            seen_text.add(row.text.lower())
            seen_shingles |= shingles(row.text)
    names = sr.given_names()
    rng = random.Random(args.seed)
    chosen = []

    def candidates(domain, docs, dev_side_only, dropped, seen):
        by = {"target": [], "rich": [], "plain": []}
        for doc_id, text in docs:
            doc = f"{domain}:{doc_id}"
            if dev_side_only and sr.pool_of(doc) != "dev":
                continue
            if doc in used_docs:
                dropped["gold-dev document"] += 1
                continue
            for k, unit in enumerate(sr.units_of(text)):
                key = unit.lower()
                if key in seen:
                    continue
                seen.add(key)
                if key in seen_text or shingles(unit) & seen_shingles:
                    dropped["shares text"] += 1
                    continue
                s = stratum(unit, names)
                by[s].append({"id": f"{doc}:{k}", "domain": domain, "rich": s != "plain", "stratum": s, "text": unit})
        for units in by.values():
            rng.shuffle(units)
        return by

    for domain, loader in sr.LOADERS.items():
        want = TARGETS[domain]
        dropped, seen = Counter(), set()
        by = candidates(domain, loader(), True, dropped, seen)
        picked = pick(by, want)
        available = f"{len(by['target'])}/{len(by['rich'])}/{len(by['plain'])}"
        topped = 0
        if len(picked) < want and domain in TOP_UPS:
            extra = pick(candidates(domain, TOP_UPS[domain](), False, dropped, seen), want - len(picked))
            topped = len(extra)
            picked += extra
        chosen += picked
        print(f"{domain:13s} {len(picked):4d} of {want} (target {sum(u['stratum'] == 'target' for u in picked)}, "
              f"rich {sum(u['stratum'] == 'rich' for u in picked)}, plain {sum(u['stratum'] == 'plain' for u in picked)}; "
              f"dev side available {available}; topped up {topped}; dropped {dict(dropped)})", flush=True)
    print(f"{len(chosen)} units")
    if args.dry_run:
        return
    rng.shuffle(chosen)
    with open(REAL / f"{args.out}.jsonl", "w", encoding="utf-8") as f:
        for u in chosen:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    print(f"-> {REAL / f'{args.out}.jsonl'}")


if __name__ == "__main__":
    main()
