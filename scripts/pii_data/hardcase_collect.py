"""Collects the hard-case generator's texts (logs, JSON, forms, notices ... with personal values
beside look-alikes) and keeps only those an independent teacher pass labels the same way.

    python scripts/pii_data/hardcase_collect.py generated --journal <generation journal> --run hard1
        writes .cache/pii_data/real/hard1.jsonl (a pool for teacher_batches.py) and
        .cache/pii_data/hardcases/hard1_generated.jsonl (the generator's own spans); --only PREFIX keeps
        the agents whose label ends in a scenario starting with PREFIX (one journal, several runs)
    python scripts/pii_data/hardcase_collect.py agreed --run hard1
        compares with labels/hard1_A.jsonl (teacher_collect.py) and writes
        .cache/pii_data/hardcases/hard1.jsonl: the texts both label identically
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teacher_collect import find_results, spans_of  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pii_train"))
import data  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / ".cache" / "pii_data" / "real"
HARD = ROOT / ".cache" / "pii_data" / "hardcases"


def generated(args):
    label_of, texts = {}, []
    for line in open(args.journal, encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("type") == "started":
            label_of[rec["key"]] = rec.get("label", "")
        elif rec.get("type") == "result":
            scenario = label_of.get(rec["key"], "generate ?").split()[-1]
            if args.only and not scenario.startswith(args.only):
                continue
            result = rec.get("result")
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except ValueError:
                    continue
            for k, t in enumerate((result or {}).get("texts", [])):
                texts.append((scenario, k, t))
    HARD.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    types = Counter()
    seen = set()
    with open(REAL / f"{args.run}.jsonl", "w", encoding="utf-8") as pool, \
         open(HARD / f"{args.run}_generated.jsonl", "w", encoding="utf-8") as gen:
        for scenario, k, t in texts:
            text = t["text"].strip()
            if not text or text in seen:
                dropped += 1
                continue
            seen.add(text)
            spans, problems = spans_of(text, t["entities"])
            if problems:
                dropped += 1
                continue
            uid = f"{args.run}:{scenario}:{k}"
            pool.write(json.dumps({"id": uid, "domain": f"hard-{scenario}", "rich": True, "text": text}, ensure_ascii=False) + "\n")
            gen.write(json.dumps({"id": uid, "text": text, "spans": spans}, ensure_ascii=False) + "\n")
            types.update(s[2] for s in spans)
            kept += 1
    print(f"{kept} texts kept, {dropped} dropped (duplicates or values not found); spans {dict(types)}")


def key(span):
    """What training sees of a span (data.keep): a URL or DATE is learnt only when personal, and
    SENSITIVE not at all; None for a span training drops."""
    s, e, label, *rest = span
    if data.keep(label, (rest[0] if rest else None) or {}) is None:
        return None
    return s, e, label


def agreed(args):
    gen = {json.loads(l)["id"]: json.loads(l) for l in open(HARD / f"{args.run}_generated.jsonl", encoding="utf-8")}
    teacher = {json.loads(l)["id"]: json.loads(l) for l in open(REAL / "labels" / f"{args.labels or args.run + '_A'}.jsonl", encoding="utf-8")}
    kept, differ = 0, Counter()
    with open(HARD / f"{args.run}.jsonl", "w", encoding="utf-8") as f:
        for uid, g in gen.items():
            t = teacher.get(uid)
            if t is None:
                differ["no teacher labels"] += 1
                continue
            a = {key(s) for s in g["spans"]} - {None}
            b = {key(s) for s in t["spans"]} - {None}
            if a != b:
                for s in a - b:
                    differ[f"generator only {s[2]}"] += 1
                for s in b - a:
                    differ[f"teacher only {s[2]}"] += 1
                continue
            f.write(json.dumps({"id": uid, "text": g["text"], "spans": t["spans"]}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"{kept} of {len(gen)} texts labelled identically; disagreements {dict(differ.most_common())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["generated", "agreed"])
    ap.add_argument("--journal")
    ap.add_argument("--run", default="hard1")
    ap.add_argument("--labels", help="the teacher label file to compare with (labels/<name>.jsonl; default <run>_A)")
    ap.add_argument("--only", help="generated: only scenarios whose name starts with this prefix")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    generated(args) if args.step == "generated" else agreed(args)


if __name__ == "__main__":
    main()
