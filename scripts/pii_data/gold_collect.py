"""Turns the adjudication workflow's answers into gold span files, and lists what the adjudicator
changed against the teacher passes so a person can check those changes.

    python scripts/pii_data/gold_collect.py --journal <adjudication transcript dir>/journal.jsonl --run gold_test --pool test
Writes .cache/pii_data/real/gold_<pool>.jsonl ({"id", "domain", "text", "spans", "note"}) and
gold_<pool>_changes.jsonl (the units whose final list differs from what both passes agreed on).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teacher_collect import find_results, spans_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / ".cache" / "pii_data" / "real"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--pool", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    units = {json.loads(l)["id"]: json.loads(l) for l in open(REAL / f"{args.pool}.jsonl", encoding="utf-8")}
    candidates = {}
    for path in (REAL / "batches" / args.run).glob("*.jsonl"):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            candidates[r["id"]] = r["candidates"]
    label_of, final = {}, {}
    for line in open(args.journal, encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("type") == "started":
            label_of[rec["key"]] = rec.get("label", "")
        elif rec.get("type") == "result":
            # labels read "adjudicate <run>/J_000.jsonl"
            label = label_of.get(rec["key"], "")
            if not label or label.split()[-1].rpartition("/")[0] != args.run:
                continue
            found = []
            find_results(rec.get("result"), found)
            for r in found:
                for u in r["units"]:
                    final[u["id"]] = u
    missing = [uid for uid in units if uid not in final]
    if missing:
        print(f"{len(missing)} unit(s) without an adjudicated answer, e.g. {missing[:3]}")
    # a person's corrections after reviewing the adjudicator's changes (gold_corrections.json)
    fixes = json.load(open(Path(__file__).with_name("gold_corrections.json"), encoding="utf-8")).get(args.pool, [])
    for fix in fixes:
        hits = [uid for uid, u in units.items() if fix["match"] in u["text"] and uid in final]
        if len(hits) != 1:
            sys.exit(f"correction {fix['match']!r} matches {len(hits)} units")
        ents = final[hits[0]]["entities"]
        for rm in fix.get("remove", []):
            before = len(ents)
            ents[:] = [e for e in ents if not (e["text"] == rm["text"] and e["type"] == rm["type"])]
            if len(ents) == before:
                sys.exit(f"correction: {rm} not found in {hits[0]}")
        ents += fix.get("add", [])
    problems = Counter()
    changed = []
    types = Counter()
    with open(REAL / f"gold_{args.pool}.jsonl", "w", encoding="utf-8") as f:
        for uid, u in units.items():
            if uid not in final:
                continue
            spans, probs = spans_of(u["text"], final[uid]["entities"])
            problems.update(p["kind"] for p in probs)
            types.update(s[2] for s in spans)
            note = final[uid].get("note", "")
            f.write(json.dumps({"id": uid, "domain": u["domain"], "text": u["text"], "spans": spans, "note": note,
                                "problems": probs}, ensure_ascii=False) + "\n")
            agreed = {(c["text"], c["type"]) for c in candidates.get(uid, []) if len(c["proposed_by"]) >= 2}
            now = {(e["text"], e["type"]) for e in final[uid]["entities"]}
            if now != agreed or note:
                changed.append({"id": uid, "text": u["text"], "added": sorted(now - agreed), "removed": sorted(agreed - now),
                                "candidates": candidates.get(uid, []), "note": note})
    with open(REAL / f"gold_{args.pool}_changes.jsonl", "w", encoding="utf-8") as f:
        for c in changed:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"gold_{args.pool}: {len(final)} units, {sum(types.values())} spans {dict(types)}; problems {dict(problems)}")
    print(f"{len(changed)} unit(s) differ from what the teacher passes agreed on")


if __name__ == "__main__":
    main()
