"""Collects the teacher workflow's labels from its journal and turns them into character spans.

The teacher returns each value as text; this finds it in its unit (every occurrence, or the one
`occurrence` names) and records values it cannot find. With two passes over the same units it also
reports how far they agree, the first measure of the teacher's reliability.

    python scripts/pii_data/teacher_collect.py --journal <workflow transcript dir>/journal.jsonl --run pilot
Writes .cache/pii_data/real/labels/<run>_<pass>.jsonl ({"id", "text", "spans", "problems"}) and
<run>_disagreements.jsonl when there are two passes.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / ".cache" / "pii_data" / "real"
ATTRS = ("public", "personal", "subtype")


def find_results(obj, found):
    """Every structured result ({"units": [...]}) anywhere in a journal record."""
    if isinstance(obj, dict):
        if isinstance(obj.get("units"), list) and all(isinstance(u, dict) and "id" in u for u in obj["units"]):
            found.append(obj)
            return
        for v in obj.values():
            find_results(v, found)
    elif isinstance(obj, list):
        for v in obj:
            find_results(v, found)
    elif isinstance(obj, str) and '"units"' in obj:
        try:
            find_results(json.loads(obj), found)
        except ValueError:
            pass


def occurrences(text: str, value: str) -> list[tuple[int, int]]:
    out, at = [], 0
    while (i := text.find(value, at)) >= 0:
        out.append((i, i + len(value)))
        at = i + 1
    if not out:  # tolerate a difference in spacing only
        pattern = r"\s+".join(map(re.escape, value.split()))
        out = [(m.start(), m.end()) for m in re.finditer(pattern, text)] if pattern else []
    if not out:
        # C1 control characters the teacher dropped: a cp1252 apostrophe read as Latin-1
        # (Terim\x92in), which it copies as "Terimin"
        keep = [i for i, c in enumerate(text) if not "\x80" <= c <= "\x9f"]
        clean = "".join(text[i] for i in keep)
        at = 0
        while (i := clean.find(value, at)) >= 0:
            out.append((keep[i], keep[i + len(value) - 1] + 1))
            at = i + 1
    return out


def spans_of(text: str, entities: list[dict]):
    spans, problems = [], []
    for ent in entities:
        value = ent.get("text", "")
        found = occurrences(text, value)
        if not found:
            problems.append({"kind": "not_found", "entity": ent})
            continue
        if ent.get("occurrence"):
            k = ent["occurrence"] - 1
            found = found[k:k + 1] if 0 <= k < len(found) else found[:1]
        attrs = {a: ent[a] for a in ATTRS if a in ent}
        for s, e in found:
            spans.append([s, e, ent["type"], attrs])
    spans.sort(key=lambda x: (x[0], -x[1]))
    kept = []
    for sp in spans:  # drop exact duplicates and spans nested in a longer one of the same type
        if any(k[0] <= sp[0] and sp[1] <= k[1] and k[2] == sp[2] for k in kept):
            continue
        kept.append(sp)
    for a, b in zip(kept, kept[1:]):
        if b[0] < a[1]:
            problems.append({"kind": "overlap", "a": a, "b": b})
    return kept, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--pool", default="test")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    texts = {}
    for line in open(REAL / f"{args.pool}.jsonl", encoding="utf-8"):
        u = json.loads(line)
        texts[u["id"]] = (u["text"], u["domain"])
    batches = {p.stem: [json.loads(l)["id"] for l in open(p, encoding="utf-8")]
               for p in (REAL / "batches" / args.run).glob("*.jsonl")}
    # the journal's "started" records name each agent's batch in their label ("label A_000.jsonl");
    # its "result" records carry the returned value under the same key
    label_of, results = {}, []
    for line in open(args.journal, encoding="utf-8"):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("type") == "started":
            label_of[rec["key"]] = rec.get("label", "")
        elif rec.get("type") == "result":
            found = []
            find_results(rec.get("result"), found)
            # labels read "label <run>/<pass>_<k>.jsonl" (or, in the pilot, "label <pass>_<k>.jsonl")
            label = label_of.get(rec["key"], "").split()[-1] if label_of.get(rec["key"]) else ""
            run, _, file = label.rpartition("/")
            if run and run != args.run:
                continue
            results += [(Path(file).stem, r) for r in found]
    # the pass is the batch file's prefix (A_000 -> A)
    by_pass = defaultdict(dict)
    unmatched = 0
    for best, r in results:
        ids = [u["id"] for u in r["units"]]
        if best not in batches or not set(ids) & set(batches[best]):
            unmatched += 1
            continue
        missing = set(batches[best]) - set(ids)
        for u in r["units"]:
            if u["id"] in texts:
                by_pass[best.split("_")[0]][u["id"]] = u["entities"]
        if missing:
            print(f"{best}: {len(missing)} unit(s) not returned")
    out = REAL / "labels"
    out.mkdir(exist_ok=True)
    labelled = {}
    for p, units in sorted(by_pass.items()):
        problems = Counter()
        rows = {}
        with open(out / f"{args.run}_{p}.jsonl", "w", encoding="utf-8") as f:
            for uid, ents in units.items():
                text, domain = texts[uid]
                spans, probs = spans_of(text, ents)
                problems.update(x["kind"] for x in probs)
                rows[uid] = spans
                f.write(json.dumps({"id": uid, "domain": domain, "text": text, "spans": spans, "problems": probs}, ensure_ascii=False) + "\n")
        labelled[p] = rows
        types = Counter(s[2] for spans in rows.values() for s in spans)
        print(f"pass {p}: {len(units)} units, {sum(types.values())} spans {dict(types)}; problems {dict(problems)}")
    if unmatched:
        print(f"{unmatched} result(s) matched no batch")
    if len(labelled) == 2:
        a, b = (labelled[k] for k in sorted(labelled))
        common = sorted(set(a) & set(b))
        exact = Counter()
        totals = Counter()
        disagreements = []
        for uid in common:
            sa = {(s[0], s[1], s[2]) for s in a[uid]}
            sb = {(s[0], s[1], s[2]) for s in b[uid]}
            for t in {x[2] for x in sa | sb}:
                ta = {x for x in sa if x[2] == t}
                tb = {x for x in sb if x[2] == t}
                exact[t] += len(ta & tb)
                totals[t] += len(ta | tb)
            if sa != sb:
                text = texts[uid][0]
                disagreements.append({"id": uid, "text": text,
                                      "only_A": [[s, e, t, text[s:e]] for s, e, t in sorted(sa - sb)],
                                      "only_B": [[s, e, t, text[s:e]] for s, e, t in sorted(sb - sa)],
                                      "both": [[s, e, t, text[s:e]] for s, e, t in sorted(sa & sb)]})
        with open(out / f"{args.run}_disagreements.jsonl", "w", encoding="utf-8") as f:
            for d in disagreements:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"\nagreement over {len(common)} units ({len(disagreements)} differ): exact span+type agreement")
        for t in sorted(totals, key=lambda t: -totals[t]):
            print(f"  {t:10s} {exact[t]:4d}/{totals[t]:4d} = {exact[t] / totals[t]:.3f}")
        print(f"  {'all':10s} {sum(exact.values()):4d}/{sum(totals.values()):4d} = {sum(exact.values()) / max(1, sum(totals.values())):.3f}")


if __name__ == "__main__":
    main()
