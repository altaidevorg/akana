"""Score a PII system on the evaluation sets with the strict scorer.

    python scripts/pii_eval/run.py                          # the installed akana, every test set
    python scripts/pii_eval/run.py --system akana+ner --pkg <dir with an akana build>
    python scripts/pii_eval/run.py --sets yusuf-said,belgin-test --bootstrap 1000 --json out.json --errors errs/

Sets and their roles are listed in datasets.py; metrics are defined in score.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets  # noqa: E402
import score  # noqa: E402
import systems  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", default="akana", choices=sorted(systems.SYSTEMS))
    ap.add_argument("--pkg", help="directory holding the akana package to test")
    ap.add_argument("--model", help="model directory for --system tagger")
    ap.add_argument("--sets", default=",".join(datasets.DEFAULT_SETS))
    ap.add_argument("--bootstrap", type=int, default=0, help="resamples for 95%% intervals (0: none)")
    ap.add_argument("--json", help="write every figure here")
    ap.add_argument("--errors", help="write error examples per set into this directory")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    factory = systems.SYSTEMS[args.system]
    extra = {"model": args.model} if args.model else {}
    plain = factory(pkg=args.pkg, **extra)
    with_tags = factory(pkg=args.pkg, saturday=True, **extra)
    results = {"system": args.system, "sets": {}}
    for name in args.sets.split(","):
        loader, role, labelled = datasets.SETS[name]
        rows = loader()
        system = with_tags if labelled == "saturday" else plain
        if labelled == "person":
            def in_scope(p):
                return p[2] == "person"
        elif labelled == "saturday":
            def in_scope(p):
                return True
        else:
            def in_scope(p):
                return p[2] is not None
        t0 = time.perf_counter()
        per_row, by_type, by_slice, examples = score.evaluate(rows, system, in_scope)
        seconds = time.perf_counter() - t0
        total = score.summary(score.total_of(per_row))
        entry = {"role": role, "rows": len(rows), "seconds": round(seconds, 2), "all": total,
                 "types": {k: score.summary(v) for k, v in by_type.items()},
                 "slices": {k: score.summary(v) for k, v in by_slice.items() if k}}
        print(f"\n=== {name} ({role}, {len(rows)} rows, {seconds:.1f}s)\n  all       {score.fmt(total)}")
        if args.bootstrap:
            entry["ci95"] = score.bootstrap(per_row, n=args.bootstrap)
            print("  95% CI    " + ", ".join(f"{m} {lo:.3f}-{hi:.3f}" for m, (lo, hi) in entry["ci95"].items()))
        if labelled == "all":
            for k in sorted(by_type):
                if by_type[k]["n_gold"]:
                    print(f"  {k:9s} {score.fmt(score.summary(by_type[k]))}")
        results["sets"][name] = entry
        if args.errors:
            os.makedirs(args.errors, exist_ok=True)
            with open(os.path.join(args.errors, f"{name}.json"), "w", encoding="utf-8") as f:
                json.dump(examples, f, ensure_ascii=False, indent=0)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
