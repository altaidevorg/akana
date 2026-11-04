"""Scores the small tagger, the big tagger, and their cascade (tagger.Cascade) at several thresholds
on the dev sets. Both models read every window once; the cascade is then simulated per threshold,
so the escalation rate (the big model's share of the work) comes with each score.

    python scripts/pii_train/sweep_cascade.py --small .cache/pii_models/electra-small-v3/best --big .cache/pii_models/berturk-v3/best
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guard  # noqa: E402

guard.limit_threads(4)

import numpy as np  # noqa: E402

import data  # noqa: E402
import tagger  # noqa: E402

sys.path.insert(0, str(HERE.parent / "pii_eval"))
import datasets as evsets  # noqa: E402
import score  # noqa: E402
import systems  # noqa: E402


def scored_summary(rows, texts, scored, bias):
    preds = tagger.decode(scored, len(texts), bias, texts)
    preds = [[(s, e, lab, lab) for s, e, lab in p if not systems.public_value(x[s:e], lab)] for x, p in zip(texts, preds)]
    ev_rows = [evsets.Row(r["text"], [(s, e, lab, lab) for s, e, lab in r["spans"]]) for r in rows]
    system = lambda text, _p=iter(preds): next(_p)
    per_row, _, _, _ = score.evaluate(ev_rows, system)
    return score.fmt(score.summary(score.total_of(per_row)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", required=True)
    ap.add_argument("--big", required=True)
    ap.add_argument("--thresholds", default="0.01,0.02,0.05,0.1,0.2,0.3")
    ap.add_argument("--bias", type=float, default=0.0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    small, big = tagger.Tagger(args.small, batch_size=16), tagger.Tagger(args.big, batch_size=16)
    assert small.tok.get_vocab() == big.tok.get_vocab()
    for name, rows in data.dev_sets().items():
        texts = [r["text"] for r in rows]
        wins = small.windows(texts)
        lp_small, lp_big = small.logprobs(wins), big.logprobs(wins)
        real = [[j for j, (a, c) in enumerate(offs) if a != c] for _, offs, _ in wins]
        unsure = [tagger.uncertainty(lp[r]) for lp, r in zip(lp_small, real)]
        print(f"\n=== {name} ({len(rows)} rows, {len(wins)} windows)")
        print(f"  small            {scored_summary(rows, texts, tagger.real_tokens(wins, lp_small), args.bias)}")
        print(f"  big              {scored_summary(rows, texts, tagger.real_tokens(wins, lp_big), args.bias)}")
        mean = [np.logaddexp(a, b) - np.log(2) for a, b in zip(lp_small, lp_big)]
        print(f"  average (all)    {scored_summary(rows, texts, tagger.real_tokens(wins, mean), args.bias)}")
        for th in (float(x) for x in args.thresholds.split(",")):
            up = [u > th for u in unsure]
            for mode, mixed in (("replace", lp_big), ("average", mean)):
                lps = [m if go else s for s, m, go in zip(lp_small, mixed, up)]
                print(f"  t={th:<5} {mode:7s} escalated {sum(up) / len(up):5.1%}  "
                      f"{scored_summary(rows, texts, tagger.real_tokens(wins, lps), args.bias)}")


if __name__ == "__main__":
    main()
