"""Scores a tagger at several operating points (tagger.Tagger's o_bias) on the dev sets: the model
runs once, only the decoding repeats.

    python scripts/pii_train/sweep_bias.py --model .cache/pii_models/electra-small-v2/best
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guard  # noqa: E402

guard.limit_threads(4)

import data  # noqa: E402
import tagger  # noqa: E402

sys.path.insert(0, str(HERE.parent / "pii_eval"))
import datasets as evsets  # noqa: E402
import score  # noqa: E402
import systems  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--biases", default="-3,-2,-1,0,1,2,3,4")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t = tagger.Tagger(args.model, batch_size=16)
    for name, rows in data.dev_sets().items():
        texts = [r["text"] for r in rows]
        scored = t.scores(texts)
        ev_rows = [evsets.Row(r["text"], [(s, e, lab, lab) for s, e, lab in r["spans"]]) for r in rows]
        print(f"\n=== {name} ({len(rows)} rows)")
        for bias in (float(b) for b in args.biases.split(",")):
            preds = t.decode(scored, len(texts), o_bias=bias, texts=texts)
            preds = [[(s, e, lab, lab) for s, e, lab in p if not systems.public_value(x[s:e], lab)]
                     for x, p in zip(texts, preds)]
            system = lambda text, _p=iter(preds): next(_p)
            per_row, _, _, _ = score.evaluate(ev_rows, system)
            print(f"  bias {bias:+.1f} {score.fmt(score.summary(score.total_of(per_row)))}")


if __name__ == "__main__":
    main()
