"""Chooses the tagger's operating point (tagger.Tagger's o_bias) by conformal risk control
(Angelopoulos et al., "Conformal Risk Control", 2022): the smallest bias λ with

    n/(n+1) · R̂(λ) + B/(n+1) ≤ α

where R̂(λ) is the mean, over the n calibration rows that hold personal data, of the share of their
PII characters left in clear (B = 1). With calibration and future rows exchangeable, the expected
leak share of a future row with personal data is then at most α. The loss is made non-increasing in
λ by taking, for each row, the largest loss at any bias at or above λ.

    python scripts/pii_train/calibrate.py --model .cache/pii_models/electra-small-v3/best --alphas 0.02,0.03,0.05
    python scripts/pii_train/calibrate.py --model <small> --big <big> --threshold -1 --labels PERSON+USERNAME+DATE
    python scripts/pii_train/calibrate.py --model <small> --labels PERSON+USERNAME+DATE --calibration gold-dev+dev2
Calibrates on the real-text dev gold (gold-dev, or with --calibration several golds pooled: dev2 is the
fresh dev gold of sample_fresh_dev.py, on which strict points calibrated on gold-dev alone leaked above
α) and prints, per α, the bias and what it costs in precision and false alarms on the dev sets. The bias is then fixed and checked once on the held-out
sets with run.py (PII_O_BIAS). With --labels, λ raises only those labels' tags (tagger.decode), so
the look-alikes of the other labels keep the default operating point.
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

BIASES = np.round(np.arange(-2.0, 8.01, 0.25), 2)


def leak_share(row: dict, preds) -> float | None:
    """Share of the row's PII characters (span cores) left unmasked; None for a row without PII."""
    text = row["text"]
    pii = set()
    for s, e, _ in row["spans"]:
        cs, ce = score.core(text, s, e)
        pii.update(range(cs, ce))
    if not pii:
        return None
    masked = set()
    for s, e, *_ in preds:
        masked.update(range(s, e))
    return len(pii - masked) / len(pii)


def crc(losses: np.ndarray, alpha: float, b: float = 1.0) -> int | None:
    """Index of the smallest λ (columns, ascending) meeting the conformal risk control bound."""
    n = len(losses)
    envelope = np.maximum.accumulate(losses[:, ::-1], axis=1)[:, ::-1]  # non-increasing in λ
    risk = envelope.mean(0)
    ok = np.nonzero(n / (n + 1) * risk + b / (n + 1) <= alpha)[0]
    return int(ok[0]) if len(ok) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--big", help="one or more further taggers ('+'-joined directories) for a cascade (tagger.Cascade)")
    ap.add_argument("--threshold", type=float, default=0.1, help="the cascade's escalation threshold (below 0: every window)")
    ap.add_argument("--labels", help="the labels λ raises, joined by '+' (default: all, by lowering O)")
    ap.add_argument("--alphas", default="0.02,0.03,0.05,0.08")
    ap.add_argument("--calibration", default="gold-dev", help="the real-text dev golds to calibrate on, joined by '+' "
                                                              "(gold-dev, dev2)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    small = tagger.Tagger(args.model, batch_size=16)
    bigs = [tagger.Tagger(path, batch_size=16) for path in args.big.split("+")] if args.big else []
    model = tagger.Cascade(small, bigs, args.threshold) if bigs else small
    labels = tuple(args.labels.split("+")) if args.labels else None
    dev = data.dev_sets()
    cal_sets = args.calibration.split("+")
    if "dev2" in cal_sets:
        dev["dev2"] = data.gold(data.REAL / "gold_dev2.jsonl")
    scored = {name: model.scores([r["text"] for r in rows]) for name, rows in dev.items()}

    def preds_at(name, bias):
        rows = dev[name]
        out = tagger.decode(scored[name], len(rows), bias, [r["text"] for r in rows], labels)
        return [[(s, e, lab) for s, e, lab in p if not systems.public_value(r["text"][s:e], lab)] for r, p in zip(rows, out)]

    losses = []
    for bias in BIASES:
        losses.append([leak_share(r, p) for name in cal_sets for r, p in zip(dev[name], preds_at(name, bias))])
    keep = [k for k, x in enumerate(losses[0]) if x is not None]
    losses = np.array([[col[k] for k in keep] for col in losses]).T  # rows x biases
    print(f"calibration: {' + '.join(cal_sets)}, {losses.shape[0]} rows with personal data; mean leak share by bias:")
    print("  " + "  ".join(f"{b:+.2f}:{m:.3f}" for b, m in zip(BIASES[::4], losses.mean(0)[::4])))
    for alpha in (float(a) for a in args.alphas.split(",")):
        k = crc(losses, alpha)
        if k is None:
            print(f"\nα={alpha}: no bias in [{BIASES[0]}, {BIASES[-1]}] meets the bound")
            continue
        bias = float(BIASES[k])
        print(f"\nα={alpha}: bias {bias:+.2f} (calibration risk {losses[:, k].mean():.4f})")
        for name, rows in dev.items():
            ev_rows = [evsets.Row(r["text"], [(s, e, lab, lab) for s, e, lab in r["spans"]]) for r in rows]
            preds = [[(s, e, lab, lab) for s, e, lab in p] for p in preds_at(name, bias)]
            per_row, _, _, _ = score.evaluate(ev_rows, lambda text, _p=iter(preds): next(_p))
            print(f"  {name:18s} {score.fmt(score.summary(score.total_of(per_row)))}")


if __name__ == "__main__":
    main()
