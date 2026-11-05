"""CPU latency and throughput of a trained tagger, the way a laptop would run it: no GPU, a batch of
one (a prompt at a time) and a batch of 16 (bulk text).

    python scripts/pii_train/bench_cpu.py .cache/pii_models/electra-small-v1/best
    python scripts/pii_train/bench_cpu.py <model> --threads 1 --prompt-chars 2000 --quantize --dev
    python scripts/pii_train/bench_cpu.py <model> --big <bigger model> --threshold -1 --prompt-chars 2000

--prompt-chars joins consecutive real units into prompts of about that length (the release gate is
a 2 KB prompt on one core); --quantize runs the linear layers in int8 (PyTorch dynamic quantization)
and --dev scores that same model on the dev sets, to show what the speed costs. --big makes a
cascade (tagger.Cascade, averaging): windows whose uncertainty exceeds --threshold are read again by
the bigger model (or by each of several, joined by '+'), and a threshold below 0 sends every window
to all of them.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guard  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--prompt-chars", type=int, default=0, help="join units into prompts of about this many characters")
ap.add_argument("--quantize", action="store_true", help="int8 dynamic quantization of the linear layers")
ap.add_argument("--dev", action="store_true", help="also score the benchmarked model on the dev sets")
ap.add_argument("--big", help="a second tagger with the same vocabulary, for a cascade")
ap.add_argument("--threshold", type=float, default=0.1, help="the cascade's escalation threshold")
ap.add_argument("--n", type=int, default=300)
args = ap.parse_args()

guard.limit_threads(args.threads)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tagger  # noqa: E402

ROOT = HERE.parents[1]


def prompts(units: list[str], size: int, n: int) -> list[str]:
    if not size:
        return units[:n]
    out, cur = [], ""
    for u in units:
        cur = f"{cur}\n{u}" if cur else u
        if len(cur) >= size:
            out.append(cur)
            cur = ""
        if len(out) == n:
            break
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    guard.wait()
    units = [json.loads(l)["text"] for l in open(ROOT / ".cache" / "pii_data" / "real" / "train.jsonl", encoding="utf-8")]
    texts = prompts(units, args.prompt_chars, args.n)
    t = tagger.Tagger(args.model, device="cpu", batch_size=1)
    if args.quantize:
        import torch
        t.model = torch.ao.quantization.quantize_dynamic(t.model, {torch.nn.Linear}, dtype=torch.qint8)
    big = [tagger.Tagger(path, device="cpu", batch_size=1) for path in args.big.split("+")] if args.big else None
    model = tagger.Cascade(t, big, args.threshold) if big else t
    model.predict(texts[:5])
    if big:
        model.windows_seen = model.windows_escalated = 0
    lat = []
    for x in texts:
        a = time.perf_counter()
        model.predict([x])
        lat.append(time.perf_counter() - a)
    chars = sum(map(len, texts))
    for m in [t] + (big or []):
        m.batch_size = 16
    a = time.perf_counter()
    model.predict(texts)
    bulk = time.perf_counter() - a
    lat.sort()
    print(f"{args.threads} thread(s){', int8' if args.quantize else ''}: {len(texts)} texts, mean {chars / len(texts):.0f} chars")
    if big:
        print(f"cascade at threshold {args.threshold}: {model.windows_escalated / model.windows_seen:.1%} of windows escalated")
    print(f"one at a time: median {1000 * statistics.median(lat):.1f} ms, p95 {1000 * lat[int(0.95 * len(lat))]:.1f} ms, "
          f"{chars / sum(lat) / 1000:.1f} KB/s")
    print(f"batches of 16: {chars / bulk / 1000:.1f} KB/s")
    if args.dev:
        import data
        sys.path.insert(0, str(HERE.parent / "pii_eval"))
        import datasets as evsets
        import score
        import systems
        for name, rows in data.dev_sets().items():
            preds = model.predict([r["text"] for r in rows])
            preds = [[(s, e, lab, lab) for s, e, lab in p if not systems.public_value(r["text"][s:e], lab)] for r, p in zip(rows, preds)]
            ev_rows = [evsets.Row(r["text"], [(s, e, lab, lab) for s, e, lab in r["spans"]]) for r in rows]
            per_row, _, _, _ = score.evaluate(ev_rows, lambda text, _p=iter(preds): next(_p))
            print(f"  {name:18s} {score.fmt(score.summary(score.total_of(per_row)))}")


if __name__ == "__main__":
    main()
