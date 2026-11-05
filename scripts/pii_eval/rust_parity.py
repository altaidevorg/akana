"""Checks that akana's Rust tagger (akana-core's pii::tagger, `akana pii-tag`) finds the same spans as
the Python tagger on the dev sets: text by text, and by the strict scorer's figures. Python runs on
the CPU in fp32, as the Rust tagger does.

    python scripts/pii_eval/rust_parity.py --akana target/release/akana.exe --model .cache/pii_models/electra-small-p3/best
    python scripts/pii_eval/rust_parity.py --akana ... --model <small> --big .cache/pii_models/berturk-p3/epoch3 \
        --points 0,2.75:PERSON+USERNAME+DATE
Each model directory needs model.onnx (scripts/pii_train/export_onnx.py) for Rust. --points lists
operating points (a bias, with the labels it raises after a colon); Python scores each set once and
decodes it at every point. Prints, per set and point, how many texts get identical spans, the first
differing texts, and both systems' scores.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "pii_train"))
import guard  # noqa: E402

guard.limit_threads(4)
import data  # noqa: E402
import datasets as evsets  # noqa: E402
import score  # noqa: E402
import systems  # noqa: E402
import tagger as tg  # noqa: E402


def dev_sets(names):
    sets = data.dev_sets()
    if (data.REAL / "gold_dev2.jsonl").is_file():
        sets["dev2"] = data.gold(data.REAL / "gold_dev2.jsonl")
    return {name: sets[name] for name in names}


def python_model(args):
    small = tg.Tagger(args.model, device="cpu", batch_size=16)
    if not args.big:
        return small
    return tg.Cascade(small, [tg.Tagger(path, device="cpu", batch_size=16) for path in args.big], -1.0)


def python_spans(scored, texts, bias, labels):
    preds = tg.decode(scored, len(texts), bias, texts, labels)
    return [[(s, e, lab) for s, e, lab in p if not systems.public_value(t[s:e], lab)] for t, p in zip(texts, preds)]


def rust_spans(args, texts, bias, labels):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "texts.jsonl"
        path.write_text("".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts), encoding="utf-8")
        cmd = [str(Path(args.akana).resolve()), "pii-tag", "--jsonl", str(path), "--model", args.model,
               "--threads", str(args.threads)]
        for big in args.big:
            cmd += ["--big", big]
        if bias:
            cmd += ["--bias", str(bias)]
        if labels:
            cmd += ["--labels", ",".join(labels)]
        out = subprocess.run(cmd, capture_output=True, check=True, encoding="utf-8").stdout
    spans = []
    for text, line in zip(texts, out.splitlines()):
        char_at, b = {}, 0  # byte offset -> character offset
        for i, ch in enumerate(text):
            char_at[b] = i
            b += len(ch.encode("utf-8"))
        char_at[b] = len(text)
        spans.append([(char_at[s], char_at[e], lab) for s, e, lab in json.loads(line)["spans"]])
    assert len(spans) == len(texts), "akana pii-tag returned a different number of lines"
    return spans


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--akana", required=True, help="the akana binary, built with --features pii-tagger")
    ap.add_argument("--model", required=True)
    ap.add_argument("--big", action="append", default=[])
    ap.add_argument("--points", default="0", help="operating points, comma-separated: BIAS or BIAS:LABEL+LABEL")
    ap.add_argument("--sets", default="gold-dev,yusuf-dev,belgin-validation,dev2")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()
    points = []
    for p in args.points.split(","):
        bias, _, labels = p.partition(":")
        points.append((float(bias), tuple(labels.split("+")) if labels else None))
    model = python_model(args)
    for name, rows in dev_sets(args.sets.split(",")).items():
        texts = [r["text"] for r in rows]
        scored = model.scores(texts)
        for bias, labels in points:
            point = f"bias {bias:+g}" + (f" on {'+'.join(labels)}" if labels else "")
            compare(args, f"{name}, {point}", rows, python_spans(scored, texts, bias, labels),
                    rust_spans(args, texts, bias, labels))


def compare(args, title, rows, py, rs):
    texts = [r["text"] for r in rows]
    same = [sorted(a) == sorted(b) for a, b in zip(py, rs)]
    print(f"{title}: identical spans on {sum(same)} of {len(texts)} texts")
    shown = 0
    for text, a, b, ok in zip(texts, py, rs, same):
        if not ok and shown < args.show:
            shown += 1
            print(f"   Python {[(text[s:e], lab) for s, e, lab in a]}\n   Rust   {[(text[s:e], lab) for s, e, lab in b]}")
    ev = [evsets.Row(r["text"], [(s, e, lab, lab) for s, e, lab in r["spans"]]) for r in rows]
    for who, preds in (("Python", py), ("Rust", rs)):
        sub = [[(s, e, lab, lab) for s, e, lab in p] for p in preds]
        per_row, _, _, _ = score.evaluate(ev, lambda text, _p=iter(sub): next(_p))
        print(f"   {who:6s} {score.fmt(score.summary(score.total_of(per_row)))}")


if __name__ == "__main__":
    main()
