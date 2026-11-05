"""Scores every candidate system on the held-out sets, one after another (one model in memory at a
time), and writes a summary table.

    python scripts/pii_eval/run_all.py --out .cache/pii_eval_final --akana-pkg <dir> \
        --system "akana" --system "tagger:model=.cache/pii_models/electra-small-v3/best" \
        --system "hybrid:model=.cache/pii_models/electra-small-v3/best,bias=1.0"

A system is `name[:key=value,...]` with keys model, big (cascade; several taggers joined by "+"),
threshold, bias and labels (the labels the bias raises, joined by "+"); each runs in its own process with run.py (the training
environment's Python for the learned ones). Writes <out>/<label>.json and .log, and
<out>/summary.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRAIN_PY = ROOT / ".cache" / "venv-train" / "Scripts" / "python.exe"
SETS = "real-gold,yusuf-test,belgin-test,btx24-test,saturday,saturday-in-scheme,wikiner-persons"
LEARNED = {"tagger", "hybrid", "hybrid+names", "ytu"}


def parse(spec: str) -> tuple[str, dict]:
    name, _, rest = spec.partition(":")
    opts = dict(kv.split("=", 1) for kv in rest.split(",") if kv)
    return name, opts


def run_name(path: str) -> str:
    """A model directory by its run: `berturk-v7/epoch3` -> berturk-v7; a soup is its own run."""
    p = Path(path)
    return p.parent.name if p.name == "best" or p.name.startswith("epoch") else p.name


def label_of(name: str, opts: dict) -> str:
    parts = [name] + [f"{k}-{'+'.join(map(run_name, v.split('+'))) if k in ('model', 'big') else v}"
                      for k, v in sorted(opts.items())]
    return "_".join(parts).replace("+", "plus").replace("/", "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--system", action="append", required=True)
    ap.add_argument("--sets", default=SETS)
    ap.add_argument("--akana-pkg")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in args.system:
        name, opts = parse(spec)
        label = label_of(name, opts)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        if args.akana_pkg:
            env["AKANA_PKG"] = args.akana_pkg
        for key, var in (("model", "PII_MODEL"), ("big", "PII_BIG"), ("threshold", "PII_THRESHOLD"), ("bias", "PII_O_BIAS"),
                         ("labels", "PII_BIAS_LABELS")):
            if key in opts:
                env[var] = opts[key]
            else:
                env.pop(var, None)
        py = str(TRAIN_PY) if name in LEARNED else sys.executable
        cmd = [py, str(HERE / "run.py"), "--system", name, "--sets", args.sets, "--bootstrap", str(args.bootstrap),
               "--json", str(out / f"{label}.json"), "--errors", str(out / f"{label}_errors")]
        if args.akana_pkg:
            cmd += ["--pkg", args.akana_pkg]
        print(f"--- {label}", flush=True)
        with open(out / f"{label}.log", "w", encoding="utf-8") as log:
            code = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        if code:
            print(f"    failed ({code}), see {out / f'{label}.log'}", flush=True)
            continue
        result = json.load(open(out / f"{label}.json", encoding="utf-8"))
        rows.append((label, result))
    lines = ["| system | set | exact P | exact R | exact F1 | full-cover R | leak | PII-free flagged |", "|---" * 8 + "|"]

    def f(x):
        return "-" if x is None else f"{x:.3f}"

    for label, result in rows:
        for name, entry in result["sets"].items():
            a = entry["all"]
            cells = [a.get(k) for k in ("exact_p", "exact_r", "exact_f1", "full_cover_r", "char_leak", "neg_flagged")]
            lines.append(f"| {label} | {name} | " + " | ".join(map(f, cells)) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
