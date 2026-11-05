"""Lists string literals of the PII code and tests that occur verbatim in the evaluation sets.

Held-out rows must not be in the code or the tests: a rule written for a row inflates the score
of that row's set. Exits 1 when a set with role "test" has hits beyond LEGACY, the rows found on
1 November 2026 (removed with the rule layer they were written for).

    python scripts/pii_eval/contamination.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets  # noqa: E402

ROOT = datasets.REPO_ROOT
SOURCES = [
    *ROOT.glob("crates/akana-core/src/pii/*.rs"),
    *ROOT.glob("crates/akana-core/tests/pii*.rs"),
    ROOT / "crates/akana-core/tests/downstream_snapshot.rs",
    *ROOT.glob("crates/akana-core/tests/data/downstream/*"),
    ROOT / "tests/test_pii.py",
]
LITERAL = re.compile(r'"((?:[^"\\\n]|\\.){20,})"')
BACKTICK = re.compile(r"`([^`\n]{20,})`")
# hits present when this check was written: number of literals per set
LEGACY = {"btx24-test": 19, "saturday": 19, "belgin-train": 19,
          "saturday-in-scheme": 19}  # the texts of "saturday", scored on fewer labels
# literals that occur in a set without being copied from it
ALLOWED = {
    "TR330006100519786457841326",  # the usual example IBAN of Turkish documentation
    # public figures of names_data.rs that real-gold texts also name
    "ahmet hamdi tanpınar",
    "mustafa kemal atatürk",
    "recep tayyip erdoğan",
    # common words of the analyzer snapshot tests/data/downstream/pii_view.tsv
    "değerlendirilmesinde",
    "değerlendirildiğinde",
}


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def literals() -> dict[str, str]:
    found = {}
    for path in SOURCES:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for rx in (LITERAL, BACKTICK):
            for m in rx.finditer(text):
                found.setdefault(m.group(1), path.name)
        if path.suffix in (".txt", ".tsv", ".jsonl"):
            for line in text.splitlines():
                if len(line) >= 20:
                    found.setdefault(line.split("\t")[0], path.name)
    return found


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    lits = {k: v for k, v in literals().items() if len(normalise(k)) >= 20 and k not in ALLOWED}
    failed = False
    for name, (loader, role, _) in datasets.SETS.items():
        joined = "\n".join(normalise(r.text) for r in loader())
        hits = [(lit, f) for lit, f in lits.items() if normalise(lit) in joined]
        new = len(hits) - LEGACY.get(name, 0)
        flag = " NEW" if role == "test" and new > 0 else ""
        failed |= bool(flag)
        print(f"{name:18s} ({role:5s}) {len(hits):3d} literal(s) found{flag}")
        for lit, f in sorted(hits, key=lambda x: -len(x[0]))[:5]:
            print(f"    [{f}] {lit[:100]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
