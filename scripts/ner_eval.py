"""akana's NER against Turkish WikiNER's gold entity types (persons, places, organisations).

    python scripts/ner_eval.py [--split test|validation] [--wikiner DIR]

with akana importable, from the repository root, after `python scripts/fetch_morph_sources.py
--only turkish-wikiNER` (CC BY-SA 4.0; the test split is for evaluation only). WikiNER's tokens
are joined with spaces into the text; an entity matches when its type and its span of tokens are
the gold's. PER is WikiNER's PERSON, LOC its GPE and LOC, ORG its ORG; its other types (dates,
amounts, works ...) are left out, as are akana's DATE, MONEY and PERCENT. An akana entity covers
the tokens its characters overlap (a name that ends before its suffix covers the whole word).
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import akana

ROOT = Path(__file__).resolve().parent.parent
TYPES = {"PERSON": "PER", "GPE": "LOC", "LOC": "LOC", "ORG": "ORG"}


def gold_spans(tokens: list[str], tags: list[str]) -> set[tuple[str, int, int]]:
    spans, start, kind = set(), None, None
    for i, tag in enumerate(tags + ["O"]):
        inside = tag.startswith("I-") and kind == TYPES.get(tag[2:])
        if not inside and start is not None:
            spans.add((kind, start, i))
            start, kind = None, None
        if tag.startswith("B-") and tag[2:] in TYPES:
            start, kind = i, TYPES[tag[2:]]
    return spans


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", default="test", choices=["test", "validation"])
    ap.add_argument("--wikiner", default=str(ROOT / ".cache" / "morph_sources" / "turkish-wikiNER"))
    args = ap.parse_args()
    import pyarrow.parquet as pq

    base = Path(args.wikiner)
    names = [l.strip() for l in (base / "tags.lst").read_text(encoding="utf-8").splitlines() if l.strip()]
    path = base / "data" / f"{args.split}-00000-of-00001.parquet"
    if not path.exists():
        sys.exit(f"missing {path}: run scripts/fetch_morph_sources.py --only turkish-wikiNER")
    c = Counter()
    for row in pq.read_table(path).to_pylist():
        tokens, tags = row["tokens"], [names[t] for t in row["tags"]]
        text, spans = "", []
        for t in tokens:
            spans.append((len(text), len(text) + len(t)))
            text += t + " "
        gold = gold_spans(tokens, tags)
        pred = set()
        for e in akana.extract_entities(text):
            if e.label not in ("PER", "LOC", "ORG"):
                continue
            covered = [i for i, (a, b) in enumerate(spans) if a < e.end and e.start < b]
            if covered:
                pred.add((e.label, covered[0], covered[-1] + 1))
        for kind in ("PER", "LOC", "ORG"):
            g = {s for s in gold if s[0] == kind}
            p = {s for s in pred if s[0] == kind}
            c[(kind, "tp")] += len(g & p)
            c[(kind, "fp")] += len(p - g)
            c[(kind, "fn")] += len(g - p)
    print(f"akana {akana.__version__}, WikiNER {args.split}:")
    for kind in ("PER", "LOC", "ORG"):
        tp, fp, fn = c[(kind, "tp")], c[(kind, "fp")], c[(kind, "fn")]
        p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
        f = 2 * p * r / max(1e-9, p + r)
        print(f"  {kind}: precision {p:.3f}, recall {r:.3f}, F1 {f:.3f} ({tp} right, {fp} wrong, {fn} missed)")


if __name__ == "__main__":
    main()
