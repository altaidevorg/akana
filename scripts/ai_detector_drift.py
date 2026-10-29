"""The AI-text detector's outputs on a fixed text sample, to compare two builds (how far a
release moves the detector between retrainings).

    python scripts/ai_detector_drift.py score OUT.json     (with one build's akana importable)
    python scripts/ai_detector_drift.py compare A.json B.json

Run `score` once with each build (each in its own process, from the repository root, which has
`.cache/morph_sample/eval` from scripts/morph_corpus_sample.py). The sample: the first paragraph of
15 sentences and 120 words or more of each decision, up to 320 from each court subset, the same of
up to 300 RagTurk articles, and the drift canary's paragraphs four at a time
(crates/akana-core/tests/data/ai_detector_canary_texts.txt). `compare` reports how far the
probability moved, how many verdicts changed and the verdicts per group.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SENTENCE = re.compile(r"(?<=[.!?…])\s+")
COURT = ["yargitay", "danistay", "emsal", "aym_bb", "aym_norm"]


def paragraph(text: str) -> str | None:
    sentences = [s.strip() for s in SENTENCE.split(text.strip()) if s.strip()]
    for i in range(0, max(1, len(sentences) - 14), 15):
        p = " ".join(sentences[i:i + 15])
        if len(p.split()) >= 120:
            return p
    return None


def sample(wt: Path) -> list[tuple[str, str]]:
    eval_dir = wt / ".cache" / "morph_sample" / "eval"
    texts = []
    for subset, cap in [(s, 320) for s in COURT] + [("ragturk", 300)]:
        n = 0
        for doc in (eval_dir / f"{subset}.txt").read_text(encoding="utf-8").split("\n\x1e\n"):
            p = paragraph(doc.replace("\n", " "))
            if p and n < cap:
                texts.append((subset, p))
                n += 1
    canary = wt / "crates" / "akana-core" / "tests" / "data" / "ai_detector_canary_texts.txt"
    paragraphs = []
    for block in canary.read_text(encoding="utf-8").split("\n\n"):
        block = " ".join(l for l in block.splitlines() if l and not l.startswith("#"))
        if block:
            paragraphs.append(block)
    # The canary's paragraphs are short; it reads them in fours as well.
    for i in range(0, len(paragraphs) - 3, 4):
        texts.append(("canary", " ".join(paragraphs[i:i + 4])))
    return texts


def score(out: Path) -> None:
    import akana

    rows = []
    for group, text in sample(ROOT):
        d = akana.audit_ai_style(text).ai_detection or {}
        rows.append({"group": group, "verdict": d.get("verdict"), "p": d.get("ai_probability")})
    out.write_text(json.dumps({"version": akana.__version__, "rows": rows}), encoding="utf-8")
    print(akana.__version__, len(rows), "texts")


def compare(a: Path, b: Path) -> None:
    ra, rb = json.loads(a.read_text(encoding="utf-8")), json.loads(b.read_text(encoding="utf-8"))
    assert len(ra["rows"]) == len(rb["rows"])
    moves, verdicts, scored = [], [], 0
    for x, y in zip(ra["rows"], rb["rows"]):
        if x["p"] is not None and y["p"] is not None:
            scored += 1
            moves.append((abs(y["p"] - x["p"]), x["group"]))
        if x["verdict"] != y["verdict"]:
            verdicts.append((x["group"], x["verdict"], y["verdict"], x["p"], y["p"]))
    moves.sort(reverse=True)
    n = len(ra["rows"])
    print(f"{ra['version']} -> {rb['version']}: {n} texts, {scored} with a probability in both")
    print(f"  probability moved: max {moves[0][0]:.4f} ({moves[0][1]}), "
          f"95th percentile {moves[int(0.05 * len(moves))][0]:.4f}, "
          f"mean {sum(m for m, _ in moves) / len(moves):.4f}; moved at all: "
          f"{sum(m > 1e-9 for m, _ in moves)}")
    print(f"  moved by more than 0.01: {sum(m > 0.01 for m, _ in moves)}, more than 0.05: "
          f"{sum(m > 0.05 for m, _ in moves)}")
    print(f"  verdicts changed: {len(verdicts)}")
    for v in verdicts[:15]:
        print("   ", v)
    for g in sorted({x["group"] for x in ra["rows"]}):
        print(f"  {g}: verdicts a {sorted_counts(ra, g)} b {sorted_counts(rb, g)}")


def sorted_counts(r, g):
    from collections import Counter

    return dict(Counter(x["verdict"] for x in r["rows"] if x["group"] == g).most_common())


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score(Path(sys.argv[2]))
    else:
        compare(Path(sys.argv[2]), Path(sys.argv[3]))
