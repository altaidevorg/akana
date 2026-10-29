"""The deasciifier on correct Turkish and on the same text typed without Turkish letters.

    python scripts/benchmark_deasciifier.py [--ragturk] [--json OUT]

with akana importable. Texts: the 1,500 committed clean court sentences
(crates/akana-core/tests/data/grammar_clean_court.txt, CC0) and, with --ragturk, the cached
RagTurk article sentences (.cache/rag_benchmarks; benchmark_rag_real_data.py fetches them).

- Correct text kept: the share of words of correct text that `deasciify` changes (it should
  change none) and of sentences with a change (review GN-10: 37 % of sentences were).
- Restored: `deasciify(asciify(text))` against the text, over the words `asciify` changed
  (how many get their Turkish letters back) and over all words.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import akana

ROOT = Path(__file__).resolve().parent.parent
COURT = ROOT / "crates" / "akana-core" / "tests" / "data" / "grammar_clean_court.txt"
WORD = re.compile(r"\S+")


def sentences_of(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def ragturk_sentences(limit: int = 3000) -> list[str]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from benchmark_rag_real_data import fetch_ragturk_dataset

    docs, _ = fetch_ragturk_dataset(max_articles=300, strict=True)
    out = []
    for d in docs:
        for s in re.split(r"(?<=[.!?])\s+", d["text"]):
            if len(s.split()) >= 4:
                out.append(s.strip())
    return out[:limit]


def measure(sentences: list[str]) -> dict:
    words = changed = sentences_changed = 0
    asciified = restored = all_words = all_right = 0
    examples = []
    for s in sentences:
        out = akana.deasciify(s)
        a, b = WORD.findall(s), WORD.findall(out)
        if len(a) != len(b):
            sentences_changed += 1
            changed += len(a)
            words += len(a)
        else:
            diff = [(x, y) for x, y in zip(a, b) if x != y]
            words += len(a)
            changed += len(diff)
            if diff:
                sentences_changed += 1
                if len(examples) < 30:
                    examples.extend(f"{x} -> {y}" for x, y in diff[:2])
        typed = akana.asciify(s)
        back = akana.deasciify(typed)
        a, t, r = WORD.findall(s), WORD.findall(typed), WORD.findall(back)
        if len(a) == len(t) == len(r):
            for x, y, z in zip(a, t, r):
                all_words += 1
                all_right += x == z
                if x != y:
                    asciified += 1
                    restored += x == z
    return {
        "sentences": len(sentences),
        "words": words,
        "correct_words_changed": changed / max(words, 1),
        "correct_sentences_changed": sentences_changed / max(len(sentences), 1),
        "asciified_words": asciified,
        "asciified_restored": restored / max(asciified, 1),
        "all_words_right": all_right / max(all_words, 1),
        "examples": examples[:30],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ragturk", action="store_true", help="also the cached RagTurk sentences")
    ap.add_argument("--json", help="write the figures here")
    args = ap.parse_args()
    sets = {"court": sentences_of(COURT)}
    if args.ragturk:
        sets["ragturk"] = ragturk_sentences()
    out = {}
    for name, sentences in sets.items():
        r = out[name] = measure(sentences)
        print(
            f"{name}: {r['sentences']} sentences, {r['words']} words; correct text changed: "
            f"{100 * r['correct_words_changed']:.2f} % of words, "
            f"{100 * r['correct_sentences_changed']:.1f} % of sentences; asciified words "
            f"restored {100 * r['asciified_restored']:.2f} % of {r['asciified_words']}, all words "
            f"right {100 * r['all_words_right']:.2f} %"
        )
        print("  e.g.", "; ".join(r["examples"][:12]))
    if args.json:
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
