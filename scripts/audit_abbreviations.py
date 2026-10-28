"""The apostrophe-suffixed forms of the lexicon's abbreviations that the corpus samples write and
the analyzer rejects, grouped by abbreviation: a pronunciation that gives the wrong harmony or
buffer letters shows as a whole paradigm rejected (OHAL'in), and a reading the lexicon lacks as
the forms of the expanded last word (SGK'nun: Kurumu'nun). Run it on every batch of
abbreviation entries (review 3 LEX3-06).

    python scripts/audit_abbreviations.py [--top N]

with akana importable (the extension built from this checkout), the court sample's type counts
in .cache/morph_sample/eval (scripts/morph_corpus_sample.py) and the prior's training types in
data/morphology/prior_train_types.tsv.gz.
"""

import argparse
import collections
import glob
import gzip
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEXICON = os.path.join(ROOT, "crates", "akana-core", "src", "morphology", "zemberek_lexicon.txt")
SAMPLE = os.path.join(ROOT, ".cache", "morph_sample", "eval")
TRAIN = os.path.join(ROOT, "data", "morphology", "prior_train_types.tsv.gz")


def turkish_lower(word: str) -> str:
    return word.replace("I", "ı").replace("İ", "i").lower()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--top", type=int, default=40, help="abbreviations to list")
    args = parser.parse_args()
    import akana

    readings: dict[str, list[str]] = collections.defaultdict(list)
    with open(LEXICON, encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if len(fields) > 3 and fields[3] == "Abbrv":
                kv = dict(t.split("=", 1) for t in fields[4:] if "=" in t)
                readings[fields[0]].append(kv.get("pr", fields[1]))
    paths = glob.glob(os.path.join(SAMPLE, "*.types.tsv"))
    if not paths:
        sys.exit(f"no type counts in {SAMPLE}: run scripts/morph_corpus_sample.py first")
    counts: collections.Counter = collections.Counter()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                word, n = line.rstrip("\n").split("\t")
                counts[word] += int(n)
    with gzip.open(TRAIN, "rt", encoding="utf-8") as f:
        for line in f:
            word, n = line.rstrip("\n").split("\t")
            counts[word] += int(n)

    morphology = akana.Morphology()
    rejected: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    accepted: collections.Counter = collections.Counter()
    for word, n in counts.items():
        if "'" not in word and "’" not in word:
            continue
        stem = re.split("['’]", word)[0]
        if not stem.isupper() or len(stem) < 2:
            continue
        key = turkish_lower(stem).replace("ı", "i")
        if key not in readings and turkish_lower(stem) not in readings:
            continue
        if morphology.analyze(word):
            accepted[stem] += n
        else:
            rejected[stem][word] += n
    total = sum(sum(c.values()) for c in rejected.values())
    print(f"rejected tokens: {total}, accepted: {sum(accepted.values())}")
    for stem, forms in sorted(rejected.items(), key=lambda kv: -sum(kv[1].values()))[: args.top]:
        key = turkish_lower(stem).replace("ı", "i")
        prs = readings.get(key, readings.get(turkish_lower(stem)))
        shown = ", ".join(f"{w} {n}" for w, n in forms.most_common(5))
        print(f"{stem} pr={prs} rejected {sum(forms.values())} accepted {accepted[stem]}: {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
