"""Writes crates/akana-core/src/grammar/frequent_locatives.txt: the lower-case word forms ending
in -ndA that the court sample writes at least 50 times (data/morphology/prior_train_types.tsv.gz).

    python scripts/frequent_locatives.py

The grammar rules take a genitive with a merged -da/-de for a misspelling of a one-letter
neighbour only when that neighbour is one of these words and a possessed locative of another
root (aranında for oranında, alınında for alanında), not when it is a rare word (seninde,
eninde; review 3 GRAM3-02). Looking neighbours up here before analysing them also bounds the
rule's cost.
"""

import gzip
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_COUNT = 50


def main():
    forms = set()
    with gzip.open(os.path.join(ROOT, "data", "morphology", "prior_train_types.tsv.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            word, _, count = line.rstrip("\n").partition("\t")
            if count.isdigit() and int(count) >= MIN_COUNT and word.islower() and word.endswith(("nda", "nde")):
                forms.add(word)
    out = os.path.join(ROOT, "crates", "akana-core", "src", "grammar", "frequent_locatives.txt")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Lower-case forms ending in -ndA seen at least {MIN_COUNT} times in the court sample; written by scripts/frequent_locatives.py.\n")
        for word in sorted(forms):
            f.write(word + "\n")
    print(f"{len(forms)} forms written to {out}")


if __name__ == "__main__":
    main()
