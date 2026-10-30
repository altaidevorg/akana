"""The general-text word counts the analyzer's frequency prior mixes with the court sample:
data/morphology/prior_general_types.tsv.gz (plan step C3, decision 3: one domain-mixed prior).

    python scripts/prior_general_types.py

with the treebanks in .cache/morph_sources/ud (scripts/fetch_morph_sources.py).

The text is the sentences (`# text =`) of the train and dev splits of the Universal
Dependencies Turkish treebanks Atis, BOUN, FrameNet, Kenet, Penn and Tourism: news, fiction,
reviews, grammar examples and flight requests, about half a million words. IMST (CC BY-NC-SA)
and every test split are left out. Words are cut from the text as scripts/morph_corpus_sample.py
cuts the court decisions (letters, with an apostrophe suffix), so the two samples count alike;
examples/morph_prior.rs weighs this one up to `GENERAL_SHARE` of the mixed counts.

RagTurk, which the plan also named, is left out: its articles are the retrieval benchmark's
documents and the source of the RagTurk gold sentences, which would then measure the prior on its
own training text. So is GECTurk, whose test split is the grammar gate's evaluation set.

The counts derive from the treebanks' sentences (CC BY-SA 4.0): the file, and the prior estimated
with it, are DERIVED FROM CC BY-SA DATA (THIRD_PARTY_NOTICES).
"""

import collections
import gzip
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UD = os.path.join(ROOT, ".cache", "morph_sources", "ud")
OUT = os.path.join(ROOT, "data", "morphology", "prior_general_types.tsv.gz")
TREEBANKS = ["Atis", "BOUN", "FrameNet", "Kenet", "Penn", "Tourism"]
SPLITS = ["train", "dev"]
# As in scripts/morph_corpus_sample.py.
LETTER = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")


def main():
    counts = collections.Counter()
    sentences = 0
    for bank in TREEBANKS:
        for split in SPLITS:
            path = os.path.join(UD, f"UD_Turkish-{bank}", f"tr_{bank.lower()}-ud-{split}.conllu")
            for line in open(path, encoding="utf-8"):
                if line.startswith("# text = "):
                    sentences += 1
                    counts.update(LETTER.findall(line[len("# text = ") :]))
    # A fixed gzip header (no file name, no time), so the file reproduces byte for byte.
    with open(OUT, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
        gz.write(
            b"# DERIVED FROM CC BY-SA DATA (Universal Dependencies Turkish treebanks, r2.18; see"
            b" THIRD_PARTY_NOTICES). Written by scripts/prior_general_types.py.\n"
        )
        for word, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            gz.write(f"{word}\t{n}\n".encode("utf-8"))
    print(f"{sentences} sentences, {sum(counts.values())} tokens, {len(counts)} types; wrote {OUT}")


if __name__ == "__main__":
    main()
