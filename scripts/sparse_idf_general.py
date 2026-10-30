"""Writes the general-domain sample the sparse encoder's function lemmas are estimated from.

Usage:
    python scripts/sparse_idf_general.py --out .cache/sparse_idf/general

The sample is the Turkish split of Global-MMLU (CohereLabs/Global-MMLU, Apache-2.0; test and
dev, 14,327 questions over 57 subjects), pinned to one revision. Each question and its four
options make one document; documents are separated by a line holding only `\\x1e`, as in
`morph_corpus_sample.py`. `examples/sparse_idf.rs` reads it next to the court sample: a lemma
counts as a function lemma only when it is frequent in both, so that words frequent in court
decisions alone (tarih, mahkeme, karar) keep their weight.
"""

import argparse
import os

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "CohereLabs/Global-MMLU"
REVISION = "0e619dbeb34206cd48705a1a0ea7fb21cae09993"
SPLITS = ["test", "dev"]
FIELDS = ["question", "option_a", "option_b", "option_c", "option_d"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=".cache/sparse_idf/general")
    args = parser.parse_args()

    docs = []
    for split in SPLITS:
        path = hf_hub_download(
            REPO, f"tr/{split}-00000-of-00001.parquet", repo_type="dataset", revision=REVISION
        )
        for row in pq.read_table(path, columns=FIELDS).to_pylist():
            docs.append(" ".join(str(row[f] or "") for f in FIELDS))
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "global_mmlu_tr.txt")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n\x1e\n".join(docs))
    print(f"{out}: {len(docs)} documents")


if __name__ == "__main__":
    main()
