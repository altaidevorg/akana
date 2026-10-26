"""Writes the fixed corpus samples that the morphology sweep and frequency prior read.

Usage:
    python scripts/morph_corpus_sample.py --root C:/Users/Gigabyte/akana --out .cache/morph_sample

`--root` is a checkout that holds the court decisions (`data/turkish_court_decisions/data/*/`)
and the cached RagTurk articles (`.cache/rag_benchmarks/ragturk/*.json`); neither is in git.
The samples are deterministic (fixed seeds), so every phase of a change is measured on the
same text:

- `eval/`: 600 decisions from each of the five court subsets, drawn from 6 random row groups
  per subset (the 2026-10-20 review's sample), each capped at 40,000 characters, plus all
  RagTurk articles. For each corpus: `<corpus>.types.tsv` (letter tokens, `word<TAB>count`),
  `<corpus>.numbers.tsv` (tokens with digits) and `<corpus>.txt` (the documents, separated by
  a line holding only `\\x1e`), which the sweep times `analyze` on.
- `train/`: decisions that are not in `eval/` (other row groups where a subset has enough,
  otherwise other rows of the same groups), for estimating the analyzer's frequency prior;
  `<corpus>.types.tsv` and `<corpus>.numbers.tsv`, and with `--train-text` also `<corpus>.txt`,
  which the sparse encoder's document frequencies are estimated from (`examples/sparse_idf.rs`).
"""

import argparse
import collections
import glob
import json
import os
import random
import re

import pyarrow.parquet as pq

SUBSETS = ["yargitay", "danistay", "emsal", "aym_bb", "aym_norm"]
EVAL_DOCS = 600
TRAIN_DOCS = {"yargitay": 4000, "danistay": 3000, "emsal": 4000, "aym_bb": 2000, "aym_norm": 1500}
CAP = 40_000
# Letter tokens with an optional apostrophe suffix, as in the review's sweep.
LETTER = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")
# Numbers with their apostrophe suffix: 1923'te, 3,5, 15., %50, 1.000.000'dan.
NUMBER = re.compile(r"%?\d+(?:[.,]\d+)*\.?(?:['’][^\W\d_]+)?")


def row_groups(root, subset):
    files = sorted(glob.glob(f"{root}/data/turkish_court_decisions/data/{subset}/*.parquet"))
    return [(f, g) for f in files for g in range(pq.ParquetFile(f).num_row_groups)]


def read_texts(path, group):
    return pq.ParquetFile(path).read_row_group(group, columns=["text"]).column("text").to_pylist()


def sample_court(root, subset, rng):
    """Returns (eval_docs, train_docs) for one subset, disjoint by (file, group, row)."""
    groups = row_groups(root, subset)
    rng.shuffle(groups)
    eval_groups = groups[:6]
    per = EVAL_DOCS // len(eval_groups) + 1
    eval_docs, taken = [], set()
    for f, g in eval_groups:
        texts = read_texts(f, g)
        for i in rng.sample(range(len(texts)), min(per, len(texts))):
            if texts[i] and len(eval_docs) < EVAL_DOCS:
                eval_docs.append(texts[i][:CAP])
                taken.add((f, g, i))
    # Training rows: other groups first; subsets with few groups reuse the eval groups' other rows.
    want = TRAIN_DOCS[subset]
    train_groups = groups[6:] or eval_groups
    train_groups = train_groups[: max(1, min(len(train_groups), 8))]
    per = want // len(train_groups) + 1
    train_docs = []
    for f, g in train_groups:
        texts = read_texts(f, g)
        free = [i for i in range(len(texts)) if (f, g, i) not in taken and texts[i]]
        for i in rng.sample(free, min(per, len(free))):
            if len(train_docs) < want:
                train_docs.append(texts[i][:CAP])
    return eval_docs, train_docs


def ragturk(root):
    docs = []
    for f in sorted(glob.glob(f"{root}/.cache/rag_benchmarks/ragturk/*.json")):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        if isinstance(d, dict) and "article" in d:
            docs.append(d["article"]["content"])
    return docs


def write_counts(path, docs, pattern):
    counts = collections.Counter(tok for d in docs for tok in pattern.findall(d))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for word, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            f.write(f"{word}\t{n}\n")
    return sum(counts.values()), len(counts)


def write_corpus(out_dir, name, docs, with_text):
    os.makedirs(out_dir, exist_ok=True)
    tokens, types = write_counts(os.path.join(out_dir, f"{name}.types.tsv"), docs, LETTER)
    write_counts(os.path.join(out_dir, f"{name}.numbers.tsv"), docs, NUMBER)
    if with_text:
        with open(os.path.join(out_dir, f"{name}.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n\x1e\n".join(docs))
    print(f"{out_dir}/{name}: {len(docs)} docs, {tokens} letter tokens, {types} types", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, help="checkout with the court data and RagTurk cache")
    parser.add_argument("--out", default=".cache/morph_sample")
    parser.add_argument("--train-text", action="store_true", help="also write the training documents")
    args = parser.parse_args()

    for subset in SUBSETS:
        # One generator per subset, so adding a subset does not change the others' samples.
        rng = random.Random(f"morph-sample-v1-{subset}")
        eval_docs, train_docs = sample_court(args.root, subset, rng)
        write_corpus(os.path.join(args.out, "eval"), subset, eval_docs, with_text=True)
        write_corpus(os.path.join(args.out, "train"), subset, train_docs, with_text=args.train_text)
    write_corpus(os.path.join(args.out, "eval"), "ragturk", ragturk(args.root), with_text=True)


if __name__ == "__main__":
    main()
