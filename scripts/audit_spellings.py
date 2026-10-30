"""Which entries of Zemberek's non-tdk.dict are misspellings of a TDK word, the selection behind
scripts/lexicon/akana-spelling.dict (review 3 LEX3-01, MORPH3-04).

    python scripts/audit_spellings.py [--tsv OUT.tsv] > candidates.dict

with akana importable (the extension built from this checkout) and the corpus sample in
.cache/morph_sample (scripts/morph_corpus_sample.py). It prints dictionary lines for review: it
only proposes, and each line needs checking by hand against TDK's dictionary, since a valid word
can be one edit from another (yuan, blog, orda). Entries with a Zemberek `Ref:`, which the build
honours, and words akana-spelling.dict already lists are skipped.

non-tdk.dict lists words TDK's dictionary lacks: most are valid technical and modern words
(aerosol, pilates), some are misspellings (şöför, orjinal, sarmısak). An entry is proposed as a
nonstandard spelling of a master-dictionary word with the same part of speech when the master
word is one edit away (a confusable letter, ö/o, ü/u, ı/i, ş/s, ç/c, ğ/g, j/c,
  k/g, b/p, d/t, or a vowel added or left out) and usage prefers it: in the prior's training
  types and the eval sample, its words are seen at least 10 times and at least four times as
  often as the variant's.
"""

import argparse
import collections
import gzip
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_full_lexicon as b  # noqa: E402

CONFUSABLE = [("ö", "o"), ("ü", "u"), ("ı", "i"), ("ş", "s"), ("ç", "c"), ("ğ", "g"), ("j", "c"),
              ("k", "g"), ("b", "p"), ("d", "t")]
VOWELS = "aeıioöuü"
MIN_MASTER = 10
RATIO = 4


def neighbours(word):
    """Words one confusable substitution, or one vowel added or left out, away."""
    out = set()
    for i, c in enumerate(word):
        for a, z in CONFUSABLE:
            for x, y in ((a, z), (z, a)):
                if c == x:
                    out.add(word[:i] + y + word[i + 1:])
        if c in VOWELS:
            out.add(word[:i] + word[i + 1:])
    for i in range(len(word) + 1):
        for v in VOWELS:
            out.add(word[:i] + v + word[i:])
    out.discard(word)
    return out


def counts_by_root(akana):
    """Tokens of the training types and the eval sample, by the root of their best reading."""
    types = collections.Counter()
    with gzip.open(os.path.join(ROOT, "data", "morphology", "prior_train_types.tsv.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            w, _, c = line.rstrip("\n").partition("\t")
            if c.isdigit():
                types[w] += int(c)
    eval_dir = os.path.join(ROOT, ".cache", "morph_sample", "eval")
    for name in os.listdir(eval_dir):
        if name.endswith(".types.tsv"):
            for line in open(os.path.join(eval_dir, name), encoding="utf-8"):
                w, _, c = line.rstrip("\n").partition("\t")
                if c.strip().isdigit():
                    types[w] += int(c)
    m = akana.Morphology()
    by_root = collections.Counter()
    for w, c in types.items():
        parses = m.analyze(w)
        if parses:
            by_root[(parses[0]["root"], parses[0]["primary_pos"])] += c
    return by_root


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tsv", help="write every candidate pair with its counts here")
    args = parser.parse_args()
    import akana

    master = collections.defaultdict(dict)  # POS -> root -> lemma of master-dictionary common words
    for e in b.read_entries(os.path.join(b.LEXICON_DIR, "zemberek", "master-dictionary.dict"), "master-dictionary.dict"):
        if e.secondary in ("None", "Time"):
            master[e.pos].setdefault(e.root, e.lemma)
    listed = set()
    for line in open(os.path.join(b.LEXICON_DIR, b.SPELLINGS), encoding="utf-8"):
        parsed = b.parse_line(line)
        if parsed is not None:
            listed.add(b.turkish_lower(parsed[0]))
    by_root = counts_by_root(akana)
    rows, lines = [], []
    for raw in open(os.path.join(b.LEXICON_DIR, "zemberek", "non-tdk.dict"), encoding="utf-8"):
        parsed = b.parse_line(raw)
        if parsed is None:
            continue
        word, meta = parsed
        e = b.make_entry(word, meta, "non-tdk.dict")
        if e is None or e.secondary not in ("None", "Time") or "Ref" in meta or b.turkish_lower(word) in listed:
            continue
        v = by_root[(e.root, e.pos)]
        best = None
        for n in neighbours(e.root):
            if n in master[e.pos]:
                c = by_root[(n, e.pos)]
                if best is None or c > best[1]:
                    best = (n, c)
        if best is None:
            continue
        rows.append((e.word, e.pos, best[0], v, best[1]))
        if best[1] < MIN_MASTER or best[1] < RATIO * v:
            continue
        # The master word's lemma: a verb's infinitive (geciştirmek: geçiştirmek).
        lemma = master[e.pos][best[0]]
        pos = f"P:{e.pos}; " if e.pos != "Noun" else ""
        lines.append(f"{e.word} [{pos}A:Nonstandard; Ref:{lemma}]  # one edit: {v} tokens against {best[1]}")
    if args.tsv:
        with open(args.tsv, "w", encoding="utf-8") as f:
            f.write("variant\tpos\tmaster\tvariant_tokens\tmaster_tokens\n")
            for r in sorted(rows, key=lambda r: -r[4]):
                f.write("\t".join(map(str, r)) + "\n")
    for line in lines:
        print(line)
    print(f"# {len(lines)} proposed", file=sys.stderr)


if __name__ == "__main__":
    main()
