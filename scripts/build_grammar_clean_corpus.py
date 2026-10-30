#!/usr/bin/env python3
"""
Builds the clean-text corpus the grammar checker's precision gate runs on
(`crates/akana-core/tests/data/grammar_clean_court.txt`), and optionally the local-only
measurement sets.

The committed corpus is correct, edited Turkish: sentences sampled deterministically from
Turkish court decisions (CC0). Sentences with anonymisation markers (initials, "...", "[BD]",
"XX"), personal data (anything the PII engine or NER flags, ID-like digit runs) or PDF
extraction damage (headings, fragments, glued or hyphen-split words) are dropped, and so are
sentences with a lower-case word the analyser does not know, which in this text is nearly
always a typo ("olmadğının") or a lost space. The local court set skips that last filter, so
the measurement there still sees unknown words. Court decisions also have real spelling
errors; the ones the grammar checker's default rules found in the sample, each confirmed by
reading it, are listed in KNOWN_ERRORS, and sentences with them are left out too.

    python scripts/build_grammar_clean_corpus.py               # rewrites the committed corpus
    python scripts/build_grammar_clean_corpus.py --local DIR   # also writes measurement sets

The corpora are read from `--data-root` (default: the repository root): the court decisions
from `data/turkish_court_decisions/data/<source>/*.parquet`, RagTurk from
`.cache/rag_benchmarks/ragturk/*.json` and GECTurk from `.cache_gecturk/`.

`--local DIR` writes, next to nothing that is committed:
- `court_local.txt`: more court sentences, disjoint from the committed ones;
- `ragturk.txt`: Wikipedia sentences from the RagTurk articles (CC BY-SA, never committed);
- `gecturk_test.tsv`, `gecturk_movie.tsv`: GECTurk rows as `labels<TAB>tokens`, the
  tokens joined by single spaces (licence: not redistributed here).

Needs pyarrow and the `akana` module (for the PII engine, NER and the analyser).
"""

import argparse
import glob
import json
import os
import random
import re

import pyarrow.parquet as pq

import akana

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SOURCES = ["aym_bb", "aym_norm", "danistay", "emsal", "yargitay"]
COMMITTED = os.path.join(
    ROOT, "crates", "akana-core", "tests", "data", "grammar_clean_court.txt"
)
PER_SOURCE_COMMITTED = 300
PER_SOURCE_LOCAL = 1200
DOCS_PER_SOURCE = 1500
SENTS_PER_DOC = 3
SEED = 20261021

UPPER = "A-ZÇĞİÖŞÜÂÎÛ"
LOWER = "a-zçğıöşüâîû"
SENT_SPLIT = re.compile(rf"(?<=[.!?])\s+(?=[{UPPER}\"(])")
FORBIDDEN = re.compile(
    r"\.\.|…|[\[\]{}*_@<>|\\#]|https?:|www\.|[XxХ]{2,}|\d{7,}|\d{3} \d{3} \d{2}"
)
INITIALS = re.compile(rf"\b[{UPPER}]\.\s?[{UPPER}]")
HYPHEN_BREAK = re.compile(rf"[{LOWER}]-\s+[{LOWER}]")
WORD = re.compile(rf"[{UPPER}{LOWER}]+")
FRAGMENT_END = re.compile(rf"(\d|\b[{UPPER}])\.$")
# Real errors in the court text (see the module docstring): "ya da", "hiçbir", "herhangi",
# "birçok", "birkaç", "birtakım", "istekte", "tarafça", "kalmayan", "olarak da", "1993'te".
KNOWN_ERRORS = re.compile(
    r"\b(yada|hiç bir|hiç biri|her hangi|bir çok|bir kaç|bir takım|istekde|tarafca|"
    r"tetkikde|çekden|uyuşmazlık ta|\w+mıyan|\w+miyen|\w+miyerek|olarak ta|kaydolunup ta|varki|isede|burda|"
    r"yukarda\w*)\b|\b\d*[345]'d[ae]",
    re.IGNORECASE,
)
HEADER = """\
# Correct, edited Turkish for the grammar checker's precision gate (tests/grammar_precision_gate.rs):
# one sentence per line, sampled from Turkish court decisions (CC0) by
# scripts/build_grammar_clean_corpus.py, which documents the filters. Lines starting with # are skipped.
"""


def paragraphs(text):
    """The text's lines, a line that does not end a sentence joined to the next one (PDF
    extraction breaks lines mid-sentence); heading-like lines stay on their own."""
    out, cur = [], ""
    for line in re.split(r"\n+", text or ""):
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        letters = [c for c in line if c.isalpha()]
        heading = letters and sum(c.isupper() for c in letters) > 0.5 * len(letters)
        if heading:
            if cur:
                out.append(cur)
            out.append(line)
            cur = ""
            continue
        cur = f"{cur} {line}" if cur else line
        if line[-1] in ".!?:;":
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


def sentences(text):
    for para in paragraphs(text):
        for s in SENT_SPLIT.split(para):
            yield s.strip()


def shape_ok(s):
    if not (40 <= len(s) <= 300) or len(s.split()) < 6:
        return False
    if not re.match(rf"[{UPPER}]", s) or not s.endswith("."):
        return False
    letters = [c for c in s if c.isalpha()]
    if sum(c.isupper() for c in letters) > 0.25 * len(letters):
        return False
    # fragments: "... yetkili kılan 27." (cut at "27. maddesi"), "2015/5612 K.", "TL. posta ..."
    if FRAGMENT_END.search(s) or s.split()[0].endswith("."):
        return False
    if "''" in s or s.count('"') % 2 or s.count("(") != s.count(")"):
        return False
    # form fields run together: "Esas No : 2015/358 Karar No : 2016/653 ..."
    if s.count(" : ") > 1:
        return False
    return not (FORBIDDEN.search(s) or INITIALS.search(s) or HYPHEN_BREAK.search(s))


class Filters:
    def __init__(self):
        self.pii = akana.TurkishPiiEngine()
        self.morph = akana.Morphology()
        self._known = {}
        self.dropped = {"shape": 0, "pii": 0, "person": 0, "glued": 0}

    def known(self, w):
        if w not in self._known:
            self._known[w] = bool(self.morph.analyze(w))
        return self._known[w]

    def glued(self, s):
        """A lower-case word the analyser does not know that splits into two words it does
        ("kadarsüreyle", "birinciparagraf"): text extraction lost a space."""
        for w in WORD.findall(s):
            if not w[0].islower() or len(w) < 6 or self.known(w):
                continue
            for i in range(2, len(w) - 1):
                if self.known(w[:i]) and self.known(w[i:]):
                    return True
        return False

    def all_known(self, s):
        """Every lower-case word is one the analyser knows."""
        return all(self.known(w) for w in WORD.findall(s) if len(w) > 1 and w[0].islower())

    def keep(self, s):
        if not shape_ok(s):
            self.dropped["shape"] += 1
            return False
        if self.pii.detect(s):
            self.dropped["pii"] += 1
            return False
        if any(e.label == "PER" for e in akana.extract_entities(s)):
            self.dropped["person"] += 1
            return False
        if self.glued(s):
            self.dropped["glued"] += 1
            return False
        return True


def court_pool(data_root, source, rng, filters):
    """Up to SENTS_PER_DOC kept sentences from each of DOCS_PER_SOURCE documents of
    `source`, documents and sentences chosen by `rng`."""
    path = sorted(
        glob.glob(
            os.path.join(data_root, "data", "turkish_court_decisions", "data", source, "*.parquet")
        )
    )[0]
    texts = pq.ParquetFile(path).read_row_group(0, columns=["text"]).column("text").to_pylist()
    docs = rng.sample(range(len(texts)), min(DOCS_PER_SOURCE, len(texts)))
    pool, seen = [], set()
    for d in docs:
        cands = [s for s in dict.fromkeys(sentences(texts[d])) if s not in seen]
        rng.shuffle(cands)
        taken = 0
        for s in cands:
            if taken == SENTS_PER_DOC:
                break
            if filters.keep(s):
                pool.append(s)
                seen.add(s)
                taken += 1
    rng.shuffle(pool)
    return pool


def ragturk(data_root, rng, n):
    out = []
    for f in sorted(glob.glob(os.path.join(data_root, ".cache", "rag_benchmarks", "ragturk", "*.json"))):
        with open(f, encoding="utf-8") as fh:
            j = json.load(fh)
        if not isinstance(j, dict):
            continue
        content = re.sub(r"==+[^=]+==+", "\n", j["article"]["content"])
        cands = [
            s
            for s in sentences(content)
            if 30 <= len(s) <= 300 and re.match(rf"[{UPPER}0-9\"]", s)
        ]
        rng.shuffle(cands)
        out.extend(cands[:20])
    rng.shuffle(out)
    return out[:n]


def gecturk(path, out):
    rows = pq.read_table(path).to_pylist()
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            toks = [t.replace("\t", " ").replace(" ", "_") for t in r["tokens"]]
            fh.write(",".join(map(str, r["labels"])) + "\t" + " ".join(toks) + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--local", help="directory for the local-only measurement sets")
    ap.add_argument("--data-root", default=ROOT, help="where the source corpora are")
    args = ap.parse_args()

    rng = random.Random(SEED)
    filters = Filters()
    committed, local = [], []
    for source in SOURCES:
        pool = court_pool(args.data_root, source, rng, filters)
        mine, rest = [], []
        for s in pool:
            if (
                len(mine) < PER_SOURCE_COMMITTED
                and filters.all_known(s)
                and not KNOWN_ERRORS.search(s)
            ):
                mine.append(s)
            else:
                rest.append(s)
        committed.extend(mine)
        local.extend(rest[:PER_SOURCE_LOCAL])
        print(f"{source}: {len(pool)} sentences kept")
    rng.shuffle(committed)
    print("dropped:", filters.dropped)

    with open(COMMITTED, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(HEADER + "\n".join(committed) + "\n")
    print(f"wrote {len(committed)} sentences to {COMMITTED}")

    if args.local:
        os.makedirs(args.local, exist_ok=True)
        rng.shuffle(local)
        with open(os.path.join(args.local, "court_local.txt"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(local) + "\n")
        rag = ragturk(args.data_root, random.Random(SEED), 5000)
        with open(os.path.join(args.local, "ragturk.txt"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(rag) + "\n")
        cache = os.path.join(args.data_root, ".cache_gecturk")
        n_test = gecturk(
            os.path.join(cache, "test-00000-of-00001-82bf0acf5b396a77.parquet"),
            os.path.join(args.local, "gecturk_test.tsv"),
        )
        n_movie = gecturk(
            os.path.join(cache, "movie_reviews-00000-of-00001-b0322cba85108ee9.parquet"),
            os.path.join(args.local, "gecturk_movie.tsv"),
        )
        print(f"local: {len(local)} court, {len(rag)} ragturk, {n_test} + {n_movie} gecturk rows")


if __name__ == "__main__":
    main()
