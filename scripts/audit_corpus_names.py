"""Which names of Zemberek's proper-from-corpus.dict are common words, the selection behind the
"Names only proper-from-corpus.dict lists" section of scripts/lexicon/akana-remove.dict (plan
step E2, review 2 MLEX2-03).

    python scripts/audit_corpus_names.py [--tsv OUT.tsv] [--person-names]

with akana importable (the extension built from this checkout), the corpus sample in
.cache/morph_sample (scripts/morph_corpus_sample.py) and the pinned sources in
.cache/morph_sources (scripts/fetch_morph_sources.py). It prints the dictionary lines to remove.

A name qualifies when only proper-from-corpus.dict lists it (not proper.dict, person-names.dict,
abbreviations.dict or akana's supplements), the analyzer gives it a common reading, and:
- seen at least 10 times in the prior's training types and the eval sample (case kept), it is
  written in lower case at least as often as capitalised, or ten times as often when TRmorph's
  person, place or proper-noun lexicon or Google's NNP list has the name (Emre, Toprak stay;
  Sona, Verdi go);
- seen fewer than 10 times, its common readings are a root with a suffix (Ruha, Satar), it is
  in no other name list, and it was seen in lower case at least as often as capitalised.

With --person-names it audits person-names.dict by the same rule instead, which lists words as
well as names (Sevgilim, Ruhsal; review 3 LEX3-03); its picks are reviewed by hand into the
"Words person-names.dict lists as names" section.
"""

import argparse
import collections
import gzip
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_full_lexicon as b  # noqa: E402

# A name whose only common reading is junk (yoo-n), kept by hand.
KEEP = {"yoon"}


def lower(w):
    return b.turkish_lower(w)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tsv", help="write every candidate with its counts here")
    parser.add_argument("--person-names", action="store_true", help="audit person-names.dict instead")
    args = parser.parse_args()
    audited = "person-names.dict" if args.person_names else "proper-from-corpus.dict"
    import akana

    # Common roots as the build's junk filter sees them, whose drops are not repeated here.
    common = {}
    for name in ("master-dictionary.dict", "non-tdk.dict"):
        for e in b.read_entries(os.path.join(b.LEXICON_DIR, "zemberek", name), name):
            if e.pos in ("Noun", "Adj") and e.secondary == "None":
                common.setdefault(e.root, set()).update(e.attrs)
    curated, corpus, raw = set(), {}, {}
    for name in b.DEFAULT_RESOURCES:
        path = os.path.join(b.LEXICON_DIR, "zemberek", name)
        for line in open(path, encoding="utf-8"):
            parsed = b.parse_line(line)
            if parsed is None:
                continue
            e = b.make_entry(parsed[0], parsed[1], name)
            if e is None or e.secondary != "ProperNoun":
                continue
            if name == audited:
                if name == "person-names.dict" or not b.junk_proper(e, common):
                    corpus.setdefault(e.root, e.word)
                    raw.setdefault(e.root, line.strip())
            elif name != "proper-from-corpus.dict":
                # A corpus name does not vouch for a person-names.dict word.
                curated.add(e.root)
    for name in b.SUPPLEMENTS:
        for e in b.read_entries(os.path.join(b.LEXICON_DIR, name), "akana:" + name):
            if e.secondary == "ProperNoun":
                curated.add(e.root)
    # Names other sections of akana-remove.dict already drop (inflected forms such as Grubu).
    # A section is a run of comment lines, named by its first.
    elsewhere, section, in_comment = set(), "", False
    for line in open(os.path.join(b.LEXICON_DIR, b.REMOVALS), encoding="utf-8"):
        if line.startswith("##"):
            if not in_comment:
                section = line.lstrip("# ")
            in_comment = True
            continue
        in_comment = False
        parsed = b.parse_line(line)
        if parsed and not section.startswith(("Names only proper-from-corpus.dict lists", "Words person-names.dict lists")):
            e = b.make_entry(parsed[0], parsed[1], "akana-remove")
            if e is not None:
                elsewhere.add(e.root)
    curated |= elsewhere

    src = os.path.join(ROOT, ".cache", "morph_sources")
    external = set()
    for f in ["person_names", "location_names", "proper_noun"]:
        for line in open(os.path.join(src, "TRmorph", "lexicon", f), encoding="utf-8"):
            if line[:1] not in "!\n " and ";" in line:
                external.add(b.normalize_circumflex(lower(line.split()[0].replace("'", ""))))
    nnp = os.path.join(src, "turkish-morphology", "src", "analyzer", "lexicon", "base", "noun_nnp.tsv")
    for line in open(nnp, encoding="utf-8"):
        cols = line.split("\t")
        if cols[0] == "NNP":
            external.add(b.normalize_circumflex(lower(cols[1].replace("'", ""))))

    case = collections.defaultdict(collections.Counter)
    sample = os.path.join(ROOT, ".cache", "morph_sample", "eval")
    sources = [gzip.open(os.path.join(ROOT, "data", "morphology", "prior_train_types.tsv.gz"), "rt", encoding="utf-8")]
    sources += [open(os.path.join(sample, f), encoding="utf-8") for f in sorted(os.listdir(sample)) if f.endswith(".types.tsv")]
    for source in sources:
        for line in source:
            w, _, c = line.rstrip("\n").partition("\t")
            stem = w.split("'")[0].split("’")[0]
            if not c or not stem:
                continue
            kind = "caps" if stem.isupper() else "upper" if stem[:1].isupper() else "lower"
            case[lower(stem)][kind] += int(c)

    morphology = akana.Morphology()
    rows, picked = [], []
    for root in sorted(corpus):
        if root in curated or root in KEEP:
            continue
        readings = [p for p in morphology.analyze(root) if p["secondary_pos"] != "Prop"]
        if not readings:
            continue
        suffixed = any(len([t for t in p["morphemes"] if t not in ("A3sg", "Nom")]) > 1 for p in readings)
        lo, up = case[root]["lower"], case[root]["upper"]
        listed = root in external
        if lo + up >= 10:
            drop = lo >= 10 * up if listed else lo >= up
        else:
            drop = suffixed and not listed and lo >= max(up, 1)
        rows.append((root, corpus[root], lo, up, int(listed), int(suffixed), int(drop)))
        if drop:
            picked.append(raw[root])
    if args.tsv:
        with open(args.tsv, "w", encoding="utf-8", newline="\n") as out:
            out.write("root\tword\tlower\tupper\tlisted\tsuffixed\tdrop\n")
            for r in rows:
                out.write("\t".join(map(str, r)) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    for line in sorted(picked, key=lower):
        print(line)


if __name__ == "__main__":
    main()
