"""How often each word form is each part of speech in the Universal Dependencies Turkish
treebanks, for the words the analyzer reads as more than one: data/morphology/pos_twins.tsv,
which examples/morph_prior.rs splits the readings of its training words by (plan step C2,
review 2 RANK2-02).

    python scripts/pos_twins.py

with akana importable (the extension built from this checkout) and the treebanks in
.cache/morph_sources/ud (scripts/fetch_morph_sources.py).

EM over a list of word types cannot tell a word's POS twins apart where they are written alike:
it gives ilk, tek and yeni to the adverb, devam and ileri to the interjection, insan and kamu
to the adjective, whose one shape explains every bare token; and it leaves readings that are in
every word that has one (git-tik past 1pl and participle) where its start put them. The
treebanks were tagged by hand. For each word form (in lower case) of the train and dev splits
of Atis, BOUN, FrameNet, Kenet, Penn and Tourism, the table counts its UPOS in fourteen classes:
NOUN, ADJ, ADV, a VERB or AUX by its VerbForm (Verb finite, VerbPart participle, VerbNoun verbal
noun, VerbAdv converb), PRON, DET, NUM, CCONJ and SCONJ (Conj), ADP (Postp), INTJ and PART. A
word written as a multiword token (yılındayız: yılında + yız) counts with its first part's tag.
It keeps the forms whose readings fall in two of the analyzer's classes or more (`prior_class`
in analyzer.rs, which `reading_class` here mirrors). IMST (CC BY-NC-SA) is used for evaluation
only, and no test split is read.

The counts derive from the treebanks' annotations (CC BY-SA 4.0), and so do the table and the
prior estimated with it: both are DERIVED FROM CC BY-SA DATA (THIRD_PARTY_NOTICES).
"""

import collections
import os
import sys

import akana

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UD = os.path.join(ROOT, ".cache", "morph_sources", "ud")
OUT = os.path.join(ROOT, "data", "morphology", "pos_twins.tsv")
TREEBANKS = ["Atis", "BOUN", "FrameNet", "Kenet", "Penn", "Tourism"]
SPLITS = ["train", "dev"]
# UPOS -> the table's class. Names (PROPN) are left out: the table is read for words written in
# lower case, and the case of a word decides its name readings (Yeni Şafak is no evidence for
# yeni the noun).
CLASSES = {
    "NOUN": "Noun",
    "ADJ": "Adj",
    "ADV": "Adv",
    "PRON": "Pron",
    "DET": "Det",
    "NUM": "Num",
    "CCONJ": "Conj",
    "SCONJ": "Conj",
    "ADP": "Postp",
    "INTJ": "Interj",
    "PART": "Part",
}
# A verb's class by its VerbForm; none (BOUN) or Fin is a finite verb.
VERB_FORMS = {"Part": "VerbPart", "Vnoun": "VerbNoun", "Conv": "VerbAdv"}
READING_CLASSES = {
    "Noun": "Noun",
    "Adj": "Adj",
    "Adv": "Adv",
    "Verb": "Verb",
    "Pron": "Pron",
    "Det": "Det",
    "Num": "Num",
    "Conj": "Conj",
    "Postp": "Postp",
    "Interj": "Interj",
    "Q": "Part",
}


def turkish_lower(word):
    return word.replace("I", "ı").replace("İ", "i").lower()


def tag_class(upos, feats):
    if upos in ("VERB", "AUX"):
        form = dict(f.split("=", 1) for f in feats.split("|") if "=" in f).get("VerbForm")
        return VERB_FORMS.get(form, "Verb")
    return CLASSES.get(upos)


# analyzer.rs TIME_ADVERBS: time words whose time noun keeps its class.
TIME_ADVERBS = {"bugün", "daha", "erken", "evvel", "önce", "sonra", "şimdi"}


def reading_class(p):
    if (
        p["secondary_pos"] == "Time"
        and p["lemma"] not in TIME_ADVERBS
        and all(t in ("A3sg", "Pnon", "Nom") for t in p["morphemes"][1:])
    ):
        return "Adv"
    if p["primary_pos"] == "Verb":
        return {"Noun": "VerbNoun", "Adj": "VerbAdj", "Adv": "VerbAdv"}.get(p["pos"], "Verb")
    return READING_CLASSES.get(p["pos"])


def main():
    counts = collections.defaultdict(collections.Counter)
    for bank in TREEBANKS:
        for split in SPLITS:
            path = os.path.join(UD, f"UD_Turkish-{bank}", f"tr_{bank.lower()}-ud-{split}.conllu")
            if not os.path.exists(path):
                continue
            # A multiword token's form and the ids of its parts: it counts once, with the tag
            # of its first part.
            word, parts = None, range(0)
            for line in open(path, encoding="utf-8"):
                if line.startswith("#"):
                    continue
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 6:
                    word, parts = None, range(0)
                    continue
                if "-" in cols[0]:
                    start, end = (int(x) for x in cols[0].split("-"))
                    word, parts = cols[1], range(start, end + 1)
                    continue
                if "." in cols[0]:
                    continue
                index = int(cols[0])
                form = cols[1]
                if index in parts:
                    if index != parts.start:
                        continue
                    form = word
                cls = tag_class(cols[3], cols[5])
                if cls:
                    counts[turkish_lower(form)][cls] += 1
    morphology = akana.Morphology()
    rows = []
    for form in sorted(counts):
        classes = {reading_class(p) for p in morphology.analyze(form)} - {None}
        if len(classes) < 2:
            continue
        seen = counts[form]
        rows.append(f"{form}\t" + " ".join(f"{c}:{n}" for c, n in sorted(seen.items())))
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("# DERIVED FROM CC BY-SA DATA (Universal Dependencies Turkish treebanks, r2.18; see\n")
        f.write("# THIRD_PARTY_NOTICES). Written by scripts/pos_twins.py: a word form the analyzer reads\n")
        f.write("# in two classes or more, and how often the treebanks' train and dev splits tag it as each.\n")
        for row in rows:
            f.write(row + "\n")
    print(f"{len(counts)} forms, {len(rows)} read in two classes or more; wrote {OUT}")


if __name__ == "__main__":
    main()
