"""The analyzer and disambiguator on held-out UD Turkish data, by treebank and genre.

Held out means never used to train anything akana ships: the test splits of the six treebanks
whose train and dev sentences feed the prior (`prior_general_types.py`, `pos_twins.py`), and
all of IMST, which is CC BY-NC-SA and evaluation only. BOUN's test split is broken down by its
genres (biographical, essays, instructional, news, popular culture).

For each word (a multiword token counts once, with its first part's tag and lemma):
- coverage: the analyzer has a reading (punctuation and symbols left out);
- class: the disambiguated reading's class matches UD's UPOS (a participle or verbal noun
  counts as a verb or as its derived class; names, punctuation, symbols and X left out);
- twins: the same, on the words whose readings fall in two classes or more, one of them UD's
  (the readings the ranking has to choose between);
- lemma: the disambiguated reading's lemma or root equals UD's lemma, lower-cased and without
  circumflexes (UD writes a verb's stem, akana its infinitive and stem; some derivation
  conventions differ, so this is agreement, not accuracy).

    # with akana importable, from the repository root:
    python scripts/ud_eval.py [--ud .cache/morph_sources/ud] [--markdown]
"""

import argparse
import collections
import os
import sys

import akana

sys.stdout.reconfigure(encoding="utf-8")

TESTED = ["Atis", "BOUN", "FrameNet", "Kenet", "Penn", "Tourism"]
BOUN_GENRES = {"bio": "biographical", "ess": "essays", "ins": "instructional", "news": "news",
               "pop": "popular culture"}
CLASSES = {
    "NOUN": "Noun", "ADJ": "Adj", "ADV": "Adv", "VERB": "Verb", "AUX": "Verb", "PRON": "Pron",
    "DET": "Det", "NUM": "Num", "CCONJ": "Conj", "SCONJ": "Conj", "ADP": "Postp",
    "INTJ": "Interj", "PART": "Part",
}
READING = {"Noun": "Noun", "Adj": "Adj", "Adv": "Adv", "Pron": "Pron", "Det": "Det", "Num": "Num",
           "Conj": "Conj", "Postp": "Postp", "Interj": "Interj", "Q": "Part", "Verb": "Verb"}
# A verb's participle or verbal noun is right as a verb or as the class it derives.
FITS = {"Noun": {"Noun", "VerbNoun"}, "Adj": {"Adj", "VerbAdj"}, "Adv": {"Adv", "VerbAdv"},
        "Verb": {"Verb", "VerbNoun", "VerbAdj", "VerbAdv"}}
FOLD = str.maketrans({"â": "a", "î": "i", "û": "u", "I": "ı", "İ": "i"})


def reading_class(p: dict) -> str | None:
    if p["primary_pos"] == "Verb":
        return {"Noun": "VerbNoun", "Adj": "VerbAdj", "Adv": "VerbAdv"}.get(p["pos"], "Verb")
    return READING.get(p["pos"])


def fits(gold: str, c: str | None) -> bool:
    return c in FITS.get(gold, {gold})


def fold(s: str) -> str:
    return s.translate(FOLD).lower()


def sentences(path: str):
    """(genre key, [(word, upos, lemma)]) for each sentence of a CoNLL-U file."""
    sent, sent_id = [], ""
    word, parts = None, range(0)
    for line in open(path, encoding="utf-8"):
        if line.startswith("# sent_id"):
            sent_id = line.split("=", 1)[1].strip()
            continue
        if line.startswith("#"):
            continue
        cols = line.rstrip("\n").split("\t")
        if len(cols) < 4:
            if sent:
                yield sent_id, sent
            sent, word, parts = [], None, range(0)
            continue
        if "-" in cols[0]:
            a, b = (int(x) for x in cols[0].split("-"))
            word, parts = cols[1], range(a, b + 1)
            continue
        if "." in cols[0]:
            continue
        i = int(cols[0])
        if i in parts:
            if i == parts.start:
                sent.append((word, cols[3], cols[2]))
        else:
            sent.append((cols[1], cols[3], cols[2]))
    if sent:
        yield sent_id, sent


def files(ud: str):
    for bank in TESTED:
        yield bank, os.path.join(ud, f"UD_Turkish-{bank}", f"tr_{bank.lower()}-ud-test.conllu")
    for split in ("train", "dev", "test"):
        yield "IMST", os.path.join(ud, "UD_Turkish-IMST", f"tr_imst-ud-{split}.conllu")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ud", default=".cache/morph_sources/ud")
    ap.add_argument("--markdown", action="store_true", help="print a Markdown table")
    args = ap.parse_args()
    morphology = akana.Morphology()
    disambiguator = akana.Disambiguator()
    # group -> [words, covered, classed, class ok, twins, twin ok, lemma ok]
    stats: dict[str, list[int]] = collections.defaultdict(lambda: [0] * 7)
    for bank, path in files(args.ud):
        if not os.path.exists(path):
            sys.exit(f"missing {path}: fetch the UD treebanks first (scripts/fetch_morph_sources.py)")
        for sent_id, sent in sentences(path):
            group = bank
            if bank == "BOUN":
                group = f"BOUN {BOUN_GENRES.get(sent_id.split('_')[0], 'other')}"
            words = [w for w, _, _ in sent]
            chosen = disambiguator.disambiguate(words)
            for (word, upos, lemma), c in zip(sent, chosen):
                if upos in ("PUNCT", "SYM"):
                    continue
                for g in (group, "all"):
                    s = stats[g]
                    s[0] += 1
                    parses = morphology.analyze(word)
                    s[1] += bool(parses)
                    gold = CLASSES.get(upos)
                    if gold:
                        s[2] += 1
                        s[3] += fits(gold, reading_class(c))
                        classes = {reading_class(p) for p in parses} - {None}
                        if len(classes) > 1 and any(fits(gold, k) for k in classes):
                            s[4] += 1
                            s[5] += fits(gold, reading_class(c))
                    if parses:
                        s[6] += fold(lemma) in (fold(c["lemma"]), fold(c["root"]))
    order = sorted((g for g in stats if g != "all"), key=lambda g: (g.split()[0], g)) + ["all"]
    pct = lambda a, b: f"{100 * a / b:.1f}" if b else "-"
    header = ["set", "words", "coverage %", "class %", "twins", "twins %", "lemma %"]
    rows = []
    for g in order:
        n, cov, classed, ok, twins, twin_ok, lemma_ok = stats[g]
        rows.append([g, str(n), pct(cov, n), pct(ok, classed), str(twins), pct(twin_ok, twins),
                     pct(lemma_ok, cov)])
    if args.markdown:
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join("---" for _ in header) + "|")
        for r in rows:
            print("| " + " | ".join(r) + " |")
    else:
        print("  ".join(f"{h:>14}" for h in header))
        for r in rows:
            print("  ".join(f"{c:>14}" for c in r))


if __name__ == "__main__":
    main()
