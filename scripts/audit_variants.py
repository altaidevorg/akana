"""Usage variants of the lexicon's alternations: the selection behind
scripts/lexicon/akana-variants.dict (plan step D5, review 2 MLEX2-05).

    python scripts/audit_variants.py > scripts/lexicon/akana-variants.dict

with akana importable (the extension built from this checkout, before the variants are added).

For each common noun and adjective of the lexicon (zemberek_lexicon.txt), a copy with one
phonological alternation flipped is read into a second analyzer:
- voicing left out (temerrüte for temerrüde) or applied (taksidi beside taksiti);
- a dropped vowel kept (fuhuşa, mühürün) or a vowel dropped;
- inverse harmony left out (sulha) or applied;
- doubling left out.
A word of the prior's training types (data/morphology/prior_train_types.tsv.gz, the training
split only, so that the evaluation samples stay unseen) that has no reading, and that a flipped
copy reads, is evidence for the variant. A variant is listed when such words make at
least MIN_TOKENS tokens. It becomes a second item of the word, with the lemma of the first and
the attribute Nonstandard: the analyzer reads the variant, marks the reading nonstandard, and
the spell checker, the grammar rules and the deasciifier keep rejecting it.

The sources disagree on which alternation is standard for some roots (TDK writes taksit, -di;
Wiktionary's declension has taksiti; Zemberek's lexicon taksit without voicing): the variant
layer reads usage without deciding that, and a root's standard spelling stays the lexicon's.
"""

import collections
import gzip
import os
import sys

import akana

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEXICON = os.path.join(ROOT, "crates", "akana-core", "src", "morphology", "zemberek_lexicon.txt")
TYPES = os.path.join(ROOT, "data", "morphology", "prior_train_types.tsv.gz")
MIN_TOKENS = 5
# Flips the evidence misleads, checked by hand: words of another root or a name.
EXCLUDE = {
    ("mas", "nodbl"): "ması, masına are the -mA-sI of a split word",
    ("başak", "ih"): "Başakçi is a surname",
    ("hazin", "vd"): "Hazni is a name",
    ("tarih", "vd"): "tarhi, tarhinde are forms of tarh (a tax's assessment)",
    ("itiyat", "ih"): "itiyadî is an adjective of its own",
}
VOWELS = set("aeıioöuü")
BACK = set("aıou")
# The flips: a tag, and what it does to the attributes (in the lexicon file's names).
DESCRIPTIONS = {
    "nov": "voicing left out",
    "v": "voicing applied",
    "novd": "the dropped vowel kept",
    "vd": "a vowel dropped",
    "noih": "inverse harmony left out",
    "ih": "inverse harmony applied",
    "nodbl": "doubling left out",
}


def fold(word):
    lower = word.replace("I", "ı").replace("İ", "i").lower()
    return lower.replace("â", "a").replace("î", "i").replace("û", "u")


# The lexicon file's attribute names -> the dictionary's (Zemberek's) names.
DICTIONARY_NAMES = {"VowelDrop": "LastVowelDrop"}


def flips(root, attrs):
    """The flipped attribute sets of a root with `attrs`, by tag."""
    vowels = [c for c in root if c in VOWELS]
    last = root[-1]
    out = {}
    if "InverseHarmony" in attrs:
        out["noih"] = attrs - {"InverseHarmony"}
    elif len(vowels) >= 2 and last in "ltkfsmhr" and vowels[-1] in BACK:
        out["ih"] = attrs | {"InverseHarmony"}
    if "Voicing" in attrs and last in "pçtkg":
        out["nov"] = (attrs - {"Voicing"}) | {"NoVoicing"}
    elif len(vowels) >= 2 and last in "pçtk":
        out["v"] = (attrs - {"NoVoicing"}) | {"Voicing"}
    if "VowelDrop" in attrs:
        out["novd"] = attrs - {"VowelDrop"}
    elif (
        len(vowels) >= 2
        and len(root) >= 4
        and root[-2] in "ıiuü"
        and root[-1] not in VOWELS
        and root[-3] not in VOWELS
    ):
        out["vd"] = attrs | {"VowelDrop"}
    if "Doubling" in attrs:
        out["nodbl"] = attrs - {"Doubling"}
    return out


def main():
    items = {}  # (lemma, pos) -> (root, attrs, display, highest index)
    for line in open(LEXICON, encoding="utf-8"):
        fields = line.split()
        if len(fields) < 4:
            continue
        lemma, root, pos, secondary = fields[:4]
        attrs = {f for f in fields[4:] if "=" not in f}
        values = dict(f.split("=", 1) for f in fields[4:] if "=" in f)
        index = int(values.get("idx", "0"))
        key = (lemma, pos)
        if key in items:
            items[key] = (*items[key][:3], max(items[key][3], index))
            continue
        display = values.get("display", lemma)
        # The entry is written as the dictionary spells the word, which must fold to its root
        # (an item spelled after another, asteroid Ref:asteroit, is left out).
        if (
            pos in ("Noun", "Adj")
            and secondary == "None"
            and root.isalpha()
            and "roots" not in values
            and fold(display) == root
        ):
            items[key] = (root, attrs, display, index)
    variants = {}  # marked lemma -> (lemma, pos, tag, attrs)
    lines = []
    for (lemma, pos), (root, attrs, _, _) in items.items():
        for tag, flipped in flips(root, attrs).items():
            marked = f"{lemma}__{tag}"
            variants[marked] = (lemma, pos, tag, flipped)
            lines.append(" ".join([marked, root, pos, "None", *sorted(flipped)]))
    print(f"# {len(variants)} flipped copies", file=sys.stderr)

    counts = collections.Counter()
    with gzip.open(TYPES, "rt", encoding="utf-8") as f:
        for line in f:
            word, _, n = line.rstrip("\n").partition("\t")
            if word and n:
                counts[word] += int(n)
    base = akana.Morphology()
    # Words the analyzer reads already, if only as old orthography (karariyle), need no variant.
    unread = [w for w in counts if not base.analyze(w)]
    print(f"# {len(unread)} of {len(counts)} training types without a reading", file=sys.stderr)

    flip = akana.Morphology()
    flip.load_dictionary_str("\n".join(lines))
    evidence = collections.defaultdict(collections.Counter)
    for word in unread:
        for p in flip.analyze(word):
            if p["lemma"] in variants:
                evidence[p["lemma"]][word] += counts[word]
                break

    chosen = sorted(
        ((sum(words.values()), marked, words) for marked, words in evidence.items()),
        key=lambda r: (-r[0], r[1]),
    )
    excluded = lambda marked: (variants[marked][0], variants[marked][2]) in EXCLUDE  # noqa: E731
    chosen = [r for r in chosen if r[0] >= MIN_TOKENS and not excluded(r[1])]
    print(f"# {len(chosen)} variants with {sum(r[0] for r in chosen)} tokens", file=sys.stderr)

    print("## Usage variants of the lexicon's alternations, generated by scripts/audit_variants.py")
    print("## (plan step D5, review 2 MLEX2-05): a second item of the word with one alternation")
    print(f"## flipped, read from at least {MIN_TOKENS} tokens of the prior's training types that have")
    print("## no reading. Its readings keep the lemma and are nonstandard. After each entry, the")
    print("## flip and the most frequent words that attest it.")
    for tokens, marked, words in chosen:
        lemma, pos, tag, flipped = variants[marked]
        root, attrs, display, index = items[(lemma, pos)]
        phonology = sorted(
            DICTIONARY_NAMES.get(a, a)
            for a in flipped & {"Voicing", "NoVoicing", "VowelDrop", "InverseHarmony", "Doubling"}
        )
        meta = [f"A:{', '.join(['Nonstandard', *phonology])}", f"Index:{index + 1}"]
        if pos == "Adj":
            meta.insert(0, "P:Adj")
        examples = " ".join(f"{w} {n}" for w, n in words.most_common(3))
        print(f"{display} [{'; '.join(meta)}]  # {DESCRIPTIONS[tag]}: {examples}")


if __name__ == "__main__":
    main()
