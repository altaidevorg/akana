"""Builds crates/akana-core/src/morphology/zemberek_lexicon.txt from Zemberek's dictionaries.

Usage:
    python scripts/build_full_lexicon.py --resources ZEMBEREK/morphology/src/main/resources/tr

The resource directory is the `tr` folder of a zemberek-nlp checkout (or of a Zeyrek install,
whose resources are copies of it). The lexicon in this repository was built from zemberek-nlp
commit 037202c7e18b977c95181a76db05cb1e95846554.

Every rule mirrors zemberek.morphology.lexicon.tr.TurkishDictionaryLoader so that the root
inventory matches Zemberek's own:

- Only Zemberek's default resources are read (`DEFAULT_DICTIONARY_RESOURCES`). The location
  gazetteer is not one of them: it holds ~28k village names, many of which are also common
  words (Kork, Bekler, Yağma), and loading it turned ordinary verbs into proper nouns.
- An explicit `P:` POS always wins. A word is inferred to be a verb only when it has no
  primary POS, starts lower-case, is longer than 3 letters and ends in -mak/-mek
  (`inferPrimaryPos`/`isVerb`); `yemek [P:Noun]` stays a noun.
- A word without a secondary POS is a proper noun when it starts upper-case
  (`inferSecondaryPos`). `Abbrv` is kept as a secondary POS instead of being dropped.
- Voicing is inferred as in `inferMorphemicAttributes`: nouns, adjectives and duplicators with
  more than one vowel ending in ç/k/p/t, or ending in -nk/-og, unless they are proper nouns,
  abbreviations, `NoVoicing` or `InverseHarmony`. Verbs are only voiced when the entry says so.

Output format (one item per line): `lemma root POS SECONDARY [ATTRIBUTES...]`.
"""

import argparse
import os

# zemberek.morphology.lexicon.tr.TurkishDictionaryLoader.DEFAULT_DICTIONARY_RESOURCES
DEFAULT_RESOURCES = [
    "master-dictionary.dict",
    "non-tdk.dict",
    "proper.dict",
    "proper-from-corpus.dict",
    "abbreviations.dict",
    "person-names.dict",
]

# Zemberek primary POS short forms -> Akana loader names.
PRIMARY = {
    "Noun": "Noun",
    "Adj": "Adj",
    "Adv": "Adv",
    "Conj": "Conj",
    "Interj": "Interj",
    "Verb": "Verb",
    "Pron": "Pron",
    "Num": "Num",
    "Det": "Det",
    "Postp": "Postp",
    "Ques": "Q",
    "Dup": "Dup",
    "Punc": "Punc",
}

# Zemberek secondary POS short forms -> Akana loader names. Postposition case government
# (PCDat, PCAbl, ...) has no Akana counterpart and is dropped.
SECONDARY = {
    "Prop": "ProperNoun",
    "Abbrv": "Abbrv",
    "Time": "Time",
    "Ord": "Ord",
    "Card": "Card",
    "Dist": "Dist",
    "Pers": "Pers",
    "Demons": "Demons",
    "Quant": "Quant",
    "Ques": "Ques",
    "Reflex": "Reflex",
    "Recip": "Recip",
    "Rel": "Rel",
}

# Zemberek attributes kept in the Akana lexicon.
ATTRIBUTES = {
    "Voicing": "Voicing",
    "LastVowelDrop": "VowelDrop",
    "Doubling": "Doubling",
    "InverseHarmony": "InverseHarmony",
    "CompoundP3sg": "CompoundP3sg",
}

VOWELS = set("aeıioöuüâîû")
STOP_CONSONANTS = set("çkpt")  # TurkishAlphabet.stopConsonants


def turkish_lower(word):
    return word.replace("I", "ı").replace("İ", "i").lower()


def normalize_circumflex(word):
    # TurkishAlphabet.normalizeCircumflex: roots are stored without â/î/û (nikâh -> nikah).
    return word.replace("â", "a").replace("î", "i").replace("û", "u")


def parse_line(line):
    """Returns (word, {key: [values]}) for a dictionary line, or None for comments/blanks."""
    line = line.strip()
    if not line or line.startswith("##"):
        return None
    if "[" in line:
        word, _, rest = line.partition("[")
        meta_str = rest.rsplit("]", 1)[0]
    else:
        word, meta_str = line, ""
    word = word.strip()
    meta = {}
    for chunk in meta_str.split(";"):
        key, sep, value = chunk.partition(":")
        if not sep:
            continue
        meta[key.strip()] = [v.strip() for v in value.split(",") if v.strip()]
    return word, meta


def is_verb(word):
    # TurkishDictionaryLoader.isVerb
    return word[:1].islower() and len(word) > 3 and (word.endswith("mek") or word.endswith("mak"))


def lexicon_item(word, meta):
    """Converts one Zemberek entry to (lemma, root, pos, secondary, attrs), or None to skip."""
    if not any(c.isalpha() for c in word):
        return None
    primary, secondary = None, None
    for token in meta.get("P", []):
        # "Ques" is both a primary and a secondary POS in Zemberek; the first one is primary.
        if primary is None and token in PRIMARY:
            primary = PRIMARY[token]
        elif token in SECONDARY and secondary is None:
            secondary = SECONDARY[token]
    if primary is None:
        primary = "Verb" if is_verb(word) else "Noun"
    if secondary is None:
        secondary = "ProperNoun" if word[:1].isupper() else "None"
    if primary == "Punc":
        return None

    raw_attrs = set(meta.get("A", []))
    attrs = {ATTRIBUTES[a] for a in raw_attrs if a in ATTRIBUTES}
    # TurkishDictionaryLoader.normalizeForDictionary: lower-case, no circumflex, dash or quote.
    lemma = normalize_circumflex(turkish_lower(word.replace("-", "").replace("'", "")))
    root = lemma
    if primary == "Verb" and lemma.endswith(("mak", "mek")):
        # The infinitive ending is not part of the root; `değil [P:Verb]` has none.
        root = lemma[:-3]
    elif primary in ("Noun", "Adj", "Dup"):
        # TurkishDictionaryLoader.inferMorphemicAttributes
        vowels = sum(1 for c in lemma if c in VOWELS)
        voicing_allowed = (
            secondary not in ("ProperNoun", "Abbrv")
            and "NoVoicing" not in raw_attrs
            and "InverseHarmony" not in raw_attrs
        )
        if vowels > 1 and lemma[-1] in STOP_CONSONANTS and voicing_allowed:
            attrs.add("Voicing")
        if (lemma.endswith("nk") or lemma.endswith("og")) and (
            "NoVoicing" not in raw_attrs and secondary != "ProperNoun"
        ):
            attrs.add("Voicing")
    if secondary == "ProperNoun":
        attrs.add("ProperNoun")
    return lemma, root, primary, secondary, tuple(sorted(attrs))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--resources", required=True, help="Zemberek resources/tr directory")
    parser.add_argument(
        "--output", default="crates/akana-core/src/morphology/zemberek_lexicon.txt"
    )
    args = parser.parse_args()

    items = set()
    for name in DEFAULT_RESOURCES:
        with open(os.path.join(args.resources, name), encoding="utf-8") as f:
            for line in f:
                parsed = parse_line(line)
                if parsed is None:
                    continue
                item = lexicon_item(*parsed)
                if item is not None:
                    items.add(item)

    with open(args.output, "w", encoding="utf-8", newline="\r\n") as f:
        for lemma, root, pos, secondary, attrs in sorted(items):
            f.write(" ".join([lemma, root, pos, secondary, *attrs]) + "\n")
    print(f"Wrote {len(items)} items to {args.output}")


if __name__ == "__main__":
    main()
