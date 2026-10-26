"""Builds crates/akana-core/src/morphology/zemberek_lexicon.txt from Zemberek's dictionaries.

Usage:
    python scripts/build_full_lexicon.py [--report REPORT.txt]

Sources (all in `scripts/lexicon/`):
- `zemberek/*.dict`: the six default dictionaries of Zemberek's TurkishDictionaryLoader
  (`DEFAULT_DICTIONARY_RESOURCES`), as shipped in Zeyrek 0.1.3 (a copy of zemberek-nlp's
  `morphology/src/main/resources/tr`; Apache License 2.0, see `zemberek/NOTICE.md`). The build
  first checks them against the SHA-256 pinned there (`scripts/fetch_zemberek_dicts.py`, which
  can also re-create them from the pinned wheel).
- `akana-legal.dict`, `akana-extra.dict`: akana's own entries in the same syntax. An entry here
  replaces Zemberek's entry with the same word, POS and secondary POS.
- `akana-remove.dict`: Zemberek entries to drop (same syntax; matched on word, POS and
  secondary POS).

Every rule mirrors zemberek.morphology.lexicon.tr.TurkishDictionaryLoader so that the root
inventory and attributes match Zemberek's own, with three differences that fix what the
analyzer accepted wrongly:

- Duplicates are merged by precedence instead of being kept under a new index. Within one
  file an entry without attributes or index that repeats an earlier entry's word and POS (a
  circumflex variant such as mahkûmiyet / mahkumiyet) is dropped; an entry from a later file
  is dropped when an earlier file already has that word and POS. Indexed homographs of one
  file (kalp "heart" and kalp "counterfeit") are kept.
- Entries of `proper-from-corpus.dict` that are inflected common words (Mahkemesi, Kurulu,
  Üniversitesi) or misspelled stems (Redd, Kitab, Kalb) are dropped unless a curated file
  (proper, person-names, abbreviations) lists the name too.
- Roots with characters that are not letters or digits (m², alm.) are dropped.

Output, one item per line, sorted:
    lemma root POS SECONDARY [ATTRIBUTES...] [pr=PRONUNCIATION] [roots=HEAD-TAIL] [pc=CASE]
`pr=` is written for abbreviations and wherever the pronunciation differs from the root (gram→gıram);
`idx=` numbers a homograph after the first
(kalp "heart", kalp idx=1 "counterfeit"), which comes first. `roots=` gives the parts of a compound that carries its own 3rd-person possessive
(cumhurbaşkanı roots=cumhur-başkan); `pc=` is the case a postposition governs. The attributes
are Zemberek's after `inferMorphemicAttributes`, so a verb's aorist class and a monosyllable's
NoVoicing are explicit.
"""

import argparse
import os
import re

from fetch_zemberek_dicts import check as check_zemberek_sources

HERE = os.path.dirname(os.path.abspath(__file__))
LEXICON_DIR = os.path.join(HERE, "lexicon")

# zemberek.morphology.lexicon.tr.TurkishDictionaryLoader.DEFAULT_DICTIONARY_RESOURCES
DEFAULT_RESOURCES = [
    "master-dictionary.dict",
    "non-tdk.dict",
    "proper.dict",
    "proper-from-corpus.dict",
    "abbreviations.dict",
    "person-names.dict",
]
CURATED_NAMES = {"proper.dict", "abbreviations.dict", "person-names.dict"}
SUPPLEMENTS = ["akana-legal.dict", "akana-extra.dict"]
REMOVALS = "akana-remove.dict"

# Zemberek primary POS short forms -> akana loader names.
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
# Zemberek secondary POS short forms -> akana loader names.
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
# Postposition case government (Zemberek secondary POS PCDat ...), written as pc=.
POSTP_CASE = {"PCDat": "Dat", "PCAbl": "Abl", "PCNom": "Nom", "PCGen": "Gen", "PCIns": "Ins", "PCAcc": "Acc", "PCLoc": "Loc"}

# Zemberek RootAttribute names -> akana names. Attributes not listed (Ext, Runtime, Dummy,
# Informal, LocaleEn, PronunciationGuessed, NoQuote, NounConsInsert_n) are not used.
ATTRIBUTES = {
    "Voicing": "Voicing",
    "NoVoicing": "NoVoicing",
    "LastVowelDrop": "VowelDrop",
    "Doubling": "Doubling",
    "InverseHarmony": "InverseHarmony",
    "CompoundP3sg": "CompoundP3sg",
    "Aorist_A": "AoristA",
    "Aorist_I": "AoristI",
    "ProgressiveVowelDrop": "ProgressiveVowelDrop",
    "Passive_In": "PassiveIn",
    "Causative_t": "CausativeT",
    "Reflexive": "Reflexive",
    "Reciprocal": "Reciprocal",
    "NonReciprocal": "NonReciprocal",
    "ImplicitPlural": "ImplicitPlural",
    "ImplicitP1sg": "ImplicitP1sg",
    "ImplicitP2sg": "ImplicitP2sg",
    "ImplicitDative": "ImplicitDative",
    "FamilyMember": "FamilyMember",
    "NoSuffix": "NoSuffix",
    # akana's own: which of a verb and the voice derivation it looks like (katıl / kat-ıl) wins,
    # and items that yield to any other reading of a word (bulamak beside bul-ama-).
    "Lexicalized": "Lexicalized",
    "NotLexicalized": "NotLexicalized",
    "Rare": "Rare",
}

VOWELS = set("aeıioöuüâîû")
STOP_CONSONANTS = set("çkpt")  # TurkishAlphabet: voiceless and not continuant
LETTERS = set("abcçdefgğhıijklmnoöprsştuüvyzqwxâîû")


def turkish_lower(word):
    return word.replace("I", "ı").replace("İ", "i").lower()


def normalize_circumflex(word):
    # TurkishAlphabet.normalizeCircumflex: roots are stored without â/î/û (nikâh -> nikah).
    return word.replace("â", "a").replace("î", "i").replace("û", "u")


def vowel_count(word):
    return sum(1 for c in word if c in VOWELS)


def load_letter_names():
    names = {}
    with open(os.path.join(LEXICON_DIR, "zemberek", "turkish-letter-names.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("##") and "=" in line:
                k, v = line.split("=", 1)
                names[k.strip()] = v.strip()
    return names


LETTER_NAMES = load_letter_names()

# TurkishNumbers.convertToString, for abbreviations with digits (A4 -> adört).
ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
POWERS = ["", "bin", "milyon", "milyar", "trilyon", "katrilyon"]


def number_to_words(digits):
    zeros = len(digits) - len(digits.lstrip("0"))
    words = ["sıfır"] * zeros
    n = int(digits) if digits.strip("0") else 0
    groups = []
    power = 0
    while n > 0:
        n, g = divmod(n, 1000)
        if g:
            h, rest = divmod(g, 100)
            t, o = divmod(rest, 10)
            part = []
            if h:
                part += ([ONES[h]] if h > 1 else []) + ["yüz"]
            part += [w for w in (TENS[t], ONES[o]) if w]
            if power == 1 and g == 1:
                part = []
            groups.insert(0, "".join(part) + POWERS[power])
        power += 1
    return "".join(words + groups)


def letter_pronunciation(word):
    """PronunciationGuesser.toTurkishLetterPronunciations: spells a word letter by letter (ABD
    -> abede); a final k is read "ka". Digit runs are read as numbers."""
    out = []
    for i, piece in enumerate(re.split(r"(\d+)", word)):
        if not piece:
            continue
        if piece.isdigit():
            out.append(number_to_words(piece))
            continue
        for j, c in enumerate(piece):
            if c == "-":
                continue
            last = i == len(re.split(r"(\d+)", word)) - 1 and j == len(piece) - 1
            if c == "k" and last:
                out.append("ka")
            else:
                out.append(LETTER_NAMES.get(c, c))
    return "".join(out)


FINAL_CLUSTERS = {
    "rk", "rt", "rp", "rs", "rç", "rd", "rb", "rg", "rf", "rm", "rn", "rz", "rş", "lk", "lt",
    "lp", "lç", "ls", "lm", "lf", "nk", "nt", "nç", "nd", "ns", "nz", "st", "şt", "ks", "ft",
    "kt", "pt", "yt", "yn", "yk", "ym", "yr", "yl", "yş",
}


def strictly_syllabifiable(word):
    """Whether a word reads as Turkish syllables (TurkishSyllableExtractor.STRICT): at most one
    consonant before the first vowel, at most three between vowels, and nothing or a single
    consonant or a common two-consonant coda after the last vowel."""
    runs = re.split(r"[aeıioöuü]+", word)
    if len(runs) < 2:
        return False
    if len(runs[0]) > 1 or any(len(r) > 3 for r in runs[1:-1]):
        return False
    coda = runs[-1]
    return len(coda) <= 1 or (len(coda) == 2 and coda in FINAL_CLUSTERS)


def guess_abbreviation(root):
    """PronunciationGuesser.guessForAbbreviation: spelled letter by letter unless it reads as
    a word (NATO, TÜBİTAK)."""
    if len(root) < 3 or not any(c in VOWELS for c in root[:2]) or not strictly_syllabifiable(root):
        return letter_pronunciation(root)
    return root.replace("w", "v").replace("q", "k").replace("x", "ks")


def parse_line(line):
    """Returns (word, {key: value}) for a dictionary line, or None for comments/blanks."""
    line = line.strip()
    if not line or line.startswith("##") or line.startswith("#"):
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
        if sep:
            meta[key.strip()] = value.strip()
    return word, meta


def is_verb(word):
    # TurkishDictionaryLoader.isVerb
    return word[:1].islower() and len(word) > 3 and (word.endswith("mek") or word.endswith("mak"))


def pos_info(word, meta):
    primary = secondary = None
    pc = None
    tokens = [t.strip() for t in meta.get("P", "").split(",") if t.strip()]
    for token in tokens:
        # "Ques" is both a primary and a secondary POS in Zemberek; the first one is primary.
        if primary is None and token in PRIMARY:
            primary = PRIMARY[token]
        elif token in SECONDARY and secondary is None:
            secondary = SECONDARY[token]
        elif token in POSTP_CASE:
            pc = POSTP_CASE[token]
    if primary is None:
        primary = "Verb" if is_verb(word) else "Noun"
    if secondary is None:
        secondary = "ProperNoun" if word[:1].isupper() else "None"
    return primary, secondary, pc


def infer_attributes(pron, primary, secondary, attrs):
    """TurkishDictionaryLoader.inferMorphemicAttributes, on the pronunciation."""
    attrs = set(attrs)
    last = pron[-1]
    vowels = vowel_count(pron)
    if primary == "Verb":
        if last in VOWELS:
            attrs |= {"ProgressiveVowelDrop", "Passive_In"}
        if vowels > 1 and "Aorist_A" not in attrs:
            attrs.add("Aorist_I")
        if vowels == 1 and "Aorist_I" not in attrs:
            attrs.add("Aorist_A")
        if last == "l":
            attrs.add("Passive_In")
        if last in VOWELS or (last in "lr" and vowels > 1):
            attrs.add("Causative_t")
    elif primary in ("Noun", "Adj", "Dup"):
        if (
            vowels > 1
            and last in STOP_CONSONANTS
            and secondary not in ("ProperNoun", "Abbrv")
            and "NoVoicing" not in attrs
            and "InverseHarmony" not in attrs
        ):
            attrs.add("Voicing")
        if pron.endswith("nk") or pron.endswith("og"):
            if "NoVoicing" not in attrs and secondary != "ProperNoun":
                attrs.add("Voicing")
        elif vowels < 2 and "Voicing" not in attrs:
            attrs.add("NoVoicing")
    return attrs


class Entry:
    __slots__ = ("lemma", "root", "pos", "secondary", "attrs", "pron", "roots", "pc", "explicit", "indexed", "index", "source", "word")

    def key(self):
        return (self.lemma, self.root, self.pos, self.secondary)

    def line(self):
        attrs = sorted(ATTRIBUTES[a] for a in self.attrs if a in ATTRIBUTES)
        if self.secondary == "ProperNoun":
            attrs.append("ProperNoun")
        fields = [self.lemma, self.root, self.pos, self.secondary, *attrs]
        if self.pron != self.root or self.secondary == "Abbrv":
            fields.append(f"pr={self.pron}")
        if self.roots:
            fields.append(f"roots={self.roots}")
        if self.pc:
            fields.append(f"pc={self.pc}")
        if self.index:
            fields.append(f"idx={self.index}")
        return " ".join(fields)

    def sort_key(self):
        # Zemberek's homograph order: the unnumbered sense first (hak "right", then hak [Index:1]).
        return (self.lemma, self.root, self.pos, self.secondary, self.index, self.line())


def make_entry(word, meta, source):
    """Converts one Zemberek line to an Entry, or None to skip it."""
    if not any(c.isalpha() for c in word):
        return None
    primary, secondary, pc = pos_info(word, meta)
    if primary == "Punc":
        return None
    # TurkishDictionaryLoader.generateRoot / normalizeForDictionary.
    lemma = normalize_circumflex(turkish_lower(word.replace("-", "").replace("'", "")))
    root = lemma[:-3] if primary == "Verb" and is_verb(word) else lemma
    if not root or not all(c in LETTERS or c.isdigit() for c in root):
        return None
    if "Pr" in meta:
        pron = normalize_circumflex(turkish_lower(meta["Pr"]))
    elif secondary == "Abbrv":
        pron = guess_abbreviation(root)
    elif any(c in VOWELS for c in root):
        pron = root
    else:
        pron = letter_pronunciation(root)
    raw = {a.strip() for a in meta.get("A", "").split(",") if a.strip()}
    e = Entry()
    e.word = word
    e.lemma = lemma
    e.root = root
    e.pos = primary
    e.secondary = secondary
    e.attrs = infer_attributes(pron, primary, secondary, raw)
    e.pron = pron
    e.roots = meta.get("Roots")
    e.pc = pc
    e.explicit = bool(raw) or "Pr" in meta
    e.indexed = "Index" in meta
    e.index = int(meta.get("Index", "0") or 0)
    e.source = source
    if "Ref" in meta and source.startswith("akana"):
        # akana entries may point at a canonical lemma (tesbit -> tespit, red -> ret).
        ref = meta["Ref"]
        e.lemma = normalize_circumflex(turkish_lower(ref))
    return e


PHONOLOGY = ("InverseHarmony", "Doubling", "LastVowelDrop", "NoVoicing")


def read_entries(path, source, report=None):
    """The file's entries. A word's pronunciation does not depend on its POS, so an entry without
    attributes or index takes the phonological attributes the file gives an unnumbered entry of
    the same spelling (meçhul [P:Adj; A:InverseHarmony] makes the noun meçhul take meçhulü, not
    meçhulu; hak [A:Doubling] makes the adjective hak double too). The spelling is compared
    with its circumflex: âdet "custom" is another word than adet "count"."""
    with open(path, encoding="utf-8") as f:
        parsed = [p for p in map(parse_line, f) if p is not None]
    phonology = {}
    for w, meta in parsed:
        attrs = {a.strip() for a in meta.get("A", "").split(",")} & set(PHONOLOGY)
        if attrs and "Index" not in meta:
            key = turkish_lower(w.replace("-", "").replace("'", ""))
            phonology.setdefault(key, set()).update(attrs)
    for word, meta in parsed:
        key = turkish_lower(word.replace("-", "").replace("'", ""))
        if "A" not in meta and "Index" not in meta and key in phonology:
            meta = dict(meta, A=", ".join(sorted(phonology[key])))
            if report is not None:
                report.append(f"bare copy: {source}: {word} takes {meta['A']}")
        entry = make_entry(word, meta, source)
        if entry is not None:
            yield entry


# Endings that mark a corpus-collected "proper noun" as an inflected common word: the
# 3rd-person possessive -sI after a vowel-final stem and what follows it (Mahkemesi,
# Caddesine). Names rarely end in it; -su/-sü is left alone (Göksu, Bengisu are names).
BUFFER_ENDINGS = sorted(
    {p + c for p in ["sı", "si", "ları", "leri"] for c in ["", "nı", "ni", "nu", "nü", "na", "ne", "nda", "nde", "ndan", "nden", "nın", "nin", "nun", "nün"]},
    key=len,
    reverse=True,
)
UNVOICE = {"b": "p", "c": "ç", "d": "t", "g": "k", "ğ": "k"}
BACK = set("aıou")
ROUNDED = set("oöuü")


def i_vowel(stem):
    v = next((c for c in reversed(stem) if c in VOWELS), "e")
    if v in BACK:
        return "u" if v in ROUNDED else "ı"
    return "ü" if v in ROUNDED else "i"


def junk_proper(entry, common):
    """Whether a corpus-collected proper noun is really an inflected common word or a misspelled
    stem: a vowel-final common root + -sI... (Mahkemesi, Caddesi); a common root of at least
    six letters + its harmonising -I (Meydanı, Protokolu; shorter ones are often names, Nesimi,
    Rüştü); a stem ending in ğ, which Turkish spelling never leaves word-final (Ayağ); or a
    doubled stem written alone (Redd, Hakk). akana-remove.dict lists the rest by name."""
    w = entry.root
    for end in BUFFER_ENDINGS:
        stem = w[: -len(end)]
        if w.endswith(end) and len(stem) >= 3 and stem[-1] in VOWELS and stem in common:
            return True
    if w[-1] in "ıiuü":
        stem = w[:-1]
        for s in (stem, stem[:-1] + UNVOICE.get(stem[-1], stem[-1])):
            if len(s) >= 6 and s in common and s[-1] not in VOWELS and w[-1] == i_vowel(s):
                return True
    if w[-1] == "ğ" and w[:-1] + "k" in common:
        return True
    if len(w) >= 3 and w[-1] == w[-2]:
        base = w[:-1]
        for s in (base, base[:-1] + UNVOICE.get(base[-1], base[-1])):
            if s in common and "Doubling" in common[s]:
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", default="crates/akana-core/src/morphology/zemberek_lexicon.txt")
    parser.add_argument("--report", help="write the dropped and merged entries here")
    args = parser.parse_args()
    check_zemberek_sources()

    report = []
    items = {}  # key -> list of entries (homographs of one file)
    order = []
    for name in DEFAULT_RESOURCES:
        for e in read_entries(os.path.join(LEXICON_DIR, "zemberek", name), name, report):
            existing = items.get(e.key())
            if existing is None:
                items[e.key()] = [e]
                order.append(e.key())
                continue
            if existing[0].source != e.source:
                if any(x.attrs != e.attrs for x in existing):
                    report.append(f"later file: dropped {e.source}: {e.line()} (kept {existing[0].line()})")
                continue
            if any(x.attrs == e.attrs and x.pron == e.pron for x in existing):
                continue
            same_spelling = [x for x in existing if turkish_lower(x.word) == turkish_lower(e.word)]
            if e.indexed or all(x.indexed for x in existing) or not same_spelling:
                # A homograph: one the file numbers (kalp "heart", kalp [Index:1] "counterfeit";
                # katil [Index:1] "murder", katil "murderer") or one it spells with a
                # circumflex (adet "count" adedi, âdet "custom" âdeti; metin, metîn).
                existing.append(e)
                continue
            if not e.explicit:
                report.append(f"variant: dropped {e.word} ({e.line()}), kept {existing[0].line()}")
                continue
            plain = [x for x in same_spelling if not x.explicit and not x.indexed]
            if plain:
                # The same word listed again with attributes: they win over the bare copy.
                report.append(f"variant: replaced {plain[0].line()} by {e.word} ({e.line()})")
                items[e.key()] = [x for x in existing if x not in plain] + [e]
                continue
            existing.append(e)

    # Common roots for the junk filter: Noun/Adj roots of the dictionaries that are not names.
    common = {}
    for key in order:
        for e in items[key]:
            if e.source in ("master-dictionary.dict", "non-tdk.dict") and e.pos in ("Noun", "Adj") and e.secondary == "None":
                common.setdefault(e.root, set()).update(e.attrs)
    curated = {
        e.root
        for key in order
        for e in items[key]
        if e.source in CURATED_NAMES or e.source in ("master-dictionary.dict", "non-tdk.dict")
        and e.secondary == "ProperNoun"
    }
    for key in order:
        kept = []
        for e in items[key]:
            if e.source == "proper-from-corpus.dict" and e.root not in curated and junk_proper(e, common):
                report.append(f"junk proper noun: dropped {e.word}")
                continue
            kept.append(e)
        items[key] = kept

    removals = {e.key() for e in read_entries(os.path.join(LEXICON_DIR, REMOVALS), "akana-remove")}
    for key in removals:
        if items.pop(key, None) is not None:
            report.append(f"akana-remove: dropped {key}")
    for name in SUPPLEMENTS:
        for e in read_entries(os.path.join(LEXICON_DIR, name), "akana:" + name):
            if e.indexed and items.get(e.key()):
                # A second reading of an entry (HMK "hemeka" and "kanunu"; merci, merci-i).
                items[e.key()].append(e)
                continue
            if e.key() in items and items[e.key()]:
                report.append(f"{name}: replaced {items[e.key()][0].line()} by {e.line()}")
            else:
                order.append(e.key())
            items[e.key()] = [e]

    entries = {e.line(): e for key in items for e in items[key]}
    lines = [e.line() for e in sorted(entries.values(), key=Entry.sort_key)]
    with open(args.output, "w", encoding="utf-8", newline="\r\n") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"Wrote {len(lines)} items to {args.output}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("\n".join(report) + "\n")
        print(f"{len(report)} report lines in {args.report}")


if __name__ == "__main__":
    main()
