"""Writes scripts/lexicon/akana-names.dict: the Turkish given names and surnames of akana's PII
lists (crates/akana-core/src/pii: gazetteer.rs GIVEN_NAMES and SURNAMES, names_data.rs
COMMON_GIVEN_NAMES) that the lexicon has no proper-noun item for, but the given names PII
itself finds more often something else (names_data.rs RARE_NAMES that are no surname: Kuzey,
Yağmur) and the names of months (Eylül, Ocak), whose capital in a date is the month's.

Without them a name such as Serpil, Gökçen or Halide read as a junk derivation (serp-il:
serpmek) and one that is also a word (Hakan, Metin, Kaya) had no name reading (review 4
MORPH4-10). Run it after changing those lists, before scripts/build_full_lexicon.py, from a
lexicon built without the file (an existing akana-names.dict is ignored, so the output does
not depend on itself):

    python scripts/names_from_pii.py
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PII = ROOT / "crates" / "akana-core" / "src" / "pii"
LEXICON = ROOT / "crates" / "akana-core" / "src" / "morphology" / "zemberek_lexicon.txt"
OUT = ROOT / "scripts" / "lexicon" / "akana-names.dict"
UPPER = {"i": "İ", "ı": "I"}
MONTHS = {
    "ocak", "şubat", "mart", "nisan", "mayıs", "haziran", "temmuz", "ağustos", "eylül", "ekim",
    "kasım", "aralık",
}


def strings(source: str, start: str, end: str) -> set[str]:
    i = source.index(start)
    return set(re.findall(r'"([^"]+)"', source[i : source.index(end, i)]))


def title(word: str) -> str:
    return UPPER.get(word[0], word[0].upper()) + word[1:]


def main() -> None:
    gazetteer = (PII / "gazetteer.rs").read_text(encoding="utf-8")
    data = (PII / "names_data.rs").read_text(encoding="utf-8")
    given = strings(gazetteer, "static ref GIVEN_NAMES:", "};") | strings(
        data, "pub(crate) const COMMON_GIVEN_NAMES", "];"
    )
    surnames = strings(gazetteer, "static ref SURNAMES:", "};")
    # A surname is written with its capital mid-sentence, as the word it also is is not (Kaya,
    # Demir), so PII's rare names stay out only when they are no surname (Kuzey, Yağmur).
    words = (strings(data, "pub(crate) const RARE_NAMES", "];") - surnames) | MONTHS
    surnames -= given
    previous = set()
    if OUT.exists():
        previous = {
            line.split(" ")[0].replace("İ", "i").replace("I", "ı").lower()
            for line in OUT.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        }
    names = set()
    for line in LEXICON.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) >= 4 and f[3] == "ProperNoun":
            names.add(f[1])
    # Names this file added are no reason to leave them out again.
    names -= previous

    def missing(candidates: set[str]) -> list[str]:
        return sorted(
            w
            for w in candidates
            if re.fullmatch(r"[a-zçğıöşüâîû]+", w) and w not in names and w not in words
        )

    lines = [
        "## Turkish given names and surnames that Zemberek's dictionaries lack as names, from akana's",
        "## own PII lists. Written by scripts/names_from_pii.py; do not edit by hand.",
        "",
        "## Given names.",
        *(f"{title(w)} [P:Noun,Prop]" for w in missing(given)),
        "",
        "## Surnames.",
        *(f"{title(w)} [P:Noun,Prop]" for w in missing(surnames)),
    ]
    OUT.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print(f"{OUT}: {len(missing(given))} given names, {len(missing(surnames))} surnames")


if __name__ == "__main__":
    main()
