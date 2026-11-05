"""Applies the guideline's "a name inside the name of an organisation, place, street, building ... is
not a PERSON" to collected labels. The collectors map a value to every place its text occurs, so a
labeller that marks the poet Yunus Emre also marks the Yunus Emre of `Yunus Emre İlkokulu`. This drops a
PERSON span when the capitalised words right after it (up to three) reach an institution or place noun.

    python scripts/pii_data/unmark_org_names.py .cache/pii_data/hardcases/hard3.jsonl
Writes <file>_org.jsonl beside the input (which stays as it is) and prints every span it drops.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# the possessive forms that close an institution's or a place's name (with any case suffix after them)
NOUNS = ("Lisesi", "İlkokulu", "Ortaokulu", "Okulu", "Koleji", "Üniversitesi", "Fakültesi", "Enstitüsü",
         "Akademisi", "Hastanesi", "Vakfı", "Derneği", "Holding", "Şirketi", "Stadı", "Stadyumu", "Salonu",
         "Kupası", "Ödülü", "Caddesi", "Cad.", "Sokağı", "Sokak", "Sk.", "Bulvarı", "Meydanı", "Parkı",
         "Köprüsü", "Havalimanı", "Havaalanı", "Limanı", "Camii", "Kilisesi", "Mahallesi", "Mah.", "Merkezi",
         "Kütüphanesi", "Müzesi", "Barajı", "Tüneli", "Gemisi", "Kışlası", "Tesisleri", "İstasyonu", "Garı",
         "Sitesi", "Apartmanı", "Konutları", "Otoyolu", "Belediyesi", "Kampüsü", "Yurdu", "Tiyatrosu",
         "Konservatuvarı", "Üssü", "Bahçesi")
WORD = re.compile(r"[^\s,;:!?()\"“”]+")


def inside_name(text: str, end: int) -> str | None:
    """The noun that makes the name ending at `end` part of a longer proper name, or None."""
    if not text[end:end + 1].isspace():
        return None
    for word in WORD.findall(text[end:end + 80])[:3]:
        if not word[:1].isupper():
            return None
        if word.startswith(NOUNS):
            return word
    return None


def main():
    for arg in sys.argv[1:]:
        path = Path(arg)
        out = path.with_name(path.stem + "_org.jsonl")
        dropped = 0
        with open(out, "w", encoding="utf-8") as f:
            for line in open(path, encoding="utf-8"):
                row = json.loads(line)
                spans = []
                for sp in row["spans"]:
                    noun = inside_name(row["text"], sp[1]) if sp[2] == "PERSON" else None
                    if noun:
                        dropped += 1
                        print(f"  {row.get('id', '')}: {row['text'][sp[0]:sp[1]]!r} + {noun}")
                    else:
                        spans.append(sp)
                f.write(json.dumps({**row, "spans": spans}, ensure_ascii=False) + "\n")
        print(f"{path.name}: {dropped} PERSON spans inside an organisation or place name dropped -> {out.name}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
