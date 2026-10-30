"""Turkish WikiNER as the NER trainer reads it: one token and its tag per line, a blank line after
each sentence, in `.cache/ner/wikiner/{train,validation,test}.tsv`.

    python scripts/prepare_wikiner.py

after `python scripts/fetch_morph_sources.py --only turkish-wikiNER`. Tags are BIO over PER
(WikiNER's PERSON), LOC (GPE and LOC) and ORG; the other types (facilities, dates, amounts,
nationalities, works ...) are O. akana's LOC has facilities and streets too, from a pattern
(`ner/mod.rs`): a tagger trained with WikiNER's FAC as LOC lost ORG F1 on the validation split. The trainer is `crates/akana-core/src/ner/model.rs`'s ignored test
`train_ner_model`.
"""

from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / ".cache" / "morph_sources" / "turkish-wikiNER"
OUT = ROOT / ".cache" / "ner" / "wikiner"
TYPES = {"PERSON": "PER", "GPE": "LOC", "LOC": "LOC", "ORG": "ORG"}


def tag(name: str) -> str:
    if name == "O":
        return "O"
    prefix, kind = name.split("-", 1)
    return f"{prefix}-{TYPES[kind]}" if kind in TYPES else "O"


def main() -> None:
    names = [l.strip() for l in (SOURCE / "tags.lst").read_text(encoding="utf-8").splitlines() if l.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        lines = []
        for row in pq.read_table(SOURCE / "data" / f"{split}-00000-of-00001.parquet").to_pylist():
            for token, t in zip(row["tokens"], row["tags"]):
                token = token.replace("\t", " ").strip()
                if token:
                    lines.append(f"{token}\t{tag(names[t])}")
            lines.append("")
        (OUT / f"{split}.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(split, sum(1 for l in lines if not l), "sentences")


if __name__ == "__main__":
    main()
