"""The CC0 word list that extends the espeak-compat lexicon (scripts/speech/espeak_compat_wordlist.json).

The words of the Common Voice Turkish sentence collector (CC0) that the train split of EchoCraft's
eSpeak fixtures lacks, from the sentences that are in none of the fixtures' splits, 40 to a text as
`w1, w2, ..., w40.`. EchoCraft's `espeak_fixtures.py` runs eSpeak NG 1.52.0 on it (about 90 s):

    cd C:/Users/Gigabyte/echocraft
    tools/oracle/.venv/Scripts/python.exe -B experiments/2026-11-07-akana-phonemiser-scripts/espeak_fixtures.py \
        OUT_DIR --corpus <akana>/scripts/speech/espeak_compat_wordlist.json

and the data builder reads OUT_DIR/espeak-plan.jsonl as AKANA_ESPEAK_EXTRA (see
crates/akana-core/src/speech/model.rs).

Usage: python scripts/speech/make_espeak_compat_wordlist.py SENTENCE_COLLECTOR.txt FIXTURES_DIR OUT.json
"""

import json
import re
import sys
from pathlib import Path

LETTERS = "A-Za-zÇĞİIÖŞÜçğıiöşüâîûÂÎÛ"
WORD_RE = re.compile(r"[%s]+(?:['’][%s]+)*(?:-[iI](?![%s]))?" % (LETTERS, LETTERS, LETTERS))
UPPER = set("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZXWQÂÎÛ")
PLAIN = re.compile(r"^[abcçdefgğhıijklmnoöprsştuüvyzâîû]+$")
VOWELS = set("aeıioöuüâîû")
JOINS = {("ister", "misin"), ("ister", "misiniz"), ("ne", "yaptın")}


def tr_lower(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").replace("Â", "â").replace("Î", "î").replace("Û", "û").lower()


def key(w: str) -> str:
    return tr_lower(w).replace("’", "'").replace("-", "")


def words(text: str) -> list[str]:
    """The phonemiser's words of a text (letters only; split at a small letter before a capital)."""
    out = []
    for m in WORD_RE.finditer(text):
        w, s = m.group(), 0
        for i in range(1, len(w)):
            if w[i] in UPPER and w[i - 1].isalpha() and w[i - 1] not in UPPER:
                out.append(w[s:i])
                s = i
        out.append(w[s:])
    return out


def main(pool_path: str, fixtures: str, out_path: str) -> None:
    pool = [line.strip() for line in Path(pool_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [json.loads(line) for line in (Path(fixtures) / "espeak-plan.jsonl").read_text(encoding="utf-8").splitlines()]
    used = {r["text"] for r in rows}
    train = set()
    for r in rows:
        if r["split"] != "train":
            continue
        ks = [key(w) for w in words(r["text"])]
        merged = []
        for k in ks:
            if merged and (merged[-1], k) in JOINS:
                merged[-1] += k
            else:
                merged.append(k)
        train.update(merged)
    vocab = set()
    for text in pool:
        if text in used:
            continue
        for w in words(text):
            k = key(w)
            if PLAIN.match(k) and len(k) >= 2 and any(c in VOWELS for c in k) and k not in train:
                vocab.add(k)
    ordered = sorted(vocab)
    clips = [
        {"id": "wl%04d" % (i // 40), "split": "wordlist", "text": ", ".join(ordered[i : i + 40]) + "."}
        for i in range(0, len(ordered), 40)
    ]
    doc = {
        "source": "Common Voice tr sentence-collector (CC0), words of sentences outside the v3 fixtures, not in v3 train",
        "clips": clips,
    }
    # LF line endings on every platform, so the committed file reproduces byte for byte
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(doc, ensure_ascii=False, indent=0))
    print(len(ordered), "words,", len(clips), "texts")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
