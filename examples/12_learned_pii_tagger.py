#!/usr/bin/env python3
"""Example 12: the learned PII tagger, next to the rule engine.

akana's learned PII tagger is two Turkish token classifiers (ELECTRA-small for the fast mode,
BERTurk averaged with it for the accurate mode) that run in Rust on the CPU. Its models come
from the Hugging Face Hub repository altaidevorg/akana-pii-tagger (CC BY-SA 4.0): the first run
downloads the fast model (57 MB) into akana's cache, and later runs work offline. The tagger is
not in the Python module yet; Python reaches it through the `akana` command-line tool built
with the `pii-tagger` feature:

    cargo install --path crates/akana-cli --features pii-tagger

Set AKANA_BIN to use an akana binary that is not on PATH, and AKANA_CACHE_DIR to choose where
the models are cached. This example:
1. Tags several texts in one call (`akana pii-tag --jsonl`) and turns the CLI's byte offsets
   into Python string offsets.
2. Masks with the tagger's spans and compares the result with the rule engine
   (`akana.pii_mask`).
3. Shows what the policy leaves unmasked: call-centre numbers and role mailboxes.
4. Compares the default and the strict operating point.
5. With `--accurate`, runs the accurate mode too (443 MB more to download, about five times
   slower).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import akana

TEXTS = [
    "Ayşe Yılmaz 12 Mart 2024'te 0532 123 45 67 numaralı telefondan aradı.",
    "Davacı Zeynep Kara'nın 14.09.2021 tarihli dilekçesi ve 2021/345 E. sayılı dosya incelendi.",
    "yarın akşam mehmetle kadıköyde buluşcaz, numaram 0533 444 55 66",
    "Bana @ayse_dev hesabından ya da ayse.kaya@ornekmail.com adresinden ulaşabilirsiniz.",
    "TC kimlik numaram 10000000146, internet bankacılığı şifrem Kedi_2024!",
    "Kemal Bey'le geçen hafta Atatürk Caddesi No: 12 D: 4 Çankaya/Ankara adresinde görüştük.",
]
PUBLIC = "Kargo sorunları için 444 0 123'ü arayın ya da destek@firma.com.tr adresine yazın."
NO_YEAR = "Ayşe Yılmaz 12 Mart'ta aradı."


def find_akana() -> str:
    """The akana binary, checked to have the tagger."""
    path = os.environ.get("AKANA_BIN") or shutil.which("akana")
    if not path:
        sys.exit("akana is not on PATH. Build it with the tagger:\n"
                 "    cargo install --path crates/akana-cli --features pii-tagger")
    probe = subprocess.run([path, "pii-tag", "--help"], capture_output=True, text=True)
    if probe.returncode != 0:
        sys.exit(f"{path} has no pii-tag command: rebuild it with --features pii-tagger")
    return path


def char_offsets(text: str, start: int, end: int) -> tuple[int, int]:
    """The CLI reports UTF-8 byte offsets; Python strings index by character."""
    data = text.encode("utf-8")
    return len(data[:start].decode("utf-8")), len(data[:end].decode("utf-8"))


def tag(akana_bin: str, texts: list[str], *flags: str) -> list[list[tuple[int, int, str]]]:
    """The tagger's spans (start, end, label) for every text, in one call of the CLI."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "texts.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        run = subprocess.run([akana_bin, "pii-tag", "--jsonl", path, *flags],
                             capture_output=True, encoding="utf-8")
    if run.returncode != 0:
        sys.exit(f"akana pii-tag failed: {run.stderr.strip()}")
    rows = [json.loads(line) for line in run.stdout.splitlines() if line.strip()]
    return [[(*char_offsets(text, s, e), label) for s, e, label in row["spans"]]
            for text, row in zip(texts, rows)]


def mask(text: str, spans: list[tuple[int, int, str]]) -> str:
    for start, end, label in sorted(spans, reverse=True):
        text = text[:start] + f"[{label}]" + text[end:]
    return text


def show(text: str, spans: list[tuple[int, int, str]]) -> None:
    print(f"  Text:   {text}")
    print(f"  Spans:  {[(text[s:e], label) for s, e, label in spans]}")
    print(f"  Masked: {mask(text, spans)}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    akana_bin = find_akana()
    print("=" * 75)
    print(" [AKANA] LEARNED PII TAGGER (altaidevorg/akana-pii-tagger)")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # 1. The fast mode on several texts, in one call
    # -------------------------------------------------------------------------
    print("\n--- 1. Fast mode: names, dates, phones, IDs, secrets, addresses ---")
    fast = tag(akana_bin, TEXTS)
    for text, spans in zip(TEXTS, fast):
        show(text, spans)
        print()

    # -------------------------------------------------------------------------
    # 2. The same texts with the rule engine
    # -------------------------------------------------------------------------
    print("--- 2. The rule engine on the same texts (akana.pii_mask, mode='tag') ---")
    for text, spans in zip(TEXTS, fast):
        print(f"  Tagger: {mask(text, spans)}")
        print(f"  Rules:  {akana.pii_mask(text, mode='tag')['masked_text']}")
        print()

    # -------------------------------------------------------------------------
    # 3. Public numbers and role mailboxes stay unmasked
    # -------------------------------------------------------------------------
    print("--- 3. The policy: call-centre numbers and role mailboxes stay ---")
    show(PUBLIC, tag(akana_bin, [PUBLIC])[0])

    # -------------------------------------------------------------------------
    # 4. The strict operating point
    # -------------------------------------------------------------------------
    print("\n--- 4. Default and strict operating point ---")
    print("  The strict point raises person, username and date tags, calibrated so that on")
    print("  average at most 4% of a text's personal characters stay in clear.")
    print(f"  Default: {mask(NO_YEAR, tag(akana_bin, [NO_YEAR])[0])}")
    print(f"  Strict:  {mask(NO_YEAR, tag(akana_bin, [NO_YEAR], '--strict')[0])}")

    # -------------------------------------------------------------------------
    # 5. The accurate mode (optional: a 443 MB download on first use)
    # -------------------------------------------------------------------------
    if "--accurate" in sys.argv[1:]:
        print("\n--- 5. Accurate mode (ELECTRA-small and BERTurk averaged) ---")
        accurate = tag(akana_bin, TEXTS + [NO_YEAR], "--accurate")
        for text, spans in zip(TEXTS + [NO_YEAR], accurate):
            print(f"  {mask(text, spans)}")
    else:
        print("\n(Run with --accurate to compare the accurate mode: 443 MB more on first use.)")


if __name__ == "__main__":
    main()
