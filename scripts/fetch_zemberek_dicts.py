"""Checks or re-creates the vendored Zemberek dictionaries in scripts/lexicon/zemberek/.

The six default dictionaries of Zemberek's TurkishDictionaryLoader and its Turkish letter names
are vendored unmodified from the Zeyrek 0.1.3 wheel (see scripts/lexicon/zemberek/NOTICE.md,
which lists their SHA-256). `build_full_lexicon.py` builds akana's lexicon from them and runs
the check first.

    python scripts/fetch_zemberek_dicts.py            # verify the vendored files (default)
    python scripts/fetch_zemberek_dicts.py --fetch    # download the pinned wheel, verify it,
                                                      # extract and verify the files
    python scripts/fetch_zemberek_dicts.py --fetch --wheel zeyrek-0.1.3-py2.py3-none-any.whl
"""

import argparse
import hashlib
import io
import os
import re
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = os.path.join(HERE, "lexicon", "zemberek")
NOTICE = os.path.join(SOURCES, "NOTICE.md")

# The PyPI release the files come from (https://pypi.org/project/zeyrek/0.1.3/).
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/7f/5b/"
    "76970fab035d2e2649ba06af037c394b9690d0e07a8faf53817ccccb3951/"
    "zeyrek-0.1.3-py2.py3-none-any.whl"
)
WHEEL_SHA256 = "23649bb49322a52d1e94959029b047fa4037bc540762819feb1096aa976b25b5"

# Vendored name -> path inside the wheel.
FILES = {
    "abbreviations.dict": "zeyrek/resources/tr/abbreviations.dict",
    "master-dictionary.dict": "zeyrek/resources/tr/master-dictionary.dict",
    "non-tdk.dict": "zeyrek/resources/tr/non-tdk.dict",
    "person-names.dict": "zeyrek/resources/tr/person-names.dict",
    "proper-from-corpus.dict": "zeyrek/resources/tr/proper-from-corpus.dict",
    "proper.dict": "zeyrek/resources/tr/proper.dict",
    "turkish-letter-names.txt": "zeyrek/resources/tr/phonetics/turkish-letter-names.txt",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def pinned_checksums():
    """The `| file | sha256 |` rows of NOTICE.md."""
    rows = re.findall(r"^\|\s*([\w.-]+)\s*\|\s*([0-9a-f]{64})\s*\|", open(NOTICE, encoding="utf-8").read(), re.M)
    checksums = dict(rows)
    missing = set(FILES) - set(checksums)
    if missing:
        raise SystemExit(f"NOTICE.md lists no checksum for {sorted(missing)}")
    return checksums


def check(directory=SOURCES):
    """Raises SystemExit naming every file whose content is not the pinned one."""
    checksums = pinned_checksums()
    bad = []
    for name in FILES:
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            bad.append(f"{name}: missing")
        elif sha256(open(path, "rb").read()) != checksums[name]:
            bad.append(f"{name}: SHA-256 differs from NOTICE.md")
    if bad:
        raise SystemExit("Zemberek dictionaries do not match the pinned release:\n  " + "\n  ".join(bad)
                         + "\nRe-create them with: python scripts/fetch_zemberek_dicts.py --fetch")


def fetch(wheel_path=None, directory=SOURCES):
    if wheel_path:
        data = open(wheel_path, "rb").read()
    else:
        with urllib.request.urlopen(WHEEL_URL, timeout=60) as response:
            data = response.read()
    if sha256(data) != WHEEL_SHA256:
        raise SystemExit(f"the wheel's SHA-256 is {sha256(data)}, expected {WHEEL_SHA256}")
    checksums = pinned_checksums()
    os.makedirs(directory, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as wheel:
        for name, member in FILES.items():
            content = wheel.read(member)
            if sha256(content) != checksums[name]:
                raise SystemExit(f"{member} in the wheel does not match NOTICE.md")
            with open(os.path.join(directory, name), "wb") as f:
                f.write(content)
    check(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fetch", action="store_true", help="re-create the files from the pinned wheel")
    parser.add_argument("--wheel", help="a local copy of the wheel instead of downloading it")
    parser.add_argument("--dest", default=SOURCES, help="directory to check or write (default: the vendored copy)")
    args = parser.parse_args()
    if args.fetch:
        fetch(args.wheel, args.dest)
    else:
        check(args.dest)
    print(f"{len(FILES)} Zemberek files match NOTICE.md in {args.dest}")


if __name__ == "__main__":
    sys.exit(main())
