"""Fetches the external morphology sources into the git-ignored `.cache/morph_sources/`, pinned by
the committed manifest `scripts/morph_sources.json` (URL, release or commit or date, SHA-256 and
licence of every file).

    python scripts/fetch_morph_sources.py            # download what is missing, verify all
    python scripts/fetch_morph_sources.py --check    # verify the cache, no network
    python scripts/fetch_morph_sources.py --pin      # resolve the pins below, rewrite the manifest
    python scripts/fetch_morph_sources.py --only UD_Turkish   # only sources whose name starts so

Sources (the user approved each on 2026-10-26):
- UD Turkish BOUN, Kenet, Penn, Tourism, Atis, FrameNet (CC BY-SA 4.0) at UD release r2.18: their
  train and dev splits may train what akana ships; their test splits are for evaluation only;
- UD Turkish IMST (CC BY-NC-SA 3.0) at r2.18: evaluation only, never in anything that ships;
- Google Research turkish-morphology lexicon (Apache-2.0), `src/analyzer/lexicon/base/*.tsv`;
- TRmorph lexicon (MIT), `lexicon/`;
- the Turkish Wiktionary extract of kaikki.org (Wiktionary content: CC BY-SA 4.0 and GFDL), an
  optional add-on kept out of the core lexicon.

Each file's `use` in the manifest says what it may feed: `train` (estimation of shipped data),
`eval` (evaluation only), `lexicon` (lexicon additions, with attribution in THIRD_PARTY_NOTICES),
`addon` (an optional add-on dictionary), `license` (the source's licence text).
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "morph_sources.json"
CACHE = HERE.parent / ".cache" / "morph_sources"
AGENT = {"User-Agent": "akana-fetch-morph-sources"}

UD_RELEASE = "r2.18"
UD_TREEBANKS = {
    "BOUN": "CC BY-SA 4.0",
    "Kenet": "CC BY-SA 4.0",
    "Penn": "CC BY-SA 4.0",
    "Tourism": "CC BY-SA 4.0",
    "Atis": "CC BY-SA 4.0",
    "FrameNet": "CC BY-SA 4.0",
    "IMST": "CC BY-NC-SA 3.0",
}
GOOGLE = ("google-research/turkish-morphology", "83d4b967dc09ff947d15a4b20a81dd328c2c4f81", "src/analyzer/lexicon/base")
TRMORPH = ("coltekin/TRmorph", "fbceb1133fbdc51bd63693d448b7cb75672e2256", "lexicon")
KAIKKI = "https://kaikki.org/trwiktionary/raw-wiktextract-data.jsonl.gz"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=300) as r:
        return r.read()


def github_files(repo: str, ref: str, path: str = "") -> list[str]:
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"
    return [f["path"] for f in json.loads(fetch(url)) if f["type"] == "file"]


def raw(repo: str, ref: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"


def pin() -> dict:
    """The manifest for the pins above: every file's URL, local path and use (hashes are added
    as the files are downloaded)."""
    sources = []
    for name, licence in UD_TREEBANKS.items():
        repo = f"UniversalDependencies/UD_Turkish-{name}"
        files = []
        for path in github_files(repo, UD_RELEASE):
            if path.startswith("LICENSE"):
                use = "license"
            elif not path.endswith(".conllu"):
                continue
            elif name == "IMST" or path.endswith("-test.conllu"):
                use = "eval"
            else:
                use = "train"
            files.append({"path": f"ud/UD_Turkish-{name}/{path}", "url": raw(repo, UD_RELEASE, path), "use": use})
        sources.append({"name": f"UD_Turkish-{name}", "version": UD_RELEASE, "license_expected": licence, "files": files})
    for repo, ref, folder in (GOOGLE, TRMORPH):
        short = repo.split("/")[1]
        files = [{"path": f"{short}/LICENSE", "url": raw(repo, ref, "LICENSE"), "use": "license"}]
        for path in github_files(repo, ref, folder):
            files.append({"path": f"{short}/{path}", "url": raw(repo, ref, path), "use": "lexicon"})
        licence = "Apache-2.0" if repo == GOOGLE[0] else "MIT"
        sources.append({"name": short, "version": ref, "license_expected": licence, "files": files})
    sources.append({
        "name": "kaikki-trwiktionary",
        "version": "extract of the trwiktionary dump of 2026-10-02, made 2026-10-25",
        "license_expected": "CC BY-SA 4.0 + GFDL",
        "files": [{"path": "kaikki/trwiktionary-raw-wiktextract-data.jsonl.gz", "url": KAIKKI, "use": "addon"}],
    })
    return {"cache": ".cache/morph_sources", "sources": sources}


def licence_found(path: Path) -> str:
    """The licence a licence file states, as a short name."""
    if not path.exists():
        return "missing"
    text = " ".join(path.read_text(encoding="utf-8", errors="replace").split())
    if "Attribution-ShareAlike 4.0" in text or "by-sa/4.0" in text:
        return "CC BY-SA 4.0"
    if "BY-NC-SA 3.0" in text or "by-nc-sa/3.0" in text:
        return "CC BY-NC-SA 3.0"
    if "BY-NC-SA 4.0" in text or "by-nc-sa/4.0" in text:
        return "CC BY-NC-SA 4.0"
    if "Apache License" in text and "Version 2.0" in text:
        return "Apache-2.0"
    if "Permission is hereby granted, free of charge" in text:
        return "MIT"
    return "unrecognised"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify the cache without network")
    parser.add_argument("--pin", action="store_true", help="resolve the pins and rewrite the manifest")
    parser.add_argument("--only", metavar="PREFIX", help="only the sources whose name starts with PREFIX")
    args = parser.parse_args()
    if args.pin and args.only:
        parser.error("--pin rewrites the whole manifest: it takes no --only")

    manifest = pin() if args.pin else json.loads(MANIFEST.read_text(encoding="utf-8"))
    if args.only:
        manifest["sources"] = [s for s in manifest["sources"] if s["name"].startswith(args.only)]
        if not manifest["sources"]:
            print(f"no source's name starts with {args.only!r}")
            return 1
    bad = 0
    for source in manifest["sources"]:
        for f in source["files"]:
            local = CACHE / f["path"]
            if not local.exists():
                if args.check:
                    print(f"missing  {f['path']}")
                    bad += 1
                    continue
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_bytes(fetch(f["url"]))
            digest = sha256(local)
            if args.pin:
                f["sha256"], f["size"] = digest, local.stat().st_size
            elif digest != f["sha256"]:
                print(f"MISMATCH {f['path']}: {digest} (manifest {f['sha256']})")
                bad += 1
    # The licence each source's own licence file states must be the one approved for it: a
    # changed licence stops the intake until someone decides about it.
    for source in manifest["sources"]:
        licence_files = [CACHE / f["path"] for f in source["files"] if f["use"] == "license"]
        found = licence_found(licence_files[0]) if licence_files else "no licence file (Wiktionary content)"
        source["license_found"] = found
        if licence_files and found != source["license_expected"]:
            print(f"LICENCE {source['name']}: the file says {found!r}, approved {source['license_expected']!r}")
            bad += 1
    if args.pin:
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {MANIFEST}")
    files = sum(len(s["files"]) for s in manifest["sources"])
    print(f"{files - bad} of {files} files verified in {CACHE}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
