"""Stages akana's PII tagger for the Hugging Face Hub and, with --upload, publishes it: the small
model (the fast mode) under electra-small/ and BERTurk (averaged with it in the accurate mode) under
berturk/, each with its ONNX export (scripts/pii_train/export_onnx.py, what akana's Rust tagger runs),
its safetensors weights, config and tokenizer (for transformers), plus the model card, the text of
its licence (CC BY-SA 4.0: the weights learnt from Wikipedia text, WikiNER and MusteriYorumlari) and
the attribution files the licence asks for. Prints what akana-core's pii::tagger::hub pins: the
commit and each file's size and SHA-256, as the Hub reports them after the upload.

    python scripts/pii_train/publish_hf.py --small .cache/pii_models/electra-small-p3/best \\
        --big .cache/pii_models/berturk-p3/epoch3 --card <model card README.md> \\
        --attribution <DATA_SOURCES.md> .cache/pii_data/real/p2/wikipedia_attribution.tsv [--upload]

The upload reads the token from HF_TOKEN (or huggingface-cli login); it is never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = "altaidevorg/akana-pii-tagger"
FILES = ("model.onnx", "model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "vocab.txt")
REQUIRED = ("model.onnx", "model.safetensors", "config.json", "tokenizer.json")
# what the Rust tagger downloads (hub.rs's SMALL and BIG)
RUST = ("electra-small/model.onnx", "electra-small/tokenizer.json", "berturk/model.onnx", "berturk/tokenizer.json")
LICENCE_URL = "https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stage(args) -> Path:
    out = Path(args.stage)
    if out.exists():
        shutil.rmtree(out)
    for name, src in (("electra-small", Path(args.small)), ("berturk", Path(args.big))):
        missing = [f for f in REQUIRED if not (src / f).is_file()]
        if missing:
            sys.exit(f"{src} lacks {', '.join(missing)}")
        (out / name).mkdir(parents=True)
        for f in FILES:
            if (src / f).is_file():
                shutil.copy2(src / f, out / name / f)
    shutil.copy2(args.card, out / "README.md")
    for extra in args.attribution:
        shutil.copy2(extra, out / Path(extra).name)
    licence = Path(args.licence) if args.licence else None
    if licence and licence.is_file():
        shutil.copy2(licence, out / "LICENSE")
    else:  # creativecommons.org refuses urllib's default User-Agent (HTTP 403)
        request = urllib.request.Request(LICENCE_URL, headers={"User-Agent": "akana-publish-hf (+https://github.com/altaidevorg/akana)"})
        with urllib.request.urlopen(request, timeout=60) as r:
            (out / "LICENSE").write_bytes(r.read())
    for path in sorted(out.rglob("*")):
        if path.is_file():
            print(f"{path.relative_to(out).as_posix():40s} {path.stat().st_size:>12,d}  {sha256(path)}")
    return out


def upload(out: Path, message: str):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
    info = api.upload_folder(repo_id=REPO, repo_type="model", folder_path=str(out), commit_message=message)
    revision = info.oid
    print(f"\nuploaded {REPO} at {revision}")
    remote = {p.path: p for p in api.get_paths_info(REPO, list(RUST), revision=revision, expand=True)}
    print("\nfor crates/akana-core/src/pii/tagger/hub.rs:")
    print(f'pub const REVISION: &str = "{revision}";')
    ok = True
    for path in RUST:
        local = out / path
        size, digest = local.stat().st_size, sha256(local)
        r = remote.get(path)
        r_size = getattr(r, "size", None)
        r_sha = getattr(getattr(r, "lfs", None), "sha256", None)
        if r is None or r_size != size or (r_sha is not None and r_sha != digest):
            ok = False
            print(f"MISMATCH {path}: local {size} {digest}, Hub {r_size} {r_sha}")
        print(f'    Asset {{ path: "{path}", size: {size}, sha256: "{digest}" }},')
    if not ok:
        sys.exit("the Hub's files differ from the staged ones")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", required=True)
    ap.add_argument("--big", required=True)
    ap.add_argument("--card", required=True, help="the model card (README.md with its YAML header)")
    ap.add_argument("--licence", help="the CC BY-SA 4.0 legal code (default: fetched from creativecommons.org)")
    ap.add_argument("--attribution", nargs="*", default=[], help="files staged beside the card (sources and their attribution)")
    ap.add_argument("--stage", default=str(ROOT / ".cache" / "pii_release" / "akana-pii-tagger"))
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--message", default="akana PII tagger")
    args = ap.parse_args()
    out = stage(args)
    if args.upload:
        upload(out, args.message)


if __name__ == "__main__":
    main()
