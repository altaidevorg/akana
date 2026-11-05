"""Averages the weights of taggers fine-tuned from one starting point (a uniform model soup): the same
pretrained encoder and, with train_tagger.py --init-seed, the same classification head. The runs differ
in data sampling, order and dropout; their average is one model, so it costs nothing at inference.

    python scripts/pii_train/soup.py OUT_DIR MODEL_DIR [MODEL_DIR ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard  # noqa: E402

guard.limit_threads(4)

import torch  # noqa: E402
from transformers import AutoModelForTokenClassification, AutoTokenizer  # noqa: E402


def main():
    out, dirs = Path(sys.argv[1]), sys.argv[2:]
    if len(dirs) < 2:
        sys.exit("give two or more model directories")
    models = [AutoModelForTokenClassification.from_pretrained(d) for d in dirs]
    vocab = AutoTokenizer.from_pretrained(dirs[0]).get_vocab()
    for d, m in zip(dirs[1:], models[1:]):
        assert m.config.id2label == models[0].config.id2label, f"{d}: another tag set"
        assert AutoTokenizer.from_pretrained(d).get_vocab() == vocab, f"{d}: another vocabulary"
    states = [m.state_dict() for m in models]
    soup = {}
    for name, value in states[0].items():
        if value.dtype.is_floating_point:
            soup[name] = (sum(s[name].float() for s in states) / len(states)).to(value.dtype)
        else:  # integer buffers (position ids) are the same in every run
            assert all(torch.equal(s[name], value) for s in states[1:]), f"{name} differs between runs"
            soup[name] = value
    models[0].load_state_dict(soup)
    out.mkdir(parents=True, exist_ok=True)
    models[0].save_pretrained(out)
    AutoTokenizer.from_pretrained(dirs[0]).save_pretrained(out)
    json.dump({"soup_of": dirs}, open(out / "soup.json", "w", encoding="utf-8"), indent=1)
    print(f"{len(dirs)} runs averaged -> {out}")


if __name__ == "__main__":
    main()
