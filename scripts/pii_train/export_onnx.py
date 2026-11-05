"""Exports a trained PII tagger (a tagger.Tagger model directory) to ONNX for akana's Rust tagger
(akana-core's pii::tagger, which runs it with RTen): model.onnx beside the directory's tokenizer.json,
with inputs input_ids and attention_mask (int64, batch x sequence, both dynamic) and output logits
(batch x sequence x 19 tags). The export is checked against PyTorch on a few dev texts with
onnxruntime when it is installed.

    python scripts/pii_train/export_onnx.py .cache/pii_models/electra-small-p3/best
    python scripts/pii_train/export_onnx.py .cache/pii_models/berturk-p3/epoch3

Needs the onnx and onnxscript packages (torch's exporter).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    import numpy as np
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    class Logits(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    model_dir = Path(sys.argv[1])
    out = model_dir / "model.onnx"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = Logits(AutoModelForTokenClassification.from_pretrained(model_dir).eval())
    enc = tok(["Ahmet Yılmaz'ın telefonu 0532 111 22 33.", "Kısa bir metin."], padding=True, return_tensors="pt")
    batch, seq = torch.export.Dim("batch", min=1, max=64), torch.export.Dim("sequence", min=2, max=512)
    torch.onnx.export(model, (enc["input_ids"], enc["attention_mask"]), str(out), input_names=["input_ids", "attention_mask"],
                      output_names=["logits"], dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
                      opset_version=18, dynamo=True, external_data=False)
    print(f"{out}: {out.stat().st_size / 1e6:.1f} MB")
    try:
        import onnxruntime as ort
    except ImportError:
        return
    texts = [json.loads(line)["text"] for line in open(ROOT / ".cache/pii_data/real/gold_dev.jsonl", encoding="utf-8")][:8]
    enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        ref = model(enc["input_ids"], enc["attention_mask"]).numpy()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()})[0]
    mask = enc["attention_mask"].numpy().astype(bool)
    print(f"onnxruntime against PyTorch on {len(texts)} dev texts: max |logit difference| "
          f"{np.abs(ref - got)[mask].max():.1e}, same best tag on {(ref.argmax(-1) == got.argmax(-1))[mask].mean():.4f}")


if __name__ == "__main__":
    main()
