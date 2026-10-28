"""The Python bindings give the golden vectors of `crates/akana-core/tests/data/embeddings/golden.jsonl`
bit for bit, as the Rust test does: a change on the binding side that moves vectors fails here,
and so does a wheel built with another toolchain or profile, which the release workflow runs
this test against (review 3 MSTAB3-06)."""

import json
import os
import struct

import akana

GOLDEN = os.path.join(
    os.path.dirname(__file__), "..", "crates", "akana-core", "tests", "data", "embeddings", "golden.jsonl"
)


def test_vectors_equal_the_golden_ones():
    with open(GOLDEN, encoding="utf-8") as f:
        header = json.loads(f.readline())
        rows = [json.loads(line) for line in f]
    assert akana.encoder_info()["sparse_spec"] == header["sparse_spec"]
    texts = [row["text"] for row in rows]
    hybrids = akana.embed_batch_hybrid(texts)
    changed = []
    for row, hybrid in zip(rows, hybrids):
        s = akana.sparse_embed(row["text"])
        sparse = [
            [i, format(struct.unpack("<I", struct.pack("<f", v))[0], "08x"), t]
            for i, v, t in zip(s["indices"], s["values"], s["terms"])
        ]
        if sparse != row["sparse"] or bytes(hybrid).hex() != row["hybrid"]:
            changed.append(row["text"][:60])
    assert not changed, f"{len(changed)} of {len(rows)} golden texts give other vectors: {changed[:5]}"
