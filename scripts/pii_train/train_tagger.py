"""Fine-tunes a Turkish encoder as a PII token classifier (data.py's labels, BIO tags).

    python scripts/pii_train/train_tagger.py --base dbmdz/electra-small-turkish-cased-discriminator --out .cache/pii_models/electra-small-v0

Runs within guard.py's limits: four CPU threads, a pause whenever memory is short or the GPU is hot.
Each epoch is saved and scored on the dev sets with the strict scorer; the best epoch is copied to
<out>/best.

With --teacher (a tagger with the same vocabulary, such as BERTurk for ELECTRA-small) the student
also learns the teacher's tag distributions, computed on the fly: a mix of cross-entropy on the
labels and temperature-scaled KL on every token, and KL alone on --unlabelled units. Several
teachers are an ensemble: the student learns the mean of their tempered distributions.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guard  # noqa: E402

guard.limit_threads(4)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

import data  # noqa: E402
import tagger  # noqa: E402

sys.path.insert(0, str(HERE.parent / "pii_eval"))
import datasets as evsets  # noqa: E402
import score  # noqa: E402
import systems  # noqa: E402


def encode(tok, rows, max_length):
    # numpy arrays, not lists of Python ints: ~0.6 KB an example instead of ~3.5 KB, which matters
    # with 125k examples on a machine with little free memory
    out = []
    for r in rows:
        enc = tok(tagger.normalize(r["text"]), return_offsets_mapping=True, truncation=True, max_length=max_length)
        if r["spans"] is None:  # unlabelled: -1 on real tokens (the teacher's distribution only)
            tags = [-1 if a != b else -100 for a, b in enc["offset_mapping"]]
        else:
            tags = tagger.token_tags(enc["offset_mapping"], r["spans"])
        out.append((np.asarray(enc["input_ids"], dtype=np.int32), np.asarray(tags, dtype=np.int16)))
    return out


def teacher_logp(teacher_logits, temp):
    """Log of the teachers' mean tag distribution at temperature `temp`."""
    logp = torch.stack([F.log_softmax(z.float() / temp, -1) for z in teacher_logits])
    return torch.logsumexp(logp, 0) - np.log(len(teacher_logits))


def loss_of(logits, lab, teacher=None, kd_weight=0.5, temp=2.0):
    """Cross-entropy on labelled tokens; with a teacher (teacher_logp at `temp`), mixed with
    T^2-scaled KL on every real token."""
    logits = logits.float()
    hard = lab >= 0
    ce = F.cross_entropy(logits[hard], lab[hard]) if hard.any() else logits.sum() * 0
    if teacher is None:
        return ce
    real = lab != -100
    t = teacher[real]
    s = F.log_softmax(logits[real] / temp, -1)
    kd = F.kl_div(s, t, log_target=True, reduction="batchmean") * temp * temp
    return (1 - kd_weight) * ce + kd_weight * kd


def batches(encoded, size, rng):
    order = list(range(len(encoded)))
    rng.shuffle(order)
    # similar lengths together: less padding, less work
    order.sort(key=lambda i: len(encoded[i][0]) // 16 + rng.random())
    chunks = [order[i:i + size] for i in range(0, len(order), size)]
    rng.shuffle(chunks)
    return chunks


def evaluate(model_dir, dev_sets, log):
    t = tagger.Tagger(model_dir, batch_size=16)
    results = {}
    for name, rows in dev_sets.items():
        preds = t.predict([r["text"] for r in rows])
        ev_rows = [evsets.Row(r["text"], [(s, e, lab, lab) for s, e, lab in r["spans"]]) for r in rows]
        # scored as deployed: call-centre numbers and role mailboxes are left to the policy
        system = lambda text, _p=iter(preds): [(s, e, lab, lab) for s, e, lab in next(_p) if not systems.public_value(text[s:e], lab)]
        per_row, by_type, _, _ = score.evaluate(ev_rows, system)
        s = score.summary(score.total_of(per_row))
        results[name] = s
        log(f"  {name:18s} {score.fmt(s)}")
    del t
    torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="dbmdz/electra-small-turkish-cased-discriminator")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--btx24", type=int, default=40000)
    ap.add_argument("--scenario", type=int, default=20000)
    ap.add_argument("--real", nargs="*", default=[], help="teacher label files (labels/<name>.jsonl) to train on")
    ap.add_argument("--no-wikiner", action="store_true")
    ap.add_argument("--hard", nargs="*", default=[], help="hard-case files (hardcases/<name>.jsonl) to train on")
    ap.add_argument("--hard-repeat", type=int, default=1)
    ap.add_argument("--real-repeat", type=int, default=1, help="copies of the teacher-labelled real text")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-seed", type=int,
                    help="seed of the classification head's initialisation (default: --seed's state); the same value "
                         "across runs gives them one starting point, so that their weights can be averaged (soup.py)")
    ap.add_argument("--teacher", nargs="*", default=[],
                    help="trained tagger directories to distil from (same vocabulary); several are averaged")
    ap.add_argument("--kd-weight", type=float, default=0.5)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--unlabelled", nargs="*", default=[], help="pools in .cache/pii_data/real for the teacher to label")
    ap.add_argument("--unlabelled-limit", type=int, help="units taken from each unlabelled pool")
    ap.add_argument("--permissive", action="store_true", help="leave out real text from sources of unclear licence")
    # the start check before the data and the GPU context load; lower it only when the page file can
    # grow (Win32_PageFileSetting MaximumSize above the current size)
    ap.add_argument("--start-ram", type=float, default=1.0, help="free RAM (GB) to wait for before loading")
    ap.add_argument("--start-commit", type=float, default=1.5, help="free commit (GB) to wait for before loading")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    # torch is already loaded here (~0.9 GB of commit); the GPU context and the data need ~0.8 GB more
    guard.wait(min_ram_gb=args.start_ram, min_commit_gb=args.start_commit, log=log)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rows = data.training_mix(seed=args.seed, btx24=args.btx24, scenario=args.scenario, real_runs=tuple(args.real),
                             wikiner=not args.no_wikiner, hard_runs=tuple(args.hard), hard_repeat=args.hard_repeat,
                             skip_domains=data.UNCLEAR_LICENCE if args.permissive else (), real_repeat=args.real_repeat)
    if args.unlabelled:
        if not args.teacher:
            ap.error("--unlabelled needs --teacher")
        rows += data.unlabelled(tuple(args.unlabelled), args.unlabelled_limit, args.seed,
                                data.UNCLEAR_LICENCE if args.permissive else ())
        rng.shuffle(rows)
    dev = data.dev_sets()
    log(f"training rows {len(rows)}; dev sets {', '.join(f'{k} ({len(v)})' for k, v in dev.items())}")
    tok = AutoTokenizer.from_pretrained(args.base)
    encoded = encode(tok, rows, args.max_length)
    del rows
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.init_seed is not None:
        torch.manual_seed(args.init_seed)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base, num_labels=len(tagger.TAGS), id2label=dict(enumerate(tagger.TAGS)), label2id=tagger.TAG_ID).to(device)
    if args.init_seed is not None:  # dropout and the rest of the run follow --seed
        torch.manual_seed(args.seed)
    teachers = []
    for path in args.teacher:
        teachers.append(AutoModelForTokenClassification.from_pretrained(path).to(device).eval())
        assert teachers[-1].config.id2label == model.config.id2label, f"{path}: teacher and student tag sets differ"
        assert AutoTokenizer.from_pretrained(path).get_vocab() == tok.get_vocab(), f"{path}: vocabularies differ"
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = sum(len(batches(encoded, args.batch, random.Random(e))) for e in range(args.epochs))
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    best, step = None, 0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for k, chunk in enumerate(batches(encoded, args.batch, random.Random(epoch))):
            if step % 50 == 0:
                guard.wait(min_ram_gb=0.4, min_commit_gb=0.5, log=log)
            width = max(len(encoded[i][0]) for i in chunk)
            ids = torch.full((len(chunk), width), tok.pad_token_id, dtype=torch.long)
            lab = torch.full((len(chunk), width), -100, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for j, i in enumerate(chunk):
                x, y = encoded[i]
                ids[j, :len(x)] = torch.from_numpy(x)
                lab[j, :len(y)] = torch.from_numpy(y)
                mask[j, :len(x)] = 1
            ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
            with torch.autocast(device, enabled=device == "cuda"):
                target = None
                if teachers:
                    with torch.no_grad():
                        target = teacher_logp([t(input_ids=ids, attention_mask=mask).logits for t in teachers], args.kd_temp)
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_of(logits, lab, target, args.kd_weight, args.kd_temp)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            step += 1
            if step % 200 == 0:
                log(f"epoch {epoch + 1} step {step}/{steps} loss {total / (k + 1):.4f} ({guard.headroom()[1]})")
        ckpt = out / f"epoch{epoch + 1}"
        model.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)
        log(f"epoch {epoch + 1} done, mean loss {total / (k + 1):.4f}; dev:")
        model.cpu()
        torch.cuda.empty_cache()
        res = evaluate(str(ckpt), dev, log)
        model.to(device)
        # chosen on real text and on yusuf-said's dev part together: names in the wild, and no
        # false alarms on look-alikes (a model can buy one with the other)
        keys = [k for k in ("gold-dev", "yusuf-dev") if k in res] or ["belgin-validation"]
        value = sum(res[k]["exact_f1"] for k in keys) / len(keys)
        if best is None or value > best[1]:
            best = (epoch + 1, value, res)
    shutil.copytree(out / f"epoch{best[0]}", out / "best", dirs_exist_ok=True)
    json.dump({"args": vars(args), "best_epoch": best[0], "dev": best[2]}, open(out / "best" / "training.json", "w"), indent=1)
    log(f"best epoch {best[0]} ({best[1]:.3f} mean exact F1 of the selection sets)")


if __name__ == "__main__":
    main()
