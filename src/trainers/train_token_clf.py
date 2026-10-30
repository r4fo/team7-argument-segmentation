"""Fine-tune a transformer encoder for discourse-element tagging (benchmark rows 4-6).

    encoder (BERT / DeBERTa ...) -> linear head -> softmax  (--decoder softmax)
                                                -> CRF      (--decoder crf)

Long essays are read as overlapping windows (default 512 subwords, 128
shared). Each word is labelled from its first subword; the other subwords
are ignored. Emissions are gathered at those first subwords, so the softmax
loss and the CRF both work on the word sequence. CRF decoding is Viterbi with
the BIO constraints (I-X only after B-X/I-X); softmax decoding is a plain
argmax, so its illegal transitions can be counted.

The best epoch on val (overlap span macro-F1) is kept and scored on
test_seen and test_unseen with src/evaluate.py.

Usage (Colab, T4 GPU)
---------------------
    python src/trainers/train_token_clf.py --model bert-base-cased --decoder softmax \
        --out-dir /content/drive/MyDrive/team7_checkpoints
    python src/trainers/train_token_clf.py --model bert-base-cased --decoder crf --smoke

    # truncation analysis: score a trained model reading only the first window
    python src/trainers/train_token_clf.py --model bert-base-cased --decoder crf \
        --predict-only --truncate --out-dir /content/drive/MyDrive/team7_checkpoints

    # offset-repair ablation: train on unrepaired labels, score on repaired tests
    python src/prepare_data.py --train-zip data/raw/persuade_corpus_2.0_train.zip \
        --test-zip data/raw/persuade_corpus_2.0_test.zip --no-repair --out-dir data/unrepaired
    python src/trainers/train_token_clf.py --model bert-base-cased --decoder crf \
        --train-data-dir data/unrepaired --run-name bert-base-cased_crf_unrepaired \
        --out-dir /content/drive/MyDrive/team7_checkpoints
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from evaluate import read_jsonl, score, write_jsonl_gz  # noqa: E402
from windows import (  # noqa: E402
    LABEL2ID,
    LABELS,
    allowed_transitions,
    make_windows,
    merge_window_emissions,
    viterbi,
)


# ---------------------------------------------------------------- data
def tokenize_essay(tokenizer, words):
    kwargs = {"is_split_into_words": True, "add_special_tokens": False}
    enc = tokenizer(words, **kwargs)
    return enc["input_ids"], enc.word_ids()


def build_windows(tokenizer, essays, max_body, stride, truncate=False):
    """Return (windows per essay, flat training items)."""
    per_essay, items = [], []
    for e_i, rec in enumerate(essays):
        ids, wids = tokenize_essay(tokenizer, rec["tokens"])
        wins = make_windows(ids, wids, max_body, stride, truncate)
        per_essay.append(wins)
        for w in wins:
            if not w.word_idx:  # a window holding only continuation subwords
                continue
            items.append(
                {
                    "essay": e_i,
                    "input_ids": w.input_ids,
                    "word_pos": w.word_pos,
                    "labels": [LABEL2ID[rec["labels"][k]] for k in w.word_idx],
                }
            )
    return per_essay, items


def collate(batch, cls_id, sep_id, pad_id, torch):
    seq = max(len(b["input_ids"]) for b in batch) + 2
    nw = max(len(b["word_pos"]) for b in batch)
    B = len(batch)
    input_ids = torch.full((B, seq), pad_id, dtype=torch.long)
    attn = torch.zeros((B, seq), dtype=torch.long)
    word_pos = torch.zeros((B, nw), dtype=torch.long)
    word_mask = torch.zeros((B, nw), dtype=torch.bool)
    labels = torch.zeros((B, nw), dtype=torch.long)
    for i, b in enumerate(batch):
        ids = [cls_id] + b["input_ids"] + [sep_id]
        input_ids[i, : len(ids)] = torch.tensor(ids)
        attn[i, : len(ids)] = 1
        k = len(b["word_pos"])
        word_pos[i, :k] = torch.tensor(b["word_pos"]) + 1  # shift past [CLS]
        word_mask[i, :k] = True
        if "labels" in b:
            labels[i, :k] = torch.tensor(b["labels"])
    return input_ids, attn, word_pos, word_mask, labels


# ---------------------------------------------------------------- model
def build_model(torch, name, num_labels, decoder, dropout=0.1):
    from torch import nn
    from transformers import AutoModel

    class TokenTagger(nn.Module):
        def __init__(self):
            super().__init__()
            # Some checkpoints (deberta-v3) load in float16; mixed-precision
            # training needs float32 master weights, so cast explicitly.
            self.encoder = AutoModel.from_pretrained(name).float()
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(self.encoder.config.hidden_size, num_labels)
            self.crf = None
            if decoder == "crf":
                from torchcrf import CRF

                self.crf = CRF(num_labels, batch_first=True)

        def emissions(self, input_ids, attention_mask, word_pos):
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            logits = self.head(self.drop(hidden))
            idx = word_pos.unsqueeze(-1).expand(-1, -1, logits.size(-1))
            return torch.gather(logits, 1, idx)  # (batch, words, labels)

        def loss(self, em, labels, word_mask):
            em = em.float()
            if self.crf is not None:
                return -self.crf(em, labels, mask=word_mask, reduction="mean")
            return nn.functional.cross_entropy(em[word_mask], labels[word_mask])

    return TokenTagger()


# ---------------------------------------------------------------- predict
def predict(torch, model, essays, per_essay, tok_ids, device, batch_size, use_amp):
    trans_ok, start_ok = allowed_transitions()
    crf = model.crf
    if crf is not None:
        T = crf.transitions.detach().float().cpu().numpy()
        S = crf.start_transitions.detach().float().cpu().numpy()
        E = crf.end_transitions.detach().float().cpu().numpy()
    flat = [(e, w_i, w) for e, wins in enumerate(per_essay) for w_i, w in enumerate(wins) if w.word_idx]
    flat.sort(key=lambda x: len(x[2].input_ids))
    out = [[[] for _ in wins] for wins in per_essay]
    model.eval()
    with torch.no_grad():
        for s in range(0, len(flat), batch_size):
            chunk = flat[s : s + batch_size]
            batch = [{"input_ids": w.input_ids, "word_pos": w.word_pos} for _, _, w in chunk]
            input_ids, attn, word_pos, _, _ = collate(batch, *tok_ids, torch)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                em = model.emissions(input_ids.to(device), attn.to(device), word_pos.to(device))
            em = em.float().cpu().numpy()
            for row, (e, w_i, w) in zip(em, chunk):
                out[e][w_i] = row[: len(w.word_idx)]
    # Merge window scores per word, then decode each essay in one pass, so a
    # CRF never produces an illegal transition where two windows meet.
    preds = {}
    for rec, wins, wem in zip(essays, per_essay, out):
        kept = [(w, x) for w, x in zip(wins, wem) if w.word_idx]
        scores = merge_window_emissions(len(rec["tokens"]), [w for w, _ in kept], [x for _, x in kept])
        if crf is not None:
            seq = viterbi(scores, T, S, E, trans_ok, start_ok)
        else:
            seq = scores.argmax(-1).tolist()
        preds[rec["essay_id"]] = [LABELS[k] for k in seq]
    return preds


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="bert-base-cased")
    ap.add_argument("--decoder", choices=["softmax", "crf"], default="softmax")
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--train-data-dir", type=Path, default=None,
                    help="read train and val from here instead (offset-repair ablation); "
                         "test sets always come from --data-dir")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out-dir", type=Path, default=Path("checkpoints"),
                    help="checkpoints (use a Google Drive folder on Colab)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3, help="head and CRF learning rate")
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue from last.pt in --out-dir")
    ap.add_argument("--predict-only", action="store_true", help="score best.pt without training")
    ap.add_argument("--truncate", action="store_true",
                    help="with --predict-only: read only the first window of each essay")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_fp16
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"

    short = args.model.split("/")[-1]
    run = args.run_name or f"{short}_{args.decoder}"
    if args.smoke:
        run += "_smoke"
    ckpt_dir = args.out_dir / run
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"run {run} on {gpu}; checkpoints in {ckpt_dir}")

    limit = {"train": 64, "val": 32, "test_seen": 32, "test_unseen": 32} if args.smoke else {}
    data = {}
    for split in ("train", "val", "test_seen", "test_unseen"):
        src = args.train_data_dir if args.train_data_dir and split in ("train", "val") else args.data_dir
        recs = list(read_jsonl(src / f"{split}.jsonl.gz"))
        if split in limit:
            recs = random.Random(args.seed).sample(recs, limit[split])
        data[split] = recs
    if args.smoke:
        args.epochs = 1

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # Essays are split into windows ourselves, so silence the "longer than
    # 512" warning the tokenizer prints for whole essays.
    tokenizer.model_max_length = 10**6
    tok_ids = (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id)
    max_body = args.max_length - 2

    model = build_model(torch, args.model, len(LABELS), args.decoder).to(device)

    def evaluate_split(split, truncate=False):
        wins, _ = build_windows(tokenizer, data[split], max_body, args.stride, truncate)
        preds = predict(torch, model, data[split], wins, tok_ids, device, args.batch_size * 2, use_amp)
        return preds, score(data[split], preds)

    t0 = time.time()
    if not args.predict_only:
        _, train_items = build_windows(tokenizer, data["train"], max_body, args.stride)
        print(f"{len(data['train']):,} train essays -> {len(train_items):,} windows")
        enc_params = list(model.encoder.parameters())
        enc_ids = {id(p) for p in enc_params}
        other = [p for p in model.parameters() if id(p) not in enc_ids]
        opt = torch.optim.AdamW(
            [{"params": enc_params, "lr": args.lr}, {"params": other, "lr": args.head_lr}],
            weight_decay=0.01,
        )
        steps_per_epoch = math.ceil(len(train_items) / (args.batch_size * args.grad_accum))
        total = steps_per_epoch * args.epochs
        sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total), total)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        start_epoch, best = 0, -1.0
        last = ckpt_dir / "last.pt"
        if args.resume and last.exists():
            st = torch.load(last, map_location=device)
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            scaler.load_state_dict(st["scaler"])
            start_epoch, best = st["epoch"] + 1, st["best"]
            print(f"resumed after epoch {st['epoch'] + 1}, best val {best:.4f}")

        for epoch in range(start_epoch, args.epochs):
            model.train()
            rng = random.Random(args.seed + epoch)
            order = list(range(len(train_items)))
            rng.shuffle(order)
            running, t_ep = 0.0, time.time()
            opt.zero_grad(set_to_none=True)
            n_batches = math.ceil(len(order) / args.batch_size)
            for b in range(n_batches):
                batch = [train_items[i] for i in order[b * args.batch_size : (b + 1) * args.batch_size]]
                input_ids, attn, word_pos, word_mask, labels = collate(batch, *tok_ids, torch)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    em = model.emissions(input_ids.to(device), attn.to(device), word_pos.to(device))
                loss = model.loss(em, labels.to(device), word_mask.to(device)) / args.grad_accum
                scaler.scale(loss).backward()
                if (b + 1) % args.grad_accum == 0 or b + 1 == n_batches:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scale_before = scaler.get_scale()
                    scaler.step(opt)
                    scaler.update()
                    # fp16 skips the optimizer step when gradients overflow
                    # (the scale drops); advance the schedule only on real steps
                    if not use_amp or scaler.get_scale() >= scale_before:
                        sched.step()
                    opt.zero_grad(set_to_none=True)
                running += loss.item() * args.grad_accum
                if (b + 1) % 200 == 0:
                    rate = (b + 1) / (time.time() - t_ep)
                    print(f"epoch {epoch + 1} batch {b + 1}/{n_batches} loss {running / (b + 1):.4f} "
                          f"({(n_batches - b - 1) / rate / 60:.0f} min left in epoch)")
            _, val = evaluate_split("val")
            f1 = val["overlap_span"]["macro_f1"]
            print(f"epoch {epoch + 1}: train loss {running / n_batches:.4f} | val overlap macro-F1 {f1:.4f} "
                  f"| exact {val['exact_span']['macro_f1']:.4f} | {(time.time() - t_ep) / 60:.1f} min")
            if f1 > best:
                best = f1
                torch.save(model.state_dict(), ckpt_dir / "best.pt")
                print("  saved best.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": epoch, "best": best}, last)
    train_minutes = round((time.time() - t0) / 60, 1)

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
    suffix = "_truncated" if args.truncate else ""
    for split in ("val", "test_seen", "test_unseen"):
        preds, result = evaluate_split(split, truncate=args.truncate)
        name = f"{run}{suffix}"
        write_jsonl_gz(args.results_dir / "predictions" / f"{name}_{split}.jsonl.gz",
                       [{"essay_id": k, "labels": v} for k, v in preds.items()])
        result.update(
            model=name,
            split=split,
            config={
                "architecture_model": args.model,
                "decoder": args.decoder,
                "epochs": args.epochs,
                "batch_size": args.batch_size * args.grad_accum,
                "lr": args.lr,
                "head_lr": args.head_lr,
                "max_length": args.max_length,
                "stride": args.stride,
                "truncate": args.truncate,
                "train_labels": "released offsets (unrepaired)" if args.train_data_dir else "repaired offsets",
                "seed": args.seed,
                "train_essays": len(data["train"]),
                "train_minutes": train_minutes,
                "hardware": gpu,
            },
        )
        out = args.results_dir / f"{name}_{split}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"{split:12} overlap macro-F1 {result['overlap_span']['macro_f1']:.4f} | "
              f"exact macro-F1 {result['exact_span']['macro_f1']:.4f} | "
              f"illegal BIO {result['illegal_bio_transitions']:,} -> {out}")


if __name__ == "__main__":
    main()
