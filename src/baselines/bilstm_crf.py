"""Baseline 3: BiLSTM-CRF over word embeddings learned from scratch.

The neural sequence labeller from the course (W6): words -> embeddings ->
bidirectional LSTM -> per-word emissions -> CRF. Unlike a 512-subword
transformer it reads the whole essay at once, so it never truncates.

Embeddings are trained from scratch on our training essays (vocabulary:
words seen at least twice, lower-cased); no pretrained vectors are used.
Decoding is constrained Viterbi (I-X only after B-X/I-X), shared with the
transformer runs.

Usage (Colab GPU)
-----------------
    python src/baselines/bilstm_crf.py --out-dir /content/drive/MyDrive/team7_checkpoints
    python src/baselines/bilstm_crf.py --smoke
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC / "trainers"))
sys.path.insert(0, str(SRC))
from evaluate import read_jsonl, score, write_jsonl_gz  # noqa: E402
from windows import LABEL2ID, LABELS, allowed_transitions, viterbi  # noqa: E402

NAME = "bilstm_crf"
PAD, UNK = 0, 1


def build_vocab(essays, min_freq=2):
    counts = Counter(t.lower() for r in essays for t in r["tokens"])
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for w, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab


def encode(rec, vocab):
    return [vocab.get(t.lower(), UNK) for t in rec["tokens"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    from torch import nn
    from torchcrf import CRF

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    name = NAME + ("_smoke" if args.smoke else "")
    ckpt_dir = args.out_dir / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data = {s: list(read_jsonl(args.data_dir / f"{s}.jsonl.gz"))
            for s in ("train", "val", "test_seen", "test_unseen")}
    if args.smoke:
        for s, n in {"train": 200, "val": 50, "test_seen": 50, "test_unseen": 50}.items():
            data[s] = random.Random(args.seed).sample(data[s], n)
        args.epochs = 1
    vocab = build_vocab(data["train"])
    print(f"vocabulary {len(vocab):,} words; device {gpu}")

    class BiLSTMCRF(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(vocab), args.emb_dim, padding_idx=PAD)
            self.lstm = nn.LSTM(args.emb_dim, args.hidden, num_layers=args.layers, batch_first=True,
                                bidirectional=True, dropout=args.dropout if args.layers > 1 else 0.0)
            self.drop = nn.Dropout(args.dropout)
            self.head = nn.Linear(2 * args.hidden, len(LABELS))
            self.crf = CRF(len(LABELS), batch_first=True)

        def emissions(self, x, lengths):
            packed = nn.utils.rnn.pack_padded_sequence(self.drop(self.emb(x)), lengths.cpu(),
                                                       batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
            return self.head(self.drop(out))

    def batches(recs, shuffle, seed=0):
        order = sorted(range(len(recs)), key=lambda i: len(recs[i]["tokens"]))
        chunks = [order[i : i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        if shuffle:
            random.Random(seed).shuffle(chunks)
        for ch in chunks:
            seqs = [encode(recs[i], vocab) for i in ch]
            L = max(len(s) for s in seqs)
            x = torch.zeros((len(ch), L), dtype=torch.long)
            y = torch.zeros((len(ch), L), dtype=torch.long)
            m = torch.zeros((len(ch), L), dtype=torch.bool)
            for j, (i, s) in enumerate(zip(ch, seqs)):
                x[j, : len(s)] = torch.tensor(s)
                y[j, : len(s)] = torch.tensor([LABEL2ID[l] for l in recs[i]["labels"]])
                m[j, : len(s)] = True
            yield ch, x, y, m, m.sum(1)

    model = BiLSTMCRF().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    trans_ok, start_ok = allowed_transitions()

    def predict(recs):
        model.eval()
        T = model.crf.transitions.detach().cpu().numpy()
        S = model.crf.start_transitions.detach().cpu().numpy()
        E = model.crf.end_transitions.detach().cpu().numpy()
        preds = {}
        with torch.no_grad():
            for ch, x, _, m, lengths in batches(recs, False):
                em = model.emissions(x.to(device), lengths).cpu().numpy()
                for j, i in enumerate(ch):
                    n = int(lengths[j])
                    seq = viterbi(em[j, :n], T, S, E, trans_ok, start_ok)
                    preds[recs[i]["essay_id"]] = [LABELS[k] for k in seq]
        return preds

    t0, best = time.time(), -1.0
    for epoch in range(args.epochs):
        model.train()
        total, nb = 0.0, 0
        for _, x, y, m, lengths in batches(data["train"], True, args.seed + epoch):
            em = model.emissions(x.to(device), lengths)
            loss = -model.crf(em, y.to(device), mask=m.to(device), reduction="mean")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total, nb = total + loss.item(), nb + 1
        val = score(data["val"], predict(data["val"]))
        f1 = val["overlap_span"]["macro_f1"]
        print(f"epoch {epoch + 1}: loss {total / nb:.3f} | val overlap macro-F1 {f1:.4f}")
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
    train_minutes = round((time.time() - t0) / 60, 1)

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
    for split in ("val", "test_seen", "test_unseen"):
        preds = predict(data[split])
        write_jsonl_gz(args.results_dir / "predictions" / f"{name}_{split}.jsonl.gz",
                       [{"essay_id": k, "labels": v} for k, v in preds.items()])
        result = score(data[split], preds)
        result.update(model=name, split=split, config={
            "vocab": len(vocab), "emb_dim": args.emb_dim, "hidden": args.hidden, "layers": args.layers,
            "dropout": args.dropout, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "seed": args.seed, "train_minutes": train_minutes, "hardware": gpu})
        out = args.results_dir / f"{name}_{split}.json"
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"{split:12} overlap macro-F1 {result['overlap_span']['macro_f1']:.4f} | "
              f"exact macro-F1 {result['exact_span']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
