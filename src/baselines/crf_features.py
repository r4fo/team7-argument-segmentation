"""Baseline 2: linear-chain CRF over hand-built token features.

The classical sequence labeller from the course (W6): each token is described
by features of itself and its neighbours, and a linear-chain CRF learns label
transitions (for example that I-Claim follows B-Claim) and decodes with Viterbi.

Features are chosen for discourse structure, not just word identity:
the word and its neighbours (±2), affixes and shape, sentence and paragraph
boundaries, and the token's relative position in the essay (Leads open an
essay, Concluding Statements close it).

Runs on CPU. By default it trains on all 11,721 training essays, like the
neural models; --train-essays N trains on a fixed random sample (seed 42).
The number used is recorded in the results file.

Usage
-----
    python src/baselines/crf_features.py --data-dir data/processed --results-dir results
    python src/baselines/crf_features.py --smoke          # 2-minute check
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import read_jsonl, score, write_jsonl_gz  # noqa: E402

NAME = "crf_features"
SENT_END = {".", "!", "?"}


def token_features(rec):
    """One feature dict per token of an essay record."""
    toks, offs, text = rec["tokens"], rec["offsets"], rec["text"]
    n = len(toks)
    feats = []
    sent_start = True
    for i, tok in enumerate(toks):
        low = tok.lower()
        gap = text[offs[i - 1][1] : offs[i][0]] if i else ""
        para_start = i == 0 or "\n" in gap
        f = {
            "bias": 1.0,
            "w": low,
            "suf3": low[-3:],
            "pre3": low[:3],
            "is_title": tok.istitle(),
            "is_upper": tok.isupper() and len(tok) > 1,
            "is_digit": tok.isdigit(),
            "is_punct": not tok[0].isalnum(),
            "sent_start": sent_start or para_start,
            "para_start": para_start,
            "pos_bucket": str(min(9, int(10 * i / max(n, 1)))),
        }
        for k in (-2, -1, 1, 2):
            j = i + k
            f[f"w{k:+d}"] = toks[j].lower() if 0 <= j < n else ("<s>" if j < 0 else "</s>")
        f["w-1|w"] = f["w-1"] + "|" + low
        f["w|w+1"] = low + "|" + f["w+1"]
        feats.append(f)
        sent_start = tok in SENT_END
    return feats


def load(path, limit=None, seed=42):
    recs = list(read_jsonl(path))
    if limit and limit < len(recs):
        recs = random.Random(seed).sample(recs, limit)
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--train-essays", type=int, default=0,
                    help="essays sampled from train (0 = all; all 11,721 train in about 30 min)")
    ap.add_argument("--max-iterations", type=int, default=100)
    ap.add_argument("--c1", type=float, default=0.1)
    ap.add_argument("--c2", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (200 train essays, 10 iterations, 100 essays per split)")
    args = ap.parse_args()

    import sklearn_crfsuite  # imported here so --help works without it

    eval_limit = None
    name = NAME
    if args.smoke:
        args.train_essays, args.max_iterations, eval_limit = 200, 10, 100
        name = NAME + "_smoke"

    t0 = time.time()
    train = load(args.data_dir / "train.jsonl.gz", args.train_essays or None, args.seed)
    X = [token_features(r) for r in train]
    y = [r["labels"] for r in train]
    print(f"training on {len(train):,} essays, {sum(map(len, y)):,} tokens")

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=args.c1,
        c2=args.c2,
        max_iterations=args.max_iterations,
        all_possible_transitions=True,
    )
    crf.fit(X, y)
    train_minutes = round((time.time() - t0) / 60, 1)
    print(f"trained in {train_minutes} min")

    for split in ("val", "test_seen", "test_unseen"):
        gold = load(args.data_dir / f"{split}.jsonl.gz", eval_limit, args.seed)
        pred_labels = crf.predict([token_features(r) for r in gold])
        preds = [{"essay_id": r["essay_id"], "labels": list(p)} for r, p in zip(gold, pred_labels)]
        write_jsonl_gz(args.results_dir / "predictions" / f"{name}_{split}.jsonl.gz", preds)
        result = score(gold, {p["essay_id"]: p["labels"] for p in preds})
        result.update(
            model=name,
            split=split,
            config={
                "train_essays": len(train),
                "max_iterations": args.max_iterations,
                "c1": args.c1,
                "c2": args.c2,
                "seed": args.seed,
                "train_minutes": train_minutes,
            },
        )
        out = args.results_dir / f"{name}_{split}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            f"{split:12} overlap macro-F1 {result['overlap_span']['macro_f1']:.4f} | "
            f"exact macro-F1 {result['exact_span']['macro_f1']:.4f} | "
            f"illegal BIO {result['illegal_bio_transitions']:,}"
        )


if __name__ == "__main__":
    main()
