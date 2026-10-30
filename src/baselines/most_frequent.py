"""Baseline 1: label each word with its most frequent training label.

The simplest reference system (W6 "most frequent class" baseline). It ignores
context entirely, so any model must beat it to show that context matters.
Words never seen in training get the most frequent label overall.

Usage
-----
    python src/baselines/most_frequent.py \
        --data-dir data/processed --results-dir results
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import read_jsonl, score, write_jsonl_gz  # noqa: E402

NAME = "most_frequent"


def train(records):
    by_word = defaultdict(Counter)
    overall = Counter()
    for r in records:
        for tok, lab in zip(r["tokens"], r["labels"]):
            by_word[tok.lower()][lab] += 1
            overall[lab] += 1
    # ties broken by label name so the result is deterministic
    pick = lambda c: max(sorted(c), key=lambda lab: c[lab])
    return {w: pick(c) for w, c in by_word.items()}, pick(overall)


def predict(model, fallback, records):
    for r in records:
        yield {
            "essay_id": r["essay_id"],
            "labels": [model.get(t.lower(), fallback) for t in r["tokens"]],
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--splits", nargs="+", default=["val", "test_seen", "test_unseen"])
    args = ap.parse_args()

    model, fallback = train(read_jsonl(args.data_dir / "train.jsonl.gz"))
    print(f"vocabulary {len(model):,} words; unseen words -> {fallback}")

    for split in args.splits:
        gold = list(read_jsonl(args.data_dir / f"{split}.jsonl.gz"))
        preds = list(predict(model, fallback, gold))
        write_jsonl_gz(args.results_dir / "predictions" / f"{NAME}_{split}.jsonl.gz", preds)
        result = score(gold, {p["essay_id"]: p["labels"] for p in preds})
        result.update(model=NAME, split=split)
        out = args.results_dir / f"{NAME}_{split}.json"
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            f"{split:12} overlap macro-F1 {result['overlap_span']['macro_f1']:.4f} | "
            f"exact macro-F1 {result['exact_span']['macro_f1']:.4f} | "
            f"token acc {result['token_accuracy']:.4f} | "
            f"illegal BIO {result['illegal_bio_transitions']:,}"
        )


if __name__ == "__main__":
    main()
