"""Two checks on the benchmark's claims, computed from saved predictions.

1. Softmax post-processing. A softmax decoder labels each word on its own and
   leaves short fragments inside spans. Is the CRF's lead over it just missing
   clean-up? We merge consecutive words of one role into a span (ignoring B/I)
   and drop spans shorter than k words, choosing k on val only, then score test.
2. Bootstrap confidence intervals. Essays are resampled with replacement
   (1,000 times, seed 0) to give 95% intervals for the change from seen to
   unseen prompts and for paired model differences on test_seen. This covers
   test-sample variation, not training-seed variation (one seed per model).

Predictions come from each model script (results/predictions/, not committed).

Usage
-----
    python src/robustness.py --results-dir results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate import TYPES, match_spans, read_jsonl, score, spans_from_bio

MODELS = ["most_frequent", "crf_features", "bilstm_crf", "bert-base-cased_softmax",
          "bert-base-cased_crf", "deberta-v3-base_crf"]
PAIRS = [("deberta-v3-base_crf", "bert-base-cased_crf"),
         ("bert-base-cased_crf", "bilstm_crf"),
         ("bert-base-cased_crf", "bert-base-cased_softmax"),
         ("bert-base-cased_crf", "bert-base-cased_softmax+postprocessed")]


def role_runs(labels):
    """Relabel so each run of one role is one span, whatever its B/I tags."""
    out, prev = [], None
    for lab in labels:
        role = None if lab == "O" else lab[2:]
        out.append("O" if role is None else ("I-" if role == prev else "B-") + role)
        prev = role
    return out


def drop_short(labels, k):
    out = list(labels)
    for _, first, last in spans_from_bio(labels):
        if last - first + 1 < k:
            out[first : last + 1] = ["O"] * (last - first + 1)
    return out


def postprocess(labels, k):
    return drop_short(role_runs(labels), k)


def span_counts(gold_recs, preds):
    """Per-essay true positive, false positive and false negative counts per type."""
    counts = np.zeros((len(gold_recs), len(TYPES), 3))
    for i, rec in enumerate(gold_recs):
        gold = spans_from_bio(rec["labels"])
        pred = spans_from_bio(preds[rec["essay_id"]])
        for gi, _ in match_spans(gold, pred, "overlap"):
            counts[i, TYPES.index(gold[gi][0]), 0] += 1
        for t, _, _ in pred:
            counts[i, TYPES.index(t), 1] += 1
        for t, _, _ in gold:
            counts[i, TYPES.index(t), 2] += 1
    counts[..., 1] -= counts[..., 0]
    counts[..., 2] -= counts[..., 0]
    return counts


def macro_f1(summed):
    tp, fp, fn = summed[..., 0], summed[..., 1], summed[..., 2]
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1)).mean(-1)


def ci(x):
    return [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--resamples", type=int, default=1000)
    args = ap.parse_args()
    pred_dir = args.results_dir / "predictions"

    def load_preds(model, split):
        base = model.removesuffix("+postprocessed")
        preds = {p["essay_id"]: p["labels"] for p in read_jsonl(pred_dir / f"{base}_{split}.jsonl.gz")}
        if base != model:
            preds = {e: postprocess(v, best_k) for e, v in preds.items()}
        return preds

    gold = {s: list(read_jsonl(args.data_dir / f"{s}.jsonl.gz")) for s in ("val", "test_seen", "test_unseen")}
    out = {}

    # 1. softmax post-processing, k chosen on val
    best_k = None
    val_preds = load_preds("bert-base-cased_softmax", "val")
    val_scores = {k: score(gold["val"], {e: postprocess(v, k) for e, v in val_preds.items()})
                  ["overlap_span"]["macro_f1"] for k in range(1, 21)}
    best_k = max(val_scores, key=val_scores.get)
    pp = {"min_span_words_chosen_on_val": best_k, "val_overlap_f1_by_k": val_scores}
    for split in ("test_seen", "test_unseen"):
        for name, fn in (("raw", lambda v: v), ("postprocessed", lambda v: postprocess(v, best_k))):
            preds = {e: fn(v) for e, v in load_preds("bert-base-cased_softmax", split).items()}
            r = score(gold[split], preds)
            pp[f"{split}_{name}"] = {
                "overlap_f1": r["overlap_span"]["macro_f1"], "exact_f1": r["exact_span"]["macro_f1"],
                "token_accuracy": r["token_accuracy"], "illegal_bio": r["illegal_bio_transitions"],
                "spans_per_essay": round(sum(len(spans_from_bio(v)) for v in preds.values()) / len(preds), 2)}
    out["softmax_postprocessing"] = pp
    print(f"softmax post-processing: k = {best_k} (val); test_seen overlap F1 "
          f"{pp['test_seen_raw']['overlap_f1']} -> {pp['test_seen_postprocessed']['overlap_f1']}")

    # 2. bootstrap
    rng = np.random.default_rng(0)
    idx = {s: rng.integers(0, len(gold[s]), (args.resamples, len(gold[s]))) for s in ("test_seen", "test_unseen")}
    boot, point = {}, {}
    for m in MODELS + ["bert-base-cased_softmax+postprocessed"]:
        for s in idx:
            c = span_counts(gold[s], load_preds(m, s))
            point[m, s] = float(macro_f1(c.sum(0)))
            boot[m, s] = macro_f1(np.stack([c[i].sum(0) for i in idx[s]]))
    out["seen_to_unseen_change"] = {
        m: {"change": round(point[m, "test_unseen"] - point[m, "test_seen"], 4),
            "ci95": ci(boot[m, "test_unseen"] - boot[m, "test_seen"])} for m in MODELS}
    out["paired_differences_test_seen"] = {
        f"{a} - {b}": {"difference": round(point[a, "test_seen"] - point[b, "test_seen"], 4),
                       "ci95": ci(boot[a, "test_seen"] - boot[b, "test_seen"])} for a, b in PAIRS}
    for m, v in out["seen_to_unseen_change"].items():
        print(f"{m:26} seen -> unseen {v['change']:+.4f}  95% CI {v['ci95']}")
    path = args.results_dir / "robustness.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
