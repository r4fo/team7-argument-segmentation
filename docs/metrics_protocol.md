# Metrics and evaluation protocol

Every model is scored by `src/evaluate.py` on the same splits, so differences
come from the method alone. No scores are reported until a script produced them.

## What is scored

Predictions are converted back to spans (type + first token + last token) and
compared with the gold spans of the 7 types.

| Metric | Match rule | Role |
| --- | --- | --- |
| **Overlap span F1 (primary)** | Same type, and the overlap covers ≥ 50% of the gold span and ≥ 50% of the predicted span | Ranks models. Tolerant of small boundary disagreements, which are common even between human raters on clause-length spans |
| Exact span F1 | Same type, same first and last token | Shows how precise the boundaries are |
| Token accuracy | Same label per token | Diagnostic only |

Each gold span matches at most one prediction and vice versa.
Precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN); a class with
no gold and no predicted spans is left out of the average.

- **Macro F1** averages the 7 per-class F1 scores equally, so Rebuttal
  counts as much as Evidence. This is the headline number.
- **Micro F1** pools TP/FP/FN over classes; reported alongside.
- Per-class F1 is always reported.

**Why not accuracy:** O, Evidence and Claim cover most tokens, so token
accuracy stays high while the rare roles (Counterclaim, Rebuttal) fail.

Our overlap rule follows the Feedback Prize 2021 convention, but our splits
differ from that competition, so our numbers are not comparable to its
leaderboard.

## Comparisons

- Splits are by essay; `test_unseen` prompts never appear in train or val.
- Tune on `val` only; score `test_seen` and `test_unseen` once per final model.
- Report for every run: model and tokenizer name, decoder, seed, epochs,
  learning rate, max length and stride, runtime, hardware, code commit.

## Analyses

1. **Seen vs. unseen prompts:** gap in macro F1 per model.
2. **Softmax vs. CRF decoding:** illegal BIO transitions, and errors split into
   boundary-only, type-only, missed and spurious.
3. **Truncation:** first-512-tokens-only vs. sliding window, F1 on
   Concluding Statement.
4. **Offset-repair ablation:** train one model on labels built from the
   released offsets and one on repaired offsets, everything else fixed, and
   score both on the same repaired test labels.
