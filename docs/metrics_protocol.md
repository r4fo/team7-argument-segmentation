# Metrics and model evaluation plan

**Planned only — no trained scores in this repository.**

- **Exact-span match**: both character/token boundaries and semantic type must match vetted gold labels.
- Span precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2PR/(P+R). Declare convention when denominators vanish.
- **Micro F1** pools TP/FP/FN; **macro F1** averages seven per-class F1 scores equally.
- Report label-specific scores and confusion/error types: boundary-only, class-only, missed, spurious, repeated-text, long-essay truncation.
- Token accuracy (including O) is secondary and can hide missed spans.
- If comparing to Kaggle Feedback Prize leaderboard metrics, implement the **same overlap scoring and official split**; do not compare overlap-F1 to your exact-span F1 as if identical.
- Distinct essays across train/validation/test; for held-out-prompt evaluation, test prompts must be unseen in training.
- Keep one independent, vetted evaluation set unchanged across experiments.
- Record dataset version/hash, seed, batch size, epochs, learning rate, tokenizer and model version, runtime and hardware.
- **Ablation idea:** compare original-offset labels and repaired-offset labels on identical essay populations, with all other settings fixed. Report retained-data quantity separately.
- Never tune models or repair heuristics against the final test set.

