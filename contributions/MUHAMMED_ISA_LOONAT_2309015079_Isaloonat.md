# Muhammed Isa Loonat · 2309015079 · Isaloonat

## Role

Benchmark row 2: the linear-chain CRF over hand-built features, the classical
sequence labeller from the course.

## What I did

- Trained the feature CRF (`src/baselines/crf_features.py`, sklearn-crfsuite,
  L-BFGS, c1 = c2 = 0.1, 100 iterations) on all 11,721 training essays on
  Colab: 43 minutes.
- Ran the first version on 3,000 essays (0.4139 seen, 0.3275 unseen). When
  the full run kept getting killed on free Colab (12.7 GB RAM), I measured it
  at about 10.7 GB and reported it; the script was changed to feed crfsuite
  one essay at a time and to save the model, after which the full run finished.
- Produced the results on val, test_seen and test_unseen, and the per-essay
  predictions.

## Results

| Split | Overlap macro-F1 | Exact macro-F1 | Illegal BIO |
| --- | --- | --- | --- |
| val | 0.4260 | 0.2340 | 0 |
| test_seen | 0.4333 | 0.2418 | 2 |
| test_unseen | 0.3260 | 0.1663 | 2 |

What they show: the CRF learns label transitions, so it almost never breaks
BIO order (2 illegal transitions against 85,050 for the BERT softmax), but its
features are words of the training topics. On unseen prompts it loses 0.107
macro-F1, most on Position (0.514 → 0.312) and Evidence (0.398 → 0.282), while
BERT + CRF loses nothing. Training on 11,721 essays instead of 3,000 raised
the seen score (0.4139 → 0.4333) but not the unseen one (0.3275 → 0.3260).

## Files I committed

- `src/baselines/crf_features.py` (memory fix)
- `results/crf_features_val.json`, `results/crf_features_test_seen.json`,
  `results/crf_features_test_unseen.json`
- Row 2 and the topic-transfer finding in `results/benchmark.md`, and the
  status lines in `README.md`

## How I used Claude

Claude (Anthropic's AI assistant) wrote `src/baselines/crf_features.py` and
its memory fix. I ran it on Colab, found and measured the memory failure,
produced the results and predictions, and studied the model so I can explain
how it works and what its results mean.
