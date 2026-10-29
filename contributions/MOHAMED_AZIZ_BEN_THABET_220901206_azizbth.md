# Mohamed Aziz Ben Thabet · 220901206 · azizbth

## Role

Benchmark row 3: the BiLSTM-CRF baseline.

## What I did

- Trained the BiLSTM-CRF (`src/baselines/bilstm_crf.py`) on a Colab Tesla T4:
  2-layer bidirectional LSTM, hidden size 256, 100-dimensional word
  embeddings learned from scratch (vocabulary of 21,799 words seen at least
  twice in training), CRF output layer with constrained Viterbi decoding,
  8 epochs, 56 minutes.
- Produced the results on val, test_seen and test_unseen, and the
  per-essay predictions used for the error analysis.

## Results (overlap span macro-F1)

| Split | Overlap macro-F1 | Exact macro-F1 | Illegal BIO |
| --- | --- | --- | --- |
| val | 0.5325 | 0.2938 | 0 |
| test_seen | 0.5372 | 0.3059 | 0 |
| test_unseen | 0.4891 | 0.2552 | 0 |

What they show: the BiLSTM-CRF is close to BERT + CRF on seen prompts
(0.537 against 0.578) but loses 0.048 on unseen prompts, where BERT loses
nothing. Its embeddings are learned only from the 12 training prompts, so
topic-specific roles drop most: Position 0.622 → 0.484, Evidence
0.473 → 0.388. It reads each essay whole, so it never truncates.

## Files I committed

- `src/baselines/bilstm_crf.py`
- `results/bilstm_crf_val.json`, `results/bilstm_crf_test_seen.json`,
  `results/bilstm_crf_test_unseen.json`

## How I used Claude

Claude (Anthropic's AI assistant) wrote `src/baselines/bilstm_crf.py`. I ran
it on Colab, produced the results and predictions, and studied the model so
I can explain how it works and what its results mean.
