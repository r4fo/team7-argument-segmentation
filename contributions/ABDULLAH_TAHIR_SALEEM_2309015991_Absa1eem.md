# Abdullah Tahir Saleem · 2309015991 · Absa1eem

## Role

Benchmark row 4: BERT with a softmax decoder, the baseline that shows what
the CRF adds.

## What I did

- Trained `bert-base-cased` with a softmax decoder
  (`src/trainers/train_token_clf.py --decoder softmax`) on a Colab Tesla T4:
  3 epochs, 29 minutes, the same data, windows (512 subwords, stride 128) and
  settings as the BERT + CRF run, so the decoder is the only difference.
- Produced the results on val, test_seen and test_unseen, and the
  per-essay predictions used for the error analysis.

## Results

| Split | Overlap macro-F1 | Exact macro-F1 | Illegal BIO |
| --- | --- | --- | --- |
| val | 0.3663 | 0.1620 | 14,572 |
| test_seen | 0.3679 | 0.1678 | 85,050 |
| test_unseen | 0.3570 | 0.1604 | 61,522 |

What they show: softmax and CRF get the same token accuracy on test_seen
(0.7647 and 0.7645), so the encoder learned the same thing. The difference is
decoding. The softmax picks each word's label on its own and switches roles
inside a span, which gives 85,050 illegal BIO transitions (10.9 per essay)
and 18.4 predicted spans per essay against 8.9 in the gold data. Those
fragments do not match the gold spans, so span macro-F1 falls from 0.578
with the CRF to 0.368.

## Files I committed

- `results/bert-base-cased_softmax_val.json`,
  `results/bert-base-cased_softmax_test_seen.json`,
  `results/bert-base-cased_softmax_test_unseen.json`

## How I used Claude

Claude (Anthropic's AI assistant) wrote the training script. I ran the
softmax configuration on Colab, produced the results and predictions, and
studied the model and the softmax-versus-CRF comparison so I can explain them.
