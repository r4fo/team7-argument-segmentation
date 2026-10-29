# Rafik Ajam · 2309116215 · r4fo

## Role

Project setup and coordination, data audit, and the BERT + CRF model
(benchmark row 5) with its analyses.

## What I did

**Task and data**
- Chose the task (argumentative discourse element segmentation) and the
  PERSUADE 2.0 dataset, checked the course sheet that no other group had them,
  and booked them for Team 7.
- Created the public repository and its initial structure, and coordinated the
  training runs across the team (one model per member, run in parallel on
  separate Colab accounts).

**Data audit**
- Reviewer A of the manual audit of repaired offsets
  (`docs/verification_sample.md`): 30 / 30 correct.
- Checked repeated-phrase cases against the full essay text in the raw CSV.
  Example: essay 8248E7E6E2A0, where "because" occurs seven times; I confirmed
  the chosen copy (start 278) is the one between the annotated Claim and
  Evidence, not the earlier one at 235.

**BERT + CRF (row 5)**
- Ran the pipeline smoke test for all five models on a Colab T4 and reported
  the DeBERTa fp16 error that was then fixed.
- Trained `bert-base-cased` with a CRF decoder: 3 epochs, 79 minutes on a
  Tesla T4. Val overlap macro-F1 rose 0.535 → 0.559 → 0.571 across epochs.
- Kept the results of the first decoding version before rescoring, so the
  effect of the window-merge fix is measured on the same trained model
  (`results/bert-base-cased_crf_labelmerge_*.json`): illegal BIO transitions
  on test_seen 665 → 0, overlap macro-F1 0.5754 → 0.5779.
- Ran the truncation analysis (same model, first 512 subwords only):
  Concluding Statement F1 0.795 → 0.609, macro-F1 on unseen prompts
  0.5787 → 0.5069.
- Downloaded the predictions used for the error analysis
  (short Claims absorbed into the following Evidence).

## Files I committed

- `results/bert-base-cased_crf_*.json` (final, `labelmerge`, `truncated`)
- `src/trainers/train_token_clf.py`, `src/trainers/windows.py`,
  `src/evaluate.py`, `tests/test_windows.py`, `requirements.txt`
- `src/verify_alignment.py`, `docs/data_pipeline.md`,
  `docs/verification_sample.md`, `data/processed/verification_report.json`
- Initial cleaning commit: `src/prepare_data.py`, `data/raw/`, `data/processed/`

## How I used Claude

Claude (Anthropic's AI assistant) drafted the code in the files above and the
documentation, and ran the CPU-side checks (unit tests, offset verification).
My part was choosing the task, reviewing the code and its outputs, running
every GPU job on Colab, catching and reporting errors from those runs,
checking the manual audit cases myself, deciding to keep the before-fix
results as evidence, and deciding what went into the repository.
