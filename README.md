# Argumentative Discourse Element Segmentation

Team 7 · COE025 Natural Language Processing · Project 2 (token classification)

We label every token of a student essay with its argumentative role and
recover the argument spans: **Lead, Position, Claim, Counterclaim, Rebuttal,
Evidence, Concluding Statement**. With BIO tagging that is 15 labels (B- and
I- for each type, plus O). Effectiveness tagging is a possible extension and
is not implemented.

**Research question:** do models learn argument structure, or only the
vocabulary of the prompts they were trained on? We test on three prompts that
no model sees during training.

## Data

[PERSUADE 2.0](https://github.com/scrosseye/persuade_corpus_2.0): 25,996
argumentative essays by US students in grades 6–12, written for 15 prompts.
Each essay was annotated by two expert raters independently and adjudicated
by a third. License CC BY 4.0; see [DATA_LICENSE.md](DATA_LICENSE.md) for
attribution, the changes we made, privacy notes and archive checksums.

**The released offsets are wrong for 68.62% of segments.** The start offset
drifts and the error accumulates along each essay: the first segment is
placed correctly 99.92% of the time, the tenth and later only 6.46%. Used as
released, 62.14% of argument spans would get wrong token boundaries. We
re-align every segment by searching for its text, which reproduces the
annotation text exactly for 100% of kept segments, and we keep 99.46% of
segments (the rest are empty, missing from the essay, or overlapping). Every
figure comes from `src/verify_alignment.py`; full details in
[docs/data_pipeline.md](docs/data_pipeline.md).

What is measured and what is not:

| Claim | Status |
| --- | --- |
| 68.62% of released offsets do not reproduce the segment text | Measured (`verify_alignment.py`, check A) |
| 99.46% of segments kept after repair | Measured; a retention rate, not an accuracy |
| Repaired offsets reproduce the annotation text: 100% | Measured (check C) |
| Repaired span is the intended copy when a phrase repeats | 100% of repeated argument segments choose the copy nearest the released offset (check H); stratified manual audit of 30 cases: two independent reviewers, 30/30 correct each |
| Repair improves model scores | Not yet measured: offset-repair ablation in `docs/metrics_protocol.md` |

| Split | Rule | Essays |
| --- | --- | --- |
| train | Official train essays from 12 prompts, minus validation | 11,721 |
| val | 10% of those essays (seed 42) | 1,302 |
| test_seen | Official test essays from the same 12 prompts | 7,819 |
| test_unseen | All essays of the held-out prompts Summer projects, Community service, Exploring Venus | 5,154 |

## Repository layout

| Path | Purpose | Status |
| --- | --- | --- |
| `data/raw/` | Official PERSUADE 2.0 train and test archives, unchanged | Done |
| `data/processed/` | Cleaned BIO data per split (`*.jsonl.gz`), `splits.json`, `stats.json`, `cleaning_report.json`, `verification_report.json` | Done |
| `src/prepare_data.py` | Raw archives → re-aligned segments → tokens + BIO labels → splits | Done |
| `src/verify_alignment.py` | Independent checks of the repair; writes the audit sample | Done |
| `src/evaluate.py` | Overlap and exact span F1 (macro, micro, per class), error types, illegal BIO count | Done |
| `src/analyze_errors.py` | Breaks unmatched gold spans into absorbed / wrong-boundary / missed, top confusions, examples | Done |
| `src/robustness.py` | Softmax post-processing tuned on val; bootstrap 95% intervals for seen→unseen changes and model differences | Done |
| `src/baselines/most_frequent.py` | Most-frequent tag per word | Done |
| `src/baselines/crf_features.py` | Linear-chain CRF with hand-built features (word, neighbours, sentence/paragraph start, position in essay) | Written; runs on Colab |
| `src/baselines/bilstm_crf.py` | BiLSTM-CRF over whole essays | Written; runs on Colab |
| `src/trainers/train_token_clf.py` | Transformer token classifier, softmax or CRF decoder, sliding windows, truncation mode | Written; runs on Colab |
| `src/trainers/windows.py` | Window splitting, first-subword alignment, prediction merging, constrained Viterbi (no torch) | Done, unit-tested |
| `tests/` | Hand-checked metric cases and window/Viterbi checks (`python -m pytest tests/`) | Done |
| `notebooks/colab_train.ipynb` | Runs the scripts on a Colab GPU: tests, smoke run, one cell per model, results export | Done |
| `docs/data_pipeline.md` | Every cleaning step, why it is needed, and its counts | Done |
| `docs/verification_sample.md` | Stratified manual audit of 30 repaired segments, two reviewers | Both reviewers 30/30 |
| `docs/metrics_protocol.md` | Metrics, why we use them, and the planned analyses | Done |
| `results/` | `benchmark.md` table and one results JSON per model and split | All 6 models done |
| `contributions/` | One file per member, named `<NAME>_<STUDENT_ID>_<GITHUB_USERNAME>.md` | Done |

## Reproduce the data

```bash
pip install -r requirements.txt

# Clean (about 2 minutes, CPU only)
python src/prepare_data.py \
    --train-zip data/raw/persuade_corpus_2.0_train.zip \
    --test-zip  data/raw/persuade_corpus_2.0_test.zip \
    --out-dir   data/processed

# Verify (about 1 minute)
python src/verify_alignment.py \
    --train-zip data/raw/persuade_corpus_2.0_train.zip \
    --test-zip  data/raw/persuade_corpus_2.0_test.zip \
    --report    data/processed/verification_report.json \
    --sample    docs/verification_sample.md
```

Both commands regenerate the committed files byte for byte (seed 42).

Each line of a processed file is one essay:

```json
{"essay_id": "...", "prompt": "Phones and driving", "task": "Independent",
 "official_set": "train", "tokens": ["..."], "offsets": [[0, 6]],
 "labels": ["B-Lead", "I-Lead", "O"], "text": "..."}
```

## Benchmark

Baselines are the vanilla methods from the course; the last row is our model.
Scores will appear in [results/benchmark.md](results/benchmark.md) as runs
finish.

| # | Architecture | Specific model | Decoder |
| --- | --- | --- | --- |
| 1 | Lookup | Most-frequent tag per word | Argmax |
| 2 | Linear-chain CRF | sklearn-crfsuite, hand features | Viterbi |
| 3 | BiLSTM | 2-layer BiLSTM, embeddings trained from scratch | CRF (constrained Viterbi) |
| 4 | BERT | bert-base-cased | Softmax |
| 5 | BERT | bert-base-cased | CRF |
| 6 | DeBERTa | deberta-v3-base | CRF |

**Primary metric:** span-level macro F1, where a prediction matches a gold
span of the same type when each covers at least 50% of the other. Macro
averaging gives rare roles (Rebuttal, Counterclaim) the same weight as
frequent ones; token accuracy would stay high while those fail. We also
report exact-match span F1 and per-class F1
([docs/metrics_protocol.md](docs/metrics_protocol.md)).

## Team

Each member documents their work in `contributions/` and commits it from
their own GitHub account.

## License

Code: [MIT](LICENSE). Data: CC BY 4.0, PERSUADE 2.0 (Crossley et al., 2024);
see [DATA_LICENSE.md](DATA_LICENSE.md).
