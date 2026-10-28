# Team 7 — Argumentative Discourse Element Segmentation

**COE025 Natural Language Processing — Project 2.** Status: **repository scaffold**, not a working training implementation. Replace the explicitly marked Python placeholders with your team's actual code and test the full pipeline.

## Objective

Identify discourse spans in English student essays and label them as **Lead, Position, Claim, Counterclaim, Rebuttal, Evidence, or Concluding Statement**. For token classification, seven categories become 15 BIO tags (B/I for each category plus O). **Effectiveness tagging** is a proposed extension, not yet implemented.

## Data

[PERSUADE 2.0 (official source)](https://github.com/scrosseye/persuade_corpus_2.0) — student essays and expert discourse annotations. Citation: Crossley et al. (2024), *A large-scale corpus for assessing written argumentation: PERSUADE 2.0*, Assessing Writing 61. Official source README currently specifies **CC BY 4.0**. Our original code uses [MIT](LICENSE), separately from the original data. See [DATA_LICENSE.md](DATA_LICENSE.md).

**Keep ZIP archives, raw student essays, private demographic attributes, processed text, and model checkpoints out of Git.** Download the authorized official files into `data/raw/` locally (ignored by Git). Do not publish data derived from minors without considering privacy. Never commit credentials or student IDs.

## Directory guide

| Path | Purpose | Current status |
|---|---|---|
| `src/prepare_data.py` | Offset audit, realignment, BIO conversion, essay/prompt splits, reports | **PLACEHOLDER — replace with Team 7's actual script** |
| `src/verify_alignment.py` | Independent validation of recovered spans | **PLACEHOLDER — replace with Team 7's actual script** |
| `src/evaluate.py` | Exact-span and macro/micro F1, per-class evaluation | TODO placeholder |
| `src/baselines/most_frequent.py` | Simple reference baseline | TODO placeholder |
| `src/baselines/crf_features.py` | Feature-based CRF baseline | TODO placeholder |
| `src/baselines/bilstm_crf.py` | BiLSTM-CRF candidate | TODO placeholder |
| `src/trainers/train_token_clf.py` | Transformer token classifier and decoder options | TODO placeholder |
| `data/raw/`, `data/processed/` | Raw archives and generated data, kept out of Git | Empty tracked folders |
| `docs/data_pipeline.md` | Cleaning details, statistics, provenance and split rules | Protocol template |
| `docs/verification_sample.md` | Independent manual QA protocol and 30-case template | Not yet verified |
| `docs/metrics_protocol.md` | Evaluation metrics and comparison rules | Proposed protocol |
| `notebooks/colab_train.ipynb` | GPU launcher to call scripts; no duplicated model implementation | Starter notebook |
| `results/benchmark.md` | Measured results once available | Empty benchmark template |
| `contributions/` | Five work logs linked to pull requests | Member templates |

## Installation (planned)

Use Python 3.11:
```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
Dependencies are **starter exact pins, not yet integration-tested** with your scripts or the Colab GPU image. Install CUDA-specific PyTorch following https://pytorch.org/get-started/locally/ if required.

## Reproduction workflow (NOT YET RUNNABLE)

1. Get the official archives through the links in the upstream README, record hashes and source version, store in ignored `data/raw/`.
2. Replace both preprocessing and alignment-validation placeholders with Team 7's actual verified scripts. Update the CLI examples below to reflect their real interfaces.
3. **Proposed commands only** (not yet verified):
```bash
python src/prepare_data.py --raw-dir data/raw --output-dir data/processed --seed 42
python src/verify_alignment.py --data-dir data/processed --output data/processed/verification_report.json
```
4. Run models with identical, disjoint essay-level train/validation/test sets, and compare seen-prompt versus held-out-prompt performance.
5. Evaluate on the **same independently vetted reference annotations**. Record model name/version, config, seed, runtime, code commit, dataset hashes, and scores in `results/`.

## Measured vs. unverified data-quality claims

- **69% official offset mismatch rate** — **measured by Team 7** under its own consistency test; document exact numerator, denominator, text normalization, and endpoint convention in `docs/data_pipeline.md`.
- **99.46% retention** — **measured by Team 7**: fraction of segments kept after realignment and discarding irrecoverable cases. **This is not a correctness/accuracy rate.**
- **Alignment correctness of retained segments** — NOT YET established independently. Follow `docs/verification_sample.md` before claiming accuracy.
- **Cleaning-related model improvement** — NOT YET measured. Require a controlled ablation before claiming it.

## Collaboration

Use one issue and feature branch per change; open a pull request for review and fill your personal contributions log. See [CONTRIBUTING.md](CONTRIBUTING.md).

