# Data pipeline

`src/prepare_data.py` turns the raw PERSUADE 2.0 release into token-level BIO
data. One command reproduces every file in `data/processed/` (about 2 minutes on a laptop CPU):

```bash
python src/prepare_data.py \
    --train-zip data/raw/persuade_corpus_2.0_train.zip \
    --test-zip  data/raw/persuade_corpus_2.0_test.zip \
    --out-dir   data/processed
```

## Raw data

| File | Rows (segments) | Essays |
| --- | --- | --- |
| `persuade_corpus_2.0_train.csv` | 173,266 | 15,594 |
| `persuade_corpus_2.0_test.csv` | 112,117 | 10,402 |

Each row is one annotated segment of an essay: the full essay text, the
segment's character offsets (`discourse_start`, `discourse_end`), the segment
text (`discourse_text`) and its type. The release is CC BY 4.0
(Crossley et al., 2024). The test archive is password-protected; the password
is published in the corpus README. Archive checksums are in
[DATA_LICENSE.md](../DATA_LICENSE.md).

Segment types: Claim 82,584 · Evidence 75,588 · Unannotated 47,410 ·
Position 25,669 · Concluding Statement 22,283 · Lead 15,098 ·
Counterclaim 9,534 · Rebuttal 7,217.

## Problem found: the released offsets are wrong

Slicing an essay with the released offsets reproduces the segment text for
only 31.38% of non-empty segments, even after ignoring surrounding whitespace
(68.62% wrong). The cause is **start drift**: the error accumulates along the
essay. The first segment of an essay is placed correctly 99.92% of the time;
the second 73.97%; the fifth 24.61%; the tenth and later only 6.46%. In 96.64%
of essays the true start is at or after the released start. The last
segment of an essay is off by 3.6 characters on average.

The end offsets are not a separate error: every annotation text carries
trailing whitespace that `discourse_end` excludes, and when the start is
correct the released end never cuts real text (0.00%).

**Impact on labels.** Building BIO labels from the released offsets would give
2.57% of tokens a different label than our repaired offsets, concentrated at
span edges (Claim 5.24%, Counterclaim 5.38%, Rebuttal 5.47% of tokens), and
62.14% of argument spans would get different token boundaries. Token-level
scores would barely notice; exact span boundaries, and therefore exact-match
span F1, would be wrong for most spans.

All figures come from `src/verify_alignment.py` (see Verification below).

## Cleaning steps

| Step | What it does | Count |
| --- | --- | --- |
| 1. Keep task columns | Drop demographic columns (gender, race, disability, ...); they are not needed and should not travel into processed data | 27 → 11 columns |
| 2. Accept correct offsets | Released start already matches the segment text | 80,996 segments |
| 3. Re-align by search | Find the stripped segment text in the essay, searching forward from the end of the previous segment (avoids matching an earlier repeated phrase) | 202,866 segments |
| 4. Whitespace-tolerant fallback | Regex match allowing different whitespace between words | 0 needed |
| 5. Drop empty segments | Segment text is empty | 1,435 segments |
| 6. Drop unlocatable segments | Text not present in the essay (e.g. "2:00" annotated where the essay says "2:") | 86 segments |
| 7. Drop overlaps | Re-aligned segment would start inside the previous one | 19 segments |
| 8. Tokenize | Words and punctuation as separate tokens, character offsets kept | 25,996 essays |
| 9. BIO labels | A token takes a segment's label when most of its characters fall inside it; first token gets B-, the rest I-. Unannotated and uncovered text get O | 35 tokens straddle a boundary and stay O |
| 10. Validate | Count illegal BIO transitions (I-X not after B-X/I-X) and check spans decode back | 0 illegal, 0 mismatches |

In total 99.46% of segments are kept (283,843 of 285,383); every dropped segment is counted in `cleaning_report.json`.

## Verification

`src/verify_alignment.py` checks the repair independently of the cleaning
run and writes `data/processed/verification_report.json`:

```bash
python src/verify_alignment.py \
    --train-zip data/raw/persuade_corpus_2.0_train.zip \
    --test-zip  data/raw/persuade_corpus_2.0_test.zip
```

| Check | Result |
| --- | --- |
| Released offsets reproduce the segment text | 31.38% |
| Same, by segment position in essay (1st / 5th / 10th+) | 99.92% / 24.61% / 6.46% |
| Released end cuts real text when the start is right | 0.00% |
| Repaired offsets reproduce the segment text | 100.00% |
| Repaired segments out of order or overlapping | 0 |
| BIO labels decode back to the same span types | 100.00% |
| Decoded spans cover the same text as repaired segments | 99.77% (rest: tokens straddling a boundary) |
| Repeated phrases: chosen copy is the one nearest the released start, argument segments (4,466) | 100.00% |
| Same, Unannotated segments (13,174; the 57 exceptions are 1–2 character fragments that become O) | 99.57% |
| Tokens labelled differently with released offsets | 2.57% |
| Spans with different token boundaries with released offsets | 62.14% |

The automatic checks prove that each repaired slice has the same text as its
annotation. They cannot prove it is the right occurrence when the phrase
appears twice in an essay (17,640 segments). For that, the script writes
`docs/verification_sample.md`: 30 segments (seed 42), stratified into 10
repeated-phrase cases, 10 with a shift of 3+ characters and 10 others, each
shown in its essay context, for two reviewers to check independently. Thirty
cases catch systematic errors; they do not estimate a population error rate.

| Reviewer | Date | Correct | Wrong |
| --- | --- | --- | --- |
| A (Rafik Ajam) | 2026-10-29 | 30 / 30 | 0 |
| B | pending | | |

Reviewer A checked repeated-phrase cases against the full essay text in the
raw CSV, for example essay 8248E7E6E2A0, where "because" occurs seven times:
the chosen copy (start 278) is the one between the annotated Claim and
Evidence, one character after the released start (277).

## Labels

15 labels: `O` plus `B-` and `I-` for Lead, Position, Claim, Counterclaim,
Rebuttal, Evidence and Concluding Statement.

## Splits

Splits are by essay, so no essay appears in two splits.

| Split | Rule | Essays |
| --- | --- | --- |
| train | Official train essays, 12 seen prompts, minus validation | 11,721 |
| val | 10% of those essays (seed 42) | 1,302 |
| test_seen | Official test essays, same 12 prompts | 7,819 |
| test_unseen | All essays of 3 held-out prompts: Summer projects, Community service (independent), Exploring Venus (text-dependent) | 5,154 |

Change the held-out prompts with `--held-out`.

## Length

Median essay: 412 word tokens; 90th percentile: 739; maximum: 1,808.
32% of essays exceed 512 word tokens before subword tokenization, so a
512-subword model sees only part of many essays unless we use sliding windows.

## Output files

| File | Content |
| --- | --- |
| `{train,val,test_seen,test_unseen}.jsonl.gz` | One essay per line: `essay_id`, `prompt`, `task`, `official_set`, `tokens`, `offsets`, `labels`, `text` |
| `splits.json` | Essay id → split |
| `stats.json` | Split sizes, length statistics, span and label counts per split |
| `cleaning_report.json` | Count of every repair and drop |
