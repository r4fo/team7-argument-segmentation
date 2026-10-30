# Benchmark

Macro F1 over the 7 discourse types (protocol: [docs/metrics_protocol.md](../docs/metrics_protocol.md)).
No row is filled until `src/evaluate.py` produced it; each filled row links its results JSON.

| # | Architecture | Specific model | Decoder | Seen: overlap F1 | Seen: exact F1 | Unseen: overlap F1 | Unseen: exact F1 | Illegal BIO (seen) | Results file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Lookup | Most-frequent tag per word | Argmax | 0.0321 | 0.0001 | 0.0311 | 0.0000 | 83,207 | [seen](most_frequent_test_seen.json), [unseen](most_frequent_test_unseen.json) |
| 2 | Linear-chain CRF | sklearn-crfsuite, hand features (3,000 training essays) | Viterbi | 0.4139 | 0.2210 | 0.3275 | 0.1659 | 3 | [seen](crf_features_test_seen.json), [unseen](crf_features_test_unseen.json) |
| 3 | BiLSTM | 2-layer BiLSTM, embeddings trained from scratch | CRF (constrained Viterbi) | 0.5372 | 0.3059 | 0.4891 | 0.2552 | 0 | [seen](bilstm_crf_test_seen.json), [unseen](bilstm_crf_test_unseen.json) |
| 4 | BERT | bert-base-cased | Softmax (argmax per word) | 0.3679 | 0.1678 | 0.3570 | 0.1604 | 85,050 | [seen](bert-base-cased_softmax_test_seen.json), [unseen](bert-base-cased_softmax_test_unseen.json) |
| 5 | BERT | bert-base-cased | CRF (constrained Viterbi) | 0.5779 | 0.3382 | 0.5787 | 0.3347 | 0 | [seen](bert-base-cased_crf_test_seen.json), [unseen](bert-base-cased_crf_test_unseen.json) |
| 6 | DeBERTa | deberta-v3-base | CRF (constrained Viterbi) | **0.6523** | **0.4027** | **0.6473** | **0.4013** | 0 | [seen](deberta-v3-base_crf_test_seen.json), [unseen](deberta-v3-base_crf_test_unseen.json) |

All transformer rows read essays as overlapping windows of 512 subwords
(stride 128), 3 epochs, seed 42, one Tesla T4. Row 4 trained in 29 minutes, row 5 in 79, row 6 in 116
(the CRF's forward algorithm runs step by step over each window).

## Findings so far

**DeBERTa-v3 is the best model, and its gain is largest on the hard roles.**
With the same CRF decoder, windows and settings, deberta-v3-base beats
bert-base-cased by 0.074 macro-F1 on seen prompts (0.6523 against 0.5779) and
0.069 on unseen prompts (0.6473 against 0.5787), and exact-match F1 rises
from 0.338 to 0.403. The biggest gains are on Evidence (0.516 → 0.655),
Counterclaim (0.417 → 0.533) and Rebuttal (0.357 → 0.460); the easy,
position-bound roles gain least (Lead +0.030, Concluding Statement +0.023).
Like BERT, it barely changes on unseen prompts (−0.005). It also cuts the
main error, a Claim absorbed into the following Evidence, from 9,702 to
6,769 cases on test_seen (−30%)
([seen](deberta-v3-base_crf_test_seen_errors.json), [unseen](deberta-v3-base_crf_test_unseen_errors.json)).

| Role (test_seen) | BERT + CRF | DeBERTa + CRF | Gain |
| --- | --- | --- | --- |
| Lead | 0.785 | 0.815 | +0.030 |
| Position | 0.646 | 0.683 | +0.037 |
| Claim | 0.529 | 0.602 | +0.073 |
| Counterclaim | 0.417 | 0.533 | +0.116 |
| Rebuttal | 0.357 | 0.460 | +0.103 |
| Evidence | 0.516 | 0.655 | +0.139 |
| Concluding Statement | 0.795 | 0.818 | +0.023 |

**The CRF decoder, not the encoder, makes the spans.** Rows 4 and 5 use the
same encoder, data and training settings; only the decoder differs. Their
token accuracy is the same (0.7647 softmax, 0.7645 CRF on test_seen), and so
is the share of words given the right role ignoring B/I (0.771 and 0.770).
Yet span macro-F1 is 0.3679 with softmax and 0.5779 with the CRF. Choosing
each word's label independently, the softmax switches roles inside a span
(for example I-Counterclaim, I-Claim, I-Counterclaim), producing 85,050
illegal BIO transitions on test_seen (10.9 per essay) and 18.4 predicted
spans per essay against 8.9 in the gold data. Each switch breaks one span
into fragments that no longer overlap the gold span enough to count. The
CRF scores the whole sequence and allows no illegal transition.

**Token accuracy hides span failure.** The most-frequent baseline reaches
53.35% token accuracy on test_seen but 0.0321 span macro-F1: it labels tokens
one by one and almost never forms a complete span.

**Rare argument roles are hardest** (row 5, test_seen, overlap F1):
Concluding Statement 0.795, Lead 0.785, Position 0.646, Claim 0.529,
Evidence 0.516, Counterclaim 0.417, Rebuttal 0.357. Roles tied to a position in
the essay are easiest; the rare rhetorical moves are hardest, which is why we
report macro-F1.

**Unseen prompts (research question).** Row 5 scores the same macro-F1 on
prompts it never saw (0.5787) as on seen prompts (0.5779), but the average
hides two movements: content-dependent roles drop on new topics (Evidence
0.516 → 0.480, Position 0.646 → 0.614, Claim 0.529 → 0.514) while structural
roles hold or rise (Counterclaim 0.417 → 0.484, Rebuttal 0.357 → 0.383).

**Word features do not transfer to new topics; BERT does.** On unseen
prompts the feature CRF (row 2) loses 0.086 macro-F1 (0.4139 → 0.3275) and
drops on every role, most on Position (−0.169) and Evidence (−0.124), whose
words are topic-specific. BERT + CRF (row 5) keeps its macro-F1
(0.5779 → 0.5787). Caveat: row 2 trained on 3,000 of the 11,721 training
essays to keep crfsuite fast, so part of its gap may come from less data.

The BiLSTM-CRF (row 3) sits in between: it learns its word embeddings only
from the 12 training prompts and loses 0.048 (0.5372 → 0.4891), with the
largest drops again on Position (0.622 → 0.484) and Evidence (0.473 → 0.388).
The less a model's word knowledge depends on the training topics, the less it
loses: hand-built word features (−0.086), embeddings learned from the training
essays (−0.048), pretrained BERT (+0.001).

| Role | CRF seen | CRF unseen | Change | BERT seen | BERT unseen | Change |
| --- | --- | --- | --- | --- | --- | --- |
| Lead | 0.641 | 0.536 | −0.105 | 0.785 | 0.767 | −0.018 |
| Position | 0.469 | 0.300 | −0.169 | 0.646 | 0.614 | −0.032 |
| Claim | 0.312 | 0.213 | −0.099 | 0.529 | 0.514 | −0.015 |
| Counterclaim | 0.181 | 0.153 | −0.027 | 0.417 | 0.484 | +0.067 |
| Rebuttal | 0.103 | 0.080 | −0.023 | 0.357 | 0.383 | +0.026 |
| Evidence | 0.429 | 0.305 | −0.124 | 0.516 | 0.480 | −0.036 |
| Concluding Statement | 0.763 | 0.706 | −0.057 | 0.795 | 0.808 | +0.013 |

**Truncation removes the ending.** Same trained model (row 5), reading only the
first 512 subwords of each essay instead of all windows:

| Class | Windows | First 512 only | Change |
| --- | --- | --- | --- |
| Concluding Statement | 0.795 | 0.609 | −0.186 |
| Rebuttal | 0.357 | 0.292 | −0.065 |
| Counterclaim | 0.417 | 0.355 | −0.062 |
| Evidence | 0.516 | 0.499 | −0.016 |
| Claim | 0.529 | 0.524 | −0.005 |
| Position | 0.646 | 0.647 | +0.001 |
| Lead | 0.785 | 0.785 | 0.000 |
| **Macro F1, test_seen** | **0.5779** | **0.5304** | **−0.048** |
| **Macro F1, test_unseen** | **0.5787** | **0.5069** | **−0.072** |

The loss grows with how late a role appears: Lead and Position (essay start)
are untouched, Concluding Statement (essay end) loses most. Files:
[seen](bert-base-cased_crf_truncated_test_seen.json),
[unseen](bert-base-cased_crf_truncated_test_unseen.json).

**The main error: short Claims absorbed into Evidence.** Of the 34,167
test_seen gold spans row 5 does not match, only 1,018 (3%) are truly missed
(predicted O). 19,839 are absorbed into or relabelled as another role, most
often a Claim swallowed by the Evidence next to it (9,702 cases), and 13,310
have the right role but wrong boundaries; 4,093 predicted spans each swallow
two or more gold spans of one role. Example (essay 041FABDBF592): gold
*[Claim: I know that kids need sports] [Evidence: some to let out there
feeling, some to let there energy out ...]*, predicted as one Evidence span.
Files: [seen](bert-base-cased_crf_test_seen_errors.json),
[unseen](bert-base-cased_crf_test_unseen_errors.json) (`src/analyze_errors.py`).
The BiLSTM-CRF (row 3) makes the same main error (Claim → Evidence 10,009
times on test_seen), so it is a property of the task, not of one model:
a short claim followed by its support reads like one paragraph
([seen](bilstm_crf_test_seen_errors.json), [unseen](bilstm_crf_test_unseen_errors.json)).

**Decoding across windows.** Our first version decoded each window separately
and joined the labels, which left illegal BIO transitions where two windows
meet even under a CRF (665 on test_seen). Merging per-word scores first and
running one constrained Viterbi over the whole essay removes them (0) and
raises macro-F1 slightly, with the same trained model:

| Split | Overlap F1 before → after | Exact F1 before → after | Illegal BIO before → after |
| --- | --- | --- | --- |
| val | 0.5706 → 0.5758 | 0.3244 → 0.3288 | 192 → 0 |
| test_seen | 0.5754 → 0.5779 | 0.3347 → 0.3382 | 665 → 0 |
| test_unseen | 0.5775 → 0.5787 | 0.3306 → 0.3347 | 429 → 0 |

Files before the fix: `bert-base-cased_crf_labelmerge_*.json`.
