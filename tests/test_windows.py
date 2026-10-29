"""Checks for src/trainers/windows.py. Run: python -m pytest tests/"""

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "trainers"))
from windows import (  # noqa: E402
    LABELS,
    allowed_transitions,
    make_windows,
    merge_window_predictions,
    viterbi,
)


def fake_tokenize(pieces_per_word):
    """input_ids and word_ids for words split into the given number of pieces."""
    ids, wids = [], []
    for w, k in enumerate(pieces_per_word):
        for j in range(k):
            ids.append(100 * w + j)
            wids.append(w)
    return ids, wids


def test_short_essay_is_one_window():
    ids, wids = fake_tokenize([1, 2, 1])
    wins = make_windows(ids, wids, max_body=10, stride=2)
    assert len(wins) == 1
    assert wins[0].word_idx == [0, 1, 2]
    assert wins[0].word_pos == [0, 1, 3]


def test_every_word_lands_in_a_window_and_positions_point_at_first_piece():
    pieces = [1, 3, 1, 2, 1, 1, 4, 1, 2, 1, 1, 3]
    ids, wids = fake_tokenize(pieces)
    wins = make_windows(ids, wids, max_body=6, stride=2)
    covered = set()
    for w in wins:
        for word, pos in zip(w.word_idx, w.word_pos):
            assert w.input_ids[pos] == 100 * word  # the first piece of that word
            covered.add(word)
        assert len(w.input_ids) <= 6
    assert covered == set(range(len(pieces)))


def test_truncate_keeps_first_window_only():
    ids, wids = fake_tokenize([1] * 20)
    wins = make_windows(ids, wids, max_body=8, stride=3, truncate=True)
    assert len(wins) == 1 and wins[0].word_idx == list(range(8))
    merged = merge_window_predictions(20, wins, [["B-Claim"] * 8])
    assert merged[:8] == ["B-Claim"] * 8 and merged[8:] == ["O"] * 12


def test_merge_prefers_window_with_more_context():
    ids, wids = fake_tokenize([1] * 10)
    wins = make_windows(ids, wids, max_body=6, stride=4)  # windows start at 0, 2, 4
    preds = [[f"w{i}"] * len(w.word_idx) for i, w in enumerate(wins)]
    merged = merge_window_predictions(10, wins, preds)
    # word 5: margin 0 in window 0 (last), 2 in window 1 (pos 3 of 6), 1 in window 2
    assert merged[5] == "w1"
    assert merged[0] == "w0" and merged[9] == "w2"


def test_merged_emissions_decode_without_illegal_seams():
    # Two windows that disagree at the seam: window 0 ends a span, window 1
    # starts mid-span. Decoding windows separately and merging labels can give
    # O -> I-Claim; merging scores and decoding once cannot.
    from windows import merge_window_emissions
    L = len(LABELS)
    ids, wids = fake_tokenize([1] * 10)
    wins = make_windows(ids, wids, max_body=6, stride=2)
    rng = np.random.default_rng(3)
    ems = [rng.normal(size=(len(w.word_idx), L)) * 4 for w in wins]
    merged = merge_window_emissions(10, wins, ems)
    assert merged.shape == (10, L)
    trans, start = allowed_transitions()
    seq = [LABELS[k] for k in viterbi(merged, np.zeros((L, L)), np.zeros(L), np.zeros(L), trans, start)]
    prev = "O"
    for lab in seq:
        assert not (lab.startswith("I-") and (prev == "O" or prev[2:] != lab[2:]))
        prev = lab
    # truncation: words in no window are decoded as O
    w0 = make_windows(ids, wids, max_body=6, stride=2, truncate=True)
    m0 = merge_window_emissions(10, w0, [ems[0]])
    assert m0[6:].argmax(-1).tolist() == [0] * 4


def test_allowed_transitions_follow_bio():
    trans, start = allowed_transitions()
    i = LABELS.index
    assert trans[i("B-Claim"), i("I-Claim")] and trans[i("I-Claim"), i("I-Claim")]
    assert not trans[i("O"), i("I-Claim")]
    assert not trans[i("B-Lead"), i("I-Claim")]
    assert trans[i("I-Claim"), i("B-Claim")] and trans[i("O"), i("B-Lead")]
    assert not start[i("I-Lead")] and start[i("B-Lead")] and start[i("O")]


def test_viterbi_matches_brute_force():
    rng = np.random.default_rng(0)
    L, T = 4, 5
    em, tr = rng.normal(size=(T, L)), rng.normal(size=(L, L))
    st, en = rng.normal(size=L), rng.normal(size=L)

    def total(seq):
        s = st[seq[0]] + en[seq[-1]] + sum(em[t, y] for t, y in enumerate(seq))
        return s + sum(tr[a, b] for a, b in zip(seq, seq[1:]))

    best = max(itertools.product(range(L), repeat=T), key=total)
    assert viterbi(em, tr, st, en) == list(best)


def test_constrained_viterbi_never_outputs_illegal_bio():
    rng = np.random.default_rng(1)
    trans, start = allowed_transitions()
    L = len(LABELS)
    for _ in range(50):
        em = rng.normal(size=(30, L)) * 3
        seq = [LABELS[k] for k in viterbi(em, np.zeros((L, L)), np.zeros(L), np.zeros(L), trans, start)]
        prev = "O"
        for lab in seq:
            if lab.startswith("I-"):
                assert prev[2:] == lab[2:] and prev != "O"
            prev = lab
