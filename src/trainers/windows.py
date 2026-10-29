"""Torch-free pieces of the transformer pipeline, unit-tested in tests/.

Essays are longer than a 512-subword encoder can read (32% exceed 512 *word*
tokens). We therefore split each essay's subword sequence into overlapping
windows, label only the first subword of every word (the rest get -100 and
are ignored, W6), and merge the per-window predictions back to one label per
word, taking each word from the window where it sits farthest from an edge.

The CRF decoder's Viterbi search is also here, in numpy, with the BIO
constraints from the course: I-X may only follow B-X or I-X.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TYPES = [
    "Lead",
    "Position",
    "Claim",
    "Counterclaim",
    "Rebuttal",
    "Evidence",
    "Concluding Statement",
]
LABELS = ["O"] + [f"{p}-{t}" for t in TYPES for p in ("B", "I")]
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}


@dataclass
class Window:
    """One encoder input. `input_ids` excludes special tokens; `word_pos[k]` is
    the position (inside `input_ids`) of the first subword of word
    `word_idx[k]`. Only words whose first subword lies in the window appear."""

    input_ids: list
    word_idx: list
    word_pos: list


def first_subword_positions(word_ids):
    """Map a tokenizer's `word_ids()` (None for specials) to {word: position}."""
    first = {}
    for pos, w in enumerate(word_ids):
        if w is not None and w not in first:
            first[w] = pos
    return first


def make_windows(input_ids, word_ids, max_body, stride, truncate=False):
    """Split one essay's subwords (no special tokens) into windows.

    max_body: subwords per window, excluding [CLS]/[SEP] (e.g. 510 for 512).
    stride:   subwords shared by consecutive windows.
    truncate: keep only the first window (the truncation baseline).
    """
    if not 0 <= stride < max_body:
        raise ValueError("stride must be in [0, max_body)")
    first = first_subword_positions(word_ids)
    n = len(input_ids)
    windows, start = [], 0
    while True:
        end = min(start + max_body, n)
        idx = [w for w, p in first.items() if start <= p < end]
        idx.sort()
        windows.append(
            Window(
                input_ids=input_ids[start:end],
                word_idx=idx,
                word_pos=[first[w] - start for w in idx],
            )
        )
        if end >= n or truncate:
            break
        start = end - stride
    return windows


def merge_window_predictions(n_words, windows, window_preds, fill="O"):
    """Combine per-window word predictions into one label per word.

    window_preds[i][k] is the prediction for windows[i].word_idx[k]. A word
    seen in several windows takes the prediction from the window where its
    first subword is farthest from either edge (most context on both sides).
    Words in no window (only possible with truncate=True) get `fill`.
    """
    best = [None] * n_words
    margin = [-1] * n_words
    for win, preds in zip(windows, window_preds):
        length = len(win.input_ids)
        for w, pos, p in zip(win.word_idx, win.word_pos, preds):
            m = min(pos, length - 1 - pos)
            if m > margin[w]:
                margin[w], best[w] = m, p
    return [fill if b is None else b for b in best]


def merge_window_emissions(n_words, windows, window_emissions, n_labels=len(LABELS), fill_label=0):
    """Combine per-window emission scores into one (n_words, n_labels) array.

    Same choice rule as merge_window_predictions: each word takes its scores
    from the window where it is farthest from an edge. Merging scores (not
    labels) lets one Viterbi pass decode the whole essay, so no illegal BIO
    transition appears where two windows meet. Words in no window get a
    score that strongly prefers `fill_label` (O).
    """
    out = np.full((n_words, n_labels), -1e4, dtype=np.float32)
    out[:, fill_label] = 0.0
    margin = np.full(n_words, -1)
    for win, em in zip(windows, window_emissions):
        length = len(win.input_ids)
        for k, (w, pos) in enumerate(zip(win.word_idx, win.word_pos)):
            m = min(pos, length - 1 - pos)
            if m > margin[w]:
                margin[w] = m
                out[w] = em[k]
    return out


def allowed_transitions(labels=LABELS):
    """(L, L) boolean matrix: may label j follow label i? Plus allowed starts."""
    n = len(labels)
    trans = np.ones((n, n), dtype=bool)
    start = np.ones(n, dtype=bool)
    for j, lab in enumerate(labels):
        if lab.startswith("I-"):
            start[j] = False
            for i, prev in enumerate(labels):
                trans[i, j] = prev[2:] == lab[2:] and prev != "O"
    return trans, start


def viterbi(emissions, transitions, start_scores, end_scores, allowed=None, allowed_start=None):
    """Best label sequence for one sequence of word emissions (T, L)."""
    emissions = np.asarray(emissions, dtype=np.float64)
    trans = np.array(transitions, dtype=np.float64)
    start = np.array(start_scores, dtype=np.float64)
    if allowed is not None:
        trans = np.where(allowed, trans, -1e9)
    if allowed_start is not None:
        start = np.where(allowed_start, start, -1e9)
    T, L = emissions.shape
    if T == 0:
        return []
    score = start + emissions[0]
    back = np.zeros((T, L), dtype=np.int64)
    for t in range(1, T):
        cand = score[:, None] + trans  # (prev, cur)
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + emissions[t]
    score = score + np.asarray(end_scores, dtype=np.float64)
    best = [int(score.argmax())]
    for t in range(T - 1, 0, -1):
        best.append(int(back[t, best[-1]]))
    return best[::-1]
