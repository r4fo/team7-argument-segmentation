"""Clean PERSUADE 2.0 and convert it to token-level BIO labels.

Pipeline
--------
1. Load the official train and test CSVs (one row per annotated segment).
2. Re-align every segment to the essay text. The released start offsets
   drift: the error accumulates along the essay, so only 31% of segments are
   placed correctly (see src/verify_alignment.py). We therefore locate each
   segment's text in the essay ourselves, searching forward from the end of
   the previous segment.
3. Tokenize each essay into words and punctuation, keeping character offsets.
4. Label tokens with BIO tags over the 7 discourse types. "Unannotated"
   segments and uncovered text become O.
5. Validate the result (illegal BIO transitions, round-trip span recovery).
6. Split by essay: held-out prompts form test_unseen; the remaining official
   train essays give train/val; the remaining official test essays give
   test_seen.
7. Write gzipped JSONL files, the split assignment, statistics and a cleaning
   report that counts every repair.

Usage
-----
    python src/prepare_data.py \
        --train-zip data/raw/persuade_corpus_2.0_train.zip \
        --test-zip data/raw/persuade_corpus_2.0_test.zip \
        --out-dir data/processed
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import random
import re
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

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

DEFAULT_HELD_OUT = ["Summer projects", "Community service", "Exploring Venus"]

TEST_ZIP_PASSWORD = b"persuade_test"  # published in the corpus README

KEEP_COLUMNS = [
    "essay_id_comp",
    "competition_set",
    "full_text",
    "discourse_start",
    "discourse_end",
    "discourse_text",
    "discourse_type",
    "task",
    "prompt_name",
    "grade_level",
    "holistic_essay_score",
]
# Demographic columns (gender, race, disability, ...) are deliberately dropped:
# they are not needed for the task and should not travel into processed data.

TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]")


# ---------------------------------------------------------------- loading
def read_zipped_csv(path: Path, password: bytes | None = None) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name, pwd=password) as fh:
            return pd.read_csv(
                io.TextIOWrapper(fh, encoding="utf-8"),
                usecols=KEEP_COLUMNS,
                low_memory=False,
            )


# ---------------------------------------------------------------- alignment
def _fuzzy_pattern(text: str) -> re.Pattern | None:
    words = text.split()
    if not words:
        return None
    return re.compile(r"\s*".join(re.escape(w) for w in words))


def align_segments(essay_text: str, segs: pd.DataFrame, report: Counter):
    """Return [(start, end, type)] with offsets that match the essay text."""
    aligned = []
    cursor = 0
    for seg in segs.itertuples(index=False):
        dtype = seg.discourse_type
        raw = seg.discourse_text if isinstance(seg.discourse_text, str) else ""
        needle = raw.strip()
        report["segments_total"] += 1

        if not needle:
            report["dropped_empty_text"] += 1
            continue

        start = end = None
        if seg.discourse_start < len(essay_text) and essay_text[
            seg.discourse_start : seg.discourse_start + len(needle)
        ] == needle:
            start = seg.discourse_start
            report["offset_already_correct"] += 1
        else:
            found = essay_text.find(needle, cursor)
            if found != -1:
                start = found
                report["realigned_exact_search"] += 1
            else:
                pattern = _fuzzy_pattern(needle)
                m = pattern.search(essay_text, cursor) if pattern else None
                if m:
                    start, end = m.start(), m.end()
                    report["realigned_whitespace_tolerant"] += 1

        if start is None:
            report["dropped_not_found"] += 1
            continue
        if end is None:
            end = start + len(needle)
        if start < cursor:
            report["dropped_overlap"] += 1
            continue

        aligned.append((start, end, dtype))
        cursor = end
    return aligned


# ---------------------------------------------------------------- labelling
def tokenize(text: str):
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def bio_labels(tokens, spans, report: Counter):
    labels = ["O"] * len(tokens)
    ti = 0
    for start, end, dtype in spans:
        if dtype not in TYPES:  # "Unannotated" segments stay O
            continue
        first = True
        while ti < len(tokens) and tokens[ti][2] <= start:
            ti += 1
        j = ti
        while j < len(tokens) and tokens[j][1] < end:
            tok_start, tok_end = tokens[j][1], tokens[j][2]
            inside = min(tok_end, end) - max(tok_start, start)
            if inside * 2 >= tok_end - tok_start:  # majority of the token
                labels[j] = f"{'B' if first else 'I'}-{dtype}"
                first = False
            else:
                report["tokens_straddling_boundary"] += 1
            j += 1
        if first:
            report["spans_with_no_token"] += 1
    return labels


def spans_from_bio(labels):
    """Decode BIO into [(type, first_token, last_token)]; lenient on I-X starts."""
    spans, cur = [], None
    for i, lab in enumerate(labels):
        if lab == "O":
            if cur:
                spans.append(tuple(cur))
            cur = None
            continue
        prefix, dtype = lab.split("-", 1)
        if prefix == "B" or cur is None or cur[0] != dtype:
            if cur:
                spans.append(tuple(cur))
            cur = [dtype, i, i]
        else:
            cur[2] = i
    if cur:
        spans.append(tuple(cur))
    return spans


def illegal_transitions(labels):
    bad, prev = 0, "O"
    for lab in labels:
        if lab.startswith("I-") and prev[2:] != lab[2:]:
            bad += 1
        prev = lab
    return bad


# ---------------------------------------------------------------- splitting
def assign_splits(essays: pd.DataFrame, held_out, val_fraction, seed):
    rng = random.Random(seed)
    split = {}
    seen_train = []
    for row in essays.itertuples(index=False):
        if row.prompt_name in held_out:
            split[row.essay_id_comp] = "test_unseen"
        elif row.competition_set == "test":
            split[row.essay_id_comp] = "test_seen"
        else:
            seen_train.append(row.essay_id_comp)
    seen_train.sort()
    rng.shuffle(seen_train)
    n_val = int(len(seen_train) * val_fraction)
    for i, eid in enumerate(seen_train):
        split[eid] = "val" if i < n_val else "train"
    return split


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--train-zip", type=Path, required=True)
    ap.add_argument("--test-zip", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--held-out", nargs="+", default=DEFAULT_HELD_OUT)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.concat(
        [
            read_zipped_csv(args.train_zip),
            read_zipped_csv(args.test_zip, TEST_ZIP_PASSWORD),
        ],
        ignore_index=True,
    )
    unknown = set(args.held_out) - set(df.prompt_name.unique())
    if unknown:
        raise SystemExit(f"Unknown held-out prompts: {sorted(unknown)}")

    essays = (
        df.drop_duplicates("essay_id_comp")
        .sort_values("essay_id_comp")
        .reset_index(drop=True)
    )
    split_of = assign_splits(essays, set(args.held_out), args.val_fraction, args.seed)

    report = Counter()
    label_counts = {s: Counter() for s in ("train", "val", "test_seen", "test_unseen")}
    span_counts = {s: Counter() for s in label_counts}
    lengths = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        # mtime=0 makes the .gz output identical on every run
        s: io.TextIOWrapper(
            gzip.GzipFile(args.out_dir / f"{s}.jsonl.gz", mode="wb", mtime=0),
            encoding="utf-8",
        )
        for s in label_counts
    }

    grouped = df.sort_values(["essay_id_comp", "discourse_start"]).groupby(
        "essay_id_comp", sort=True
    )
    for eid, segs in grouped:
        meta = segs.iloc[0]
        text = meta.full_text
        spans = align_segments(text, segs, report)
        tokens = tokenize(text)
        labels = bio_labels(tokens, spans, report)

        bad = illegal_transitions(labels)
        report["illegal_bio_transitions"] += bad
        expected = sum(1 for s in spans if s[2] in TYPES)
        recovered = len(spans_from_bio(labels))
        if recovered != expected:
            report["essays_span_roundtrip_mismatch"] += 1

        split = split_of[eid]
        label_counts[split].update(labels)
        span_counts[split].update(t for t, _, _ in spans_from_bio(labels))
        lengths.append(len(tokens))
        record = {
            "essay_id": eid,
            "prompt": meta.prompt_name,
            "task": meta.task,
            "official_set": meta.competition_set,
            "tokens": [t for t, _, _ in tokens],
            "offsets": [[s, e] for _, s, e in tokens],
            "labels": labels,
            "text": text,
        }
        writers[split].write(json.dumps(record, ensure_ascii=False) + "\n")
        report["essays_written"] += 1

    for w in writers.values():
        w.close()

    split_sizes = Counter(split_of.values())
    lengths_sorted = sorted(lengths)
    stats = {
        "labels": LABELS,
        "held_out_prompts": args.held_out,
        "seed": args.seed,
        "essays_per_split": dict(split_sizes),
        "tokens_per_essay": {
            "median": lengths_sorted[len(lengths_sorted) // 2],
            "p90": lengths_sorted[int(len(lengths_sorted) * 0.9)],
            "max": lengths_sorted[-1],
            "share_over_512_word_tokens": round(
                sum(l > 512 for l in lengths) / len(lengths), 4
            ),
        },
        "spans_per_split": {s: dict(c) for s, c in span_counts.items()},
        "token_labels_per_split": {s: dict(c) for s, c in label_counts.items()},
    }
    (args.out_dir / "splits.json").write_text(
        json.dumps(split_of, indent=0, sort_keys=True)
    )
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    (args.out_dir / "cleaning_report.json").write_text(
        json.dumps(dict(report), indent=2, sort_keys=True)
    )
    print(json.dumps(dict(report), indent=2, sort_keys=True))
    print(json.dumps(stats["essays_per_split"], indent=2))


if __name__ == "__main__":
    main()
