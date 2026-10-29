"""Independent checks of the offset repair in prepare_data.py.

Every number we report about the cleaning comes from this script, so anyone
can rerun it and get the same figures.

Checks
------
A. Official offsets: how often does essay[start:end] reproduce the segment
   text? Broken down by the segment's position in its essay, to test the
   "drift grows along the essay" diagnosis.
B. End offsets: for segments whose start is correct, does the released end
   ever cut real text? (The annotation text carries trailing whitespace that
   the end offset excludes; that alone is not an error.)
C. Repaired offsets: does essay[start:end] equal the segment text for every
   kept segment, and are the kept segments in order without overlap?
D. Drift shape: is the repaired-minus-official shift non-negative and
   non-decreasing within each essay?
E. Token labels: decode our BIO labels back into spans and compare their text
   with the repaired segment text.
F. Impact: build BIO labels from the official offsets as released and count
   how many tokens get a different label than with the repaired offsets.
H. Repeated phrases: when a segment's text occurs more than once in its
   essay, is the copy we chose the one nearest to the released start offset?
G. Manual audit sample: write a stratified sample of segments (repeated
   phrases, large shifts, others) with their essay context to a Markdown
   file for two reviewers to check.

Usage
-----
    python src/verify_alignment.py \
        --train-zip data/raw/persuade_corpus_2.0_train.zip \
        --test-zip data/raw/persuade_corpus_2.0_test.zip \
        --report data/processed/verification_report.json \
        --sample docs/verification_sample.md
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from prepare_data import (
    TEST_ZIP_PASSWORD,
    TYPES,
    align_segments,
    bio_labels,
    read_zipped_csv,
    spans_from_bio,
    tokenize,
)


def norm(s: str) -> str:
    return " ".join(s.split())


def pct(a: int, b: int) -> float:
    return round(100 * a / b, 2) if b else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--train-zip", type=Path, required=True)
    ap.add_argument("--test-zip", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=Path("data/processed/verification_report.json"))
    ap.add_argument("--sample", type=Path, default=Path("docs/verification_sample.md"))
    ap.add_argument("--sample-size", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.concat(
        [read_zipped_csv(args.train_zip), read_zipped_csv(args.test_zip, TEST_ZIP_PASSWORD)],
        ignore_index=True,
    )

    c = Counter()
    by_position = defaultdict(lambda: [0, 0])  # position bucket -> [correct, total]
    impact_by_type = defaultdict(lambda: [0, 0])  # gold type -> [changed, total]
    audit_pool = []

    for eid, segs in df.sort_values(["essay_id_comp", "discourse_start"]).groupby("essay_id_comp"):
        text = segs.iloc[0].full_text

        # ---- A + B: official offsets as released
        official = []
        for pos, seg in enumerate(segs.itertuples(index=False)):
            raw = seg.discourse_text if isinstance(seg.discourse_text, str) else ""
            if not raw.strip():
                continue
            c["A_segments"] += 1
            ok = text[seg.discourse_start : seg.discourse_end].strip() == raw.strip()
            c["A_official_correct"] += ok
            bucket = str(pos + 1) if pos < 9 else "10+"
            by_position[bucket][0] += ok
            by_position[bucket][1] += 1
            lead = len(raw) - len(raw.lstrip())
            stripped = raw.strip()
            c["B_trailing_whitespace"] += raw != raw.rstrip()
            if text[seg.discourse_start + lead : seg.discourse_start + lead + len(stripped)] == stripped:
                c["B_start_correct"] += 1
                c["B_end_cuts_text"] += seg.discourse_end < seg.discourse_start + lead + len(stripped)
            official.append((seg.discourse_start, seg.discourse_end, seg.discourse_type, raw.strip()))

        # ---- C: repaired offsets
        repaired = align_segments(text, segs, Counter())
        prev_end = 0
        for start, end, _ in repaired:
            c["C_kept"] += 1
            c["C_overlap_or_disorder"] += start < prev_end
            prev_end = end
        # match each repaired span to its annotation text (same order, empties removed)
        texts = [o[3] for o in official]
        j = 0
        shifts = []
        for start, end, dtype in repaired:
            while j < len(official) and not norm(text[start:end]) == norm(texts[j]):
                j += 1
            if j == len(official):
                c["C_text_mismatch"] += 1
                break
            c["C_text_equal"] += 1
            shifts.append(start - official[j][0])
            occurrences, m = [], text.find(texts[j])
            while m != -1:
                occurrences.append(m)
                m = text.find(texts[j], m + 1)
            repeated = len(occurrences) > 1
            if repeated:
                # H: is the chosen copy the one nearest to where the annotator pointed?
                nearest = min(occurrences, key=lambda m: abs(m - official[j][0]))
                kind = "arg" if dtype in TYPES else "unannotated"
                c[f"H_repeated_{kind}"] += 1
                c[f"H_nearest_{kind}"] += start == nearest
                c["H_repeated_segments"] += 1
                c["H_chose_nearest_copy"] += start == nearest
            audit_pool.append(
                (eid, dtype, official[j][0], official[j][1], start, end, repeated, texts[j], occurrences)
            )
            j += 1

        # ---- D: drift shape
        if shifts:
            c["D_essays"] += 1
            c["D_all_shifts_nonnegative"] += min(shifts) >= 0
            c["D_monotone_nondecreasing"] += all(b >= a for a, b in zip(shifts, shifts[1:]))
            c["D_last_segment_shift_sum"] += shifts[-1]

        # ---- E: BIO round trip
        tokens = tokenize(text)
        labels = bio_labels(tokens, repaired, Counter())
        decoded = spans_from_bio(labels)
        typed = [(s, e, t) for s, e, t in repaired if t in TYPES]
        c["E_spans"] += len(typed)
        if len(decoded) == len(typed):
            for (dt, a, b), (s, e, t) in zip(decoded, typed):
                tok_text = text[tokens[a][1] : tokens[b][2]]
                c["E_type_equal"] += dt == t
                c["E_text_equal"] += norm(tok_text) == norm(text[s:e])
        else:
            c["E_essays_count_mismatch"] += 1

        # ---- F: impact of using official offsets directly
        official_spans = [(s, e, t) for s, e, t, _ in official]
        labels_official = bio_labels(tokens, official_spans, Counter())
        spans_rep = {(t, a, b) for t, a, b in decoded}
        spans_off = {(t, a, b) for t, a, b in spans_from_bio(labels_official)}
        c["F_spans"] += len(spans_rep)
        c["F_spans_boundary_changed"] += len(spans_rep - spans_off)
        for lo, lr in zip(labels_official, labels):
            gold_type = lr[2:] if lr != "O" else "O"
            impact_by_type[gold_type][1] += 1
            if lo != lr:
                impact_by_type[gold_type][0] += 1
                c["F_tokens_changed"] += 1
            c["F_tokens"] += 1

    # ---- G: manual audit sample
    # Stratified: the hard cases (phrase repeated in the essay, large shift)
    # are oversampled because a plain random sample would rarely contain them.
    rng = random.Random(args.seed)
    texts_by_id = df.drop_duplicates("essay_id_comp").set_index("essay_id_comp").full_text
    strata = {
        "repeated phrase": [r for r in audit_pool if r[6]],
        "shift of 3+ characters": [r for r in audit_pool if not r[6] and r[4] - r[2] >= 3],
        "other": [r for r in audit_pool if not r[6] and r[4] - r[2] < 3],
    }
    per = args.sample_size // len(strata)
    rows = []
    for name, pool in strata.items():
        rows += [(name, r) for r in rng.sample(pool, min(per, len(pool)))]
    lines = [
        "# Manual audit of repaired offsets",
        "",
        f"{len(rows)} segments (seed {args.seed}), stratified: {per} where the segment text "
        f"appears more than once in the essay ({len(strata['repeated phrase']):,} such segments), "
        f"{per} with a shift of 3+ characters ({len(strata['shift of 3+ characters']):,}), "
        f"{per} others ({len(strata['other']):,}).",
        "",
        "**How to review.** Two reviewers check each case independently. A case is "
        "correct when both hold:",
        "",
        "1. The repaired span (between ⟦ ⟧) starts and ends at the same words as the "
        "annotation text, with nothing missing or extra.",
        "2. It is the right copy: when the phrase occurs more than once, the chosen copy "
        "is the one at the annotator's position (the official start), not another one.",
        "",
        "The official slice is shown to make the drift visible; it is expected to be off by "
        "the shift.",
        "",
    ]

    def show(t, s, e, width=60):
        span = t[s:e].replace("\n", " ")
        if len(span) > 200:
            span = span[:100] + " … " + span[-100:]
        before = t[max(0, s - width) : s].replace("\n", " ")
        after = t[e : e + width].replace("\n", " ")
        return f"…{before}⟦{span}⟧{after}…"

    for k, (stratum, row) in enumerate(rows, 1):
        eid, dtype, os_, oe, rs, re_, _, ann, occ = row
        t = texts_by_id[eid]
        ann_short = ann if len(ann) <= 200 else ann[:100] + " … " + ann[-100:]
        lines += [
            f"## {k}. {dtype} · {stratum} · essay {eid}",
            "",
            f"- **Annotation text:** {ann_short!r}",
            f"- **Official slice** [{os_}:{oe}]: {show(t, os_, oe)}",
            f"- **Repaired span** [{rs}:{re_}], shift {rs - os_:+d}: {show(t, rs, re_)}",
        ]
        if len(occ) > 1:
            where = ", ".join(f"{m}{' (chosen)' if m == rs else ''}" for m in occ)
            lines.append(f"- **Copies of this text start at:** {where}; official start {os_}")
        lines += [
            "- [ ] Reviewer A: correct   - [ ] Reviewer B: correct",
            "",
        ]
    lines += ["## Result", "", "- Reviewer A correct: __ / " + str(len(rows)),
              "- Reviewer B correct: __ / " + str(len(rows)),
              "- Disagreements and resolution: ", ""]
    args.sample.parent.mkdir(parents=True, exist_ok=True)
    args.sample.write_text("\n".join(lines), encoding="utf-8")

    report = {
        "A_official_offsets_correct_pct": pct(c["A_official_correct"], c["A_segments"]),
        "A_official_offsets_wrong_pct": pct(c["A_segments"] - c["A_official_correct"], c["A_segments"]),
        "A_correct_pct_by_segment_position": {
            k: pct(v[0], v[1]) for k, v in sorted(by_position.items(), key=lambda kv: (kv[0] == "10+", int(kv[0]) if kv[0] != "10+" else 0))
        },
        "B_annotation_text_has_trailing_whitespace_pct": pct(c["B_trailing_whitespace"], c["A_segments"]),
        "B_end_cuts_real_text_when_start_correct_pct": pct(c["B_end_cuts_text"], c["B_start_correct"]),
        "C_repaired_text_equal_pct": pct(c["C_text_equal"], c["C_kept"]),
        "C_overlap_or_disorder": c["C_overlap_or_disorder"],
        "D_essays_shift_nonnegative_pct": pct(c["D_all_shifts_nonnegative"], c["D_essays"]),
        "D_essays_shift_monotone_pct": pct(c["D_monotone_nondecreasing"], c["D_essays"]),
        "D_mean_shift_of_last_segment_chars": round(c["D_last_segment_shift_sum"] / max(c["D_essays"], 1), 2),
        "E_bio_roundtrip_type_equal_pct": pct(c["E_type_equal"], c["E_spans"]),
        "E_bio_roundtrip_text_equal_pct": pct(c["E_text_equal"], c["E_spans"]),
        "E_essays_span_count_mismatch": c["E_essays_count_mismatch"],
        "F_tokens_with_different_label_pct": pct(c["F_tokens_changed"], c["F_tokens"]),
        "F_spans_with_different_token_boundaries_pct": pct(c["F_spans_boundary_changed"], c["F_spans"]),
        "H_repeated_phrase_segments": c["H_repeated_segments"],
        "H_chose_copy_nearest_official_start_pct": pct(c["H_chose_nearest_copy"], c["H_repeated_segments"]),
        "H_argument_segments_repeated": c["H_repeated_arg"],
        "H_argument_segments_nearest_copy_pct": pct(c["H_nearest_arg"], c["H_repeated_arg"]),
        "H_unannotated_segments_repeated": c["H_repeated_unannotated"],
        "H_unannotated_segments_nearest_copy_pct": pct(c["H_nearest_unannotated"], c["H_repeated_unannotated"]),
        "F_changed_pct_by_true_label": {
            t: pct(v[0], v[1]) for t, v in sorted(impact_by_type.items())
        },
        "counts": dict(c),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "counts"}, indent=2))


if __name__ == "__main__":
    main()
