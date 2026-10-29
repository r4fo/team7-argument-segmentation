"""Score token-level BIO predictions as argument spans.

Every model writes its predictions in the same format and is scored here, so
results are comparable. The protocol is in docs/metrics_protocol.md.

Prediction file: JSON lines (optionally .gz), one essay per line, labels
aligned with the tokens of the gold file:

    {"essay_id": "003FDC7E6F20", "labels": ["B-Position", "I-Position", ...]}

Usage
-----
    python src/evaluate.py \
        --gold data/processed/test_seen.jsonl.gz \
        --pred results/predictions/most_frequent_test_seen.jsonl.gz \
        --out  results/most_frequent_test_seen.json
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

TYPES = [
    "Lead",
    "Position",
    "Claim",
    "Counterclaim",
    "Rebuttal",
    "Evidence",
    "Concluding Statement",
]


# ---------------------------------------------------------------- io
def read_jsonl(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_jsonl_gz(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(path, mode="wb", mtime=0) as raw:
        for rec in records:
            raw.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))


# ---------------------------------------------------------------- spans
def spans_from_bio(labels):
    """Decode BIO into [(type, first, last)]. An I-X that does not continue an
    X span starts a new span (lenient decoding, so softmax outputs with
    illegal transitions are still scored)."""
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


def _overlap(a, b):
    """Token overlap of two (first, last) inclusive ranges."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def _length(a):
    return a[1] - a[0] + 1


def overlap_match(a, b, threshold=0.5):
    inter = _overlap(a, b)
    return inter >= threshold * _length(a) and inter >= threshold * _length(b)


def match_spans(gold, pred, rule):
    """One-to-one matching of same-type spans. Returns matched (g, p) index pairs."""
    pairs = []
    for gi, (gt, gs, ge) in enumerate(gold):
        for pi, (pt, ps, pe) in enumerate(pred):
            if gt != pt:
                continue
            if rule == "exact":
                ok = (gs, ge) == (ps, pe)
            else:
                ok = overlap_match((gs, ge), (ps, pe))
            if ok:
                pairs.append((_overlap((gs, ge), (ps, pe)), gi, pi))
    pairs.sort(reverse=True)
    used_g, used_p, matched = set(), set(), []
    for _, gi, pi in pairs:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            matched.append((gi, pi))
    return matched


def classify_errors(gold, pred, matched):
    """Label unmatched spans: type error, boundary error, missed, spurious."""
    errors = Counter()
    mg = {g for g, _ in matched}
    mp = {p for _, p in matched}
    for gi, (gt, gs, ge) in enumerate(gold):
        if gi in mg:
            continue
        cands = [(pt, ps, pe) for pi, (pt, ps, pe) in enumerate(pred) if pi not in mp]
        if any(pt != gt and overlap_match((gs, ge), (ps, pe)) for pt, ps, pe in cands):
            errors["type_error"] += 1
        elif any(pt == gt and _overlap((gs, ge), (ps, pe)) > 0 for pt, ps, pe in cands):
            errors["boundary_error"] += 1
        else:
            errors["missed"] += 1
    for pi, (pt, ps, pe) in enumerate(pred):
        if pi in mp:
            continue
        if not any(_overlap((gs, ge), (ps, pe)) > 0 for _, gs, ge in gold):
            errors["spurious"] += 1
    return errors


# ---------------------------------------------------------------- metrics
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def summarize(counts):
    per_class = {}
    for t in TYPES:
        tp, fp, fn = counts[t]["tp"], counts[t]["fp"], counts[t]["fn"]
        if tp + fp + fn == 0:
            continue  # no gold and no predicted spans of this type
        p, r, f = prf(tp, fp, fn)
        per_class[t] = {"precision": p, "recall": r, "f1": f, "support": tp + fn}
    macro = round(sum(v["f1"] for v in per_class.values()) / len(per_class), 4) if per_class else 0.0
    tp = sum(counts[t]["tp"] for t in TYPES)
    fp = sum(counts[t]["fp"] for t in TYPES)
    fn = sum(counts[t]["fn"] for t in TYPES)
    _, _, micro = prf(tp, fp, fn)
    return {"macro_f1": macro, "micro_f1": micro, "per_class": per_class}


def score(gold_records, pred_by_id):
    counts = {rule: {t: Counter() for t in TYPES} for rule in ("overlap", "exact")}
    errors = Counter()
    tokens = correct = illegal = 0
    missing = []
    for g in gold_records:
        eid = g["essay_id"]
        if eid not in pred_by_id:
            missing.append(eid)
            continue
        gl, pl = g["labels"], pred_by_id[eid]
        if len(gl) != len(pl):
            raise ValueError(f"{eid}: {len(pl)} predicted labels for {len(gl)} tokens")
        tokens += len(gl)
        correct += sum(a == b for a, b in zip(gl, pl))
        illegal += illegal_transitions(pl)
        gold, pred = spans_from_bio(gl), spans_from_bio(pl)
        for rule in ("overlap", "exact"):
            matched = match_spans(gold, pred, rule)
            mg = {g_ for g_, _ in matched}
            mp = {p_ for _, p_ in matched}
            for gi, (t, _, _) in enumerate(gold):
                counts[rule][t]["tp" if gi in mg else "fn"] += 1
            for pi, (t, _, _) in enumerate(pred):
                if pi not in mp:
                    counts[rule][t]["fp"] += 1
            if rule == "overlap":
                errors.update(classify_errors(gold, pred, matched))
    if missing:
        raise ValueError(f"{len(missing)} essays have no prediction, e.g. {missing[:3]}")
    return {
        "overlap_span": summarize(counts["overlap"]),
        "exact_span": summarize(counts["exact"]),
        "token_accuracy": round(correct / tokens, 4) if tokens else 0.0,
        "illegal_bio_transitions": illegal,
        "overlap_errors": dict(errors),
        "essays": len(pred_by_id),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    pred_by_id = {r["essay_id"]: r["labels"] for r in read_jsonl(args.pred)}
    result = score(read_jsonl(args.gold), pred_by_id)
    result["gold_file"] = str(args.gold)
    result["pred_file"] = str(args.pred)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
