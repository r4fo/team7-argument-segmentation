"""Explain where a model's span errors come from.

For every gold span the model did not match (overlap rule, as in
src/evaluate.py), look at what the model predicted over that span's tokens
and take the majority label:

    other type   the span was absorbed into, or relabelled as, another role
    same type    right role, wrong boundaries (merged with or split from a
                 neighbouring span of the same role)
    O            the model predicted nothing there (truly missed)

Also counts predicted spans that swallow two or more gold spans of the same
role, the most common confusions (gold role -> predicted role), and writes a
few example essays for the presentation.

Usage
-----
    python src/analyze_errors.py \
        --gold data/processed/test_seen.jsonl.gz \
        --pred results/predictions/bert-base-cased_crf_test_seen.jsonl.gz \
        --out  results/bert-base-cased_crf_test_seen_errors.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluate import match_spans, read_jsonl, spans_from_bio


def analyze(gold_records, pred_by_id, n_examples=3):
    fate = Counter()
    confusion = Counter()
    swallow = 0
    unmatched_total = gold_total = 0
    examples = {"absorbed": [], "clean": []}
    for g in gold_records:
        labels = pred_by_id[g["essay_id"]]
        G, P = spans_from_bio(g["labels"]), spans_from_bio(labels)
        matched = {gi for gi, _ in match_spans(G, P, "overlap")}
        gold_total += len(G)
        absorbed_here = []
        for gi, (t, s, e) in enumerate(G):
            if gi in matched:
                continue
            unmatched_total += 1
            kinds = Counter(lab[2:] if lab != "O" else "O" for lab in labels[s : e + 1])
            dom = max(sorted(kinds), key=lambda k: kinds[k])
            if dom == "O":
                fate["missed (predicted O)"] += 1
            elif dom == t:
                fate["right role, wrong boundaries"] += 1
            else:
                fate["absorbed into / labelled as another role"] += 1
                confusion[f"{t} -> {dom}"] += 1
                absorbed_here.append((t, dom, s, e))
        for pt, ps, pe in P:
            if sum(1 for t, s, e in G if t == pt and s >= ps and e <= pe) >= 2:
                swallow += 1
        n = len(g["tokens"])
        if absorbed_here and 150 <= n <= 300 and len(examples["absorbed"]) < n_examples:
            t, dom, s, e = absorbed_here[0]
            examples["absorbed"].append(
                {"essay_id": g["essay_id"], "gold": t, "predicted": dom,
                 "text": " ".join(g["tokens"][s : e + 1])}
            )
        if not absorbed_here and len(G) >= 6 and len(matched) == len(G) and 150 <= n <= 300 \
                and len(examples["clean"]) < n_examples:
            examples["clean"].append({"essay_id": g["essay_id"], "spans": len(G)})
    return {
        "gold_spans": gold_total,
        "unmatched_gold_spans": unmatched_total,
        "unmatched_by_prediction": dict(fate.most_common()),
        "predicted_spans_swallowing_2plus_gold": swallow,
        "top_confusions": dict(confusion.most_common(10)),
        "examples": examples,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    pred = {r["essay_id"]: r["labels"] for r in read_jsonl(args.pred)}
    result = analyze(read_jsonl(args.gold), pred)
    result.update(gold_file=str(args.gold), pred_file=str(args.pred))
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
