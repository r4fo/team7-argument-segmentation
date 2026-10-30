"""Hand-checked cases for src/evaluate.py. Run: python -m pytest tests/"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import illegal_transitions, score, spans_from_bio  # noqa: E402


def run(gold, pred):
    return score([{"essay_id": "e", "labels": gold}], {"e": pred})


def test_perfect_prediction():
    gold = ["B-Claim", "I-Claim", "O", "B-Evidence", "I-Evidence"]
    r = run(gold, gold)
    assert r["overlap_span"]["macro_f1"] == 1.0
    assert r["exact_span"]["macro_f1"] == 1.0
    assert r["token_accuracy"] == 1.0


def test_lecture_example_exact_match():
    # W6 practice slide: gold Ada (PER), New York (LOC); predicted Ada, York.
    # Here as Claim / Evidence: TP=1, FP=1, FN=1 -> micro F1 = 2/(2+1+1) = 0.5
    gold = ["B-Claim", "O", "B-Evidence", "I-Evidence", "O"]
    pred = ["B-Claim", "O", "O", "B-Evidence", "O"]
    r = run(gold, pred)
    assert r["exact_span"]["micro_f1"] == 0.5
    # York covers 50% of New York and 100% of itself -> overlap match
    assert r["overlap_span"]["micro_f1"] == 1.0


def test_overlap_threshold():
    gold = ["B-Claim"] + ["I-Claim"] * 3 + ["O"] * 4  # tokens 0-3
    short = ["O"] * 3 + ["B-Claim"] + ["O"] * 4  # covers 1/4 of gold
    r = run(gold, short)
    assert r["overlap_span"]["micro_f1"] == 0.0
    # right type, overlapping, too short: a boundary error, not a spurious span
    assert r["overlap_errors"] == {"boundary_error": 1}


def test_type_error_and_spurious():
    gold = ["B-Claim", "I-Claim", "O", "O"]
    pred = ["B-Rebuttal", "I-Rebuttal", "O", "B-Lead"]
    errs = run(gold, pred)["overlap_errors"]
    assert errs["type_error"] == 1
    assert errs["spurious"] == 1


def test_lenient_decoding_and_illegal_count():
    labels = ["O", "I-Claim", "I-Claim", "B-Claim", "I-Evidence"]
    assert spans_from_bio(labels) == [("Claim", 1, 2), ("Claim", 3, 3), ("Evidence", 4, 4)]
    assert illegal_transitions(labels) == 2


def test_adjacent_same_type_spans_stay_separate():
    gold = ["B-Claim", "I-Claim", "B-Claim", "I-Claim"]
    merged = ["B-Claim", "I-Claim", "I-Claim", "I-Claim"]
    r = run(gold, merged)
    assert r["exact_span"]["per_class"]["Claim"]["recall"] == 0.0
