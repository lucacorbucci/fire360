import re
from collections import Counter

import pytest
from fire360.explanations.decision_tree import compute_stability_dt, parse_explanation_dt


@pytest.mark.parametrize(
    "exp_1, exp_2, output",
    [
        (["a", "b"], ["a", "b"], 1.0),
        (["f", "c"], ["a", "b"], 0.0),
        (["d", "e"], ["d", "c"], 0.5),
        (["a", "b", "c", "d"], ["a", "b", "c", "e"], 0.75),
        (["a", "b", "a"], ["a", "b", "d"], 0.67),
        (["a", "c", "a"], ["a", "b", "d"], 0.33),
        (["a", "c", "a"], ["a", "c", "a"], 1.0),
        (["a", "b", "a"], ["a", "c", "a"], 0.67),
        (["a", "a", "a"], ["a", "a", "a"], 1.0),
        (["a", "g", "a"], ["a", "f", "s"], 0.33),
    ],
)
def test_stability(exp_1: list[str], exp_2: list[str], output: float) -> None:
    # Two identical explanations should have maximum stability (i.e. 1.0)
    explanations = [exp_1, exp_2]
    stability = compute_stability_dt(explanations)
    assert round(stability, 2) == output


@pytest.mark.parametrize(
    "explanation, expected",
    [
        (
            "(['(time = 41.0) <= 51.5', '(bvp_open = 43.0) <= 55.5', 'Leaf node 2 reached, prediction: 0.0'], 0)",
            ["time", "bvp_open"],
        ),
        ("'(age = 45)'", ["age"]),
        ("'(height = 180)' and '(weight = 75)'", ["height", "weight"]),
        (
            "'(brand = nike)' '(price = 100)'",
            ["brand", "price"],
        ),
        ("no match here", []),
        ("   '(  temp    = 30)' extra text", ["temp"]),
        # Multiple matches with mixed spacing.
        ("'(pressure=101)' and '(humidity = 80)'", ["pressure", "humidity"]),
        # Tuple input with a single match in list.
        (
            "['(speed = 50) ']",
            ["speed"],
        ),
        (
            "'(speed = 50)' and '(speed = 10)' and '(test=111)'",
            ["speed", "speed", "test"],
        ),
        # Tuple input with no match.
        (
            "this text has no match",
            [],
        ),
        (
            "'(marital-status_ Married-civ-spouse = False)' <= 0.5', '(hours-per-week = 40)' <= 44.5', 'Leaf node 2 reached, prediction: 0",
            ["marital-status_ Married-civ-spouse", "hours-per-week"],
        ),
    ],
)
def test_parse_explanation(explanation: str, expected: list[str]) -> None:
    # Adjust the test input when explanation is a tuple.
    explanation = str(explanation)
    result = parse_explanation_dt(explanation)
    assert result == expected
