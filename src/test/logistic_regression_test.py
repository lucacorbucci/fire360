import re

import pytest

from synth_xai.explanations.logistic_regression import compute_stability_lr, parse_explanation_lr


@pytest.mark.parametrize(
    "explanation, expected",
    [
        (" 'sex_binary   : coefficient=-12.52348470819837, value=1", ["sex_binary"]),
        ("age: coefficient=5.3, value=42", ["age"]),
        (
            "sex_binary: coefficient=-12.52348470819837, value=1', 'edu_level: coefficient=-7.3348013481681456, value=5', 'citizenship: coefficient=4.31405029089262, value=1",
            ["sex_binary", "edu_level", "citizenship"],
        ),
        ("   'income : coefficient=-0.2, value=35000", ["income"]),
        ("coefficient=5, value=42", []),
        ("\t' education '   : coefficient=3.14, value=16", []),
        (
            "['sex_binary: coefficient=-12.52348470819837, value=1', 'edu_level: coefficient=-7.3348013481681456, value=5', 'citizenship: coefficient=4.31405029089262, value=1', 'economic_status: coefficient=3.7491310198873933, value=111', 'age: coefficient=-3.350037337090565, value=6', 'country_birth: coefficient=2.267282994351346, value=1', 'Marital_status: coefficient=-1.4862653095013403, value=1', 'household_size: coefficient=-0.7562493114228797, value=112', 'prev_residence_place: coefficient=0.5042693628520929, value=1', 'household_position: coefficient=-0.11770076803090257, value=1131', 'cur_eco_activity: coefficient=-0.09708095043104509, value=122']",
            [
                "sex_binary",
                "edu_level",
                "citizenship",
                "economic_status",
                "age",
                "country_birth",
                "Marital_status",
                "household_size",
                "prev_residence_place",
                "household_position",
                "cur_eco_activity",
            ],
        ),
    ],
)
def test_parse_explanation_lr(explanation: str, expected: list) -> None:
    assert parse_explanation_lr(explanation) == expected


@pytest.mark.parametrize(
    "explanations, expected",
    [
        (
            [
                [
                    "sex_binary",
                    "edu_level",
                    "citizenship",
                    "economic_status",
                ],
                [
                    "sex_binary",
                    "edu_level",
                    "citizenship",
                    "economic_status",
                ],
            ],
            1.0,
        ),
        (
            [
                [
                    "sex_binary",
                    "citizenship",
                    "edu_level",
                    "economic_status",
                ],
                [
                    "sex_binary",
                    "edu_level",
                    "citizenship",
                    "economic_status",
                ],
            ],
            0.5,
        ),
        (
            [
                [
                    "economic_status",
                    "sex_binary",
                    "citizenship",
                    "edu_level",
                ],
                [
                    "sex_binary",
                    "edu_level",
                    "citizenship",
                    "economic_status",
                ],
            ],
            0.25,
        ),
        (
            [
                [
                    "economic_status",
                    "sex_binary",
                    "citizenship",
                    "edu_level",
                ],
                [
                    "sex_binary",
                    "edu_level",
                    "economic_status",
                    "citizenship",
                ],
            ],
            0,
        ),
    ],
)
def test_compute_stability_lr(explanations: list[str], expected: float) -> None:
    assert compute_stability_lr(explanations) == expected
