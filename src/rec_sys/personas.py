bank_guarantor = {
    "features": ["model_confidence", "stability", "fidelity_neighbours"],
    "explanation_type": ["model", "rules_explanation", "rules_explanation"],
    "threshold": [0.95, 0.95, 0.95],
    "left": {
        "prediction": ["Rules", "Counterfactual"],
    },
    "right": {
        "prediction": ["Rules"],
    },
}

bank_teller = {
    "features": ["model_confidence"],
    "explanation_type": ["model"],
    "threshold": [0.95],
    "left": {
        "prediction": ["Confidence", "Counterexamples"],
    },
    "right": {
        "features": ["fidelity_neighbours"],
        "explanation_type": ["fi_explanation"],
        "threshold": 0.95,
        "left": {"prediction": ["Rules"]},
        "right": {"prediction": ["FI"]},
    },
}

manager = {
    "features": ["model_confidence"],
    "explanation_type": ["model"],
    "threshold": [0.95],
    "left": {
        "prediction": ["Confidence", "Counterfactual"],
    },
    "right": {
        "features": ["fidelity_neighbours", "stability"],
        "explanation_type": ["fi_explanation", "fi_explanation"],
        "threshold": [0.95, 0.95],
        "left": {
            "features": ["stability", "fidelity_neighbours"],
            "explanation_type": ["rules_explanation", "rules_explanation"],
            "threshold": [0.95, 0.95],
            "left": {"prediction": ["Exemplars"]},
            "right": {"prediction": ["Rules"]},
        },
        "right": {"prediction": ["FI"]},
    },
}

bank_scenario_personas = {
    "bank_guarantor": bank_guarantor,
    "bank_teller": bank_teller,
    "manager": manager,
}
