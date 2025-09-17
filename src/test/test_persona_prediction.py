import pytest

from rec_sys.data_preparation import PersonaRecommenderDatasetGenerator
from rec_sys.personas import bank_guarantor, bank_scenario_personas, bank_teller, manager


@pytest.fixture
def generator():
    """Test fixture to create a PersonaRecommenderDatasetGenerator instance."""
    return PersonaRecommenderDatasetGenerator(
        personas=bank_scenario_personas,
        max_rejections=2,
    )


def test_bank_guarantor_prediction_scenario_1(generator):
    """Test bank guarantor with high confidence, high fidelity, high stability -> Rules only."""
    bank_guarantor_data_1 = {
        "rules_explanation": {
            "confidence_bb": 0.99,
            "fidelity_neighbours": 1.0,
            "stability": 1.0,
        },
    }
    result = generator.get_persona_prediction(bank_guarantor, bank_guarantor_data_1)
    assert result == ["Rules"]


def test_bank_guarantor_prediction_scenario_2(generator):
    """Test bank guarantor with high confidence, medium fidelity, high stability -> Rules + Counterfactual."""
    bank_guarantor_data_2 = {
        "rules_explanation": {
            "confidence_bb": 0.99,
            "fidelity_neighbours": 0.5,
            "stability": 1.0,
        },
    }
    result = generator.get_persona_prediction(bank_guarantor, bank_guarantor_data_2)
    assert result == ["Rules", "Counterfactual"]


def test_bank_guarantor_prediction_scenario_3(generator):
    """Test bank guarantor with medium confidence, high fidelity, high stability -> Rules + Counterfactual."""
    bank_guarantor_data_3 = {
        "rules_explanation": {
            "confidence_bb": 0.94,
            "fidelity_neighbours": 1.0,
            "stability": 1.0,
        },
    }
    result = generator.get_persona_prediction(bank_guarantor, bank_guarantor_data_3)
    assert result == ["Rules", "Counterfactual"]


def test_bank_teller_prediction_scenario_1(generator):
    """Test bank teller with medium confidence -> Confidence + Counterexamples."""
    bank_teller_data_1 = {
        "rules_explanation": {
            "confidence_bb": 0.93,
            "fidelity_neighbours": 1.0,
            "stability": 1.0,
        },
    }
    result = generator.get_persona_prediction(bank_teller, bank_teller_data_1)
    assert result == ["Confidence", "Counterexamples"]


def test_bank_teller_prediction_scenario_2(generator):
    """Test bank teller with high confidence rules, good FI explanation -> FI."""
    bank_teller_data_2 = {
        "rules_explanation": {
            "confidence_bb": 0.99,
            "fidelity_neighbours": 0.5,
            "stability": 1.0,
        },
        "fi_explanation": {
            "confidence_bb": 0.99,
            "fidelity_neighbours": 0.97,
            "stability": 0.95,
        },
    }
    result = generator.get_persona_prediction(bank_teller, bank_teller_data_2)
    assert result == ["FI"]


def test_bank_teller_prediction_scenario_3(generator):
    """Test bank teller with high confidence rules, medium FI explanation -> Rules."""
    bank_teller_data_3 = {
        "rules_explanation": {
            "confidence_bb": 0.99,
            "fidelity_neighbours": 0.5,
            "stability": 1.0,
        },
        "fi_explanation": {
            "confidence_bb": 0.99,
            "fidelity_neighbours": 0.94,
            "stability": 0.95,
        },
    }
    result = generator.get_persona_prediction(bank_teller, bank_teller_data_3)
    assert result == ["Rules"]


def test_manager_prediction_scenario_1(generator):
    """Test manager with medium confidence -> Confidence + Counterfactual."""
    manager_data_1 = {
        "rules_explanation": {
            "confidence_bb": 0.94,
            "fidelity_neighbours": 0.5,
            "stability": 1.0,
        },
    }
    result = generator.get_persona_prediction(manager, manager_data_1)
    assert result == ["Confidence", "Counterfactual"]


def test_manager_prediction_scenario_2(generator):
    """Test manager with high confidence rules, good FI explanation -> FI."""
    manager_data_2 = {
        "rules_explanation": {
            "confidence_bb": 0.99,
            "fidelity_neighbours": 0.5,
            "stability": 1.0,
        },
        "fi_explanation": {
            "confidence_bb": 0.99,
            "fidelity_neighbours": 0.97,
            "stability": 0.96,
        },
    }
    result = generator.get_persona_prediction(manager, manager_data_2)
    assert result == ["FI"]


def test_manager_prediction_scenario_3(generator):
    """Test manager with high confidence, high fidelity rules -> Rules."""
    manager_data_3 = {
        "rules_explanation": {
            "confidence_bb": 0.99,
            "fidelity_neighbours": 0.96,
            "stability": 1.0,
        },
        "fi_explanation": {
            "confidence_bb": 0.99,
            "fidelity_neighbours": 0.93,
            "stability": 0.96,
        },
    }
    result = generator.get_persona_prediction(manager, manager_data_3)
    assert result == ["Rules"]


def test_manager_prediction_scenario_4(generator):
    """Test manager with high confidence, high fidelity rules (variant) -> Rules."""
    manager_data_4 = {
        "rules_explanation": {
            "confidence_bb": 0.99,
            "fidelity_neighbours": 0.96,
            "stability": 1.0,
        },
        "fi_explanation": {
            "confidence_bb": 0.99,
            "fidelity_neighbours": 0.96,
            "stability": 0.93,
        },
    }
    result = generator.get_persona_prediction(manager, manager_data_4)
    assert result == ["Rules"]


def test_manager_prediction_scenario_5(generator):
    """Test manager with medium stability -> Exemplars."""
    manager_data_5 = {
        "rules_explanation": {
            "confidence_bb": 0.99,
            "fidelity_neighbours": 0.96,
            "stability": 0.93,
        },
        "fi_explanation": {
            "confidence_bb": 0.99,
            "fidelity_neighbours": 0.96,
            "stability": 0.93,
        },
    }
    result = generator.get_persona_prediction(manager, manager_data_5)
    assert result == ["Exemplars"]


def test_manager_prediction_scenario_6(generator):
    """Test manager with medium rules fidelity -> Exemplars."""
    manager_data_6 = {
        "rules_explanation": {
            "confidence_bb": 0.99,
            "fidelity_neighbours": 0.93,
            "stability": 1.0,
        },
        "fi_explanation": {
            "confidence_bb": 0.99,
            "fidelity_neighbours": 0.96,
            "stability": 0.93,
        },
    }
    result = generator.get_persona_prediction(manager, manager_data_6)
    assert result == ["Exemplars"]
