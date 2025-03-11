import re
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold

warnings.simplefilter("ignore")


def grid_search_lr(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> LogisticRegression:
    """
    Performs grid search hyperparameter tuning for logistic regression.
    We tune the penalty type, the inverse regularization strength C, and the class_weight.
    """
    param_grid = {
        "penalty": ["l1", "l2"],
        "C": [0.01, 0.1, 1],
        "class_weight": [None, "balanced"],
    }
    # We use the 'liblinear' solver since it supports both l1 and l2 penalties.
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    grid_search = GridSearchCV(
        LogisticRegression(solver="liblinear", max_iter=1000, random_state=seed),
        param_grid,
        cv=cv,
        scoring="accuracy",
    )
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best model accuracy: {grid_search.best_score_}")
    best_model.fit(x_train, y_train)
    return best_model


def extract_logistic_explanation(
    model: LogisticRegression, sample: pd.DataFrame, y_name: str
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """
    Extracts an explanation for the logistic regression prediction by reporting each feature's coefficient and its contribution.
    The contribution is computed as feature_value * coefficient.
    """
    # Drop the outcome variable if present in the sample
    sample_input = sample.drop(columns=[y_name]) if y_name in sample.columns else sample.copy()
    sample_pred = model.predict(sample_input)
    # For binary classification, model.coef_ has shape (1, n_features)
    coeffs = model.coef_
    feature_names = sample_input.columns.tolist()

    explanation_data = []
    for i, feature in enumerate(feature_names):
        explanation_data.append((feature, coeffs[0][i], sample_input.iloc[0, i]))

    # Sort the explanation based on the absolute value of the contribution in descending order
    explanation_data.sort(key=lambda x: abs(x[1]), reverse=True)
    explanation = [f"{feature}: coefficient={coef}, value={value}" for feature, coef, value in explanation_data]
    return sample_pred, explanation, coeffs[0], feature_names


def compute_robustness_lr(explanations: list[list[str]], top_k: list) -> list:
    """
    Computes the stability of the explanations. Given a list of
    explanations, one for each sample, compute the stability.
    the first explanation of the sample we are explaining.
    The other explanations are the explanations of the samples
    that are close to the one we want to explain.
    The number of these explanation is variable and depends on a
    parameter that we can set.
    Given the first explanation and each of the other, compute the
    amount of features that are in the same order and then divide by
    the total amount of features.
    In the end compute the average of these metric for all the
    possible combinations of <sample to be explainer, close explanation>.

    Args:
        explanations: List of explanations for each sample.

    Returns:
        float: The stability of the explanations.

    """
    explanation_sample = explanations[0]
    robustness = []
    for explanation in explanations[1:]:
        # 1 if the two features are in the same position
        features_in_the_same_order = sum(
            [1 for i in range(len(explanation_sample)) if explanation_sample[i] == explanation[i]]
        )
        total_features = len(explanation_sample)
        robustness.append(features_in_the_same_order / total_features)

    mean_robustness = []
    for top in top_k:
        mean_robustness.append(np.mean(robustness[:top]))
    return mean_robustness


def compute_stability_lr(explanations: list[list[str]]) -> float:
    """
    The stability for the logistic regression (feature importance model)
    is computed by considering the number of features that occur in the same order
    across all explanation lists.

    Args:
        explanations: List of explanations for each sample. In this case
        we should have a list with two lists inside

    Returns:
        float: The stability of the explanations.
    """
    first_explanation = explanations[0]
    second_explanation = explanations[1]
    for i in range(len(first_explanation)):
        if not (len(first_explanation) == len(second_explanation)):
            print("Not the same length")
            print(first_explanation, second_explanation)

    features_in_the_same_order = sum(
        [1 for i in range(len(first_explanation)) if first_explanation[i] == second_explanation[i]]
    )
    total_features = len(first_explanation)

    return float(features_in_the_same_order / total_features)


def parse_explanation_lr(explanation: str) -> list[str]:
    # Regex to capture the feature name that might be enclosed in quotes with optional spaces
    matches = re.findall(r"'([^']*)'", explanation)
    features = []

    for match in matches:
        end = match.find(":")
        feature = match[:end]
        features.append(feature)
    return features


def parse_coefficients_lr(explanation: str) -> list[float]:
    # Regex to extract the coefficients of the explanations
    # For instance given 'sex_binary: coefficient=-2.0002829218120355, value=0',
    # 'edu_level: coefficient=-1.9999999998997144, value=2'
    # I want to extract a list with [-2.0002829218120355, -1.9999999998997144]
    coefficients = re.findall(r"coefficient=([-+]?\d*\.\d+)", explanation)
    return [float(coef) for coef in coefficients]
