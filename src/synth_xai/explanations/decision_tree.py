import re
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

from collections import Counter

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


def grid_search_dt(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> DecisionTreeClassifier:
    # Define the parameter grid
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 7, 10],
        "min_samples_leaf": [1, 2, 5, 10],
        "class_weight": [None, "balanced"],
    }

    # Initialize the grid search
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=seed), param_grid, cv=5, scoring="accuracy")

    # Fit the grid search to the data
    grid_search.fit(x_train, y_train)

    # Get the best estimator
    clf = grid_search.best_estimator_
    clf.fit(x_train, y_train)
    return clf


def extract_rule_dt(
    clf: DecisionTreeClassifier, y_name: str, sample: pd.DataFrame
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    # Get the path of the sample row in the tree:
    # Get the prediction for the sample row
    sample = sample.drop(columns=[y_name])
    sample_pred = clf.predict(sample)

    # Get the decision path for the sample row
    node_indicator = clf.decision_path(sample)
    leave_id = clf.apply(sample)

    # Get the feature and threshold used at each node
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    # Get the feature names
    feature_names = sample.columns

    # Print the path from the root to the leaf
    node_index = node_indicator.indices[node_indicator.indptr[0] : node_indicator.indptr[1]]
    rule = []
    # print("Decision path for the sample row:")
    for node_id in node_index:
        if leave_id[0] == node_id:
            # print(f"Leaf node {node_id} reached, prediction: {sample_pred[0]}")
            rule.append(f"Leaf node {node_id} reached, prediction: {sample_pred[0]}")
        else:
            threshold_sign = "<=" if sample.iloc[0, feature[node_id]] <= threshold[node_id] else ">"
            # print(
            #     f"Node {node_id}: ({feature_names[feature[node_id]]} = {sample.iloc[0, feature[node_id]]}) {threshold_sign} {threshold[node_id]}"
            # )
            rule.append(
                f"({feature_names[feature[node_id]]} = {sample.iloc[0, feature[node_id]]}) {threshold_sign} {threshold[node_id]}"
            )
    return sample_pred, rule, threshold, feature


def compute_robustness_dt(explanations: list[list[str]]) -> float:
    sample_explanation = explanations[0]
    robustness = []
    for explanation in explanations[1:]:
        robustness.append(compute_stability_dt(explanations=[sample_explanation, explanation]))
    return float(np.mean(robustness))


def compute_stability_dt(explanations: list[list[str]]) -> float:
    if not isinstance(explanations, list) or len(explanations) != 2:
        raise ValueError("Expected exactly two explanations for stability computation")

    list1 = explanations[0]
    list2 = explanations[1]

    counter1, counter2 = Counter(list1), Counter(list2)
    common = sum((counter1 & counter2).values())
    total = len(list1) + len(list2)
    return 2 * common / total if total > 0 else 0.0


def compute_robustness_dt_lipschitz(explanations: list[list[str]]) -> float:
    """
    Differently from the other compute_robustness_dt function, this one computes the robustness
    using the Lipschitz constant.

    Args:
        explanations (list[list[str]]): List of explanations for two samples

    Returns:
        float: The Lipschitz robustness between the two norm_explanations

    """
    sample_explanation = explanations[0]
    robustness = [
        compute_stability_dt_lipschitz(explanations=[sample_explanation, explanation])
        for explanation in explanations[1:]
    ]
    return float(np.mean(robustness))


def compute_stability_dt_lipschitz(explanations: list[list[str]]) -> float:
    """
    The function computes the Lipschitz constant based on the definition
    presented in https://arxiv.org/abs/1806.07538.
    The stability is computed as:
    L = ‖ e_x - e_x' ‖ / ‖ x - x' ‖

    Args:
        explanations (list[list[str]]): List of explanations for two samples

    Returns:
        float: The Lipschitz loss between the two explanations

    """
    if not isinstance(explanations, list) or len(explanations) != 2:
        msg = "Expected exactly two explanations for stability computation"
        raise ValueError(msg)

    list1 = explanations[0]
    list2 = explanations[1]

    print(list1, list2)
    norm_explanations = np.linalg.norm(np.array(list1) - np.array(list2))
    norm_samples = np.linalg.norm(np.array(list1) - np.array(list2))

    return float(norm_explanations / norm_samples) if norm_samples > 0 else 0.0


def parse_explanation_dt(explanation: str) -> list[str]:
    pattern = re.compile(r"\(\s*(\w+)\s*=")
    matches = []

    # If explanation is a tuple and its first element is a list of strings
    if isinstance(explanation, tuple) and isinstance(explanation[0], list):
        for text in explanation[0]:
            matches.extend(pattern.findall(text))
    # Otherwise, if explanation is a string
    elif isinstance(explanation, str):
        matches = pattern.findall(explanation)

    return matches
