import re
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

import math
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


def compute_robustness_dt(explanations: list[list[str]], top_k: list) -> list:
    sample_explanation = explanations[0]
    robustness = []
    for index, explanation in enumerate(explanations[1:]):
        stability = compute_stability_dt(explanations=[sample_explanation, explanation])
        if not math.isnan(stability):
            robustness.append(stability)
    mean_robustness = []
    for top in top_k:
        mean_robustness.append(np.mean(robustness[:top]))
    return mean_robustness


def compute_stability_dt(
    explanations: list[list[str]],
) -> float:
    if not isinstance(explanations, list) or len(explanations) != 2:
        raise ValueError("Expected exactly two explanations for stability computation")

    list1 = explanations[0]
    list2 = explanations[1]
    counter1, counter2 = Counter(list1), Counter(list2)
    common = sum((counter1 & counter2).values())
    total = len(list1) + len(list2)
    return 2 * common / total if total > 0 else 0.0


def parse_explanation_dt(explanation: str) -> list[str]:
    feature_names = []
    while explanation.find("'(") >= 0 and explanation.find("=") >= 0:
        start = explanation.find("(") + 1  # Find position after '('
        if explanation[start] == "[":
            explanation = explanation[start + 1 :]
        else:
            end = explanation.find("=")  # Find position of '='
            if start > 0 and end > start:  # Ensure valid positions
                feature_names.append(explanation[start:end].strip())  # Extract and strip spaces
            explanation = explanation[end + 1 :]

    return feature_names
