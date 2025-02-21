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


# def extract_alternative(
#     clf: DecisionTreeClassifier,
#     sample_pred: np.ndarray,
#     old_x: pd.DataFrame,
#     threshold: np.ndarray,
#     feature: np.ndarray,
# ) -> list[str]:
#     # Find the closest path that gives a different prediction
#     different_pred = abs(1 - sample_pred[0])
#     for i in range(len(clf.tree_.value)):
#         if clf.tree_.value[i][0][different_pred] > clf.tree_.value[i][0][sample_pred[0]]:
#             different_node_id = i
#             break

#     # Print the path for the different prediction
#     node_indicator_diff = clf.decision_path(old_x.iloc[[different_node_id]])
#     node_index_diff = node_indicator_diff.indices[node_indicator_diff.indptr[0] : node_indicator_diff.indptr[1]]

#     feature_names = sample.columns

#     logger.info("\nClosest path with a different prediction:")
#     rule = []
#     for node_id in node_index_diff:
#         if different_node_id == node_id:
#             logger.info(f"Leaf node {node_id} reached, prediction: {different_pred}")
#         else:
#             threshold_sign = "<=" if old_x.iloc[different_node_id, feature[node_id]] <= threshold[node_id] else ">"
#             logger.info(
#                 f"Node {node_id}: ({feature_names[feature[node_id]]} = {old_x.iloc[different_node_id, feature[node_id]]}) {threshold_sign} {threshold[node_id]}"
#             )
#             rule.append(
#                 f"({feature_names[feature[node_id]]} = {old_x.iloc[different_node_id, feature[node_id]]}) {threshold_sign} {threshold[node_id]}"
#             )
#     return rule
