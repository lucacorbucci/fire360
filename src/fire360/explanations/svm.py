import re
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC

warnings.simplefilter("ignore")


def grid_search_svm(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> SVC:
    """
    Performs grid search hyperparameter tuning for a linear SVM.
    We tune the regularization parameter C and the class_weight.
    """
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "class_weight": [None, "balanced"],
    }
    # We use a linear kernel so that the model has interpretable coefficients.
    grid_search = GridSearchCV(
        SVC(kernel="linear", max_iter=1000, random_state=seed),
        param_grid,
        cv=KFold(n_splits=5, shuffle=True, random_state=seed),
        scoring="accuracy",
    )
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    best_model.fit(x_train, y_train)
    return best_model


def extract_svm_explanation(
    model: SVC, sample: pd.DataFrame, y_name: str
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """
    Extracts an explanation for the SVM prediction by reporting each feature's coefficient and its contribution.
    The contribution is computed as feature_value * coefficient.
    """
    # Drop the outcome variable if present in the sample.
    sample_input = sample.drop(columns=[y_name]) if y_name in sample.columns else sample.copy()
    sample_pred = model.predict(sample_input)
    # For a linear SVM, model.coef_ has shape (1, n_features)
    coeffs = model.coef_
    feature_names = sample_input.columns.tolist()

    explanation_data = []
    for i, feature in enumerate(feature_names):
        explanation_data.append((feature, coeffs[0][i], sample_input.iloc[0, i]))

    # Sort the explanation based on the absolute value of the coefficient (contribution) in descending order.
    explanation_data.sort(key=lambda x: abs(x[1]), reverse=True)
    explanation = [f"{feature}: coefficient={coef}, value={value}" for feature, coef, value in explanation_data]
    return sample_pred, explanation, coeffs[0], feature_names


def compute_robustness_svm(explanations: list[list[str]], top_k: list) -> list:
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


def compute_stability_svm(explanations: list[list[str]]) -> float:
    """
    The stability for the SVM (feature importance model)
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
    features_in_the_same_order = sum(
        [1 for i in range(len(first_explanation)) if first_explanation[i] == second_explanation[i]]
    )
    total_features = len(first_explanation)

    return float(features_in_the_same_order / total_features)


def parse_explanation_svm(explanation: str) -> list[str]:
    # Regex to capture the feature name that might be enclosed in quotes with optional spaces
    matches = re.findall(r"'([^']*)'", explanation)
    features = []

    for match in matches:
        end = match.find(":")
        feature = match[:end]
        features.append(feature)
    return features


def parse_coefficients_svm(explanation: str) -> list[float]:
    # Regex to extract the coefficients of the explanations
    # For instance given 'sex_binary: coefficient=-2.0002829218120355, value=0',
    # 'edu_level: coefficient=-1.9999999998997144, value=2'
    # I want to extract a list with [-2.0002829218120355, -1.9999999998997144]
    coefficients = re.findall(r"coefficient=([-+]?\d*\.\d+)", explanation)
    return [float(coef) for coef in coefficients]
