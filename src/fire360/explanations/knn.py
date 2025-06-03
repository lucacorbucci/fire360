import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

warnings.simplefilter("ignore")


def grid_search_knn(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> KNeighborsClassifier:
    """
    Performs grid search hyperparameter tuning for KNN.
    The search includes the number of neighbors, the weighting scheme, and the distance metric.
    """
    param_grid = {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grid_search = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=cv,
        scoring="accuracy",
    )
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    # Fit the best estimator on the training data.
    best_model.fit(x_train, y_train)
    return best_model


def extract_knn_explanation(
    model: KNeighborsClassifier, sample: pd.DataFrame, outcome_variable: str
) -> tuple[np.ndarray, list[str], Any, list[str]]:
    """
    Extracts an explanation for the KNN prediction by identifying its nearest neighbors.
    Since KNN does not have feature weights like logistic regression, we explain the prediction
    by reporting the indices, distances, and labels of the k nearest neighbors.
    """
    # Ensure we only use feature columns.
    sample_input = sample.drop(columns=[outcome_variable], errors="ignore")
    sample_pred = model.predict(sample_input)
    feature_names = sample_input.columns.tolist()
    # Retrieve the indices and distances of the k nearest neighbors.
    distances, indices = model.kneighbors(sample_input)
    explanation = []
    explanation.append(f"KNN prediction: {sample_pred[0]}")
    explanation.append("Nearest neighbors (index, distance, label):")
    # Accessing the training labels from the fitted model.
    for dist, idx in zip(distances[0], indices[0]):
        # Note: model._y holds the training labels and model._fit_X holds the training samples after fitting.
        neighbor_label = model._y[idx]
        neighbor_sample = model._fit_X[idx]
        explanation.append(f"Index: {idx}, distance: {dist:.4f}, label: {neighbor_label}, sample: {neighbor_sample}")
    return sample_pred, explanation, None, feature_names
