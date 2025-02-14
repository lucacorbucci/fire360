import argparse
import random
import signal
import sys
import warnings
from pathlib import Path
from types import FrameType

import numpy as np
import pandas as pd
import torch
from explanation_utils import (
    evaluate_bb,
    find_top_closest_rows,
    get_test_data,
    label_synthetic_data,
    load_bb,
    load_synthetic_data,
    make_predictions,
    prepare_neighbours,
    transform_input_data,
)
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--dataset_name", type=str, default=None, required=True)
parser.add_argument("--synthetic_dataset_path", type=str, default=None, required=True)
parser.add_argument("--bb_path", type=str, default=None, required=True)
parser.add_argument("--model_name", type=str, default=False)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--top_k", type=int, default=42)
parser.add_argument("--debug", type=bool, default=False)


def signal_handler(_sig: int, frame: FrameType | None) -> None:
    """
    Function used to handle the SIGINT signal.
    """
    logger.info("Gracefully stopping your experiment! Keep calm!")
    sys.exit(0)


def grid_search_knn(x_train: np.ndarray, y_train: np.ndarray) -> KNeighborsClassifier:
    """
    Performs grid search hyperparameter tuning for KNN.
    The search includes the number of neighbors, the weighting scheme, and the distance metric.
    """
    param_grid = {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    }
    grid_search = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=5,
        scoring="accuracy",
    )
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    # Fit the best estimator on the training data.
    best_model.fit(x_train, y_train)
    return best_model


def extract_knn_explanation(
    model: KNeighborsClassifier, sample: pd.DataFrame, outcome_variable: str
) -> tuple[np.ndarray, list[str], None, list[str]]:
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


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    args = parser.parse_args()

    current_script_path = Path(__file__).resolve().parent
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(args.seed)
    bb = load_bb(args.bb_path)

    train_data, test_data, outcome_variable = get_test_data(args)

    x, y, scaler = transform_input_data(train_data=train_data, test_data=test_data, outcome_variable=outcome_variable)

    evaluate_bb(x, y, bb)
    synthetic_data = load_synthetic_data(args.synthetic_dataset_path)
    synthetic_data_labels = label_synthetic_data(
        synthetic_data=synthetic_data, outcome_variable=outcome_variable, bb=bb, scaler=scaler
    )
    synthetic_data[outcome_variable] = synthetic_data_labels

    predictions_bb = []
    predictions_knn = []
    for index_sample in range(0, 1000):
        sample = test_data.iloc[[index_sample]]

        # Make prediction with the black-box model.
        x_sample = torch.tensor([x[index_sample]])
        y_sample = torch.tensor([y[index_sample]])
        sample_pred_bb = make_predictions(x_sample, y_sample, bb)

        top_k_samples = find_top_closest_rows(
            synthetic_data=synthetic_data,
            sample=sample,
            k=args.top_k,
            y_name=outcome_variable,
        )

        X, Y, old_x = prepare_neighbours(top_k_samples=top_k_samples, y_name=outcome_variable)

        x_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Hyperparameter tuning with KNN.
        best_model = grid_search_knn(x_train=x_train, y_train=y_train)

        # Evaluate the KNN model on a test split.
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        # Uncomment the next line if you wish to log the performance.
        # logger.info(f"KNN - Accuracy: {accuracy} - F1: {f1}")

        # Extract explanation for the sample row.
        sample_pred, explanation, _, feature_names = extract_knn_explanation(
            model=best_model, sample=sample, outcome_variable=outcome_variable
        )
        logger.info(f"Explanation for the sample row: {explanation}")

        predictions_bb.append(sample_pred_bb[0].item())
        predictions_knn.append(sample_pred[0])

    # Compute the fidelity between the black-box and the KNN predictions.
    fidelity = accuracy_score(predictions_bb, predictions_knn)
    logger.info(f"Fidelity: {fidelity}")
