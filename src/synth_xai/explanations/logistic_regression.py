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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split

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


def grid_search_lr(x_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    Performs grid search hyperparameter tuning for logistic regression.
    We tune the penalty type, the inverse regularization strength C, and the class_weight.
    """
    param_grid = {
        "penalty": ["l1", "l2"],
        "C": [0.01, 0.1, 1, 10, 100],
        "class_weight": [None, "balanced"],
    }
    # We use the 'liblinear' solver since it supports both l1 and l2 penalties.
    grid_search = GridSearchCV(
        LogisticRegression(solver="liblinear", max_iter=1000, random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
    )
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
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
    predictions_lr = []
    for index_sample in range(0, 1000):
        logger.info(f"Processing sample {index_sample}")
        sample = test_data.iloc[[index_sample]]
        logger.info(f"Sample: {sample}")

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

        # Hyperparameter tuning with logistic regression.
        best_model = grid_search_lr(x_train=x_train, y_train=y_train)

        # Evaluate the logistic regression model on a test split.
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        # Uncomment the next line if you wish to log the performance.
        # logger.info(f"Logistic Regression - Accuracy: {accuracy} - F1: {f1}")

        # Extract explanation for the sample row.
        sample_pred, explanation, coefficients, feature_names = extract_logistic_explanation(
            model=best_model, sample=sample, y_name=outcome_variable
        )
        logger.info(f"Explanation for the sample row: {explanation}")

        predictions_bb.append(sample_pred_bb[0].item())
        predictions_lr.append(sample_pred[0])

    # Compute the fidelity between the black-box and the logistic regression predictions.
    fidelity = accuracy_score(predictions_bb, predictions_lr)
    logger.info(f"Fidelity: {fidelity}")
