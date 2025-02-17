import argparse
import random
import signal
import sys
import warnings
from collections import Counter
from functools import partial
from pathlib import Path
from types import FrameType

import multiprocess
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
from multiprocess import Pool
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
)
from sklearn.tree import DecisionTreeClassifier, export_text

import wandb
from synth_xai.bb_architectures import MultiClassModel, SimpleModel

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
    Function used to handle the SIGINT signal
    """
    logger.info("Gracefully stopping your experiment! Keep calm!")
    # global wandb_run
    # if wandb_run:
    #     wandb_run.finish()
    sys.exit(0)


def grid_search(x_train: np.ndarray, y_train: np.ndarray) -> DecisionTreeClassifier:
    # Define the parameter grid
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 7, 10],
        "min_samples_leaf": [1, 2, 5, 10],
        "class_weight": [None, "balanced"],
    }

    # Initialize the grid search
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring="accuracy")

    # Fit the grid search to the data
    grid_search.fit(x_train, y_train)

    # Get the best estimator
    clf = grid_search.best_estimator_
    clf.fit(x_train, y_train)
    return clf


def extract_rule(
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


def setup_wandb(args: argparse.Namespace, synthetic_data_file: Path) -> wandb.sdk.wandb_run.Run:
    wandb_run = wandb.init(
        # set the wandb project where this run will be logged
        project="tango_eval",
        dir="/raid/lcorbucci/wandb_tmp",
        name=f"{args.dataset_name}",
        config={
            "dataset_name": args.dataset_name,
            "Model": args.model_name,
            "Synthetic Data File": synthetic_data_file.name,
            "Synthetiser": synthetic_data_file.name.split("_")[-1],
            "epochs": synthetic_data_file.name.split("_")[4],
            "num_samples_generated": synthetic_data_file.name.split("_")[2],
        },
    )
    return wandb_run


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

    synthetic_data_path = Path(args.synthetic_dataset_path)
    if not synthetic_data_path.exists():
        logger.error(f"File {synthetic_data_path} does not exist!")
        sys.exit(1)
    # get the list of csv files in the directory
    synthetic_data_files = list(synthetic_data_path.glob("*.csv"))

    def process_file(
        synthetic_data_file: Path,
        args: argparse.Namespace,
        outcome_variable: str,
        bb: torch.nn.Module,
        scaler: MinMaxScaler,
        test_data: pd.DataFrame,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[str, float]:
        logger.info(f"Processing synthetic file: {synthetic_data_file}")
        synthetic_data = load_synthetic_data(synthetic_data_file)
        logger.info(f"Loaded synthetic data: {synthetic_data_file}")
        synthetic_data_labels = label_synthetic_data(
            synthetic_data=synthetic_data, outcome_variable=outcome_variable, bb=bb, scaler=scaler
        )
        logger.info(f"Labeled synthetic dataset: {synthetic_data_file}")
        synthetic_data[outcome_variable] = synthetic_data_labels

        predictions_bb = []
        predictions_dt = []
        for index_sample in range(0, min(4000, len(test_data))):
            sample = test_data.iloc[[index_sample]]

            # make prediction with bb
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

            best_model = grid_search(x_train=x_train, y_train=y_train)

            # Predict on the test set
            y_pred = best_model.predict(X_test)
            # Calculate the accuracy
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            sample_pred, rule, threshold, feature = extract_rule(clf=best_model, y_name=outcome_variable, sample=sample)

            logger.info(f"Rule for the sample row: {rule}")

            predictions_bb.append(sample_pred_bb[0].item())
            predictions_dt.append(sample_pred[0])

        # compute the fidelity
        fidelity = accuracy_score(predictions_bb, predictions_dt)
        logger.info(f"Fidelity for file {synthetic_data_file.name}: {fidelity}")
        wandb_run = setup_wandb(args, synthetic_data_file)
        wandb_run.log(
            {
                "Fidelity": fidelity,
                "Dataset": args.dataset_name,
                "Model": args.model_name,
                "Synthetic Data File": synthetic_data_file.name,
                "Synthetiser": synthetic_data_file.name.split("_")[-1],
            }
        )
        wandb_run.finish()

        return synthetic_data_file.name, fidelity

    multiprocess.set_start_method("spawn")
    # Create the argument tuples for each process_file call.
    pool_args = [(file, args, outcome_variable, bb, scaler, test_data, x, y) for file in synthetic_data_files]
    with Pool(10) as pool:
        results = pool.starmap(process_file, pool_args)

    for file_name, fidelity in results:
        logger.info(f"File: {file_name} - Final Fidelity: {fidelity}")
