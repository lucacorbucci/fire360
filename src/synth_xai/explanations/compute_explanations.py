import argparse
import datetime
import random
import signal
import sys
import warnings
from collections import Counter
from functools import partial
from pathlib import Path
from types import FrameType
from typing import Any

import dill
import multiprocess
import numpy as np
import pandas as pd
import torch
from explanation_utils import (
    evaluate_bb,
    find_top_closest_rows,
    get_test_data,
    is_explainer_supported,
    label_synthetic_data,
    load_bb,
    load_synthetic_data,
    make_predictions,
    prepare_neighbours,
    setup_wandb,
    transform_input_data,
)
from loguru import logger
from multiprocess import Pool
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
)

from synth_xai.bb_architectures import MultiClassModel, SimpleModel
from synth_xai.explanations.explainer_model import ExplainerModel

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--dataset_name", type=str, default=None, required=True)
parser.add_argument("--synthetic_dataset_path", type=str, default=None, required=True)
parser.add_argument("--bb_path", type=str, default=None, required=True)
parser.add_argument("--model_name", type=str, default=None, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--validation_seed", type=int, default=11)
parser.add_argument("--top_k", type=int, default=42)
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--explanation_type", type=str, default=None, required=True)
parser.add_argument("--store_path", type=str, default=None, required=True)
parser.add_argument("--num_processes", type=int, default=None, required=True)


def signal_handler(_sig: int, frame: FrameType | None) -> None:
    """
    Function used to handle the SIGINT signal
    """
    logger.info("Gracefully stopping your experiment! Keep calm!")
    # global wandb_run
    # if wandb_run:
    #     wandb_run.finish()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    args = parser.parse_args()

    if not is_explainer_supported(args.explanation_type):
        logger.error(f"Explanation type {args.explanation_type} is not supported!")
        sys.exit(1)

    current_script_path = Path(__file__).resolve().parent

    bb = load_bb(args.bb_path)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(args.seed)

    train_data, test_data, outcome_variable = get_test_data(args)

    random.seed(args.validation_seed)
    torch.manual_seed(args.validation_seed)
    torch.cuda.manual_seed(args.validation_seed)
    torch.cuda.manual_seed_all(args.validation_seed)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(args.validation_seed)

    x, y, scaler = transform_input_data(train_data=train_data, test_data=test_data, outcome_variable=outcome_variable)

    evaluate_bb(x, y, bb)

    synthetic_data_path = Path(args.synthetic_dataset_path)
    if not synthetic_data_path.exists():
        logger.error(f"File {synthetic_data_path} does not exist!")
        sys.exit(1)
    logger.info(f"Processing synthetic file: {synthetic_data_path.name}")
    synthetic_data = load_synthetic_data(synthetic_data_path)
    logger.info(f"Loaded synthetic data: {synthetic_data_path.name}")

    synthetic_data_labels = label_synthetic_data(
        synthetic_data=synthetic_data, outcome_variable=outcome_variable, bb=bb, scaler=scaler
    )
    logger.info(f"Labeled synthetic dataset: {synthetic_data_path.name}")
    synthetic_data[outcome_variable] = synthetic_data_labels

    multiprocess.set_start_method("spawn")

    def process_sample(
        index: int,
        test_data: pd.DataFrame,
        x: np.ndarray,
        y: np.ndarray,
        bb: Any,
        args: argparse.Namespace,
        synthetic_data: pd.DataFrame,
        outcome_variable: str,
    ) -> tuple[int, int, float, float, list[str], int, dict[str, int]]:
        # Process one sample from the test data.
        start_time = datetime.datetime.now()
        sample = test_data.iloc[[index]]
        x_sample = torch.tensor([x[index]])
        y_sample = torch.tensor([y[index]])
        sample_pred_bb = make_predictions(x_sample, y_sample, bb)

        start_ranking_time = datetime.datetime.now()
        top_k_samples = find_top_closest_rows(
            synthetic_data=synthetic_data,
            sample=sample,
            k=args.top_k,
            y_name=outcome_variable,
        )
        finish_ranking_time = datetime.datetime.now()

        ranking_time = (finish_ranking_time - start_ranking_time).microseconds
        logger.info(f"Ranking time: {ranking_time}")

        X, Y, old_x = prepare_neighbours(top_k_samples=top_k_samples, y_name=outcome_variable)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        explainer_model = ExplainerModel(explainer_type=args.explanation_type)
        explainer_model.grid_search(x_train=x_train, y_train=y_train, seed=args.validation_seed)

        y_pred = explainer_model.predict(x_test)
        accuracy_val = accuracy_score(y_test, y_pred)
        f1_val = f1_score(y_test, y_pred, average="weighted")

        start_explanation_time = datetime.datetime.now()
        sample_pred, explanation, threshold, feature = explainer_model.extract_explanation(
            clf=explainer_model.best_model, y_name=outcome_variable, sample=sample
        )
        finish_explanation_time = datetime.datetime.now()
        explanation_time = (finish_explanation_time - start_explanation_time).microseconds
        # logger.info(f"Extracted explanation: {explanation}")

        end_time = datetime.datetime.now()
        total_time = (end_time - start_time).microseconds

        time = {"total_time": total_time, "ranking_time": ranking_time, "explanation_time": explanation_time}

        return sample_pred_bb[0].item(), sample_pred[0], accuracy_val, f1_val, explanation, index, time

    # Run the processing in parallel.

    num_samples = min(20000, len(test_data))
    args_list = [(i, test_data, x, y, bb, args, synthetic_data, outcome_variable) for i in range(num_samples)]
    with Pool(args.num_processes) as pool:
        results = pool.starmap(process_sample, args_list)

    predictions_bb, predictions_dt, accuracy, f1_score, explanations, indexes, times = map(
        list, zip(*results, strict=False)
    )

    # compute the fidelity
    fidelity = accuracy_score(predictions_bb, predictions_dt)
    logger.info(f"Fidelity for file {synthetic_data_path.name}: {fidelity}")
    wandb_run = setup_wandb(args, synthetic_data_path, num_samples)

    synthetiser = synthetic_data_path.name.split("_")[-1].split(".")[0]
    num_samples_generated = synthetic_data_path.name.split("_")[2]
    epochs = synthetic_data_path.name.split("_")[4]

    total_time = [time["total_time"] for time in times]
    ranking_time = [time["ranking_time"] for time in times]
    explanation_time = [time["explanation_time"] for time in times]

    wandb_run.log(
        {
            "Fidelity": fidelity,
            "Tree Accuracy": np.mean(accuracy),
            "Tree F1 Score": np.mean(f1_score),
            "Tree Accuracy Std": np.std(accuracy),
            "Tree F1 Score Std": np.std(f1_score),
            "Total Time": np.mean(total_time),
            "Total Time Std": np.std(total_time),
            "Ranking Time": np.mean(ranking_time),
            "Ranking Time Std": np.std(ranking_time),
            "Explanation Time": np.mean(explanation_time),
            "Explanation Time Std": np.std(explanation_time),
            "Total Time (sec)": np.mean(total_time) / 1e6,
            "Total Time Std (sec)": np.std(total_time) / 1e6,
            "Ranking Time (sec)": np.mean(ranking_time) / 1e6,
            "Ranking Time Std (sec)": np.std(ranking_time) / 1e6,
            "Explanation Time (sec)": np.mean(explanation_time) / 1e6,
            "Explanation Time Std (sec)": np.std(explanation_time) / 1e6,
        }
    )
    wandb_run.finish()

    computed_explanations = {}
    for index in indexes:
        computed_explanations[index] = (explanations[index], predictions_bb[index])
    # serialize the explanations
    file_name = (
        args.explanation_type
        + "_"
        + synthetiser
        + "_"
        + str(num_samples_generated)
        + "_"
        + str(epochs)
        + "_"
        + str(args.validation_seed)
        + ".pkl"
    )
    store_path = Path(args.store_path) / file_name
    with Path(store_path).open("wb") as f:
        # store the computed_explanations dictionary as a json file
        dill.dump(computed_explanations, f)

    logger.info(f"Final Fidelity: {fidelity}")
