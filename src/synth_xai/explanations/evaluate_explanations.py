import argparse
import ast
import math
import os
import random
import signal
import sys
import warnings
from pathlib import Path
from types import FrameType

import dill
import numpy as np
import pandas as pd
import torch
from explanation_utils import (
    evaluate_bb,
    find_similar_samples,
    get_test_data,
    load_bb,
    setup_wandb,
    transform_input_data,
)
from loguru import logger
from multiprocess import Pool

from synth_xai.bb_architectures import MultiClassModel, SimpleModel  # DO NOT REMOVE
from synth_xai.explanations.explainer_model import ExplainerModel

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--dataset_name", type=str, default=None, required=True)
parser.add_argument("--bb_path", type=str, default=None, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--explanation_type", type=str, default=None, required=True)
parser.add_argument("--top_k", type=int, nargs="+", default=None, required=True)
parser.add_argument("--explanations", type=str, nargs="+", default=None, required=True)
parser.add_argument("--artifacts_path", type=str, default=None, required=True)
parser.add_argument("--wandb_project_name", type=str, default=None, required=True)
parser.add_argument("--neigh_size", type=int, default=-1)


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

    if args.neigh_size == -1 and not args.explanation_type in ["shap"]:
        raise ValueError("Neigh size is required for all explanations except for SHAP")

    current_script_path = Path(__file__).resolve().parent

    bb = load_bb(args.bb_path)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(args.seed)

    train_data, test_data, outcome_variable = get_test_data(args)

    x, y, scaler = transform_input_data(train_data=train_data, test_data=test_data, outcome_variable=outcome_variable)

    predictions = evaluate_bb(x, y, bb)
    test_data[outcome_variable] = predictions

    explanation_data = []
    for explanation in args.explanations:
        logger.info(f"Opening explanation file {explanation}")

        with open(args.artifacts_path + explanation, "rb") as f:
            explanation_data.append(dill.load(f))
    total_explanations = len(explanation_data[0])
    logger.info(f"Total explanations: {total_explanations}")
    is_our_explanation = args.explanation_type in ["logistic", "svm", "dt", "knn"]

    explanation_mapping = {
        "lime": "logistic",
        "shap": "logistic",
        "lore": "dt",
        "lore_genetic": "dt",
    }

    explainer_model = ExplainerModel(
        explainer_type=args.explanation_type
        if args.explanation_type not in explanation_mapping
        else explanation_mapping[args.explanation_type]
    )
    explained_indexes = list(explanation_data[0].keys())

    wandb_run = setup_wandb(args, num_samples=total_explanations, project_name=args.wandb_project_name)

    metrics = {}

    def preprocess_explanations(args: argparse.Namespace, explanation_data: list) -> list:
        pre_processed_data = [{}, {}]
        if Path(args.artifacts_path + f"pre_processed_{args.explanation_type}_{args.neigh_size}.pkl").exists():
            logger.info("Loading explanations from disk")
            with open(args.artifacts_path + f"pre_processed_{args.explanation_type}_{args.neigh_size}.pkl", "rb") as f:
                pre_processed_data = dill.load(f)
        else:
            logger.info("Preprocessing explanations")

            def parse_explanation(index: int) -> tuple[int, list, list] | bool:
                try:
                    return (
                        index,
                        explainer_model.parse_explanation(str(explanation_data[0][index])),
                        explainer_model.parse_explanation(str(explanation_data[1][index])),
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing index {index}: {e}, Explanation 0: {explanation_data[0][index]}, Explanation 1: {explanation_data[1][index]}"
                    )
                    return False

            with Pool(20) as pool:
                results = pool.map(parse_explanation, explained_indexes)

            if False in results:
                logger.error("Some explanations failed to process.")

            for index, explanation_0, explanation_1 in results:
                pre_processed_data[0][index] = explanation_0
                pre_processed_data[1][index] = explanation_1
            with open(args.artifacts_path + f"pre_processed_{args.explanation_type}_{args.neigh_size}.pkl", "wb") as f:
                dill.dump(pre_processed_data, f)

        logger.info("Preprocessing explanations done!")
        return pre_processed_data

    def preprocess_distances(
        args: argparse.Namespace,
        explanation_data: list,
        test_data: pd.DataFrame,
        top_k: int,
        outcome_variable: str,
    ) -> dict:
        all_closest_rows = {}
        if os.path.exists(args.artifacts_path + f"closest_rows_{args.neigh_size}.pkl"):
            with open(args.artifacts_path + f"closest_rows_{args.neigh_size}.pkl", "rb") as f:
                all_closest_rows = dill.load(f)
        else:

            def compute_closest_rows(index: int) -> tuple[int, list[int]]:
                sample = test_data.iloc[[index]]
                closest_rows = find_similar_samples(test_data, sample, top_k, outcome_variable, index)
                return index, closest_rows

            with Pool(20) as pool:
                results = pool.map(compute_closest_rows, explained_indexes)
                all_closest_rows = dict(results)

            with open(args.artifacts_path + f"closest_rows_{args.neigh_size}.pkl", "wb") as f:
                dill.dump(all_closest_rows, f)
        return all_closest_rows

    def pre_process_coefficients(explanation_data: list) -> dict[int, list]:
        if Path(args.artifacts_path + f"coefficients_{args.explanation_type}_{args.neigh_size}.pkl").exists():
            logger.info("Loading coefficients from disk")
            with open(args.artifacts_path + f"coefficients_{args.explanation_type}_{args.neigh_size}.pkl", "rb") as f:
                coefficients = dill.load(f)
        else:

            def extract_coefficients(index: int) -> tuple[int, list]:
                return index, explainer_model.parse_coefficients(str(explanation_data[0][index]))

            with Pool(20) as pool:
                results = pool.map(extract_coefficients, explained_indexes)

            coefficients = dict(results)
            with open(args.artifacts_path + f"coefficients_{args.explanation_type}_{args.neigh_size}.pkl", "wb") as f:
                dill.dump(coefficients, f)

        return coefficients

    def pre_process_shap_explanations(explanation_data: list) -> tuple[list[dict], dict]:
        if (
            Path(args.artifacts_path + f"pre_processed_shap_values_{args.neigh_size}.pkl").exists()
            and Path(args.artifacts_path + f"pre_processed_coefficients_{args.neigh_size}.pkl").exists()
        ):
            logger.info("Loading shap values from disk")
            with open(args.artifacts_path + f"pre_processed_shap_values_{args.neigh_size}.pkl", "rb") as f:
                pre_processed_data = dill.load(f)
            with open(args.artifacts_path + f"pre_processed_coefficients_{args.neigh_size}.pkl", "rb") as f:
                coefficients = dill.load(f)
        else:
            pre_processed_data = [{}, {}]
            coefficients = {}
            for index in explanation_data[0].keys():
                coefficients[index] = [coeff for _, coeff in explanation_data[0][index][0]]
                pre_processed_data[0][index] = [
                    feature_name
                    for feature_name, _ in sorted(explanation_data[0][index][0], key=lambda x: abs(x[1]), reverse=True)
                ]
                pre_processed_data[1][index] = [
                    feature_name
                    for feature_name, _ in sorted(explanation_data[1][index][0], key=lambda x: abs(x[1]), reverse=True)
                ]

            with open(args.artifacts_path + f"pre_processed_shap_values_{args.neigh_size}.pkl", "wb") as f:
                dill.dump(pre_processed_data, f)
        return pre_processed_data, coefficients

    def preprocess_explanations_lore(explanation_data: list) -> list:
        pre_processed_data = [{}, {}]
        for index in list(explanation_data[0].keys()):
            pre_processed_data[0][index] = [
                premise["attr"] for premise in explanation_data[0][index]["rule"]["premises"]
            ]
            pre_processed_data[1][index] = [
                premise["attr"] for premise in explanation_data[1][index]["rule"]["premises"]
            ]
        return pre_processed_data

    test_data = test_data.iloc[explained_indexes]

    if args.explanation_type in ["logistic", "svm"]:
        coefficients = pre_process_coefficients(explanation_data)
    elif args.explanation_type in ["lime"]:
        coefficients = {}
        pre_processed_data = [{}, {}]
        for index in explanation_data[0].keys():
            coefficients[index] = [coeff for _, coeff in explanation_data[0][index][0]]
            pre_processed_data[0][index] = explanation_data[0][index][2]
            pre_processed_data[1][index] = explanation_data[1][index][2]
        explanation_data = pre_processed_data

    if is_our_explanation:
        explanation_data = preprocess_explanations(args, explanation_data)
        logger.info("Preprocessing explanations done!")

    if args.explanation_type in ["shap"]:
        explanation_data, coefficients = pre_process_shap_explanations(explanation_data)
        logger.info("Preprocessing shap values done!")
    if args.explanation_type in ["lore", "lore_genetic"]:
        explanation_data = preprocess_explanations_lore(explanation_data)
        logger.info("Preprocessing explanations done!")

    all_closest_rows = preprocess_distances(args, explanation_data, test_data, max(args.top_k) * 2, outcome_variable)
    logger.info("Preprocessing distances done!")

    # Stability computation
    logger.info("Computing stability")
    stabilities = []

    def compute_stability(index: int, explanation_data: list, is_our_explanation: bool) -> float | bool:
        explanations = []
        if not is_our_explanation:
            explanations.append(explanation_data[0][index])
            explanations.append(explanation_data[1][index])
        else:
            explanations.append(explanation_data[0][index])
            explanations.append(explanation_data[1][index])
        stability = explainer_model.compute_stability(explanations)
        return stability

    args_list = [(i, explanation_data, is_our_explanation) for i in explained_indexes]
    with Pool(20) as pool:
        stabilities = pool.starmap(compute_stability, args_list)
    stabilities = [stability for stability in stabilities if stability is not False]
    logger.info(f"Stabilities: {np.mean(stabilities)} +/- {np.std(stabilities)}")
    logger.info("Stability computation done!")
    metrics["stability"] = np.mean(stabilities)
    metrics["stability_std"] = np.std(stabilities)

    # Robustness computation
    robustness_list: np.ndarray = np.array([])

    def compute_robustness(index: int, explanation_data: list, is_our_explanation: bool) -> list | bool:
        closest_rows = [i for i in all_closest_rows[index] if i in explained_indexes]
        explanations = []
        if not is_our_explanation:
            explanations.append(explanation_data[0][index])
            for closest_row_index in closest_rows:
                explanations.append(explanation_data[1][closest_row_index])
        else:
            explanations = [explanation_data[0][index]]
            for closest_row_index in closest_rows:
                explanations.append(explanation_data[0][closest_row_index])
        robustness = explainer_model.compute_robustness(explanations=explanations, top_k=args.top_k)
        return robustness

    args_list = [(i, explanation_data, is_our_explanation) for i in explained_indexes]
    with Pool(20) as pool:
        robustness_list = pool.starmap(compute_robustness, args_list)

    robustness_list = [robustness for robustness in robustness_list if robustness is not False]

    robustness_list = np.array(robustness_list)
    for index, top in enumerate(args.top_k):
        # get all the samples in position index in robustness_list
        logger.info(f"Computing robustness for top {top} on real test set")
        valid_robustness = robustness_list[:, index][~np.isnan(robustness_list[:, index])]
        logger.info(f"Robustness: {np.mean(valid_robustness)} +/- {np.std(valid_robustness)}")
        metrics[f"robustness_top_{top}"] = np.mean(valid_robustness)
        metrics[f"robustness_std_top_{top}"] = np.std(valid_robustness)
    logger.info("Robustness computation done!")

    # Faithfulness computation
    if args.explanation_type in ["logistic", "svm", "lime", "shap"]:
        base_value = np.mean(x)
        mean_faithfulness, std_faithfulness = explainer_model.compute_faithfulness(
            model=bb,
            dataset=x,
            explanations=[coefficients[index] for index in explained_indexes],
            base_value=base_value,
        )

        logger.info(f"Faithfulness: {mean_faithfulness} +/- {std_faithfulness}")
        metrics["faithfulness"] = mean_faithfulness
        metrics["faithfulness_std"] = std_faithfulness

    metrics["neigh_size"] = args.neigh_size
    wandb_run.log(metrics)
    wandb_run.finish()
