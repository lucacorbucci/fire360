import argparse
import ast
import random
import re
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
    find_similar_samples,
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
parser.add_argument("--bb_path", type=str, default=None, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--explanation_type", type=str, default=None, required=True)
parser.add_argument("--top_k", type=int, default=1)
parser.add_argument("--explanations", type=str, nargs="+", default=None, required=True)
parser.add_argument("--artifacts_path", type=str, default=None, required=True)


def signal_handler(_sig: int, frame: FrameType | None) -> None:
    """
    Function used to handle the SIGINT signal
    """
    logger.info("Gracefully stopping your experiment! Keep calm!")
    # global wandb_run
    # if wandb_run:
    #     wandb_run.finish()
    sys.exit(0)


# def compute_stability(explanations: list):
#     """
#     Function to compute the stability of the explanations.

#     For feature importance explanations, stability is measured as the number
#     of features that occur in the same order across all explanation lists.
#     For rule-based explanations, it is measured as the number of features that
#     are present in every rule.
#     """
#     logger.info("Computing stability")

#     # Check if we have a feature importance explanation:
#     # we assume that if every explanation is a list of strings (features) with the same length,
#     # it represents a feature ranking.
#     if all(isinstance(exp, list) for exp in explanations) and all(isinstance(feat, str) for feat in explanations[0]):
#         baseline = explanations[0]
#         stable_count = 0
#         # Compare the feature at each position in the baseline with that of all other explanations.
#         for i, feature in enumerate(baseline):
#             if all(exp[i] == feature for exp in explanations):
#                 stable_count += 1
#         logger.info(f"Feature importance stability: {stable_count}/{len(baseline)} features in same order")
#         return stable_count

#     # Otherwise, treat explanations as rule sets.
#     # Each explanation is assumed to be a list (or iterable) of features extracted from a rule.
#     rule_sets = []
#     for exp in explanations:
#         try:
#             # Ensure exp is iterable and convert to a set.
#             rule_sets.append(set(exp))
#         except TypeError:
#             logger.error("Rule-based explanation is not iterable.")
#             return None

#     common_features = set.intersection(*rule_sets)
#     stability = len(common_features)
#     logger.info(f"Rule-based stability: {stability} common features in all rules")
#     return stability


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    args = parser.parse_args()

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
    explainer_model = ExplainerModel(explainer_type=args.explanation_type)
    explained_indexes = list(range(total_explanations))
    # Stability computation
    logger.info("Computing stability")
    stabilities = []
    for index in explained_indexes:
        # get the explanations from all the files
        explanations = []
        explanations.append(explainer_model.parse_explanation(str(explanation_data[0][index])))
        explanations.append(explainer_model.parse_explanation(str(explanation_data[1][index])))

        stabilities.append(explainer_model.compute_stability(explanations))

    logger.info(f"Stabilities: {np.mean(stabilities)} +/- {np.std(stabilities)}")
    logger.info("Stability computation done!")

    # Robustness computation
    robustness_list = []
    test_data = test_data.iloc[explained_indexes]
    for index in explained_indexes:
        # get sample index from test_data
        sample = test_data.iloc[[index]]

        # find top-k closest rows
        top_k = args.top_k
        closest_rows = find_similar_samples(test_data, sample, top_k, outcome_variable, index)

        explanations = [explainer_model.parse_explanation(str(explanation_data[0][index]))]

        for closest_row_index in closest_rows:
            explanations.append(explainer_model.parse_explanation(str(explanation_data[0][closest_row_index])))
        robustness_list.append(explainer_model.compute_robustness(explanations=explanations))

    logger.info(f"Computing robustness: {np.mean(robustness_list)} +/- {np.std(robustness_list)}")
    logger.info("Robustness computation done!")

    # Faithfulness computation
    if args.explanation_type in ["logistic", "svm"]:
        base_value = np.mean(x)
        mean_faithfulness, std_faithfulness = explainer_model.compute_faithfulness(
            model=bb,
            dataset=x,
            explanations=[
                explainer_model.parse_coefficients(str(explanation_data[0][index])) for index in explained_indexes
            ],
            base_value=base_value,
            outcome_variable=outcome_variable,
        )

        logger.info(f"Faithfulness: {mean_faithfulness} +/- {std_faithfulness}")
