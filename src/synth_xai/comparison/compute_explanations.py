import os

default_n_threads = 64
os.environ["OPENBLAS_NUM_THREADS"] = f"{default_n_threads}"
os.environ["MKL_NUM_THREADS"] = f"{default_n_threads}"
os.environ["OMP_NUM_THREADS"] = f"{default_n_threads}"

import argparse
import datetime
import random
import signal
import sys
import warnings
from collections import Counter
from pathlib import Path
from types import FrameType

import dill
import multiprocess
import numpy as np
import torch
from loguru import logger
from multiprocess import Pool
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import accuracy_score

from synth_xai.bb_architectures import MultiClassModel, SimpleModel
from synth_xai.comparison.explainer import Explainer
from synth_xai.comparison.utils_comparison import prepare_data
from synth_xai.explanations.explanation_utils import (
    load_bb,
    make_predictions,
    setup_wandb,
)
from synth_xai.utils import (
    aix_model,
)

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--dataset_name", type=str, default=None, required=True)
parser.add_argument("--bb_path", type=str, default=None, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--explanation_type", type=str, default=None, required=True)
parser.add_argument("--sweep", type=bool, default=False)
parser.add_argument("--validation_seed", type=int, default=None)
parser.add_argument("--num_processes", type=int, default=20, required=True)
parser.add_argument("--k_means_k", type=int, default=100)
parser.add_argument("--store_path", type=str, default=None, required=True)


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

    current_script_path = Path(__file__).resolve().parent

    bb = load_bb(args.bb_path)
    model = aix_model(model=bb)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(args.seed)
    (
        x_train,
        y_train,
        x_test,
        y_test,
        _,
        _,
        train_df,
        test_data,
        feature_names,
        categorical_feature_names,
        class_names,
        outcome_variable_name,
    ) = prepare_data(args=args, current_path=current_script_path)

    print(f"Using validation seed {args.validation_seed}")
    random.seed(args.validation_seed)
    torch.manual_seed(args.validation_seed)
    torch.cuda.manual_seed(args.validation_seed)
    torch.cuda.manual_seed_all(args.validation_seed)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(args.validation_seed)

    categorical_features = [
        train_df.columns.get_loc(col) for col in categorical_feature_names if col in train_df.columns
    ]

    explainer = Explainer(
        args=args,
        x_train=x_train,
        feature_names=feature_names,
        categorical_feature_names=categorical_feature_names,
        class_names=class_names,
        model=model,
        k_means_k=args.k_means_k,
        train_df=train_df,
        target_name=outcome_variable_name,
    )
    multiprocess.set_start_method("spawn")
    num_samples = min(1000, len(test_data))

    def explain_sample(
        explainer: Explainer,
        model: aix_model,
        bb: SimpleModel,
        sample: np.ndarray,
        y_sample: int,
        num_features: int,
        index: int,
        args: argparse.Namespace,
    ) -> tuple[list, int, int, list]:
        if index % 100 == 0:
            logger.info(f"Explaining sample {index}")
        x_sample = torch.tensor([sample])
        y_sample = torch.tensor([y_sample], dtype=torch.float32)

        bb.to("cuda" if torch.cuda.is_available() else "cpu")
        sample_pred_bb = make_predictions(x_sample, y_sample, bb)

        def predict_fn(x: np.ndarray) -> np.ndarray:
            model.model.to("cuda" if torch.cuda.is_available() else "cpu")
            prediction = model.predict_proba(x)
            return prediction

        start_explanation_time = datetime.datetime.now()

        if args.explanation_type == "lore":
            explanation, fidelity_sample, feat_in_the_rule = explainer.explain_instance(
                sample, predict_fn, sample_pred_bb[0].item()
            )
        else:
            explanation, local_pred, feat_in_the_rule = explainer.explain_instance(
                sample, predict_fn, sample_pred_bb[0].item()
            )

        end_time = datetime.datetime.now()
        total_time = (end_time - start_explanation_time).microseconds
        if args.explanation_type == "lore":
            return explanation, sample_pred_bb[0].item(), fidelity_sample, feat_in_the_rule, total_time, index
        else:
            return explanation, sample_pred_bb[0].item(), local_pred, feat_in_the_rule, total_time, index

    if args.explanation_type == "lore":
        args_list = [
            (
                explainer,
                model,
                bb,
                explainer.dataset.df.iloc[sample_idx][:-1],
                explainer.dataset.df.iloc[sample_idx][-1],
                len(feature_names),
                sample_idx,
                args,
            )
            for sample_idx in range(num_samples)
        ]
    else:
        args_list = [
            (
                explainer,
                model,
                bb,
                x_test[sample_idx],
                y_test[sample_idx],
                len(feature_names),
                sample_idx,
                args,
            )
            for sample_idx in range(num_samples)
        ]

    with Pool(40) as pool:
        results = pool.starmap(explain_sample, args_list)

    explanations, predictions_bb, local_predictions, features_in_the_rule, times, indexes = map(
        list, zip(*results, strict=False)
    )
    computed_explanations = {}
    if args.explanation_type == "lore":
        for index in indexes:
            computed_explanations[index] = explanations[index]
        fidelity = np.mean(local_predictions)
        fidelity_std = np.std(local_predictions)
    else:
        for index in indexes:
            computed_explanations[index] = (explanations[index], predictions_bb[index], features_in_the_rule[index])
        fidelity = accuracy_score(predictions_bb, local_predictions)

    file_name = args.explanation_type + f"_{args.validation_seed}" + ".pkl"
    store_path = Path(args.store_path) / file_name

    with Path(store_path).open("wb") as f:
        # store the computed_explanations dictionary as a json file
        dill.dump(computed_explanations, f)

    # print(Counter(predictions_bb), Counter(local_predictions))
    wandb_run = setup_wandb(args, project_name="comparison_tango", num_samples=num_samples)
    wandb_run.log(
        {
            "fidelity": fidelity,
            "fidelity_std": fidelity_std,
            "Total Time": np.mean(times),
            "Total Time Std": np.std(times),
            "Total Time (sec)": np.mean(times) / 1e6,
            "Total Time Std (sec)": np.std(times) / 1e6,
        }
    )
    logger.info(f"Final Fidelity: {fidelity}")
