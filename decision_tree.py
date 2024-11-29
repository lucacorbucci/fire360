import argparse
import os
import random
import signal
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from scipy.io import arff
from sdv.evaluation.single_table import (
    evaluate_quality,
    run_diagnostic,
)
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer

from utils import (
    prepare_adult,
    prepare_brest_cancer,
    prepare_diabetes,
    prepare_dutch,
    prepare_pima,
)

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--dataset_name", type=str, default=False)
parser.add_argument("--seed", type=int, default=42)


def signal_handler(sig, frame):
    print("Gracefully stopping your experiment! Keep calm!")
    global wandb_run
    if wandb_run:
        wandb_run.finish()
    sys.exit(0)


def get_test_data(dataset_name):
    current_script_path = Path(__file__).resolve().parent
    if dataset_name == "pima":
        _, _, _, _, _, _, _, test_data = prepare_pima(
            sweep=False, seed=args.seed, current_path=current_script_path
        )
        outcome_variable = "Outcome"
    elif dataset_name == "adult":
        _, _, _, _, _, _, _, test_data = prepare_adult(
            sweep=False, seed=args.seed, current_path=current_script_path
        )
        outcome_variable = "income"
    elif dataset_name == "breast_cancer":
        _, _, _, _, _, _, _, test_data = prepare_brest_cancer(
            sweep=False, seed=args.seed, current_path=current_script_path
        )
        outcome_variable = "Status"
    elif dataset_name == "diabetes":
        _, _, _, _, _, _, _, test_data = prepare_diabetes(
            sweep=False, seed=args.seed, current_path=current_script_path
        )
        outcome_variable = "readmitted"
    elif dataset_name == "dutch":
        _, _, _, _, _, _, _, test_data = prepare_dutch(
            sweep=False, seed=args.seed, current_path=current_script_path
        )
        outcome_variable = "occupation_binary"
    else:
        raise ValueError("Invalid dataset name")
    return test_data, outcome_variable


def load_synthetic_data(dataset_name):
    synthetic_data_path = f"synthetic_data/{dataset_name}.csv"
    synthetic_data = pd.read_csv(synthetic_data_path)
    return synthetic_data


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    args = parser.parse_args()

    current_script_path = Path(__file__).resolve().parent
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    test_data, outcome_variable = get_test_data(args.dataset_name)

    sample = test_data.sample(1)

    synthetic_data = load_synthetic_data(args.dataset_name)

    top_100 = find_top_closest_rows()

    bb = load_bb()
