import argparse
import os

os.environ["OPENBLAS_NUM_THREADS"] = "64"
# os.environ["NUM_THREADS"] = "64"

import signal
import sys
import warnings
from pathlib import Path

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
parser.add_argument("--samples_to_generate", type=int, default=False)


def signal_handler(sig, frame):
    print("Gracefully stopping your experiment! Keep calm!")
    global wandb_run
    if wandb_run:
        wandb_run.finish()
    sys.exit(0)


def get_real_data(dataset_name):
    current_script_path = Path(__file__).resolve().parent
    if dataset_name == "pima":
        _, _, _, _, _, _, real_data, _ = prepare_pima(
            sweep=False, seed=args.seed, current_path=current_script_path
        )
        outcome_variable = "Outcome"
    elif dataset_name == "adult":
        _, _, _, _, _, _, real_data, _ = prepare_adult(
            sweep=False, seed=args.seed, current_path=current_script_path
        )
        outcome_variable = "income"
    elif dataset_name == "breast_cancer":
        _, _, _, _, _, _, real_data, _ = prepare_brest_cancer(
            sweep=False, seed=args.seed, current_path=current_script_path
        )
        outcome_variable = "Status"
    elif dataset_name == "diabetes":
        _, _, _, _, _, _, real_data, _ = prepare_diabetes(
            sweep=False, seed=args.seed, current_path=current_script_path
        )
        outcome_variable = "readmitted"
    elif dataset_name == "dutch":
        _, _, _, _, _, _, real_data, _ = prepare_dutch(
            sweep=False, seed=args.seed, current_path=current_script_path
        )
        outcome_variable = "occupation_binary"
    else:
        raise ValueError("Invalid dataset name")
    return real_data, outcome_variable


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    args = parser.parse_args()

    real_data, outcome_variable = get_real_data(args.dataset_name)

    y = real_data[outcome_variable]
    X = real_data.drop(columns=[outcome_variable])

    # smote = SMOTE(sampling_strategy=0.83, random_state=42)
    # real_data, y = smote.fit_resample(X, y)
    # real_data[outcome_variable] = y

    metadata = Metadata.detect_from_dataframe(
        data=real_data, table_name=args.dataset_name
    )

    synthesizer = CTGANSynthesizer(metadata)

    print("Starting fitting the model")
    synthesizer.fit(real_data)

    synthetic_data = synthesizer.sample(num_rows=args.samples_to_generate)

    # save synthetic data
    synthetic_data.to_csv(f"./data/{args.dataset_name}/synthetic_data.csv", index=False)

    diagnostic_report = run_diagnostic(
        real_data=real_data, synthetic_data=synthetic_data, metadata=metadata
    )
    print(diagnostic_report)

    quality_report = evaluate_quality(
        real_data=real_data, synthetic_data=synthetic_data, metadata=metadata
    )

    print(quality_report)
