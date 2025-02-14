import argparse
import copy
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import (
    MinMaxScaler,
)
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from synth_xai.bb_architectures import MultiClassModel, SimpleModel
from synth_xai.utils import (
    prepare_adult,
    prepare_breast_cancer,
    prepare_covertype,
    prepare_diabetes,
    prepare_dutch,
    prepare_house16,
    prepare_letter,
    prepare_pima,
    prepare_shuttle,
)


def get_test_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Get the preprocessed real data based on the dataset name. We will use
    these data to train the synthesizer and evaluate the quality of the synthetic data

    Args:
        args: The arguments passed to the script

    Returns:
        real_data: The preprocessed real data
        outcome_variable: The outcome variable of the dataset

    """
    current_script_path = Path(__file__).resolve().parent
    dataset_name = args.dataset_name
    match dataset_name:
        case "pima":
            _, _, _, _, _, _, _, test_data = prepare_pima(sweep=False, seed=args.seed, current_path=current_script_path)
            outcome_variable = "Outcome"
        case "adult":
            _, _, _, _, _, _, _, test_data = prepare_adult(
                sweep=False, seed=args.seed, current_path=current_script_path
            )

            outcome_variable = "income_binary"
        case "breast_cancer":
            _, _, _, _, _, _, _, test_data = prepare_breast_cancer(
                sweep=False, seed=args.seed, current_path=current_script_path
            )
            outcome_variable = "Status"
        case "diabetes":
            _, _, _, _, _, _, _, test_data = prepare_diabetes(
                sweep=False, seed=args.seed, current_path=current_script_path
            )
            outcome_variable = "readmitted"
        case "dutch":
            _, _, _, _, _, _, train_df, test_data = prepare_dutch(
                sweep=False, seed=args.seed, current_path=current_script_path
            )
            outcome_variable = "occupation_binary"
        case "letter":
            _, _, _, _, _, _, _, test_data = prepare_letter(sweep=False, seed=args.seed)
            outcome_variable = "letter"
        case "shuttle":
            _, _, _, _, _, _, _, test_data = prepare_shuttle(sweep=False, seed=args.seed)
            outcome_variable = "class"
        case "covertype":
            _, _, _, _, _, _, _, test_data = prepare_covertype(sweep=False, seed=args.seed)
            outcome_variable = "cover_type"
        case "house16":
            _, _, _, _, _, _, _, test_data = prepare_house16(sweep=False, seed=args.seed)
            outcome_variable = "class"
        case _:
            msg = "Invalid dataset name"
            raise ValueError(msg)
    return train_df, test_data, outcome_variable


def load_synthetic_data(synthetic_data_path: str) -> pd.DataFrame:
    return pd.read_csv(synthetic_data_path)


def load_bb(bb_path: str) -> torch.nn.Module:
    return torch.load(bb_path)


def create_torch_loader(batch_size: int, x_train: np.ndarray, y_train: np.ndarray) -> DataLoader:
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

    return DataLoader(train_dataset, batch_size=batch_size)


def find_top_closest_rows(synthetic_data: pd.DataFrame, sample: pd.DataFrame, k: int, y_name: str) -> pd.DataFrame:
    """
    Find the top k closest rows in the synthetic data to the sample.
    We use cosine similarity to find the closest rows.
    We make sure that in the neighbours we have at least 30% of the minority class.

    Args:
        synthetic_data: The synthetic data
        sample: The sample row
        k: The number of closest rows to find
        y_name: The outcome variable name

    Returns:
        top_k_samples: The top k closest rows to the sample

    """
    sample_class = sample[y_name].to_numpy()[0]
    sample = sample.drop(columns=[y_name])
    synthetic_data_ = synthetic_data.drop(columns=[y_name])
    similarity = cosine_similarity(sample, synthetic_data_)
    top_k_indices = np.argsort(similarity[0])[::-1]

    maximum_size_class = 0.7
    top_k_samples: list[pd.DataFrame] = []
    class_counts: Counter[int] = Counter()
    for idx in top_k_indices:
        if len(top_k_samples) >= k:
            break
        synthetic_class = synthetic_data.iloc[idx][y_name]

        # we convert the non binary problem to a binary one
        current_class = 1 if synthetic_class != sample_class else 0
        if class_counts[current_class] / k <= maximum_size_class:
            top_k_samples.append(synthetic_data.iloc[idx])
            class_counts[current_class] += 1
    logger.debug(class_counts)
    return pd.DataFrame(top_k_samples)


def make_predictions(x: pd.DataFrame, y: pd.DataFrame, bb: torch.nn.Module) -> list[torch.tensor]:
    train_loader = create_torch_loader(batch_size=16, x_train=x, y_train=y)
    predictions = []
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to("cuda"), target.to("cuda")
            outputs = bb(data)
            predicted = outputs.argmax(dim=1, keepdim=True)
            predictions.extend(predicted)

    return predictions


def evaluate_bb(x: np.array, y: np.array, bb: torch.nn.Module) -> None:
    """
    Evaluate the quality of the black-box model on the data x, y
    passed as arguments. We calculate the accuracy and F1 score of the model.

    Args:
        x: The features
        y: The labels
        bb: The black-box model

    Returns:
        None
    """
    predictions = [prediction.item() for prediction in make_predictions(x, y, bb)]
    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions)
    logger.info(f"Accuracy: {accuracy} - F1: {f1}")


def transform_input_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame, outcome_variable: str
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    x = test_data.drop(columns=[outcome_variable])
    y = test_data[outcome_variable].to_numpy()

    train_data = train_data.drop(columns=[outcome_variable])

    scaler = MinMaxScaler()
    _ = scaler.fit_transform(train_data)
    x = scaler.transform(x)

    return x, y, scaler


def label_synthetic_data(
    synthetic_data: pd.DataFrame, outcome_variable: str, bb: torch.nn.Module, scaler: MinMaxScaler
) -> pd.DataFrame:
    x = synthetic_data.drop(columns=[outcome_variable])
    y = synthetic_data[outcome_variable].to_numpy()
    x = scaler.transform(x)
    train_loader = create_torch_loader(batch_size=16, x_train=x, y_train=y)
    predictions = []
    with torch.no_grad():
        for data, target in train_loader:
            samples, _ = data.to("cuda"), target.to("cuda")
            outputs = bb(samples)
            predicted = outputs.argmax(dim=1, keepdim=True)
            predictions.extend(predicted)
    return [item.item() for item in predictions]


def prepare_neighbours(
    top_k_samples: pd.DataFrame,
    y_name: str,
) -> tuple[np.ndarray, list[int], pd.DataFrame]:
    y = top_k_samples[y_name]
    x = top_k_samples.drop(y_name, axis=1)
    old_x = copy.deepcopy(x)
    logger.debug(f"Targets of the synthetic dataset: {Counter(y)}")

    return x, y, old_x
