import argparse
import copy
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy.io import arff
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    StandardScaler,
)
from torch import optim
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

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


def prepare_data(args: argparse.Namespace, current_path: Path) -> tuple[DataLoader, DataLoader, DataLoader]:
    match args.dataset_name:
        # case "breast_cancer":
        #     x_train, x_val, x_test, y_train, y_val, y_test, train_df, test_data = prepare_breast_cancer(
        #         args.sweep, args.seed, current_path, args.validation_seed
        #     )
        # case "pima":
        #     x_train, x_val, x_test, y_train, y_val, y_test, train_df, test_data = prepare_pima(
        #         args.sweep, args.seed, current_path, args.validation_seed
        #     )
        # case "diabetes":
        #     x_train, x_val, x_test, y_train, y_val, y_test, train_df, test_data = prepare_diabetes(
        #         args.sweep, args.seed, current_path, args.validation_seed
        #     )
        case "adult":
            x_train, x_val, x_test, y_train, y_val, y_test, train_df, test_data = prepare_adult(
                args.sweep,
                args.seed,
                current_path,
                args.validation_seed,
            )
            outcome_variable = "income_binary"
            feature_names = list(train_df.drop(columns=[outcome_variable]).columns)
            categorical_feature_names = [
                "workclass",
                "education",
                "marital-status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "native-country",
            ]
            class_names = ["<=50K", ">50K"]

        case "dutch":
            x_train, x_val, x_test, y_train, y_val, y_test, train_df, test_data = prepare_dutch(
                args.sweep, args.seed, current_path, args.validation_seed
            )
            outcome_variable = "occupation_binary"
            feature_names = list(train_df.drop(columns=[outcome_variable]).columns)
            categorical_feature_names = [
                "sex",
                "age_category",
                "citizenship",
                "household_position",
                "household_size",
                "country_of_birth",
            ]
            class_names = ["Low Skill", "High Skill"]
        case "house16":
            x_train, x_val, x_test, y_train, y_val, y_test, train_df, test_data = prepare_house16(
                sweep=args.sweep,
                seed=args.seed,
                validation_seed=args.validation_seed,
            )
            outcome_variable = "class"
            feature_names = list(train_df.drop(columns=[outcome_variable]).columns)

            categorical_feature_names = []  # Assuming all features are numerical
            class_names = ["No House", "House"]
        case "letter":
            x_train, x_val, x_test, y_train, y_val, y_test, train_df, test_data = prepare_letter(
                sweep=args.sweep,
                seed=args.seed,
                validation_seed=args.validation_seed,
            )
            outcome_variable = "letter"
            feature_names = list(train_df.drop(columns=[outcome_variable]).columns)

            categorical_feature_names = []  # Assuming all features are numerical
            class_names = [chr(i) for i in range(65, 91)]  # A-Z letters

        case "shuttle":
            x_train, x_val, x_test, y_train, y_val, y_test, train_df, test_data = prepare_shuttle(
                sweep=args.sweep,
                seed=args.seed,
                validation_seed=args.validation_seed,
            )
            outcome_variable = "class"
            feature_names = list(train_df.drop(columns=[outcome_variable]).columns)

            categorical_feature_names = []  # Shuttle dataset is fully numerical
            class_names = ["1", "2", "3", "4", "5", "6", "7"]  # Adjust if needed

        case "covertype":
            x_train, x_val, x_test, y_train, y_val, y_test, train_df, test_data = prepare_covertype(
                sweep=args.sweep,
                seed=args.seed,
                validation_seed=args.validation_seed,
            )
            outcome_variable = "cover_type"
            feature_names = list(train_df.drop(columns=[outcome_variable]).columns)
            categorical_feature_names = []
            class_names = ["4", "1", "0", "6", "2", "5", "3"]
        case _:
            raise ValueError("Dataset not recognized")

    return (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        train_df,
        test_data,
        feature_names,
        categorical_feature_names,
        class_names,
    )
