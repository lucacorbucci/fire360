import copy
import random
import time
from collections import Counter
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy.io import arff
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    TensorDataset,
)


def get_optimizer(optimizer, model, lr):
    if optimizer == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError("Optimizer not recognized")


def dataset_to_numpy(
    _df,
    _feature_cols: list,
    _metadata: dict,
):
    """Args:
    _df: pandas dataframe
    _feature_cols: list of feature column names
    _metadata: dictionary with metadata
    num_sensitive_features: number of sensitive features to use
    sensitive_features_last: if True, then sensitive features are encoded as last columns
    """

    # transform features to 1-hot
    _X = _df[_feature_cols]
    # take sensitive features separately
    if "dummy_cols" in _metadata.keys():
        dummy_cols = _metadata["dummy_cols"]
    else:
        dummy_cols = None
    _X2 = pd.get_dummies(_X, columns=dummy_cols, drop_first=False)
    # esc = MinMaxScaler()
    # _X = esc.fit_transform(_X2)

    return _X


def create_torch_loader(
    sweep, batch_size, x_train, x_val, x_test, y_train, y_val, y_test
):
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    if sweep:
        print("Validation set is used", sweep)
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if sweep:
        val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    return train_loader, val_loader, test_loader


def prepare_pima(sweep, seed, current_path, validation_seed=None):
    file_path = current_path / "data/pima/pima.csv"
    df = pd.read_csv(file_path)
    y = df["Outcome"]
    X = df.drop(columns=["Outcome"])

    columns_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    imputer = SimpleImputer(missing_values=0, strategy="mean")
    X[columns_with_zeros] = imputer.fit_transform(X[columns_with_zeros])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    train_df = None
    test_df = None

    if sweep:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=0.2,
            random_state=validation_seed,
            stratify=y_train,
        )
    else:
        x_val = None
        y_val = None

    smote = SMOTE(sampling_strategy=0.83, random_state=42)
    X_smote, y_smote = smote.fit_resample(x_train, y_train)
    print("After SMOTE:", Counter(y_smote))

    minority_class_count = Counter(y_smote)[1]
    under_sampler = RandomUnderSampler(sampling_strategy={0: minority_class_count})
    x_train, y_train = under_sampler.fit_resample(X_smote, y_smote)

    y_train = y_train.values
    y_test = y_test.values
    if sweep:
        y_val = y_val.values

    return x_train, x_val, x_test, y_train, y_val, y_test, train_df, test_df


def prepare_brest_cancer(sweep, seed, current_path, validation_seed=None):
    file_path = current_path / "data/breast_cancer/breast_cancer.csv"
    df = pd.read_csv(file_path)
    encoder = LabelEncoder()
    for col in df.select_dtypes("object"):
        df[col] = encoder.fit_transform(df[[col]])
    x = df.drop(columns="Status")
    y = df.loc[:, "Status"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True, random_state=seed, stratify=y
    )

    train_df = copy.copy(x_train)
    train_df["Status"] = y_train

    if sweep:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=0.2,
            shuffle=True,
            random_state=validation_seed,
            stratify=y_train,
        )
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    if sweep:
        x_val = scaler.transform(x_val)
        y_val = y_val.values
    else:
        x_val = None
        y_val = None
    y_test = y_test.values

    return x_train, x_val, x_test, y_train, y_val, y_test, train_df


def prepare_diabetes(sweep, seed, current_path, validation_seed=None):
    file_path = current_path / "data/diabetes/diabetic_data.csv"
    df = pd.read_csv(file_path)

    # Drop columns with too many unique values or low relevance for prediction
    drop_columns = [
        "encounter_id",
        "patient_nbr",
        "weight",
        "payer_code",
        "medical_specialty",
    ]

    df.drop(columns=drop_columns, inplace=True)

    # Handling missing values
    df.replace("?", np.nan, inplace=True)
    df.fillna(df.mean().iloc[0], inplace=True)

    high_cardinality_cols = ["diag_1", "diag_2", "diag_3"]
    for col in high_cardinality_cols:
        freq = df[col].value_counts(normalize=True)
        rare_categories = freq[freq < 0.01].index
        df[col] = df[col].replace(rare_categories, "Other")

    # Apply ordinal encoding to high-cardinality columns
    ordinal_encoder = LabelEncoder()
    for col in high_cardinality_cols:
        df[col] = ordinal_encoder.fit_transform(df[col].astype(str))

    # Encode the 'readmitted' column as binary target variable
    df["readmitted"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

    # Encode categorical features
    categorical_features = df.select_dtypes(include=["object"]).columns
    for col in categorical_features:
        df[col] = LabelEncoder().fit_transform(df[col])

    df = pd.get_dummies(df, columns=None, drop_first=False)

    y = df["readmitted"]
    X = df.drop(columns=["readmitted"])

    # Standardize numerical features fitting the scaler with the training data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    if sweep:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=0.2,
            random_state=validation_seed,
            stratify=y_train,
        )

    train_df = None
    test_df = None

    smote = SMOTE(sampling_strategy=0.50, random_state=seed)
    X_smote, y_smote = smote.fit_resample(x_train, y_train)

    minority_class_count = Counter(y_smote)[1]
    under_sampler = RandomUnderSampler(sampling_strategy={0: minority_class_count})
    x_train, y_train = under_sampler.fit_resample(X_smote, y_smote)

    print("Final class distribution after undersampling:", Counter(y_train))

    y_test = y_test.values
    return x_train, x_val, x_test, y_train, y_val, y_test, train_df, test_df


def prepare_adult(
    sweep,
    seed,
    current_path,
    validation_seed=None,
):
    adult_feat_cols = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "sex_binary",
        "race_binary",
        "age_binary",
    ]

    adult_columns_names = (
        "age",
        "workclass",  # Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
        "fnlwgt",  # "weight" of that person in the dataset (i.e. how many people does that person represent) -> https://www.kansascityfed.org/research/datamuseum/cps/coreinfo/keyconcepts/weights
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    )

    file_path = current_path / "data/adult/adult.data"
    df_adult = pd.read_csv(file_path, names=adult_columns_names)
    df_adult["sex_binary"] = np.where(df_adult["sex"] == " Male", 1, 0)
    df_adult["race_binary"] = np.where(df_adult["race"] == " White", 1, 0)
    df_adult["age_binary"] = np.where(
        (df_adult["age"] > 25) & (df_adult["age"] < 60), 1, 0
    )

    y = np.zeros(len(df_adult))

    y[df_adult["income"] == " >50K"] = 1
    df_adult["income_binary"] = y
    del df_adult["income"]

    df_adult_original = copy.copy(df_adult)
    del df_adult["income_binary"]

    df_adult = df_adult[adult_feat_cols]

    df_adult = pd.get_dummies(df_adult, columns=None, drop_first=False)

    x_train, x_test, y_train, y_test = train_test_split(
        df_adult, y, test_size=0.2, random_state=seed, stratify=y
    )

    train_df = df_adult_original.loc[x_train.index]
    test_df = df_adult_original.loc[x_test.index]

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    if sweep:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=0.2,
            random_state=validation_seed,
            stratify=y_train,
        )
    else:
        x_val = None
        y_val = None

    return x_train, x_val, x_test, y_train, y_val, y_test, train_df, test_df


def prepare_dutch(sweep, seed, current_path, validation_seed=None):
    file_path = current_path / "data/dutch/dutch_census.arff"
    data = arff.loadarff(file_path)
    dutch_df = pd.DataFrame(data[0]).astype("int32")

    dutch_df["sex_binary"] = np.where(dutch_df["sex"] == 1, 1, 0)
    dutch_df["occupation_binary"] = np.where(dutch_df["occupation"] >= 300, 1, 0)

    del dutch_df["sex"]
    del dutch_df["occupation"]

    y = dutch_df["occupation_binary"].astype(int).values

    dutch_df_feature_columns = [
        "age",
        "household_position",
        "household_size",
        "prev_residence_place",
        "citizenship",
        "country_birth",
        "edu_level",
        "economic_status",
        "cur_eco_activity",
        "Marital_status",
        "sex_binary",
    ]

    x_train, x_test, y_train, y_test = train_test_split(
        dutch_df, y, test_size=0.2, random_state=seed, stratify=y
    )

    train_df = copy.copy(x_train)
    train_df["occupation_binary"] = y_train
    test_df = copy.copy(x_test)
    test_df["occupation_binary"] = y_test

    if sweep:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=0.2,
            random_state=validation_seed,
            stratify=y_train,
        )
    else:
        x_val = None
        y_val = None

    x_train = dataset_to_numpy(
        x_train, dutch_df_feature_columns, {"target_variable": "occupation_binary"}
    )
    x_test = dataset_to_numpy(
        x_test, dutch_df_feature_columns, {"target_variable": "occupation_binary"}
    )

    if x_val is not None:
        x_val = dataset_to_numpy(
            x_val, dutch_df_feature_columns, {"target_variable": "occupation_binary"}
        )

    return x_train, x_val, x_test, y_train, y_val, y_test, train_df, test_df


def prepare_data(args, current_path) -> (DataLoader, DataLoader, DataLoader):
    # Load and split data

    if args.dataset_name == "breast_cancer":
        x_train, x_val, x_test, y_train, y_val, y_test, _, _ = prepare_brest_cancer(
            args.sweep, args.seed, current_path, args.validation_seed
        )
    elif args.dataset_name == "pima":
        x_train, x_val, x_test, y_train, y_val, y_test, _, _ = prepare_pima(
            args.sweep, args.seed, current_path, args.validation_seed
        )
    elif args.dataset_name == "diabetes":
        x_train, x_val, x_test, y_train, y_val, y_test, _, _ = prepare_diabetes(
            args.sweep, args.seed, current_path, args.validation_seed
        )
    elif args.dataset_name == "adult":
        x_train, x_val, x_test, y_train, y_val, y_test, _, _ = prepare_adult(
            args.sweep,
            args.seed,
            current_path,
            args.validation_seed,
        )
    elif args.dataset_name == "dutch":
        x_train, x_val, x_test, y_train, y_val, y_test, _, _ = prepare_dutch(
            args.sweep, args.seed, current_path, args.validation_seed
        )
    else:
        raise ValueError("Dataset not recognized")

    train_loader, val_loader, test_loader = create_torch_loader(
        args.sweep, args.batch_size, x_train, x_val, x_test, y_train, y_val, y_test
    )

    return train_loader, val_loader, test_loader
