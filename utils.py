import random
import time
from collections import Counter

import dill
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
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



def prepare_data(args):
    # Load and split data
    
    if args.dataset_name == "breast_cancer":
        df = pd.read_csv("../../data/cancer/breast_cancer.csv")
        encoder = LabelEncoder()
        for col in df.select_dtypes("object"):
            df[col] = encoder.fit_transform(df[[col]])
        x = df.drop(columns="Status")
        y = df.loc[:, "Status"]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, shuffle=True, random_state=args.seed, stratify=y
        )
        if args.sweep:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, test_size=0.2, shuffle=True, random_state=args.validation_seed, stratify=y_train
            )
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        if args.sweep:
            x_val = scaler.transform(x_val)

    elif args.dataset_name == "pima":
        df = pd.read_csv('../../data/pima/pima.csv')
        y = df["Outcome"]
        X = df.drop(columns=["Outcome"])
        

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)

        if args.sweep:
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=args.validation_seed, stratify=y)

        smote = SMOTE(sampling_strategy=0.83, random_state=args.seed)
        X_smote, y_smote = smote.fit_resample(x_train, y_train)
        minority_class_count = Counter(y_smote)[1]
        under_sampler = RandomUnderSampler(sampling_strategy={0: minority_class_count})
        x_train, y_train = under_sampler.fit_resample(X_smote, y_smote)
        print("Final class distribution after undersampling:", Counter(y_train))
        x_train = x_train.values
        x_test = x_test.values
        if args.sweep:
            x_val = x_val.values
    elif args.dataset_name == "diabetes":
        df = pd.read_csv("../../data/diabetes/diabetic_data.csv")

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
        df.fillna(df.mode().iloc[0], inplace=True)

        # Encode the 'readmitted' column as binary target variable
        df["readmitted"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)


        # Encode categorical features
        categorical_features = df.select_dtypes(include=["object"]).columns
        for col in categorical_features:
            df[col] = LabelEncoder().fit_transform(df[col])
        
        # undersample and oversample the data
        y = df["readmitted"]
        X = df.drop(columns=["readmitted"])

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.seed, stratify=y
        )
        if args.sweep:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, test_size=0.2, random_state=args.validation_seed, stratify=y
            )

        # Standardize numerical features fitting the scaler with the training data
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        if args.sweep:
            x_val = scaler.transform(x_val)

        smote = SMOTE(sampling_strategy=0.50, random_state=42)
        X_smote, y_smote = smote.fit_resample(x_train, y_train)

        minority_class_count = Counter(y_smote)[1]
        under_sampler = RandomUnderSampler(sampling_strategy={0: minority_class_count})
        x_train, y_train = under_sampler.fit_resample(X_smote, y_smote)

        print("Final class distribution after undersampling:", Counter(y_train))
        print("Final class distribution of test set:", Counter(y_test))

        
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    if args.sweep:
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.sweep:
        val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        val_loader = None


    return train_loader, val_loader, test_loader

