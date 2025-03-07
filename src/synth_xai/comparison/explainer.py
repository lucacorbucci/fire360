import argparse
import copy
import re
import typing
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from lime.lime_tabular import LimeTabularExplainer
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import (
    MinMaxScaler,
)

from lore_sa.bbox import sklearn_classifier_bbox
from lore_sa.dataset import TabularDataset
from lore_sa.lore import TabularGeneticGeneratorLore, TabularRandomGeneratorLore
from synth_xai.bb_architectures import MultiClassModel, SimpleModel
from synth_xai.explanations.explanation_utils import load_bb
from synth_xai.utils import prepare_adult


class MyModel:
    def __init__(self, model: torch.nn.Module, scaler) -> None:
        self.model = model
        self.scaler = scaler

    def predict(self, x: np.ndarray) -> torch.Tensor:
        x = self.scaler.transform(x)

        predictions = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        for sample in x:
            sample_tensor = torch.Tensor(sample).to(device)
            predictions.append(self.model(sample_tensor).argmax().item())
        return np.array(predictions)


class Explainer:
    def __init__(
        self,
        args: argparse.Namespace,
        x_train: pd.DataFrame,
        feature_names: list[str],
        categorical_feature_names: list[str],
        class_names: list[str],
        model: torch.nn.Module = None,
        k_means_k: int = 100,
        train_df: pd.DataFrame = None,
        target_name: str = None,
        lore_generator: str = "random",
    ) -> None:
        self.explanation_type = args.explanation_type
        self.feature_names = feature_names
        self.model = model
        self.model.model.to("cuda" if torch.cuda.is_available() else "cpu")
        match args.explanation_type:
            case "lime":
                # Initialize LimeTabularExplainer
                self.explainer = LimeTabularExplainer(
                    x_train,  # Unscaled training data
                    mode="classification",
                    feature_names=feature_names,
                    categorical_features=categorical_feature_names,
                    class_names=class_names,
                    discretize_continuous=True,  # Discretize continuous features for better interpretability
                    random_state=args.validation_seed,
                )
            case "shap":
                self.explainer = shap.KernelExplainer(
                    model.predict_proba,
                    data=shap.kmeans(x_train, k_means_k),
                )
            case "lore":
                train_df_tmp = copy.copy(train_df)
                train_df_tmp = train_df_tmp.drop(columns=[target_name])

                scaler = MinMaxScaler()
                _ = scaler.fit_transform(train_df_tmp)

                model = MyModel(model=model.model, scaler=scaler)
                bbox = sklearn_classifier_bbox.sklearnBBox(model)

                train_df[target_name] = [int(x) for x in train_df[target_name]]
                train_df[target_name] = train_df[target_name].astype("category")
                self.dataset = TabularDataset.from_dict(train_df, class_name=target_name)
                self.dataset.df.dropna(inplace=True)

                if lore_generator == "genetic":
                    logger.info("Using genetic generator")
                    self.explainer = TabularGeneticGeneratorLore(bbox, self.dataset)
                else:
                    self.explainer = TabularRandomGeneratorLore(bbox, self.dataset)

    def explain_instance(
        self,
        instance: pd.Series,
        predict_fn: typing.Callable,
        prediction_bb: int = None,
    ) -> tuple[list, int, list]:
        match self.explanation_type:
            case "lime":
                # Explain instance using LimeTabularExplainer
                explanation = self.explainer.explain_instance(
                    instance,
                    predict_fn,
                    num_features=len(self.feature_names),
                )
                feature_names = [feature for feature, weight in explanation.as_list()]
                clean_features = [re.sub(r"[<>]=?|\d+(\.\d+)?", "", feature).strip() for feature in feature_names]

                local_pred = 0 if explanation.local_pred[0] < 0.5 else 1
                return explanation.as_list(), local_pred, clean_features
            case "shap":
                # Explain instance using SHAP
                shap_values = self.explainer(instance)

                feature_importance = shap_values.values[:, prediction_bb]

                return (
                    list(zip(self.feature_names, feature_importance)),
                    prediction_bb,
                    self.feature_names,
                )
            case "lore":
                explanation = self.explainer.explain(instance, num_instances=5000)
                prediction_surrogate = explanation["prediction_surrogate"]
                return (
                    explanation,
                    prediction_surrogate,
                    [premise["attr"] for premise in explanation["rule"]["premises"]],
                )
            case _:
                raise ValueError("Invalid explainer name")
                return []
