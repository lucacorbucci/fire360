import argparse
import re
import typing
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from lime.lime_tabular import LimeTabularExplainer
from sklearn.cluster import KMeans

from synth_xai.bb_architectures import MultiClassModel, SimpleModel
from synth_xai.explanations.explanation_utils import load_bb
from synth_xai.utils import prepare_adult


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
            case _:
                raise ValueError("Invalid explainer name")
                return []
