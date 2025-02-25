import argparse

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer


class Explainer:
    def __init__(
        self,
        args: argparse.Namespace,
        train_df: pd.DataFrame,
        feature_names: list[str],
        categorical_feature_names: list[str],
        class_names: list[str],
    ) -> None:
        self.explainer_name = args.explainer_name
        match args.explainer_name:
            case "lime":
                # Initialize LimeTabularExplainer
                self.explainer = LimeTabularExplainer(
                    train_df.drop(columns=["income_binary"]).values,  # Unscaled training data
                    mode="classification",
                    feature_names=feature_names,
                    categorical_features=categorical_feature_names,
                    class_names=class_names,
                    discretize_continuous=True,  # Discretize continuous features for better interpretability
                )

    def explain_instance(
        self,
        instance: pd.Series,
        predict_fn: callable,
        num_features: int,
    ) -> list:
        match self.explainer_name:
            case "lime":
                # Explain instance using LimeTabularExplainer
                explanation = self.explainer.explain_instance(
                    instance,
                    predict_fn,
                    num_features=10,
                )

                return explanation.as_list(), int(np.argmax(explanation.local_pred))
            case _:
                raise ValueError("Invalid explainer name")
                return []
