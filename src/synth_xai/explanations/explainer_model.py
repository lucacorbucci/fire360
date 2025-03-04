import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from aix360.metrics.local_metrics import faithfulness_metric
from sklearn.tree import DecisionTreeClassifier

from synth_xai.explanations.decision_tree import (
    compute_robustness_dt,
    compute_stability_dt,
    extract_rule_dt,
    grid_search_dt,
    parse_explanation_dt,
)
from synth_xai.explanations.knn import extract_knn_explanation, grid_search_knn
from synth_xai.explanations.logistic_regression import (
    compute_robustness_lr,
    compute_stability_lr,
    extract_logistic_explanation,
    grid_search_lr,
    parse_coefficients_lr,
    parse_explanation_lr,
)
from synth_xai.explanations.svm import (
    compute_robustness_svm,
    compute_stability_svm,
    extract_svm_explanation,
    grid_search_svm,
    parse_coefficients_svm,
    parse_explanation_svm,
)
from synth_xai.utils import aix_model


class ExplainerModel:
    def __init__(self, explainer_type: str):
        self.explainer_type = explainer_type
        self.best_model = None

    def grid_search(self, x_train: pd.DataFrame, y_train: pd.Series, seed: int) -> None:
        match self.explainer_type:
            case "logistic":
                self.best_model = grid_search_lr(x_train=x_train, y_train=y_train, seed=seed)
            case "dt":
                self.best_model = grid_search_dt(x_train=x_train, y_train=y_train, seed=seed)
            case "svm":
                self.best_model = grid_search_svm(x_train=x_train, y_train=y_train, seed=seed)
            case "knn":
                self.best_model = grid_search_knn(x_train=x_train, y_train=y_train, seed=seed)
            case _:
                msg = "Invalid explainer type"
                raise ValueError(msg)

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("The model has not been trained. Please run grid_search first.")
        match self.explainer_type:
            case "logistic" | "dt" | "svm" | "knn":
                return np.array(self.best_model.predict(x_test))
            case _:
                msg = "Invalid explainer type"
                raise ValueError(msg)

    def extract_explanation(
        self,
        clf: DecisionTreeClassifier,
        y_name: str,
        sample: pd.DataFrame,
    ) -> tuple[np.ndarray, list[str], Any, Any]:
        match self.explainer_type:
            case "logistic":
                result = extract_logistic_explanation(model=clf, y_name=y_name, sample=sample)
                return (
                    result[0],
                    result[1],
                    result[2],
                    np.array(result[3]) if isinstance(result[3], list) else result[3],
                )
            case "dt":
                return extract_rule_dt(clf=clf, y_name=y_name, sample=sample)
            case "svm":
                return extract_svm_explanation(model=clf, sample=sample, y_name=y_name)
            case "knn":
                return extract_knn_explanation(model=clf, sample=sample, outcome_variable=y_name)
            case _:
                raise ValueError("Invalid explainer type")

    def compute_stability(self, explanations: list[list[str]]) -> float:
        """
        Compute the stability of the explanations
        """
        match self.explainer_type:
            case "logistic":
                return compute_stability_lr(explanations=explanations)
            case "dt":
                return compute_stability_dt(explanations=explanations)
            case "svm":
                return compute_stability_svm(explanations=explanations)
            case "knn":
                raise ValueError("Invalid explainer type")
            case _:
                raise ValueError("Invalid explainer type")

    def compute_robustness(
        self,
        explanations: list[list[str]],
        top_k: list,
    ) -> list:
        """
        Compute the stability of the explanations
        """
        match self.explainer_type:
            case "logistic":
                return compute_robustness_lr(explanations=explanations, top_k=top_k)
            case "dt":
                return compute_robustness_dt(explanations=explanations, top_k=top_k)
            case "svm":
                return compute_robustness_svm(explanations=explanations, top_k=top_k)
            case "knn":
                raise ValueError("Invalid explainer type")
            case _:
                raise ValueError("Invalid explainer type")

    def parse_explanation(self, explanation: str) -> list[str]:
        match self.explainer_type:
            case "dt":
                return parse_explanation_dt(explanation)
            case "logistic":
                return parse_explanation_lr(explanation)
            case "svm":
                return parse_explanation_svm(explanation)
            case _:
                raise ValueError("Invalid explainer type")

    def parse_coefficients(self, explanation: str) -> list[float]:
        match self.explainer_type:
            case "logistic":
                return parse_coefficients_lr(explanation)
            case "svm":
                return parse_coefficients_svm(explanation)
            case _:
                raise ValueError("Invalid explainer type")

    def compute_faithfulness(
        self,
        model: torch.nn.Module,
        dataset: pd.DataFrame,
        explanations: list[list[float]],
        base_value: float,
    ) -> tuple[float, float]:
        """
        Compute the faithfulness of the explanation
        """
        faithfulnesses = []
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        for index, explanation in enumerate(explanations):
            x = np.array(dataset[index])
            coefs = np.array(explanation)
            model_aix = aix_model(model)

            faithfulness = faithfulness_metric(
                model=model_aix,
                x=x,
                coefs=coefs,
                base=base_value * np.ones(shape=coefs.shape[0]),
            )
            # check if faithfulness is nan
            if np.isnan(faithfulness):
                continue
            faithfulnesses.append(faithfulness)

        return float(np.mean(faithfulnesses)), float(np.std(faithfulnesses))
