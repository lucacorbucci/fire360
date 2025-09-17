import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import dill
import numpy as np
import pandas as pd
from loguru import logger
from personas_1 import bank_scenario_personas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--rule_explanations", type=str, default=None, required=True)
parser.add_argument("--fi_explanations", type=str, default=None, required=True)
parser.add_argument("--dataset_name", type=str, default=None, required=True)


class PersonaRecommenderDatasetGenerator:
    """
    Generates a dataset simulating interactions between different personas
    and a recommender system for ML explanations.
    """

    def __init__(
        self,
        personas: dict,
        all_explanations: dict,
        max_rejections: int = 2,
        min_rejections=1,
        probability_early_accept: float = 0.1,
    ):  # Added probability_early_accept
        # Define personas with their decision tree logic
        self.personas = personas
        self.max_rejections = max_rejections
        self.min_rejections = min_rejections
        self.all_explanations = all_explanations
        self.preferences = {}
        self.interactions = {}
        self.probability_early_accept = probability_early_accept  # Store the probability

    def get_metric_value(self, explanation_type, feature, sample_data):
        """Extract metric value from sample data based on explanation type and feature"""
        if explanation_type == "model":
            # For model explanation type, we look at confidence_bb from any explanation
            if "rules_explanation" in sample_data:
                return sample_data["rules_explanation"]["confidence_bb"]
            elif "fi_explanation" in sample_data:
                return sample_data["fi_explanation"]["confidence_bb"]
            else:
                raise ValueError("No explanation data found for model confidence")
        elif explanation_type in sample_data:
            return sample_data[explanation_type][feature]
        else:
            raise ValueError(f"Explanation type {explanation_type} not found in sample data")

    def traverse_tree(self, node, sample_data):
        """Recursively traverse the decision tree"""
        # Base case: if we have a prediction, return it
        if "prediction" in node:
            return node["prediction"]

        # If we have features to check, evaluate them
        if "features" in node:
            features = node["features"]
            explanation_types = node["explanation_type"]
            thresholds = node["threshold"]

            # Check all features (assuming AND logic for multiple features)
            all_conditions_met = True

            for i, feature in enumerate(features):
                explanation_type = explanation_types[i] if isinstance(explanation_types, list) else explanation_types
                threshold = thresholds[i] if isinstance(thresholds, list) else thresholds

                try:
                    metric_value = self.get_metric_value(explanation_type, feature, sample_data)

                    # If any condition fails (metric < threshold), go left
                    if metric_value < threshold:
                        all_conditions_met = False
                        break

                except (KeyError, ValueError) as e:
                    print(f"Warning: Could not evaluate {feature} with {explanation_type}: {e}")
                    all_conditions_met = False
                    break

            # Navigate based on conditions
            if all_conditions_met:
                # All conditions met, go right
                if "right" in node:
                    return self.traverse_tree(node["right"], sample_data)
                else:
                    raise ValueError("No right branch found when conditions are met")
            else:
                # At least one condition failed, go left
                if "left" in node:
                    return self.traverse_tree(node["left"], sample_data)
                else:
                    raise ValueError("No left branch found when conditions fail")

        # Handle single feature case (for nested nodes)
        elif "feature" in node:
            feature = node["feature"]
            explanation_type = node["explanation_type"]
            threshold = node["threshold"]

            try:
                metric_value = self.get_metric_value(explanation_type, feature, sample_data)

                if metric_value < threshold:
                    # Condition failed, go left
                    if "left" in node:
                        return self.traverse_tree(node["left"], sample_data)
                    else:
                        raise ValueError("No left branch found when condition fails")
                elif "right" in node:
                    return self.traverse_tree(node["right"], sample_data)
                else:
                    raise ValueError("No right branch found when condition is met")

            except (KeyError, ValueError) as e:
                logger.info(f"Warning: Could not evaluate {feature} with {explanation_type}: {e}")
                # Default to left branch on error
                if "left" in node:
                    return self.traverse_tree(node["left"], sample_data)
                else:
                    raise ValueError("No left branch found for error case")

        else:
            raise ValueError("Invalid node structure: no features, feature, or prediction found")

    def get_persona_prediction(self):
        """
        Traverses a persona decision tree to determine the appropriate prediction
        based on sample explanation metrics.
        """
        for persona_name, persona in self.personas.items():
            self.preferences[persona_name] = {}
            for index, sample in self.all_explanations.items():
                self.preferences[persona_name][index] = self.traverse_tree(persona, sample)

    def simulate_interactions(self):
        """
        Simulates interactions between personas and the recommender system.
        Includes a probability to accept a suggestion before max_rejections.
        """
        if not self.preferences:
            raise ValueError("Preferences must be generated before simulating interactions.")
        possible_suggestions = [
            "Rules",
            "Counterfactual",
            "Counterexamples",
            "FI",
            "Exemplars",
        ]

        for persona_name, persona_prefs in self.preferences.items():
            self.interactions[persona_name] = {}
            for index, preferred_suggestion in persona_prefs.items():
                self.interactions[persona_name][index] = []
                current_timestamp = datetime.now()

                num_actual_rejections = 0
                potential_refusals = [sugg for sugg in possible_suggestions if sugg != preferred_suggestion]
                random.shuffle(potential_refusals)

                for i in range(self.max_rejections):
                    if not potential_refusals:
                        break

                    # Check for early acceptance based on probability
                    if random.random() < self.probability_early_accept and num_actual_rejections >= self.min_rejections:
                        break  # Persona decides to accept earlier than max_rejections

                    refused_suggestion = potential_refusals.pop(0)

                    self.interactions[persona_name][index].append(
                        {"suggestion": refused_suggestion, "refused": True, "timestamp": current_timestamp}
                    )
                    current_timestamp += timedelta(seconds=1)
                    num_actual_rejections += 1

                self.interactions[persona_name][index].append(
                    {"suggestion": preferred_suggestion, "refused": False, "timestamp": current_timestamp}
                )

    def create_dataset(self):
        """
        Creates a pandas dataset from the simulated interactions.
        The dataset is structured with the following columns:
        - the sample with all the column of the sample. This is in the self.all_explanations
        dictionary
        - The model confidence, this is again in the self.all_explanations dictionary
        - The stability of the sample, this is again in the self.all_explanations
        - The fidelity neighbours of the sample, this is again in the self.all_explanations
        - The suggested explanation made by the recommender system that is stored in the
        self.interactions dictionary
        - The preference for that suggestion (Refused/Accepted)
        - A unique incremental ID for each persona interaction with a sample
        """
        if not self.interactions:
            raise ValueError("Interactions must be simulated before creating dataset.")

        dataset_rows = []
        interaction_id = 0  # Initialize unique ID counter

        for persona_name, persona_interactions in self.interactions.items():
            for sample_index, interactions_list in persona_interactions.items():
                # Get the sample data from all_explanations
                sample_data = self.all_explanations[sample_index]

                # Extract sample features from rules_explanation
                sample = sample_data["rules_explanation"]["sample"]

                # Extract metrics from explanations
                model_confidence = sample_data["rules_explanation"]["confidence_bb"]
                stability = sample_data["rules_explanation"]["stability"]
                fidelity_neighbours = sample_data["rules_explanation"]["fidelity_neighbours"]

                # Process each interaction for this sample
                for interaction in interactions_list:
                    row = {}
                    row["interaction_id"] = interaction_id

                    # Add sample features (assuming sample is a dict or array-like)
                    if isinstance(sample, dict):
                        row.update(sample)
                    elif hasattr(sample, "__iter__") and not isinstance(sample, str):
                        # If sample is array-like, create feature columns
                        if hasattr(sample, "index") and hasattr(sample, "values"):
                            # Handle pandas Series
                            for feature_name, feature_value in zip(sample, sample.values[0]):
                                row[feature_name] = feature_value
                        elif hasattr(sample, "columns") and hasattr(sample, "iloc"):
                            # Handle pandas DataFrame (use first row)
                            row.update(sample.iloc[0].to_dict())
                        else:
                            # Handle other iterables (list, array, etc.)
                            row.update({f"feature_{i}": value for i, value in enumerate(sample)})
                    else:
                        row["sample"] = sample

                    # Add metadata
                    row["persona"] = persona_name
                    row["sample_index"] = sample_index
                    row["model_confidence"] = model_confidence
                    row["stability"] = stability
                    row["fidelity_neighbours"] = fidelity_neighbours

                    # Add interaction data
                    suggestion = interaction["suggestion"]
                    row["suggested_explanation"] = [suggestion] if isinstance(suggestion, str) else suggestion
                    row["timestamp"] = interaction["timestamp"]
                    row["role"] = persona_name
                    row["preference"] = "Accepted" if not interaction["refused"] else "Refused"

                    dataset_rows.append(row)
                interaction_id += 1

        # Create DataFrame
        self.dataset = pd.DataFrame(dataset_rows)
        return self.dataset

    def pre_process_dataset(
        self,
        categorical_columns=None,
        target_column="preference",
        suggested_explanation_column="suggested_explanation",
    ):
        """
        Preprocess the dataset to ensure all columns are in the correct format.
        The categorical columns are converted using one hot encoding.
        The target column is converted to a binary format.
        The suggested explanation column is converted to a list of strings even in the case
        in which it is a single string. Then, we use the one hot encoding
        to convert the suggested explanation column to a binary format where the
        suggested explanation is 1 and the others are 0.
        We split the dataset into train and test set, by first splitting the dataset
        based on the role, then for each role we split the dataset into 80% train and 20% test set
        based on the timestamp, the last 20% of the dataset is used as test set. We also
        ensure that the last row of the training dataset is an accepted suggestion.
        In the end we remove the "role" column.
        """
        if not hasattr(self, "dataset") or self.dataset is None:
            raise ValueError("Dataset must be created before preprocessing.")

        df = self.dataset.copy()

        # Convert target column to binary format (Accepted=1, Refused=0)
        df[target_column] = (df[target_column] == "Accepted").astype(int)
        # Handle suggested explanation column - get all unique explanations from lists
        # First, collect all unique individual strings from all lists
        all_explanations = set()

        # Store original lists for later processing
        original_suggested_explanations = df[suggested_explanation_column].copy()

        for explanations in df[suggested_explanation_column]:
            if isinstance(explanations, list):
                all_explanations.update(explanations)
            elif isinstance(explanations, str):
                # Handle case where it might be a single string
                all_explanations.add(explanations)
            else:
                # Convert other types to string
                all_explanations.add(str(explanations))

        # One-hot encode suggested explanations
        # For each unique explanation, check if it's present in each row's list
        for explanation in all_explanations:
            df[f"{suggested_explanation_column}_{explanation}"] = original_suggested_explanations.apply(
                lambda x: 1
                if (isinstance(x, list) and explanation in x) or (not isinstance(x, list) and str(x) == explanation)
                else 0
            )

        # Handle categorical columns with one-hot encoding
        if categorical_columns:
            for col in categorical_columns:
                if col in df.columns:
                    # Get dummies and join with original dataframe
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df, dummies], axis=1)
                    df.drop(col, axis=1, inplace=True)

        # Sort by role and timestamp for proper train/test splitting
        df = df.sort_values(["role", "interaction_id", "timestamp"])

        train_dfs = []
        test_dfs = []

        # Split by role, then by timestamp
        for role in df["role"].unique():
            role_data = df[df["role"] == role].copy()

            role_data.reset_index(drop=True, inplace=True)

            # Sort by timestamp within role
            role_data = role_data.sort_values(["interaction_id", "timestamp"])

            unique_interaction_ids = sorted(role_data["interaction_id"].unique())
            # get the first 80% of the interaction IDs for training
            split_idx = int(len(unique_interaction_ids) * 0.8)
            train_interaction_ids = unique_interaction_ids[split_idx]

            train_interaction_ids = (
                role_data[role_data["interaction_id"] == unique_interaction_ids[split_idx - 1]].index[-1] + 1
            )

            train_role = role_data.iloc[:train_interaction_ids]
            test_role = role_data.iloc[train_interaction_ids:]
            print(
                role,
                len(unique_interaction_ids),
                len(unique_interaction_ids) * 0.8,
                split_idx,
                train_interaction_ids,
                len(role_data),
                len(train_role),
                len(test_role),
            )
            train_dfs.append(train_role)
            test_dfs.append(test_role)

        # Combine all role data
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)

        # Remove role column
        columns_to_drop = ["role", "suggested_explanation"]
        train_df = train_df.drop(columns=[col for col in columns_to_drop if col in test_df.columns])
        test_df = test_df.drop(columns=[col for col in columns_to_drop if col in test_df.columns])

        return train_df, test_df


if __name__ == "__main__":
    args = parser.parse_args()
    base_path = Path("/home/lcorbucci/fire360/artifacts/")

    explanation_path = base_path / args.dataset_name / f"explanations/{args.rule_explanations}.pkl"
    with explanation_path.open("rb") as f:
        rule_explanations = dill.load(f)

    if args.fi_explanations is None:
        raise ValueError("Feature importance explanations must be provided.")
    fi_explanation_path = base_path / args.dataset_name / f"explanations/{args.fi_explanations}.pkl"
    with fi_explanation_path.open("rb") as f:
        fi_explanations = dill.load(f)

    indexes = set(rule_explanations.keys()) & set(fi_explanations.keys())

    all_explanations_data = {
        index: {
            "rules_explanation": rule_explanations[index],
            "fi_explanation": fi_explanations[index],
        }
        for index in indexes
    }

    generator = PersonaRecommenderDatasetGenerator(
        personas=bank_scenario_personas,
        max_rejections=5,
        min_rejections=5,
        all_explanations=all_explanations_data,
        probability_early_accept=0.1,
    )

    logger.info("Generating persona predictions...")
    generator.get_persona_prediction()
    logger.info("Simulating interactions...")
    generator.simulate_interactions()
    logger.info("Creating dataset...")
    dataset = generator.create_dataset()
    dataset.to_csv("dataset.csv", index=False)

    logger.info("Preprocessing dataset...")
    train, test = generator.pre_process_dataset()

    train.to_csv(f"train_{args.dataset_name}_test.csv", index=False)
    test.to_csv(f"test_{args.dataset_name}_test.csv", index=False)
