import argparse
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

import wandb

warnings.filterwarnings("ignore")


def preprocess_data_with_one_hot(
    df, max_sequence_length=5, scaler=None, target_scaler=None, include_previous_targets=True, personas_mapping=None
):
    """
    Preprocess the interaction data by creating sequences of interactions for each interaction_id.
    For many-to-one LSTM: use history as input, predict only for the last timestep.
    Includes past target values in the input features.
    """
    print(f"Processing {len(df)} total interactions...")

    # Handle missing values
    missing_values_columns = df.columns[df.isna().any()].tolist()
    for column in missing_values_columns:
        df[column] = df[column].fillna(df[column].mode()[0])

    if len(df.columns[df.isna().any()].tolist()) != 0:
        error_message = "There are still missing values in the dataset"
        raise ValueError(error_message)

    # Define columns to drop and target columns
    cols_to_drop = [
        "interaction_id",
        "timestamp",
        "preference",
    ]

    target_cols = [
        "suggested_explanation_Counterexamples",
        "suggested_explanation_Counterfactual",  # -> questo funziona
        "suggested_explanation_Exemplars",
        "suggested_explanation_FI",  # -> questo funziona
        "suggested_explanation_Rules",
        "suggested_explanation_Confidence",  # -> questo funziona
    ]

    persona_column = "persona"

    # handle categorical features
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        # use target encoding for categorical features
        if col not in cols_to_drop + target_cols:
            df[col] = df[col].astype("category")
            if col == persona_column:
                original_categories = df[col].cat.categories
                personas_mapping = dict(zip(original_categories, range(len(original_categories))))
                inverse_personas_mapping = {v: k for k, v in personas_mapping.items()}
            df[col] = df[col].cat.codes

    # Convert timestamp to datetime for proper sorting
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Get feature columns (everything except what we drop and targets)
    original_feature_cols = [col for col in df.columns if col not in cols_to_drop + target_cols]

    print(f"Original Feature columns ({len(original_feature_cols)}): {original_feature_cols}")
    print(f"Target columns ({len(target_cols)}): {target_cols}")

    # Group by interaction_id to create sequences
    X_sequences = []
    y_last = []

    grouped = df.groupby("interaction_id")
    print(f"Processing {len(grouped)} unique interaction groups...")
    personas = []

    for interaction_id, group in grouped:
        # Sort by timestamp to maintain temporal order
        group = group.sort_values("timestamp")
        # extract persona
        if persona_column in group.columns:
            current_persona = group[persona_column].iloc[0]
            personas.append(current_persona)
        # Extract original features and targets
        original_features_group = group[original_feature_cols].values.astype(np.float32)
        targets_group = group[target_cols].values.astype(np.float32)

        # Initialize an empty list to store features with concatenated targets
        sequence_with_targets = []

        # For each step in the sequence, concatenate the original features with the *previous* targets
        # The first step will have zero-padded targets
        for i in range(len(original_features_group)):
            current_original_features = original_features_group[i]
            if include_previous_targets:
                if i == 0:
                    # For the first element, use zeros for previous targets
                    previous_targets = np.zeros(len(target_cols), dtype=np.float32)
                else:
                    previous_targets = targets_group[i - 1]  # Use target from the previous step

                combined_features = np.concatenate((current_original_features, previous_targets))
            else:
                combined_features = current_original_features

            sequence_with_targets.append(combined_features)

        sequence_with_targets = np.array(sequence_with_targets, dtype=np.float32)

        # Handle sequence length - pad or truncate to max_sequence_length
        if len(sequence_with_targets) < max_sequence_length:
            # Pad with zeros at the beginning
            padding_length = max_sequence_length - len(sequence_with_targets)
            feature_padding = np.zeros((padding_length, sequence_with_targets.shape[1]), dtype=np.float32)
            sequence_with_targets = np.vstack([feature_padding, sequence_with_targets])
            last_target = (
                targets_group[-1] if len(targets_group) > 0 else np.zeros(targets_group.shape[1], dtype=np.float32)
            )
        elif len(sequence_with_targets) > max_sequence_length:
            # Take the last max_sequence_length interactions
            sequence_with_targets = sequence_with_targets[-max_sequence_length:]
            last_target = targets_group[-1]
        else:
            last_target = targets_group[-1]

        X_sequences.append(sequence_with_targets)
        y_last.append(last_target)

    # Convert to numpy arrays
    X_sequences = np.array(X_sequences, dtype=np.float32)
    y_last = np.array(y_last, dtype=np.float32)

    print(f"Created {len(X_sequences)} sequences with shape {X_sequences.shape}")
    print(f"Target shape (last timestep only): {y_last.shape}")

    # Apply scaling to features
    if scaler is None:
        scaler = MinMaxScaler()
        # Reshape to 2D for scaler (combine batch and sequence dimensions)
        X_reshaped = X_sequences.reshape(-1, X_sequences.shape[-1])
        X_scaled = scaler.fit_transform(X_reshaped)
        # Reshape back to original shape
        X_sequences = X_scaled.reshape(X_sequences.shape)
    else:
        # Use existing scaler for transform only
        X_reshaped = X_sequences.reshape(-1, X_sequences.shape[-1])
        X_scaled = scaler.transform(X_reshaped)
        X_sequences = X_scaled.reshape(X_sequences.shape)

    # Print target distribution for debugging
    print("Target distribution:")
    for i, col in enumerate(target_cols):
        pos_count = np.sum(y_last[:, i])
        total_count = len(y_last)
        print(f"  {col}: {pos_count}/{total_count} ({pos_count / total_count:.3f})")

    # We return the scaler for features, and potentially a target_scaler if you wanted to scale targets too (though for binary, often not needed)
    return X_sequences, y_last, scaler, target_cols, personas, inverse_personas_mapping


class GRUMultiClassMultiLabel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(GRUMultiClassMultiLabel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        gru_out, hidden = self.gru(x)
        # Take the last output from the sequence (many-to-one)
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_size)
        last_output = self.dropout(last_output)
        output = self.fc(last_output)  # (batch_size, num_classes)
        return output


def evaluate_example_based_metrics(
    ground_truth: np.ndarray, predictions: np.ndarray, personas: np.ndarray = None, persona_mapping: dict = None
):
    """
    Calculate example-based metrics, optionally including per-persona metrics
    """
    exact_match_acc = np.mean((predictions == ground_truth).all(axis=1)) * 100
    hamming_loss = np.mean(ground_truth != predictions)

    # compute the hamming score that is defined as
    # the sum over all the samples of the number of items that are 1
    # both in the ground truth and the predictions divided by the total number
    # of items that are 1 in the ground truth union the predictions.
    # This is then divided by the number of samples.
    hamming_score = np.mean(
        [
            np.sum(predictions[i].astype(bool) & ground_truth[i].astype(bool))
            / np.sum(predictions[i].astype(bool) | ground_truth[i].astype(bool))
            for i in range(ground_truth.shape[0])
        ]
    )

    micro_f1 = f1_score(ground_truth, predictions, average="micro")
    macro_f1 = f1_score(ground_truth, predictions, average="macro")

    precision_micro = precision_score(ground_truth, predictions, average="micro")
    precision_macro = precision_score(ground_truth, predictions, average="macro")

    recall_micro = recall_score(ground_truth, predictions, average="micro")
    recall_macro = recall_score(ground_truth, predictions, average="macro")

    result = {
        "exact_match_accuracy": exact_match_acc,
        "hamming_loss": hamming_loss,
        "hamming_score": hamming_score,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "micro_precision": precision_micro,
        "macro_precision": precision_macro,
        "micro_recall": recall_micro,
        "macro_recall": recall_macro,
    }

    # Add per-persona metrics if personas are provided
    if personas is not None:
        unique_personas = np.unique(personas)
        per_persona_metrics = {}

        for persona in unique_personas:
            persona_mask = personas == persona
            if np.sum(persona_mask) == 0:
                continue

            persona_gt = ground_truth[persona_mask]
            persona_pred = predictions[persona_mask]

            # Calculate per-persona metrics
            persona_exact_match = np.mean((persona_pred == persona_gt).all(axis=1)) * 100
            persona_hamming_loss = np.mean(persona_gt != persona_pred)
            persona_hamming_score = np.mean(
                [
                    np.sum(persona_pred[i].astype(bool) & persona_gt[i].astype(bool))
                    / np.sum(persona_pred[i].astype(bool) | persona_gt[i].astype(bool))
                    for i in range(persona_gt.shape[0])
                ]
            )

            persona_micro_f1 = f1_score(persona_gt, persona_pred, average="micro")
            persona_macro_f1 = f1_score(persona_gt, persona_pred, average="macro")
            persona_precision_micro = precision_score(persona_gt, persona_pred, average="micro")
            persona_precision_macro = precision_score(persona_gt, persona_pred, average="macro")
            persona_recall_micro = recall_score(persona_gt, persona_pred, average="micro")
            persona_recall_macro = recall_score(persona_gt, persona_pred, average="macro")

            per_persona_metrics[f"persona_{persona_mapping[persona]}"] = {
                "exact_match_accuracy": persona_exact_match,
                "hamming_loss": persona_hamming_loss,
                "hamming_score": persona_hamming_score,
                "micro_f1": persona_micro_f1,
                "macro_f1": persona_macro_f1,
                "micro_precision": persona_precision_micro,
                "macro_precision": persona_precision_macro,
                "micro_recall": persona_recall_micro,
                "macro_recall": persona_recall_macro,
            }

        result["per_persona_metrics"] = per_persona_metrics

    return result


def evaluate_label_based_metrics(
    ground_truth: np.ndarray, predictions: np.ndarray, personas: np.ndarray = None, persona_mapping: dict = None
):
    # Per-class accuracy
    per_class_acc = (predictions == ground_truth).astype(np.float32).mean(axis=0) * 100

    # Per-label metrics
    per_label_f1 = f1_score(ground_truth, predictions, average=None)
    per_label_precision = precision_score(ground_truth, predictions, average=None)
    per_label_recall = recall_score(ground_truth, predictions, average=None)

    result = {
        "per_class_accuracy": per_class_acc,
        "per_label_f1": per_label_f1,
        "per_label_precision": per_label_precision,
        "per_label_recall": per_label_recall,
    }

    # Add per-persona metrics if personas are provided
    if personas is not None and persona_mapping is not None:
        unique_personas = np.unique(personas)
        per_persona_metrics = {}

        for persona in unique_personas:
            persona_mask = personas == persona
            if np.sum(persona_mask) == 0:
                continue

            persona_gt = ground_truth[persona_mask]
            persona_pred = predictions[persona_mask]

            # Calculate per-persona label-based metrics
            persona_per_class_acc = (persona_pred == persona_gt).astype(np.float32).mean(axis=0) * 100
            persona_per_label_f1 = f1_score(persona_gt, persona_pred, average=None)
            persona_per_label_precision = precision_score(persona_gt, persona_pred, average=None)
            persona_per_label_recall = recall_score(persona_gt, persona_pred, average=None)

            per_persona_metrics[f"persona_{persona_mapping[persona]}"] = {
                "per_class_accuracy": persona_per_class_acc,
                "per_label_f1": persona_per_label_f1,
                "per_label_precision": persona_per_label_precision,
                "per_label_recall": persona_per_label_recall,
            }

        result["per_persona_metrics"] = per_persona_metrics

    return result


class MyDataSet(Dataset):
    def __init__(self, X, y, personas=None):
        self.X = X
        self.y = y
        self.personas = personas

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.personas is not None:
            return self.X[idx], self.y[idx], self.personas[idx]
        else:
            return self.X[idx], self.y[idx]


def compute_baseline(y_train_baseline, y_test_baseline):
    # Calculate positive proportions from training data
    positive_proportions = np.mean(y_train_baseline, axis=0)

    # Generate random predictions for test set
    random_predictions = np.random.rand(y_test_baseline.shape[0], y_test_baseline.shape[1])

    # Apply thresholds based on positive proportions
    random_labels = (random_predictions < positive_proportions).astype(int)

    # Evaluate using your metrics
    random_metrics = evaluate_example_based_metrics(y_test_baseline, random_labels)

    return random_metrics


def train_model(lstm_model, train_loader, scheduler):
    lstm_model.train()
    total_loss_train = 0

    for batch_X, batch_y, personas in train_loader:
        batch_X = batch_X.to("cuda" if torch.cuda.is_available() else "cpu")
        batch_y = batch_y.to("cuda" if torch.cuda.is_available() else "cpu")
        lstm_optimizer.zero_grad()

        # Forward pass
        outputs = lstm_model(batch_X)
        loss = lstm_criterion(outputs, batch_y)

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)

        lstm_optimizer.step()
        total_loss_train += loss.item()
        # probs = torch.sigmoid(outputs) # This line was not used, can be removed

    # Step the scheduler
    scheduler.step()

    avg_loss_train = total_loss_train / len(train_loader)

    if epoch % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss_train:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    return avg_loss_train


def eval_model(lstm_model, dataloader):
    # Evaluation on validation set
    lstm_model.eval()
    total_loss_eval = 0
    all_predictions = []
    all_ground_truth = []
    all_personas = []
    with torch.no_grad():
        for batch_X, batch_y, personas in dataloader:
            batch_X = batch_X.to("cuda" if torch.cuda.is_available() else "cpu")
            batch_y = batch_y.to("cuda" if torch.cuda.is_available() else "cpu")
            personas = personas.to("cuda" if torch.cuda.is_available() else "cpu")
            predictions = lstm_model(batch_X)
            predicted_probs = torch.sigmoid(predictions)
            predicted_labels = (predicted_probs > 0.5).float()
            loss = lstm_criterion(predictions, batch_y)
            all_predictions.append(predicted_labels.cpu())
            all_ground_truth.append(batch_y.cpu())
            total_loss_eval += loss.item()
            all_personas.append(personas.cpu())
    return all_ground_truth, all_predictions, total_loss_eval, all_personas


def log_results(
    sweep: bool,
    unique_personas: list[str],
    epoch: int,
    example_based_metrics: dict,
    label_based_metrics: dict,
    y_train: np.ndarray,
    y_val: np.ndarray | None,
    y_test: np.ndarray | None,
    target_cols: list[str],
    wandb,
) -> None:
    if sweep:
        for cla, acc in enumerate(label_based_metrics["per_class_accuracy"]):
            wandb.log(
                {
                    f"label {target_cols[cla]} Validation accuracy": acc.item() / 100,
                    "epoch": epoch + 1,
                }
            )
        for cla, f1 in enumerate(label_based_metrics["per_label_f1"]):
            wandb.log(
                {
                    f"label {target_cols[cla]} Validation F1": f1.item(),
                    "epoch": epoch + 1,
                }
            )
        for cla, prec in enumerate(label_based_metrics["per_label_precision"]):
            wandb.log(
                {
                    f"label {target_cols[cla]} Validation Precision": prec.item(),
                    "epoch": epoch + 1,
                }
            )
        for cla, rec in enumerate(label_based_metrics["per_label_recall"]):
            wandb.log(
                {
                    f"label {target_cols[cla]} Validation Recall": rec.item(),
                    "epoch": epoch + 1,
                }
            )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_loss_train,
                "val_loss": total_loss_eval,
                "val_exact_match_accuracy": example_based_metrics["exact_match_accuracy"],
                "val_hamming_loss": example_based_metrics["hamming_loss"],
                "val_hamming_score": example_based_metrics["hamming_score"],
                "val_micro_f1": example_based_metrics["micro_f1"],
                "val_macro_f1": example_based_metrics["macro_f1"],
                "val_micro_precision": example_based_metrics["micro_precision"],
                "val_macro_precision": example_based_metrics["macro_precision"],
                "val_micro_recall": example_based_metrics["micro_recall"],
                "val_macro_recall": example_based_metrics["macro_recall"],
            }
        )
        random_example_based_metrics = compute_baseline(y_train, y_val)

    else:
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_loss_train,
                "test_exact_match_accuracy": example_based_metrics["exact_match_accuracy"],
                "test_hamming_loss": example_based_metrics["hamming_loss"],
                "test_hamming_score": example_based_metrics["hamming_score"],
                "test_micro_f1": example_based_metrics["micro_f1"],
                "test_macro_f1": example_based_metrics["macro_f1"],
                "test_micro_precision": example_based_metrics["micro_precision"],
                "test_macro_precision": example_based_metrics["macro_precision"],
                "test_micro_recall": example_based_metrics["micro_recall"],
                "test_macro_recall": example_based_metrics["macro_recall"],
            }
        )

        for cla, acc in enumerate(label_based_metrics["per_class_accuracy"]):
            wandb.log({"epoch": epoch + 1, f"label {target_cols[cla]} Test accuracy": acc.item() / 100})
        for cla, f1 in enumerate(label_based_metrics["per_label_f1"]):
            wandb.log({"epoch": epoch + 1, f"label {target_cols[cla]} Test F1": f1.item()})
        for cla, prec in enumerate(label_based_metrics["per_label_precision"]):
            wandb.log({"epoch": epoch + 1, f"label {target_cols[cla]} Test Precision": prec.item()})
        for cla, rec in enumerate(label_based_metrics["per_label_recall"]):
            wandb.log({"epoch": epoch + 1, f"label {target_cols[cla]} Test Recall": rec.item()})

        random_example_based_metrics = compute_baseline(y_train, y_test)
    wandb.log(
        {
            "random_example_based_exact_match_accuracy": random_example_based_metrics["exact_match_accuracy"],
            "random_example_based_hamming_loss": random_example_based_metrics["hamming_loss"],
            "random_example_based_hamming_score": random_example_based_metrics["hamming_score"],
            "random_example_based_micro_f1": random_example_based_metrics["micro_f1"],
            "random_example_based_macro_f1": random_example_based_metrics["macro_f1"],
            "random_example_based_micro_precision": random_example_based_metrics["micro_precision"],
            "random_example_based_macro_precision": random_example_based_metrics["macro_precision"],
            "random_example_based_micro_recall": random_example_based_metrics["micro_recall"],
            "random_example_based_macro_recall": random_example_based_metrics["macro_recall"],
        }
    )

    for persona in unique_personas:
        persona_metrics = example_based_metrics["per_persona_metrics"].get(f"persona_{persona}", {})
        wandb.log(
            {
                "epoch": epoch + 1,
                f"persona_{persona}_exact_match_accuracy": persona_metrics.get("exact_match_accuracy", 0),
                f"persona_{persona}_hamming_loss": persona_metrics.get("hamming_loss", 0),
                f"persona_{persona}_hamming_score": persona_metrics.get("hamming_score", 0),
                f"persona_{persona}_micro_f1": persona_metrics.get("micro_f1", 0),
                f"persona_{persona}_macro_f1": persona_metrics.get("macro_f1", 0),
                f"persona_{persona}_micro_precision": persona_metrics.get("micro_precision", 0),
                f"persona_{persona}_macro_precision": persona_metrics.get("macro_precision", 0),
                f"persona_{persona}_micro_recall": persona_metrics.get("micro_recall", 0),
                f"persona_{persona}_macro_recall": persona_metrics.get("macro_recall", 0),
            }
        )

        persona_label_metrics = label_based_metrics["per_persona_metrics"].get(f"persona_{persona}", {})
        for cla, acc in enumerate(persona_label_metrics.get("per_class_accuracy", [])):
            wandb.log(
                {
                    f"persona_{persona}_label_{target_cols[cla]}_accuracy": acc.item() / 100,
                    "epoch": epoch + 1,
                }
            )
        for cla, f1 in enumerate(persona_label_metrics.get("per_label_f1", [])):
            wandb.log(
                {
                    f"persona_{persona}_label_{target_cols[cla]}_F1": f1.item(),
                    "epoch": epoch + 1,
                }
            )
        for cla, prec in enumerate(persona_label_metrics.get("per_label_precision", [])):
            wandb.log(
                {
                    f"persona_{persona}_label_{target_cols[cla]}_Precision": prec.item(),
                    "epoch": epoch + 1,
                }
            )
        for cla, rec in enumerate(persona_label_metrics.get("per_label_recall", [])):
            wandb.log(
                {
                    f"persona_{persona}_label_{target_cols[cla]}_Recall": rec.item(),
                    "epoch": epoch + 1,
                }
            )


if __name__ == "__main__":
    # Argument parser (same as before)
    parser = argparse.ArgumentParser(description="History-based Recommendation System Training")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size for LSTM")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--validation_seed", type=int, default=None, help="Validation split seed")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--project_name", type=str, default="history-rec-sys", help="Wandb project name")
    parser.add_argument("--sweep", type=bool, default=False, help="Whether this is a sweep run")
    parser.add_argument("--zeros", type=bool, default=False, help="Whether to use zeros for padding")
    parser.add_argument(
        "--include_previous_targets", type=bool, default=True, help="Include previous targets in input features"
    )
    parser.add_argument("--max_sequence_length", type=int, default=5, help="Maximum sequence length")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for optimizer")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer type")
    parser.add_argument(
        "--model",
        type=str,
        default="lstm",
        choices=["lstm", "bilstm", "gru", "lstm_attention", "lstm_deeper"],
        help="Model type",
    )
    args = parser.parse_args()

    max_sequence_length = 5

    ####################################################################################
    #                                   Prepare Data.                                  #
    ####################################################################################

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    if args.validation_seed is None:
        validation_seed = int(str(time.time()).split(".")[1]) * args.seed
        args.validation_seed = validation_seed

    if args.sweep:
        # Get unique interaction_ids from training data
        unique_interaction_ids = train_df["interaction_id"].unique()

        # Split interaction_ids instead of individual samples
        train_ids, val_ids = train_test_split(unique_interaction_ids, test_size=0.2, random_state=args.validation_seed)

        # Create boolean masks for training and validation sets
        train_mask = np.isin(train_df["interaction_id"], train_ids)
        val_mask = np.isin(train_df["interaction_id"], val_ids)

        # Filter the original dataframes
        train_df_split = train_df[train_mask]
        val_df_split = train_df[val_mask]

        # Preprocess the split data separately
        X_train, y_train, scaler, target_cols, personas_train, persona_mapping = preprocess_data_with_one_hot(
            train_df_split, max_sequence_length, include_previous_targets=args.include_previous_targets
        )
        X_val, y_val, _, _, personas_val, _ = preprocess_data_with_one_hot(
            val_df_split,
            max_sequence_length,
            scaler,
            include_previous_targets=args.include_previous_targets,
            personas_mapping=persona_mapping,
        )
    else:
        X_val = None
        y_val = None
        X_train, y_train, scaler, target_cols, personas_train, persona_mapping = preprocess_data_with_one_hot(
            train_df,
            max_sequence_length,
            include_previous_targets=args.include_previous_targets,
        )

    X_test, y_test, _, _, personas_test, _ = preprocess_data_with_one_hot(
        test_df,
        max_sequence_length,
        scaler,
        include_previous_targets=args.include_previous_targets,
        personas_mapping=persona_mapping,
    )
    print(persona_mapping)

    # Determine num_classes based on the preprocessed target data
    num_classes = y_train.shape[1]

    # input_size needs to be updated to include the concatenated targets
    # X_train.shape[2] now represents original_feature_size + num_classes
    input_size = X_train.shape[2]

    print(f"Size: {X_train.shape}, Y Size: {y_train.shape}, Personas: {len(personas_train)}")

    train_dataset = MyDataSet(torch.tensor(X_train), torch.tensor(y_train), torch.tensor(personas_train))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    if X_val is not None and y_val is not None:
        val_dataset = MyDataSet(torch.tensor(X_val), torch.tensor(y_val), torch.tensor(personas_val))
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        val_loader = None
        test_dataset = MyDataSet(torch.tensor(X_test), torch.tensor(y_test), torch.tensor(personas_test))
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    ####################################################################################
    #                             Prepare Model Training                               #
    ####################################################################################

    lstm_model = GRUMultiClassMultiLabel(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout,
    )

    # Use better optimizer settings with lower learning rate
    lstm_criterion = nn.BCEWithLogitsLoss()
    if args.optimizer == "adam":
        lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # Reduced LR
    else:
        lstm_optimizer = optim.SGD(
            lstm_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    scheduler = optim.lr_scheduler.StepLR(lstm_optimizer, step_size=20, gamma=0.5)  # Add scheduler

    # Training loop
    num_epochs = args.num_epochs
    best_loss = float("inf")

    lstm_model.to("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=args.project_name,
        name=f"{args.model}-hidden{args.hidden_size}-layers{args.num_layers}-dropout{args.dropout}-batch{args.batch_size}-epochs{num_epochs}-lr{args.lr}",
        config={
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "num_epochs": num_epochs,
            "learning_rate": args.lr,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "max_sequence_length": max_sequence_length,
            "input_size": input_size,  # Log the new input size
            "num_classes": num_classes,  # Log the number of classes
        },
    )

    ####################################################################################
    #                                   Training Loop                                  #
    ####################################################################################

    for epoch in range(num_epochs):
        avg_loss_train = train_model(lstm_model=lstm_model, train_loader=train_loader, scheduler=scheduler)
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_loss_train,
            }
        )

        if args.sweep:
            all_ground_truth, all_predictions, total_loss_eval, all_personas = eval_model(lstm_model, dataloader=val_loader)
        else:
            all_ground_truth, all_predictions, total_loss_eval, all_personas = eval_model(lstm_model, dataloader=test_loader)

        example_based_metrics = evaluate_example_based_metrics(
            ground_truth=torch.cat(all_ground_truth).numpy(),
            predictions=torch.cat(all_predictions).numpy(),
            personas=torch.cat(all_personas).numpy(),
            persona_mapping=persona_mapping,
        )
        label_based_metrics = evaluate_label_based_metrics(
            ground_truth=torch.cat(all_ground_truth).numpy(),
            predictions=torch.cat(all_predictions).numpy(),
            personas=torch.cat(all_personas).numpy(),
            persona_mapping=persona_mapping,
        )

        unique_personas = [persona_mapping[persona] for persona in list(np.unique(torch.cat(all_personas).numpy()))]

        log_results(
            sweep=args.sweep,
            unique_personas=unique_personas,
            epoch=epoch,
            example_based_metrics=example_based_metrics,
            label_based_metrics=label_based_metrics,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            target_cols=target_cols,
            wandb=wandb,
        )
