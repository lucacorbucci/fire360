"""
Unit tests for history_rec_sys.py

Tests cover:
- Neural network output shapes
- Data preprocessing correctness
- Model architecture validation
- Input/output tensor dimensions
"""

import os
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add the source directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rec_sys")))

from history_rec_sys import (
    HistoryRecommenderModel,
    calculate_exact_match_accuracy,
    evaluate_model,
    preprocess_data_with_one_hot,
    train_model,
)


class TestDataPreprocessing:
    """Test data preprocessing functions"""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing"""
        np.random.seed(42)
        data = {
            "interaction_id": [1, 1, 1, 2, 2, 3, 3, 3, 3],
            "timestamp": [1, 2, 3, 1, 2, 1, 2, 3, 4],
            "persona": ["A", "A", "A", "B", "B", "C", "C", "C", "C"],
            "feature1": np.random.rand(9),
            "feature2": np.random.rand(9),
            "feature3": np.random.randint(0, 5, 9),
            "suggested_explanation_Counterexamples": np.random.randint(0, 2, 9),
            "suggested_explanation_Counterfactual": np.random.randint(0, 2, 9),
            "suggested_explanation_Exemplars": np.random.randint(0, 2, 9),
            "suggested_explanation_FI": np.random.randint(0, 2, 9),
            "suggested_explanation_Rules": np.random.randint(0, 2, 9),
            "suggested_explanation_Confidence": np.random.randint(0, 2, 9),
            "preference": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
        return pd.DataFrame(data)

    def test_preprocess_data_basic_functionality(self, sample_dataframe):
        """Test basic preprocessing functionality"""
        X_sequences, y_last = preprocess_data_with_one_hot(sample_dataframe, max_sequence_length=5)

        # Check output types
        assert isinstance(X_sequences, np.ndarray)
        assert isinstance(y_last, np.ndarray)

        # Check output shapes
        assert X_sequences.ndim == 3  # (num_sequences, seq_len, num_features)
        assert y_last.ndim == 2  # (num_sequences, num_targets)

        # Check sequence length is respected
        assert X_sequences.shape[1] == 5  # max_sequence_length

        # Check target dimensions
        assert y_last.shape[1] == 6  # 6 target columns

        # Check number of sequences matches unique interaction_ids
        unique_ids = sample_dataframe["interaction_id"].nunique()
        assert X_sequences.shape[0] == unique_ids
        assert y_last.shape[0] == unique_ids

    def test_preprocess_data_sequence_length_padding(self):
        """Test sequence length padding for short sequences"""
        # Create data with short sequences
        data = {
            "interaction_id": [1, 1, 2],  # interaction 1 has 2 steps, interaction 2 has 1 step
            "timestamp": [1, 2, 1],
            "persona": ["A", "A", "B"],
            "feature1": [0.1, 0.2, 0.3],
            "feature2": [0.4, 0.5, 0.6],
            "suggested_explanation_Counterexamples": [1, 0, 1],
            "suggested_explanation_Counterfactual": [0, 1, 0],
            "suggested_explanation_Exemplars": [1, 1, 0],
            "suggested_explanation_FI": [0, 0, 1],
            "suggested_explanation_Rules": [1, 0, 1],
            "suggested_explanation_Confidence": [0, 1, 0],
            "preference": [1, 1, 1],
        }
        df = pd.DataFrame(data)

        X_sequences, y_last = preprocess_data_with_one_hot(df, max_sequence_length=4)

        # Check that sequences are padded correctly
        assert X_sequences.shape == (2, 4, 4)  # 2 interactions, 4 timesteps, 4 features (including UserID_A)
        assert y_last.shape == (2, 6)  # 2 interactions, 6 targets

        # Check that padding is at the beginning (zeros at start)
        # First interaction should have 2 zeros, then actual data
        assert np.all(X_sequences[0, :2, :] == 0)  # First 2 timesteps should be padding
        assert not np.all(X_sequences[0, 2:, :] == 0)  # Last 2 timesteps should have data

    def test_preprocess_data_sequence_length_truncation(self):
        """Test sequence length truncation for long sequences"""
        # Create data with long sequences
        data = {
            "interaction_id": [1] * 7,  # 7 interactions for same ID
            "timestamp": list(range(1, 8)),
            "persona": ["A"] * 7,
            "feature1": np.arange(0.1, 0.8, 0.1),
            "feature2": np.arange(0.2, 0.9, 0.1),
            "suggested_explanation_Counterexamples": [1, 0, 1, 0, 1, 0, 1],
            "suggested_explanation_Counterfactual": [0, 1, 0, 1, 0, 1, 0],
            "suggested_explanation_Exemplars": [1, 1, 0, 0, 1, 1, 0],
            "suggested_explanation_FI": [0, 0, 1, 1, 0, 0, 1],
            "suggested_explanation_Rules": [1, 0, 1, 0, 1, 0, 1],
            "suggested_explanation_Confidence": [0, 1, 0, 1, 0, 1, 0],
            "preference": [1] * 7,
        }
        df = pd.DataFrame(data)

        X_sequences, y_last = preprocess_data_with_one_hot(df, max_sequence_length=3)

        # Check that sequences are truncated correctly
        assert X_sequences.shape == (1, 3, 4)  # 1 interaction, 3 timesteps, 4 features
        assert y_last.shape == (1, 6)  # 1 interaction, 6 targets

        # Check that we kept the last 3 interactions
        # The last feature1 values should be 0.5, 0.6, 0.7 (indices 4, 5, 6)
        expected_last_feature1_values = [0.5, 0.6, 0.7]
        actual_feature1_values = X_sequences[0, :, 0].tolist()  # feature1 is first feature column
        np.testing.assert_array_almost_equal(actual_feature1_values, expected_last_feature1_values, decimal=5)

    def test_preprocess_data_missing_values(self):
        """Test handling of missing values"""
        data = {
            "interaction_id": [1, 1, 2],
            "timestamp": [1, 2, 1],
            "persona": ["A", "A", None],  # Missing persona
            "feature1": [0.1, None, 0.3],  # Missing feature
            "feature2": [0.4, 0.5, 0.6],
            "suggested_explanation_Counterexamples": [1, 0, 1],
            "suggested_explanation_Counterfactual": [0, 1, 0],
            "suggested_explanation_Exemplars": [1, 1, 0],
            "suggested_explanation_FI": [0, 0, 1],
            "suggested_explanation_Rules": [1, 0, 1],
            "suggested_explanation_Confidence": [0, 1, 0],
            "preference": [1, 1, 1],
        }
        df = pd.DataFrame(data)

        # Should not raise an exception
        X_sequences, y_last = preprocess_data_with_one_hot(df, max_sequence_length=3)

        # Check that data was processed
        assert X_sequences.shape[0] == 2  # 2 unique interaction_ids
        assert y_last.shape[0] == 2

    def test_preprocess_data_feature_columns(self, sample_dataframe):
        """Test that feature columns are correctly identified"""
        X_sequences, y_last = preprocess_data_with_one_hot(sample_dataframe, max_sequence_length=5)

        # Expected features: feature1, feature2, feature3, UserID_A (4 total)
        # Dropped: interaction_id, persona, timestamp
        # Targets: 6 explanation columns
        expected_num_features = 4  # 3 original features + UserID_A
        assert X_sequences.shape[2] == expected_num_features

    def test_preprocess_data_temporal_ordering(self):
        """Test that temporal ordering is preserved"""
        data = {
            "interaction_id": [1, 1, 1],
            "timestamp": [3, 1, 2],  # Out of order
            "persona": ["A", "A", "A"],
            "feature1": [0.3, 0.1, 0.2],  # Should be reordered to [0.1, 0.2, 0.3]
            "suggested_explanation_Counterexamples": [1, 0, 1],
            "suggested_explanation_Counterfactual": [0, 1, 0],
            "suggested_explanation_Exemplars": [1, 1, 0],
            "suggested_explanation_FI": [0, 0, 1],
            "suggested_explanation_Rules": [1, 0, 1],
            "suggested_explanation_Confidence": [0, 1, 0],
            "preference": [1, 1, 1],
        }
        df = pd.DataFrame(data)

        X_sequences, y_last = preprocess_data_with_one_hot(df, max_sequence_length=5)

        # Check that the sequence is ordered by timestamp
        # After sorting by timestamp: feature1 should be [0.1, 0.2, 0.3]
        expected_sequence = [0.0, 0.0, 0.1, 0.2, 0.3]  # 2 padding zeros + 3 actual values
        actual_sequence = X_sequences[0, :, 0].tolist()  # feature1 values
        np.testing.assert_array_almost_equal(actual_sequence, expected_sequence, decimal=5)


class TestHistoryRecommenderModel:
    """Test the neural network model architecture"""

    @pytest.fixture
    def model_params(self):
        """Standard model parameters for testing"""
        return {
            "input_size": 10,
            "hidden_size": 32,
            "num_layers": 2,
            "output_size": 6,
            "dropout": 0.2,
        }

    def test_model_initialization(self, model_params):
        """Test model initialization"""
        model = HistoryRecommenderModel(**model_params)

        # Check that all components are initialized
        assert isinstance(model.lstm, nn.LSTM)
        assert isinstance(model.dropout, nn.Dropout)
        assert isinstance(model.fc, nn.Linear)
        assert isinstance(model.sigmoid, nn.Sigmoid)

        # Check layer dimensions
        assert model.lstm.input_size == model_params["input_size"]
        assert model.lstm.hidden_size == model_params["hidden_size"]
        assert model.lstm.num_layers == model_params["num_layers"]
        assert model.fc.in_features == model_params["hidden_size"]
        assert model.fc.out_features == model_params["output_size"]

    def test_model_forward_pass_shape(self, model_params):
        """Test forward pass output shapes"""
        model = HistoryRecommenderModel(**model_params)
        model.eval()

        batch_size = 16
        seq_length = 5
        input_size = model_params["input_size"]

        # Create random input
        x = torch.randn(batch_size, seq_length, input_size)

        # Forward pass
        output = model(x)

        # Check output shape - should be many-to-one (only last timestep)
        expected_shape = (batch_size, model_params["output_size"])
        assert output.shape == expected_shape

        # Check output range (should be between 0 and 1 due to sigmoid)
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

    def test_model_forward_pass_different_batch_sizes(self, model_params):
        """Test forward pass with different batch sizes"""
        model = HistoryRecommenderModel(**model_params)
        model.eval()

        seq_length = 5
        input_size = model_params["input_size"]

        # Test different batch sizes
        for batch_size in [1, 8, 32, 64]:
            x = torch.randn(batch_size, seq_length, input_size)
            output = model(x)

            expected_shape = (batch_size, model_params["output_size"])
            assert output.shape == expected_shape

    def test_model_gradient_flow(self, model_params):
        """Test that gradients flow through the model correctly"""
        model = HistoryRecommenderModel(**model_params)

        batch_size = 8
        seq_length = 5
        input_size = model_params["input_size"]
        output_size = model_params["output_size"]

        x = torch.randn(batch_size, seq_length, input_size, requires_grad=True)
        target = torch.randint(0, 2, (batch_size, output_size)).float()

        # Forward pass
        output = model(x)

        # Calculate loss
        criterion = nn.BCELoss()
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Check that gradients are computed
        assert x.grad is not None
        for param in model.parameters():
            assert param.grad is not None

    def test_model_device_compatibility(self, model_params):
        """Test model works on different devices"""
        model = HistoryRecommenderModel(**model_params)

        # Test CPU
        x_cpu = torch.randn(4, 5, model_params["input_size"])
        output_cpu = model(x_cpu)
        assert output_cpu.device == torch.device("cpu")

        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            x_cuda = x_cpu.cuda()
            output_cuda = model_cuda(x_cuda)
            assert output_cuda.device.type == "cuda"

    def test_model_single_layer_no_dropout(self):
        """Test model with single layer and no dropout in LSTM"""
        model_params = {
            "input_size": 10,
            "hidden_size": 32,
            "num_layers": 1,  # Single layer
            "output_size": 6,
            "dropout": 0.2,
        }

        model = HistoryRecommenderModel(**model_params)

        # LSTM dropout should be 0 for single layer
        assert model.lstm.dropout == 0

        # Test forward pass still works
        x = torch.randn(4, 5, 10)
        output = model(x)
        assert output.shape == (4, 6)


class TestModelTrainingAndEvaluation:
    """Test training and evaluation functions"""

    @pytest.fixture
    def sample_data_loaders(self):
        """Create sample data loaders for testing"""
        # Create synthetic data
        batch_size = 8
        seq_length = 5
        input_size = 10
        output_size = 6
        num_samples = 32

        X = torch.randn(num_samples, seq_length, input_size)
        y = torch.randint(0, 2, (num_samples, output_size)).float()

        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, input_size, output_size

    def test_evaluate_model_basic(self, sample_data_loaders):
        """Test basic model evaluation"""
        train_loader, test_loader, input_size, output_size = sample_data_loaders

        model = HistoryRecommenderModel(
            input_size=input_size, hidden_size=16, num_layers=1, output_size=output_size, dropout=0.1
        )

        device = torch.device("cpu")
        metrics = evaluate_model(model, test_loader, device)

        # Check that all expected metrics are returned
        expected_metrics = ["Accuracy", "Precision", "Recall", "F1_Score"]
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1  # Metrics should be between 0 and 1

    def test_calculate_exact_match_accuracy(self, sample_data_loaders):
        """Test exact match accuracy calculation"""
        train_loader, test_loader, input_size, output_size = sample_data_loaders

        model = HistoryRecommenderModel(
            input_size=input_size, hidden_size=16, num_layers=1, output_size=output_size, dropout=0.1
        )

        device = torch.device("cpu")
        exact_match_acc = calculate_exact_match_accuracy(model, test_loader, device)

        # Should return a float between 0 and 1
        assert isinstance(exact_match_acc, float)
        assert 0 <= exact_match_acc <= 1

    @patch("wandb.log")
    def test_train_model_basic(self, mock_wandb_log, sample_data_loaders):
        """Test basic model training"""
        train_loader, test_loader, input_size, output_size = sample_data_loaders

        model = HistoryRecommenderModel(
            input_size=input_size, hidden_size=16, num_layers=1, output_size=output_size, dropout=0.1
        )

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = torch.device("cpu")

        # Train for a few epochs
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=2,
            wandb_logger=None,  # No wandb logging for test
            validation_loader=None,
        )

        # Check that model is returned and parameters have changed
        assert trained_model is not None
        assert isinstance(trained_model, HistoryRecommenderModel)

    def test_train_model_with_validation(self, sample_data_loaders):
        """Test model training with validation"""
        train_loader, test_loader, input_size, output_size = sample_data_loaders

        model = HistoryRecommenderModel(
            input_size=input_size, hidden_size=16, num_layers=1, output_size=output_size, dropout=0.1
        )

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = torch.device("cpu")

        # Mock wandb logger
        class MockWandbLogger:
            def __init__(self):
                self.logged_data = []

            def log(self, data):
                self.logged_data.append(data)

        wandb_logger = MockWandbLogger()

        # Train with validation
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=2,
            wandb_logger=wandb_logger,
            validation_loader=test_loader,
        )

        # Check that validation metrics were logged
        assert len(wandb_logger.logged_data) > 0
        # Should have training metrics and validation metrics
        logged_keys = set()
        for log_entry in wandb_logger.logged_data:
            logged_keys.update(log_entry.keys())

        assert "Train_Accuracy" in logged_keys
        assert "Train_Loss" in logged_keys


class TestIntegration:
    """Integration tests combining preprocessing and model"""

    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data preprocessing to model prediction"""
        # Create sample data
        data = {
            "interaction_id": [1, 1, 1, 2, 2, 3],
            "timestamp": [1, 2, 3, 1, 2, 1],
            "persona": ["A", "A", "A", "B", "B", "C"],
            "feature1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "feature2": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            "feature3": [1, 2, 3, 4, 5, 6],
            "suggested_explanation_Counterexamples": [1, 0, 1, 0, 1, 0],
            "suggested_explanation_Counterfactual": [0, 1, 0, 1, 0, 1],
            "suggested_explanation_Exemplars": [1, 1, 0, 0, 1, 1],
            "suggested_explanation_FI": [0, 0, 1, 1, 0, 0],
            "suggested_explanation_Rules": [1, 0, 1, 0, 1, 0],
            "suggested_explanation_Confidence": [0, 1, 0, 1, 0, 1],
            "preference": [1, 1, 1, 1, 1, 1],
        }
        df = pd.DataFrame(data)

        # Preprocessing
        X_sequences, y_targets = preprocess_data_with_one_hot(df, max_sequence_length=4)

        # Model creation
        input_size = X_sequences.shape[2]
        output_size = y_targets.shape[1]

        model = HistoryRecommenderModel(
            input_size=input_size, hidden_size=16, num_layers=1, output_size=output_size, dropout=0.1
        )

        # Convert to tensors
        X_tensor = torch.tensor(X_sequences, dtype=torch.float32)
        y_tensor = torch.tensor(y_targets, dtype=torch.float32)

        # Forward pass
        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor)

        # Check output shapes
        assert predictions.shape == (3, 6)  # 3 sequences, 6 target columns
        assert torch.all(predictions >= 0)
        assert torch.all(predictions <= 1)

    def test_data_consistency_shapes(self):
        """Test that all shapes are consistent throughout the pipeline"""
        # Create larger sample data
        np.random.seed(42)
        interaction_ids = np.repeat(range(1, 21), 4)  # 20 interactions, 4 timesteps each
        data = {
            "interaction_id": interaction_ids,
            "timestamp": np.tile(range(1, 5), 20),
            "persona": np.random.choice(["A", "B", "C"], len(interaction_ids)),
            "feature1": np.random.rand(len(interaction_ids)),
            "feature2": np.random.rand(len(interaction_ids)),
            "feature3": np.random.randint(0, 10, len(interaction_ids)),
            "suggested_explanation_Counterexamples": np.random.randint(0, 2, len(interaction_ids)),
            "suggested_explanation_Counterfactual": np.random.randint(0, 2, len(interaction_ids)),
            "suggested_explanation_Exemplars": np.random.randint(0, 2, len(interaction_ids)),
            "suggested_explanation_FI": np.random.randint(0, 2, len(interaction_ids)),
            "suggested_explanation_Rules": np.random.randint(0, 2, len(interaction_ids)),
            "suggested_explanation_Confidence": np.random.randint(0, 2, len(interaction_ids)),
            "preference": np.ones(len(interaction_ids)),
        }
        df = pd.DataFrame(data)

        max_seq_len = 5
        X_sequences, y_targets = preprocess_data_with_one_hot(df, max_sequence_length=max_seq_len)

        # Check preprocessing output shapes
        num_sequences = df["interaction_id"].nunique()
        assert X_sequences.shape[0] == num_sequences
        assert X_sequences.shape[1] == max_seq_len
        assert y_targets.shape[0] == num_sequences
        assert y_targets.shape[1] == 6  # 6 target columns

        # Test with DataLoader
        dataset = TensorDataset(torch.tensor(X_sequences), torch.tensor(y_targets))
        data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

        # Create model
        model = HistoryRecommenderModel(
            input_size=X_sequences.shape[2],
            hidden_size=32,
            num_layers=2,
            output_size=y_targets.shape[1],
            dropout=0.2,
        )

        # Test one batch
        for batch_X, batch_y in data_loader:
            predictions = model(batch_X)

            # Check batch shapes
            assert batch_X.shape[1] == max_seq_len  # Sequence length
            assert batch_X.shape[2] == X_sequences.shape[2]  # Feature size
            assert batch_y.shape[1] == 6  # Target size
            assert predictions.shape == batch_y.shape  # Prediction matches target shape
            break  # Only test one batch

    def test_model_deterministic_behavior(self):
        """Test that model produces consistent results with same input"""
        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Create deterministic input
        X = torch.randn(4, 5, 8)

        model = HistoryRecommenderModel(input_size=8, hidden_size=16, num_layers=1, output_size=6, dropout=0.0)

        # Run model twice in eval mode
        model.eval()
        with torch.no_grad():
            output1 = model(X)
            output2 = model(X)

        # Results should be identical
        torch.testing.assert_close(output1, output2)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
