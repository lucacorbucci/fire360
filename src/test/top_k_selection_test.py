from collections import Counter

import numpy as np
import pandas as pd
import pytest
from fire360.explanations.explanation_utils import find_top_closest_rows


def test_find_top_closest_rows() -> None:
    """
    Test the find_top_closest_rows function
    """
    synthetic_data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [5, 4, 3, 2, 1, 0, -1, -2, -3, -4],
            "outcome": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    # Create a sample row
    sample = pd.DataFrame({"feature1": [2], "feature2": [4], "outcome": [1]})
    # Define the number of closest rows to find
    k = 4
    y_name = "outcome"
    # Call the function
    top_k_samples = find_top_closest_rows(synthetic_data, sample, k, y_name)
    # Check the output
    assert len(top_k_samples) == k
    assert isinstance(top_k_samples, pd.DataFrame)
    assert "feature1" in top_k_samples.columns
    assert "feature2" in top_k_samples.columns
    assert "outcome" in top_k_samples.columns
    # Check that there are at least 30% of the minority classes
    class_counts = Counter(top_k_samples[y_name])
    minority_class = 0
    minimum_percentage = 0.3
    assert class_counts[minority_class] / len(top_k_samples) >= minimum_percentage


def test_find_top_closest_rows_multiclass() -> None:
    """
    Test the find_top_closest_rows function
    """
    synthetic_data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "feature2": [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9],
            "outcome": [0, 1, 2, 3, 0, 1, 0, 3, 0, 3, 2, 1, 2, 3, 0],
        }
    )

    # Create a sample row
    sample = pd.DataFrame({"feature1": [2], "feature2": [4], "outcome": [1]})
    # Define the number of closest rows to find
    k = 4
    y_name = "outcome"
    # Call the function
    top_k_samples = find_top_closest_rows(synthetic_data, sample, k, y_name)
    # Check the output
    assert len(top_k_samples) == k
    assert isinstance(top_k_samples, pd.DataFrame)
    assert "feature1" in top_k_samples.columns
    assert "feature2" in top_k_samples.columns
    assert "outcome" in top_k_samples.columns
    # Check that there are at least 30% of the minority classes
    class_counts = Counter(top_k_samples[y_name])
    minority_classes = [0, 2, 3]
    minimum_percentage = 0.3

    total_minority_classes = 0
    for cl in minority_classes:
        total_minority_classes += class_counts[cl]

    assert total_minority_classes / len(top_k_samples) >= minimum_percentage


@pytest.mark.parametrize("k", [50, 100, 1000, 10000])
@pytest.mark.parametrize("dataset_size", [100000, 1000000, 1000000])
def test_find_top_closest_rows_big(k: int, dataset_size: int) -> None:
    """
    Test the find_top_closest_rows function
    """
    rng = np.random.default_rng()

    synthetic_data_large = pd.DataFrame(
        {
            "feature1": rng.integers(1, 20, size=dataset_size),  # Random integers between 1 and 20
            "feature2": rng.integers(-10, 10, size=dataset_size),  # Random integers between -10 and 10
            "outcome": rng.integers(0, 22, size=dataset_size),  # Random integers between 0 and 3 (4 classes)
        }
    )

    # Create a sample row
    sample = pd.DataFrame({"feature1": [2], "feature2": [4], "outcome": [1]})
    # Define the number of closest rows to find
    y_name = "outcome"
    # Call the function
    top_k_samples = find_top_closest_rows(synthetic_data_large, sample, k, y_name)
    # Check the output
    assert len(top_k_samples) == k
    assert isinstance(top_k_samples, pd.DataFrame)
    assert "feature1" in top_k_samples.columns
    assert "feature2" in top_k_samples.columns
    assert "outcome" in top_k_samples.columns
    # Check that there are at least 30% of the minority classes
    class_counts = Counter(top_k_samples[y_name])
    minority_classes = list(range(0, 22))
    del minority_classes[1]  # Remove the majority
    minimum_percentage = 0.3

    total_minority_classes = 0
    for cl in minority_classes:
        total_minority_classes += class_counts[cl]

    assert total_minority_classes / len(top_k_samples) >= minimum_percentage


if __name__ == "__main__":
    pytest.main()
