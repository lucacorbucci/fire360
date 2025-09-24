# FIRE360

<!-- markdownlint-disable MD033 MD041 MD002 -->
<!-- markdownlint-disable commands-show-output no-duplicate-heading -->
<!-- spell-checker:ignore markdownlint ; (options) DESTDIR UTILNAME manpages reimplementation oranda -->
<div align="center">

![fire360 logo](logo.png)

A comprehensive and fast local explanation approach tailored for tabular data.

[![PyPI](https://img.shields.io/pypi/v/fire360)](https://pypi.org/project/fire360/)
[![Python](https://img.shields.io/pypi/pyversions/fire360)](https://pypi.org/project/fire360/)
[![License](https://img.shields.io/github/license/lucacorbucci/fire360)](https://github.com/lucacorbucci/fire360/blob/main/LICENSE)

</div>

## Description

FIRE360 is a Python library for generating local explanations of black-box machine learning models on tabular data. It uses surrogate models trained on synthetic data neighborhoods to provide interpretable explanations while maintaining high fidelity to the original black-box predictions.

## Features

- **Multiple Surrogate Models**: Support for Decision Trees, Logistic Regression, SVM, and KNN as surrogate explainers
- **Synthetic Data Generation**: Uses CTGAN and TVAE for generating synthetic datasets to create explanation neighborhoods
- **Evaluation Metrics**: Computes fidelity, robustness, and stability of explanations
- **Benchmarking**: Includes comparison tools with state-of-the-art explainers like LIME, SHAP, and LORE
- **Extensible Architecture**: Easy to add new surrogate models and evaluation metrics
- **Comprehensive Datasets**: Pre-configured support for Adult, Dutch, Covertype, House16, Letter, and Shuttle datasets

## Installation

### Prerequisites

- Python >= 3.10, < 3.13
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management

### Install uv

You can install uv using one of the following methods:

```bash
# Using pipx
pipx install uv

# Using curl
curl -LsSf https://astral.sh/uv/install.sh | sh

# Using wget
wget -qO- https://astral.sh/uv/install.sh | sh
```

For more details, visit the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

### Install FIRE360

```bash
# Clone the repository
git clone https://github.com/lucacorbucci/fire360.git
cd fire360

# Install dependencies and the package
uv sync
```

## Quick Start

Here's a basic example of how to use FIRE360 to explain a black-box model prediction:

```python
from fire360.explanations.explainer_model import ExplainerModel
from fire360.explanations.explanation_utils import (
    load_bb, load_synthetic_data, label_synthetic_data,
    find_top_closest_rows, prepare_neighbours
)
import pandas as pd

# Load your trained black-box model
bb_model = load_bb("path/to/black_box_model.pth")

# Load and prepare synthetic data
synthetic_data = load_synthetic_data("path/to/synthetic_data.csv")
synthetic_data = label_synthetic_data(synthetic_data, "target_column", bb_model, scaler)

# Select a sample to explain
sample = test_data.iloc[[0]]  # Your test sample

# Find similar samples from synthetic data
neighborhood = find_top_closest_rows(synthetic_data, sample, k=1000, y_name="target_column")
X_neigh, y_neigh, _ = prepare_neighbours(neighborhood, "target_column")

# Create explainer and generate explanation
explainer = ExplainerModel(explainer_type="dt")  # or "logistic", "svm", "knn"
explainer.grid_search(X_neigh, y_neigh, seed=42)

sample_pred, explanation, threshold, feature = explainer.extract_explanation(
    explainer.best_model, "target_column", sample
)

print(f"Explanation: {explanation}")
```

For a complete working example, see [`src/examples/fire360_example.ipynb`](src/examples/fire360_example.ipynb).

## API Reference

### Core Classes

#### `ExplainerModel`

Main class for generating explanations using different surrogate models.

**Parameters:**
- `explainer_type` (str): Type of surrogate model ("dt", "logistic", "svm", "knn")

**Methods:**
- `grid_search(x_train, y_train, seed)`: Perform hyperparameter tuning
- `extract_explanation(model, y_name, sample)`: Generate explanation for a sample
- `predict(x_test)`: Make predictions with the surrogate model
- `compute_stability(explanations)`: Calculate explanation stability
- `compute_robustness(top_k_samples)`: Calculate explanation robustness
- `compute_faithfulness(x_test, y_test)`: Calculate faithfulness metric

### Utility Functions

Located in `fire360.explanations.explanation_utils`:

- `load_bb(model_path)`: Load a pre-trained black-box model
- `load_synthetic_data(data_path)`: Load synthetic dataset
- `label_synthetic_data(synthetic_data, outcome_variable, bb, scaler)`: Label synthetic data with BB predictions
- `find_top_closest_rows(synthetic_data, sample, k, y_name)`: Find k most similar samples
- `prepare_neighbours(top_k_samples, y_name)`: Prepare neighborhood data for training
- `evaluate_bb(x, y, bb)`: Evaluate black-box model accuracy



## Generating Synthetic Data

Synthetic datasets are generated using CTGAN and TVAE from the SDV library. Recommended training epochs:
- CTGAN: 2500 epochs
- TVAE: 2500 epochs

Scripts are available in `src/experiments/train_synth/` for generating synthetic data for each dataset.

## Evaluation and Comparison

FIRE360 includes comprehensive evaluation tools:

- **Explanation Quality**: Fidelity, robustness, stability
- **Comparison with Baselines**: LIME, SHAP, LORE (genetic)

Scripts for evaluation are in `src/experiments/evaluate_explanations/`.

## Visualization Dashboard

A dashboard for visualizing explanations is available at: [FIRE360 Dashboard](https://heroic-dasik-a43ca2.netlify.app/).

Code for the dashboard can be found in the `UI` folder.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

Paper will be available soon.