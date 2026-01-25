# Recommender Systems Guide

A comprehensive guide and implementation repository for state-of-the-art recommender systems, featuring matrix factorization methods with practical Python implementations.

## Overview

This repository provides both theoretical foundations and practical implementations of recommender systems, with a focus on matrix factorization techniques. The codebase includes multiple matrix factorization algorithms implemented from scratch and using established libraries, along with comprehensive evaluation metrics and recommendation generation capabilities.

### Key Features

- **Multiple Matrix Factorization Implementations**:
  - ALS (Alternating Least Squares) using the `implicit` library
  - ALS implementation from scratch using NumPy (educational)
  - SVD (Singular Value Decomposition) using `scikit-surprise`
  - Factorization Machines (FM) with user and item features

- **Comprehensive Evaluation**:
  - Rating prediction metrics (RMSE, MAE)
  - Ranking metrics (Precision@K, Recall@K, NDCG@K, Hit Rate@K)
  - Cold start evaluation capabilities

- **Production-Ready Pipeline**:
  - Complete end-to-end workflow from data loading to recommendations
  - Automatic dataset download and preprocessing
  - Results saving and comparison

- **Well-Tested**:
  - Comprehensive test suite with pytest
  - Integration tests for the complete pipeline

## Project Structure

```
recommnder_systems_guide/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── recom_sys_guide.md          # Comprehensive theoretical guide
│
└── matrix_factorization/       # Matrix factorization implementations
    ├── README.md                # Detailed documentation for MF module
    ├── environment.yml         # Conda environment file (recommended)
    ├── main.py                 # Main pipeline script
    │
    ├── data_loader.py          # MovieLens 100K data loading and preprocessing
    ├── mf_als.py              # ALS using implicit library
    ├── mf_als_from_scratch.py # Educational ALS implementation
    ├── mf_svd.py              # SVD implementation
    ├── mf_fm.py               # Factorization Machines
    ├── evaluation.py          # Evaluation metrics
    ├── recommend.py           # Recommendation generation
    │
    ├── data/                  # Data directory
    ├── results/               # Output directory for results
    │
    └── tests/                 # Test suite
        ├── conftest.py
        ├── test_data_loader.py
        ├── test_evaluation.py
        ├── test_integration.py
        ├── test_mf_als.py
        ├── test_mf_als_from_scratch.py
        ├── test_mf_svd.py
        ├── test_mf_fm.py
        └── test_recommend.py
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- conda (recommended) or pip package manager

### Installation

#### Option 1: Using Conda (Recommended)

The easiest way to set up the environment is using conda, which avoids C++ compilation issues on Windows:

```bash
# Navigate to the matrix_factorization directory
cd matrix_factorization

# Create and activate the conda environment
conda env create -f environment.yml
conda activate recom_sys

# You're ready to run the code!
```

#### Option 2: Using pip (Manual Installation)

If you prefer pip, install dependencies manually:

```bash
# First install scikit-surprise via conda (avoids C++ build requirements on Windows)
conda install -c conda-forge scikit-surprise

# Then install remaining dependencies via pip
pip install implicit numpy pandas matplotlib scikit-learn scipy myfm
```

**Note**: On Windows, `scikit-surprise` requires Microsoft Visual C++ 14.0 or greater when installing via pip. Using conda avoids this requirement.

### Running the Code

#### Complete Pipeline

Run the main pipeline script to train all models, evaluate them, and generate recommendations:

```bash
cd matrix_factorization
python main.py
```

This will:
1. Download the MovieLens 100K dataset (if not already present)
2. Load user and item features
3. Split data into train/test sets (80/20)
4. Train four models:
   - ALS (implicit library)
   - ALS (from scratch)
   - SVD
   - Factorization Machine (with features)
5. Evaluate all models with comprehensive metrics
6. Generate sample recommendations
7. Save results to `results/matrix_factorization_results.txt`

#### Individual Components

You can also run individual modules for specific tasks:

**Train ALS Model:**
```bash
python mf_als.py
```

**Train SVD Model:**
```bash
python mf_svd.py
```

**Train Factorization Machine:**
```bash
python mf_fm.py
```

**Educational ALS from Scratch:**
```bash
python mf_als_from_scratch.py
```

### Running Tests

To run the test suite:

```bash
cd matrix_factorization

# Install test dependencies
pip install -r tests/requirements-test.txt

# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=. --cov-report=html
```

## Dataset

This implementation uses the **MovieLens 100K dataset**, which contains:
- 100,000 ratings from 943 users on 1,682 movies
- Rating scale: 1-5 (integer ratings)
- Sparsity: ~93.7% (very sparse)
- User features: age, gender, occupation, zip code
- Item features: genres (19 categories), release year

The dataset is automatically downloaded on first use via the Surprise library and cached in your home directory under `.surprise_data/ml-100k/`.

## Usage Examples

### Basic Usage

```python
from data_loader import load_movielens_100k, get_train_test_split
from mf_als import train_als_model
from evaluation import evaluate_model

# Load data
data = load_movielens_100k()
trainset, testset = get_train_test_split(data, test_size=0.2)

# Train model
model = train_als_model(trainset, n_factors=50, reg=0.1, n_iter=50)

# Make predictions
predictions = []
for uid, iid, true_r in testset:
    pred_r = model.predict(uid, iid)
    predictions.append((uid, iid, true_r, pred_r))

# Evaluate
results = evaluate_model(predictions, k=10, threshold=4.0)
print(f"RMSE: {results['rmse']:.4f}")
print(f"Precision@10: {results['precision@10']:.4f}")
```

### Factorization Machines with Features

```python
from data_loader import load_user_features, load_item_features
from mf_fm import train_fm_model

# Load features
user_features = load_user_features()
item_features = load_item_features()

# Train FM model (handles cold start!)
fm_model = train_fm_model(
    trainset, user_features, item_features,
    n_factors=50, learning_rate=0.1, reg_lambda=0.01, n_epochs=30
)

# Can predict even for new users/items with features
prediction = fm_model.predict(new_user_id, item_id)
```

### Generating Recommendations

```python
from recommend import generate_top_n_recommendations, print_recommendations

# Generate top-10 recommendations for a user
recommendations = generate_top_n_recommendations(
    model, trainset, user_id="123", n=10, exclude_rated=True
)

# Print recommendations
print_recommendations(user_id="123", recommendations, max_display=10)
```

## Documentation

- **Comprehensive Guide**: See [`recom_sys_guide.md`](recom_sys_guide.md) for detailed theoretical foundations covering:
  - Classical approaches (Collaborative Filtering, Content-Based, Matrix Factorization)
  - Deep learning methods (Neural CF, Autoencoders, RNNs)
  - Modern state-of-the-art (Transformers, GNNs, LLMs, Reinforcement Learning)
  - Advanced topics (Cold Start, Diversity, Fairness, Multi-Objective Optimization)
  - Practical implementation considerations

- **Matrix Factorization Module**: See [`matrix_factorization/README.md`](matrix_factorization/README.md) for:
  - Detailed algorithm explanations
  - Hyperparameter tuning guides
  - Feature engineering documentation
  - Model comparison tables

## Key Algorithms Implemented

### 1. Alternating Least Squares (ALS)

Two implementations:
- **Production**: Using `implicit` library (`mf_als.py`)
- **Educational**: From scratch with NumPy (`mf_als_from_scratch.py`)

**Loss Function**:
$$\mathcal{L} = \sum_{(u,i) \in \mathcal{O}} (r_{ui} - p_u^T q_i)^2 + \lambda(||p_u||^2 + ||q_i||^2)$$

### 2. Singular Value Decomposition (SVD)

Using `scikit-surprise` library with bias terms:
$$\hat{r}_{ui} = \mu + b_u + b_i + p_u^T q_i$$

### 3. Factorization Machines (FM)

Generalizes matrix factorization to incorporate user and item features:
$$\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

**Advantages**: Handles cold start excellently by using features for new users/items.

## Evaluation Metrics

The implementation includes comprehensive evaluation metrics:

**Rating Prediction**:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)

**Ranking Metrics**:
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- Hit Rate@K

**Cold Start Metrics** (for FM):
- Cold User RMSE
- Cold Item RMSE
- Coverage

## Expected Results

Typical performance on MovieLens 100K:

**Standard Models (ALS, SVD)**:
- RMSE: ~0.92-0.96
- MAE: ~0.72-0.75
- Precision@10: ~0.35-0.40 (threshold=4.0)
- Recall@10: ~0.10-0.15 (threshold=4.0)

**Factorization Machines (with features)**:
- Overall RMSE: ~0.88-0.92 (5-8% improvement)
- Cold Start User RMSE: ~0.95-1.05
- Cold Start Item RMSE: ~0.90-1.00

*Note: Results vary with random seed and hyperparameters*

## Dependencies

Core dependencies (from `environment.yml`):
- `scikit-surprise>=1.1.3` - Matrix factorization library
- `implicit>=0.6.0` - Implicit feedback algorithms
- `numpy>=1.24.0` - Numerical computations
- `pandas>=2.0.0` - Data manipulation
- `scikit-learn>=1.3.0` - Machine learning utilities
- `scipy>=1.10.0` - Scientific computing
- `myfm` - Factorization Machines library

## Troubleshooting

### Import Errors

Ensure all dependencies are installed:
```bash
conda env create -f matrix_factorization/environment.yml
conda activate recom_sys
```

### Dataset Download Issues

The Surprise library downloads MovieLens 100K automatically. If it fails:
- Check internet connection
- Dataset may already be cached in `~/.surprise_data/ml-100k/`

### Memory Issues

If running out of memory:
- Reduce `n_factors` (e.g., from 50 to 20)
- Use SVD instead of ALS
- Process in smaller batches

### Windows C++ Build Errors

If you encounter C++ compilation errors when installing `scikit-surprise`:
- Use conda instead: `conda install -c conda-forge scikit-surprise`
- Or install Microsoft Visual C++ 14.0 or greater

## Contributing

This is an educational repository. Contributions, improvements, and bug fixes are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- **Comprehensive Guide**: See [`recom_sys_guide.md`](recom_sys_guide.md) for extensive references to papers, books, and resources
- **Matrix Factorization Guide**: See [`matrix_factorization/README.md`](matrix_factorization/README.md) for algorithm-specific references

## Next Steps

1. **Experiment with hyperparameters**: Try different values of `n_factors`, `reg`, etc.
2. **Try larger datasets**: MovieLens 1M, 10M, or 25M
3. **Explore extensions**: See `matrix_factorization/feature_extensions.md` for advanced feature engineering
4. **Read the comprehensive guide**: Deep dive into theory in `recom_sys_guide.md`

---

**Note**: This is a learning implementation following comprehensive theoretical foundations. For production systems, consider additional optimizations, distributed training, and real-time serving infrastructure.
