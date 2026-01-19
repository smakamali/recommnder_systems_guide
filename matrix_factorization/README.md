# Matrix Factorization for Recommender Systems

This directory contains implementations of matrix factorization methods for recommendation systems using the MovieLens 100K dataset. The implementations follow the comprehensive guide in the parent directory's README.md.

## Overview

Matrix factorization decomposes the user-item interaction matrix into lower-dimensional user and item representations, enabling the prediction of missing ratings and generation of recommendations.

### Core Concept

The rating matrix $R \in \mathbb{R}^{m \times n}$ is approximated as:

$$R \approx P^T Q$$

where:
- $P \in \mathbb{R}^{k \times m}$ represents user factors
- $Q \in \mathbb{R}^{k \times n}$ represents item factors
- $k \ll \min(m,n)$ is the number of latent factors

**Prediction**: $\hat{r}_{ui} = p_u^T q_i$

For detailed theory, see: [`../README.md`](../README.md) Section 2.3 - Matrix Factorization (lines 330-419)

## Algorithm Approaches

This implementation includes two optimization approaches with three implementations:

### 1. Alternating Least Squares (ALS)

**Two implementations provided:**
- **ALS (implicit library)**: Production-ready implementation using the `implicit` library (`mf_als.py`)
- **ALS (from scratch)**: Educational implementation using only NumPy, following the guide's pseudocode exactly (`mf_als_from_scratch.py`)

**Loss Function**:
$$\mathcal{L} = \sum_{(u,i) \in \mathcal{O}} (r_{ui} - p_u^T q_i)^2 + \lambda(||p_u||^2 + ||q_i||^2)$$

ALS alternates between:
1. Fixing $Q$, optimizing $P$ (solving for each user)
2. Fixing $P$, optimizing $Q$ (solving for each item)

**Advantages**:
- Parallelizable (can update users/items independently)
- Fast convergence
- No learning rate to tune

**Reference**: Guide lines 368-399

### 2. Stochastic Gradient Descent (SGD)

SGD optimizes the same loss function using gradient descent:
- Updates parameters for each training example
- Requires learning rate tuning
- More memory-efficient for very large datasets

## Installation

### Prerequisites

- Python 3.8 or higher
- conda (recommended) or pip package manager

### Setup

#### Option 1: Using Conda (Recommended - avoids C++ build issues)

1. Navigate to this directory:
```bash
cd matrix_factorization
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate recom_sys
```

This will install all dependencies, including `scikit-surprise` via conda (which avoids C++ compilation requirements on Windows).

#### Option 2: Using pip (Manual installation)

1. Navigate to this directory:
```bash
cd matrix_factorization
```

2. Install `scikit-surprise` via conda first (to avoid C++ build tools requirement):
```bash
conda install -c conda-forge scikit-surprise
```

3. Install remaining dependencies via pip:
```bash
pip install implicit numpy pandas matplotlib scikit-learn scipy
```

**Note**: On Windows, `scikit-surprise` requires Microsoft Visual C++ 14.0 or greater when installing via pip. Using conda avoids this requirement.

### Required Packages

- `scikit-surprise` - Matrix factorization library (install via conda)
- `implicit` - Implicit feedback recommendation algorithms
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `matplotlib` - Visualization
- `scikit-learn` - Machine learning utilities
- `scipy` - Scientific computing library

## Dataset

This implementation uses the **MovieLens 100K dataset**, which contains:
- 100,000 ratings from 943 users on 1,682 movies
- Rating scale: 1-5 (integer ratings)
- Sparsity: ~93.7% (very sparse)

The dataset is automatically downloaded on first use via the Surprise library.

## Project Structure

```
matrix_factorization/
├── README.md                   # This file
├── environment.yml             # Conda environment file (recommended)
├── data_loader.py             # Load and preprocess MovieLens 100K
├── mf_als.py                  # ALS implementation (using implicit library)
├── mf_sgd.py                  # SGD implementation
├── mf_als_from_scratch.py     # Educational ALS from scratch (NumPy)
├── evaluation.py              # Evaluation metrics (RMSE, MAE, etc.)
├── recommend.py               # Generate top-N recommendations
├── main.py                    # Complete pipeline script
└── results/                   # Output directory for results
```

## Usage

### Quick Start: Complete Pipeline

Run the complete pipeline that trains three models, evaluates them, and generates recommendations:

```bash
python main.py
```

This will:
1. Load MovieLens 100K dataset
2. Split into train/test (80/20)
3. Train ALS model (using implicit library)
4. Train ALS from scratch model (educational NumPy implementation)
5. Train SGD model
6. Evaluate all three models (RMSE, MAE, Precision@K, Recall@K, NDCG@K, Hit Rate@K)
7. Generate sample recommendations
8. Save results to `results/`

### Individual Components

#### Train ALS Model

```bash
python mf_als.py
```

Or use in code:
```python
from data_loader import load_movielens_100k, get_train_test_split
from mf_als import train_als_model

data = load_movielens_100k()
trainset, testset = get_train_test_split(data)

model = train_als_model(trainset, n_factors=50, reg=0.1, n_iter=50)
prediction = model.predict(user_id="123", item_id="456")
```

#### Train SGD Model

```bash
python mf_sgd.py
```

Or use in code:
```python
from mf_sgd import train_sgd_model

model = train_sgd_model(trainset, n_factors=50, n_epochs=20)
predictions = model.test(testset)
```

#### Educational ALS from Scratch

See the implementation that directly follows the guide's pseudocode:

```bash
python mf_als_from_scratch.py
```

This implements the exact algorithm from guide lines 370-399 using only NumPy.

#### Evaluate Models

```python
from evaluation import evaluate_model

predictions = model.test(testset)
results = evaluate_model(predictions, k=10, threshold=4.0)
# Results include: RMSE, MAE, Precision@K, Recall@K, NDCG@K, Hit Rate@K
```

#### Generate Recommendations

```python
from recommend import generate_top_n_recommendations, print_recommendations

recommendations = generate_top_n_recommendations(
    model, trainset, user_id="123", n=10, exclude_rated=True
)
print_recommendations(user_id="123", recommendations)
```

## Evaluation Metrics

This implementation includes metrics from the guide's Section 1.3:

### Rating Prediction Metrics

- **RMSE** (Root Mean Square Error): $\sqrt{\frac{1}{N}\sum(r_{ui} - \hat{r}_{ui})^2}$
- **MAE** (Mean Absolute Error): $\frac{1}{N}\sum|r_{ui} - \hat{r}_{ui}|$

Lower values indicate better performance.

### Ranking Metrics

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain (considers ranking position)
- **Hit Rate@K**: Percentage of users with at least one relevant item in top-K

See: [`../README.md`](../README.md) Section 1.3 - Evaluation Metrics (lines 104-160)

## Hyperparameters

### Default Parameters

**ALS Model** (`mf_als.py`):
- `n_factors=50`: Number of latent factors $k$
- `reg=0.1`: Regularization parameter $\lambda$
- `n_iter=50`: Number of ALS iterations

**SGD Model** (`mf_sgd.py`):
- `n_factors=50`: Number of latent factors $k$
- `n_epochs=20`: Number of training epochs
- `lr_all=0.005`: Learning rate
- `reg_all=0.02`: Regularization parameter

### Tuning Tips

- **Increase `n_factors`**: Captures more complex patterns, but may overfit
- **Increase `reg` (regularization)**: Reduces overfitting, but may underfit
- **For ALS**: More iterations usually improve results (up to convergence)
- **For SGD**: More epochs help, but watch for overfitting

## Results Interpretation

After running `main.py`, check `results/matrix_factorization_results.txt` for:
- RMSE and MAE (lower is better)
- Precision@10, Recall@10, NDCG@10, and Hit Rate@10 (higher is better)
- Comparison between all three models: ALS (implicit), ALS (from scratch), and SGD

**Typical Results** (MovieLens 100K):
- RMSE: ~0.92-0.96
- MAE: ~0.72-0.75
- Precision@10: ~0.35-0.40 (threshold=4.0)
- Recall@10: ~0.10-0.15 (threshold=4.0)

*Note: Results vary with random seed and hyperparameters*

## Key Files Explained

- **`data_loader.py`**: Handles MovieLens dataset loading and train/test splitting
- **`mf_als.py`**: ALS implementation using implicit library (efficient production-ready implementation)
- **`mf_sgd.py`**: SGD implementation using Surprise's SVD class
- **`mf_als_from_scratch.py`**: Educational implementation following guide pseudocode exactly (NumPy only)
- **`evaluation.py`**: Implements RMSE, MAE, Precision@K, Recall@K, NDCG@K, Hit Rate@K
- **`recommend.py`**: Generates top-N recommendations with cold start handling
- **`main.py`**: Orchestrates the complete pipeline (trains all three models)

## Cold Start Handling

The `recommend.py` module handles cold start scenarios:

- **New users**: Recommends most popular items (by average rating)
- **New items**: Uses global mean or model's default prediction

For production systems, consider:
- Content-based features
- Demographic-based recommendations
- Hybrid methods

See: [`../README.md`](../README.md) Section 5.1 - Cold Start Problem (lines 1252-1316)

## Comparison: ALS vs SGD

| Aspect | ALS (implicit) | ALS (from scratch) | SGD |
|--------|----------------|-------------------|-----|
| **Implementation** | Production library | Educational NumPy | Surprise SVD |
| **Convergence** | Fast (few iterations) | Fast (few iterations) | Slower (more epochs needed) |
| **Parallelization** | Excellent (independent updates) | Good (independent updates) | Limited (sequential gradient updates) |
| **Memory** | Higher (stores full matrices) | Higher (stores full matrices) | Lower (updates per example) |
| **Hyperparameters** | Fewer (no learning rate) | Fewer (no learning rate) | More (learning rate tuning) |
| **Best For** | Medium datasets, need speed | Learning/understanding | Large datasets, limited memory |

All three implementations achieve similar accuracy on MovieLens 100K. The ALS from scratch implementation is primarily for educational purposes to understand the algorithm.

## Next Steps

1. **Experiment with hyperparameters**: Try different values of `n_factors`, `reg`, etc.
2. **Try larger datasets**: MovieLens 1M, 10M, or 25M
3. **Compare with other methods**: Collaborative filtering, content-based, neural networks
4. **Explore extensions**: SVD++, TimeSVD++, Factorization Machines

## References

- **Guide Section 2.3**: Matrix Factorization (lines 330-419)
- **Guide Section 1.3**: Evaluation Metrics (lines 104-160)
- **Guide Section 5.1**: Cold Start Problem (lines 1252-1316)

Main guide: [`../README.md`](../README.md)

## Troubleshooting

### Import Errors

Ensure all dependencies are installed. Use conda for easiest setup:
```bash
conda env create -f environment.yml
conda activate recom_sys
```

Or manually:
```bash
conda install -c conda-forge scikit-surprise
pip install implicit numpy pandas matplotlib scikit-learn scipy
```

### Dataset Download Issues

The Surprise library downloads MovieLens 100K automatically. If it fails:
- Check internet connection
- The dataset may already be cached in your home directory under `.surprise_data/`

### Memory Issues

If running out of memory:
- Reduce `n_factors` (e.g., from 50 to 20)
- Use SGD instead of ALS
- Process in smaller batches

### Poor Results

- Try different random seeds
- Tune hyperparameters (especially regularization)
- Ensure train/test split is appropriate
- Check for data preprocessing issues

## License

See parent directory LICENSE file.

---

**Note**: This is a learning implementation following the comprehensive guide. For production systems, consider additional optimizations, distributed training, and real-time serving infrastructure.

