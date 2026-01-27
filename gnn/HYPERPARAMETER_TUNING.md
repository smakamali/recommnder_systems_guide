# GraphSAGE Hyperparameter Tuning Guide

This guide explains how to use `param_tuning.py` for general-purpose hyperparameter tuning of GraphSAGE recommender systems.

## Overview

The parallel tuning script allows you to efficiently explore different hyperparameters including:
- **Hidden layer dimensions** (e.g., 32, 64, 128, 256)
- **Number of layers** (e.g., 1, 2, 3, 4)
- **Loss functions** (MSE, BPR, Combined)
- **Loss weights** (MSE weight, BPR weight)
- **Learning rates**
- **Batch sizes**
- **Number of epochs**

The script runs multiple configurations in parallel on GPU(s) or CPU, significantly reducing experiment time.

## Quick Start

### 1. Run Pre-defined Tuning Modes

```bash
# Run example configurations (diverse set)
python gnn/param_tuning.py --mode example

# Architecture search (hidden_dim x num_layers)
python gnn/param_tuning.py --mode architecture

# Loss function comparison
python gnn/param_tuning.py --mode loss

# Full grid search
python gnn/param_tuning.py --mode grid

# With custom memory settings for larger models
python gnn/param_tuning.py --mode architecture --model-memory 2.5 --gpu-buffer 3.0

# With automatic memory estimation
python gnn/param_tuning.py --mode architecture --auto-memory
```

**Command-line Options:**
- `--mode {example,architecture,loss,grid}` - Tuning mode (default: example)
- `--model-memory GB` - GPU memory per model in GB (default: 1.5)
- `--gpu-buffer GB` - GPU memory buffer in GB (default: 2.0)
- `--auto-memory` - Auto-estimate memory from configurations (recommended)

### 2. Custom Configurations in Python

```python
from gnn.param_tuning import tune_hyperparameters_parallel
from common.data_loader import load_movielens_100k, get_train_test_split, load_user_features, load_item_features

# Load data
data = load_movielens_100k()
trainset, testset = get_train_test_split(data, test_size=0.2)
user_features = load_user_features()
item_features = load_item_features()

# Define custom configurations
configurations = [
    {
        'name': 'Baseline',
        'hidden_dim': 64,
        'num_layers': 2,
        'loss_type': 'mse',
        'learning_rate': 0.001,
        'batch_size': 512,
        'num_epochs': 20
    },
    {
        'name': 'Deeper',
        'hidden_dim': 64,
        'num_layers': 4,
        'loss_type': 'mse',
        'learning_rate': 0.001,
        'batch_size': 512,
        'num_epochs': 20
    },
    {
        'name': 'Wider',
        'hidden_dim': 256,
        'num_layers': 2,
        'loss_type': 'mse',
        'learning_rate': 0.001,
        'batch_size': 512,
        'num_epochs': 20
    }
]

# Run tuning
results = tune_hyperparameters_parallel(
    trainset, testset, user_features, item_features,
    configurations=configurations,
    num_gpus=1,  # Auto-detected if None
    verbose=True
)
```

## Helper Functions

### 1. Grid Search

Create all combinations of parameters:

```python
from gnn.param_tuning import create_grid_search_configs

configurations = create_grid_search_configs(
    param_grid={
        'hidden_dim': [64, 128, 256],
        'num_layers': [2, 3, 4],
        'learning_rate': [0.001, 0.0001]
    },
    base_config={
        'loss_type': 'mse',
        'batch_size': 512,
        'num_epochs': 20
    }
)

# This creates 3 * 3 * 2 = 18 configurations
```

### 2. Architecture Search

Search over hidden dimensions and layer counts:

```python
from gnn.param_tuning import create_architecture_search_configs

configurations = create_architecture_search_configs(
    hidden_dims=[32, 64, 128, 256],
    num_layers_list=[1, 2, 3, 4],
    base_config={'loss_type': 'mse', 'num_epochs': 20}
)

# This creates 4 * 4 = 16 configurations
```

### 3. Loss Function Comparison

Compare different loss function weights:

```python
from gnn.param_tuning import create_loss_function_configs

configurations = create_loss_function_configs(
    bpr_weights=[0.0, 0.05, 0.1, 0.15, 0.2],  # 0.0 = MSE only
    base_config={'hidden_dim': 64, 'num_layers': 2}
)

# This creates 5 configurations with different BPR weights
```

## Configuration Parameters

Each configuration dictionary can include:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | **Required** | Configuration name for identification |
| `hidden_dim` | int | 64 | Hidden layer dimension |
| `num_layers` | int | 2 | Number of GraphSAGE layers |
| `loss_type` | str | 'mse' | Loss function: 'mse', 'bpr', or 'combined' |
| `mse_weight` | float | 1.0 | Weight for MSE loss (if combined) |
| `bpr_weight` | float | 0.0 | Weight for BPR loss (if combined) |
| `learning_rate` | float | 0.001 | Learning rate for optimizer |
| `batch_size` | int | 512 | Training batch size |
| `num_epochs` | int | 20 | Number of training epochs |

## Output

### Console Output

The script provides detailed progress information:
- Configuration details
- Training progress for each configuration
- Evaluation metrics (RMSE, MAE, NDCG@10, Precision@10, etc.)
- Best configurations by different metrics

### Saved Results

Results are automatically saved to:
```
gnn/tuning/tuning_results_{mode}_{timestamp}.json
```

Load saved results:
```python
from gnn.param_tuning import load_results

results = load_results('gnn/tuning/tuning_results_architecture_20260126_143022.json')
```

## Results Summary

The output includes:

1. **Configuration Details Table**
   - All hyperparameters for each configuration

2. **Rating Prediction Metrics**
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)

3. **Ranking Metrics (K=10)**
   - Precision@10
   - Recall@10
   - NDCG@10
   - Hit Rate@10

4. **Cold Start Metrics** (if applicable)
   - RMSE for cold start items
   - MAE for cold start items
   - Coverage

5. **Best Configurations**
   - Best RMSE configuration
   - Best NDCG@10 configuration
   - Best Precision@10 configuration

## Performance Tips

### GPU Memory Management

For single GPU with limited memory:
- The script automatically detects available GPU memory
- Uses ThreadPoolExecutor with CUDA streams for efficient parallel execution
- Automatically calculates max parallel workers based on memory requirements

#### Option 1: Limit Workers Manually
```python
results = tune_hyperparameters_parallel(
    trainset, testset, user_features, item_features,
    configurations=configurations,
    max_workers=4,  # Limit to 4 parallel workers
    num_gpus=1
)
```

#### Option 2: Auto-Estimate Memory (Recommended)
Let the script automatically calculate memory from your configurations:

```python
from gnn.param_tuning import tune_hyperparameters_parallel, auto_estimate_memory_from_configs

# Define your configurations (varying sizes)
configurations = [
    {'name': 'Small', 'hidden_dim': 64, 'num_layers': 2},
    {'name': 'Large', 'hidden_dim': 256, 'num_layers': 4},
]

# Auto-estimate memory from largest config
memory_gb = auto_estimate_memory_from_configs(
    configurations,
    num_users=943,
    num_items=1682
)
print(f"Auto-estimated memory: {memory_gb:.1f}GB per model")

results = tune_hyperparameters_parallel(
    trainset, testset, user_features, item_features,
    configurations=configurations,
    model_memory_gb=memory_gb,  # Auto-calculated
    num_gpus=1
)
```

Or use the command-line flag:
```bash
python gnn/param_tuning.py --mode architecture --auto-memory
```

#### Option 3: Manual Memory Estimate
If you encounter OOM errors or want to tune larger models:

```python
# For larger models (e.g., hidden_dim=256, num_layers=4)
from gnn.param_tuning import estimate_model_memory

# Estimate memory for your model
memory_per_model = estimate_model_memory(
    hidden_dim=256, 
    num_layers=4,
    num_users=943,
    num_items=1682
)
print(f"Estimated memory: {memory_per_model:.1f}GB per model")

results = tune_hyperparameters_parallel(
    trainset, testset, user_features, item_features,
    configurations=configurations,
    model_memory_gb=memory_per_model,  # Use estimated value
    gpu_memory_buffer_gb=3.0,  # Increase buffer if needed
    num_gpus=1
)
```

**Memory Parameters:**
- `model_memory_gb` (default: 1.5): Estimated GPU memory per model in GB
  - Increase for larger models (higher `hidden_dim`, more `num_layers`)
  - Decrease for smaller models to fit more parallel workers
- `gpu_memory_buffer_gb` (default: 2.0): Memory buffer for PyTorch overhead
  - Increase if you see OOM errors during initialization
  - Decrease to allow more parallel workers (risky)

### Multi-GPU Support

For multiple GPUs:
- Configurations are automatically distributed across GPUs in round-robin fashion
- Each GPU runs configurations independently

### CPU-Only Mode

For CPU-only systems:
- The script automatically detects no GPU and uses CPU
- Uses half of available CPU cores by default

## Examples

### Example 1: Find Optimal Architecture

```python
# Test different architectures
configurations = []

for hidden_dim in [32, 64, 128, 256]:
    for num_layers in [2, 3, 4]:
        configurations.append({
            'name': f'h{hidden_dim}-l{num_layers}',
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'loss_type': 'mse',
            'num_epochs': 30
        })

results = tune_hyperparameters_parallel(
    trainset, testset, user_features, item_features,
    configurations=configurations
)
```

### Example 2: Learning Rate Search

```python
configurations = []

for lr in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
    configurations.append({
        'name': f'lr-{lr}',
        'hidden_dim': 64,
        'num_layers': 2,
        'learning_rate': lr,
        'num_epochs': 20
    })

results = tune_hyperparameters_parallel(
    trainset, testset, user_features, item_features,
    configurations=configurations
)
```

### Example 3: Combined Loss Weight Tuning

```python
configurations = []

# Test MSE only
configurations.append({
    'name': 'MSE-Only',
    'hidden_dim': 64,
    'num_layers': 2,
    'loss_type': 'mse'
})

# Test different BPR weights
for bpr_weight in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    configurations.append({
        'name': f'BPR-{bpr_weight}',
        'hidden_dim': 64,
        'num_layers': 2,
        'loss_type': 'combined',
        'mse_weight': 1.0,
        'bpr_weight': bpr_weight
    })

results = tune_hyperparameters_parallel(
    trainset, testset, user_features, item_features,
    configurations=configurations
)
```

## Interpreting Results

### For Recommendation Systems

- **High RMSE, High NDCG**: Good for ranking, less accurate ratings
- **Low RMSE, Low NDCG**: Accurate ratings, poor ranking
- **Balance**: Look for configurations with good NDCG and acceptable RMSE

### Architecture Insights

- **Deeper models** (more layers): Better feature propagation, risk of over-smoothing
- **Wider models** (larger hidden_dim): More capacity, risk of overfitting
- **Combined loss**: Better ranking at the cost of slightly higher RMSE

### Loss Function Trade-offs

- **MSE only**: Best for rating prediction accuracy
- **Combined (low BPR weight)**: Improved ranking with minimal rating loss
- **Combined (high BPR weight)**: Best ranking, but higher rating errors

## Best Practices

1. **Start small**: Test a few configurations first to estimate runtime
2. **Use grid search**: For systematic exploration of parameter space
3. **Monitor GPU memory**: Adjust `max_workers` if needed
4. **Save results**: Results are automatically saved for later analysis
5. **Compare metrics**: Consider multiple metrics (RMSE, NDCG, Precision)
6. **Cold start**: Include cold start metrics if applicable to your use case

## Troubleshooting

### Out of Memory (OOM)

```python
# Option 1: Reduce parallel workers
results = tune_hyperparameters_parallel(..., max_workers=2)

# Option 2: Increase memory estimate per model
results = tune_hyperparameters_parallel(..., model_memory_gb=2.5)

# Option 3: Increase GPU memory buffer
results = tune_hyperparameters_parallel(..., gpu_memory_buffer_gb=3.0)

# Option 4: Reduce batch size in configurations
config['batch_size'] = 256  # instead of 512

# Option 5: Use the memory estimator
from gnn.param_tuning import estimate_model_memory
mem = estimate_model_memory(hidden_dim=128, num_layers=3, 
                            num_users=943, num_items=1682)
results = tune_hyperparameters_parallel(..., model_memory_gb=mem)
```

### Slow Training

```python
# Reduce number of epochs for faster experimentation
config['num_epochs'] = 10  # instead of 20

# Or reduce number of configurations
configurations = configurations[:5]  # Test first 5 only
```

### No GPU Detected

The script automatically falls back to CPU. To force GPU:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

## References

- [GraphSAGE Paper](https://arxiv.org/abs/1706.02216)
- [BPR: Bayesian Personalized Ranking](https://arxiv.org/abs/1205.2618)
- Main training script: `gnn/train_graphsage.py`
- Model implementation: `gnn/graphsage_model.py`

---

## Experimental Results Analysis

### Large-Scale Architecture Search Results

Based on comprehensive tuning experiments testing 30 different configurations (hidden dimensions: 16, 32, 64, 128, 256, 512; layers: 1-5), the following insights were discovered:

### Top Performers

#### 1. Best Overall Configuration: `h64-l3` (hidden_dim=64, num_layers=3)

**Performance Metrics:**
- **NDCG@10**: 0.8442 (highest)
- **RMSE**: 0.968 (4th best)
- **MAE**: 0.787 (4th best)
- **Precision@10**: 0.678 (highest)
- **Recall@10**: 0.735 (highest)
- **Cold Item RMSE**: 1.021 (2nd best)
- **Cold Item MAE**: 0.829 (2nd best)

**Configuration:**
```python
{
    'hidden_dim': 64,
    'num_layers': 3,
    'loss_type': 'mse',
    'learning_rate': 0.001,
    'batch_size': 512,
    'num_epochs': 20
}
```

**Rationale:**
- Highest ranking quality (NDCG@10)
- Strong cold item generalization
- Best precision and recall
- Efficient model size (good memory/compute balance)
- Balanced across all metrics

#### 2. Close Second: `h128-l5` (hidden_dim=128, num_layers=5)

**Performance Metrics:**
- **NDCG@10**: 0.8442 (2nd, very close)
- **RMSE**: 1.384
- **MAE**: 1.189
- **Precision@10**: 0.677 (2nd)
- **Recall@10**: 0.735 (tied for highest)
- **Cold Item RMSE**: 1.193
- **Cold Item MAE**: 0.927

**Note:** While achieving similar NDCG, this configuration has higher RMSE/MAE and is less efficient due to more layers.

#### 3. Best Accuracy: `h256-l1` (hidden_dim=256, num_layers=1)

**Performance Metrics:**
- **NDCG@10**: 0.8370
- **RMSE**: 0.954 (best)
- **MAE**: 0.752 (best)
- **Precision@10**: 0.672
- **Recall@10**: 0.731
- **Cold Item RMSE**: 1.020 (best)
- **Cold Item MAE**: 0.817 (best)

**Note:** Best for rating prediction accuracy and cold item generalization, but slightly lower ranking quality than `h64-l3`.

### Key Findings

#### 1. Architecture Insights

**Optimal Range:**
- **Hidden dimensions**: 64-128 provide the best balance
- **Layer depth**: 3 layers is optimal for 64-128 hidden dims
- **Wide and shallow**: 256 hidden dim with 1 layer also performs well

**Poor Performers:**
- Large models (256+ hidden dim, 4+ layers) show poor performance (RMSE ~2.76, NDCG ~0.697)
- Very small models (16 dim) or very large models (512 dim) with many layers underperform
- Configurations with identical poor results suggest convergence issues or insufficient capacity

#### 2. Layer Depth Analysis

- **3 layers**: Optimal for moderate hidden dimensions (64-128)
- **1 layer**: Works well with larger hidden dimensions (256, 512) - "wide and shallow" approach
- **5 layers**: Can work with 128 hidden dim but less efficient
- **4+ layers**: Generally poor performance, especially with large hidden dims (256+)

#### 3. Hidden Dimension Analysis

- **64-128**: Best overall balance of performance and efficiency
- **256**: Works well with 1 layer (wide, shallow architecture)
- **512**: Only works with 1 layer, otherwise poor performance
- **16**: Too small, consistently underperforms
- **32**: Acceptable but not optimal

#### 4. Cold Item Generalization

- **Best**: `h256-l1` (RMSE: 1.020, MAE: 0.817)
- **Second**: `h64-l3` (RMSE: 1.021, MAE: 0.829)
- Both configurations show strong generalization to unseen items

#### 5. Ranking Quality (NDCG@10)

Top configurations:
1. `h64-l3`: 0.8442
2. `h128-l5`: 0.8442 (very close)
3. `h32-l3`: 0.8379
4. `h256-l1`: 0.8370
5. `h512-l1`: 0.8343

### Summary Statistics

- **Total configurations tested**: 30
- **Successful runs**: 30 (100% success rate)
- **Best NDCG@10**: 0.8442 (`h64-l3`)
- **Best RMSE**: 0.954 (`h256-l1`)
- **Best cold item RMSE**: 1.020 (`h256-l1`)
- **Worst performers**: Large models (256+ dim, 4+ layers) with RMSE ~2.76

### Recommendations

#### For Production Use

**Primary Recommendation: `h64-l3`**
- Best overall ranking quality
- Strong cold item generalization
- Efficient model size
- Balanced across all metrics

**Alternative: `h256-l1`** (if accuracy is priority)
- Best RMSE/MAE and cold item performance
- Slightly lower ranking quality (NDCG: 0.837 vs 0.844)
- Larger model (more memory)

#### Architecture Selection Guidelines

1. **For ranking quality**: Use 64-128 hidden dim with 3 layers
2. **For accuracy**: Use 256 hidden dim with 1 layer
3. **Avoid**: 256+ hidden dim with 4+ layers (poor convergence)
4. **Efficiency**: Smaller models (64 dim, 3 layers) provide best performance/efficiency trade-off

### Experimental Data

Results from experiments conducted on January 27, 2026:
- **Experiment 1**: 12 configurations (hidden_dim: 32-256, layers: 2-4)
- **Experiment 2**: 30 configurations (hidden_dim: 16-512, layers: 1-5)

Results files:
- `gnn/tuning/tuning_results_architecture_20260127_035134.json`
- `gnn/tuning/tuning_results_architecture_20260127_041554.json`
