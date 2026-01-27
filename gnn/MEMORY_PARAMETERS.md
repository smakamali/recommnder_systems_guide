# GPU Memory Parameters Guide

This guide explains the parameterized memory management system for GraphSAGE hyperparameter tuning.

## Overview

The tuning script now allows you to customize memory requirements instead of using hard-coded values. This enables better control over parallel execution and supports tuning larger models.

## New Parameters

### `model_memory_gb` (default: 1.5)

Estimated GPU memory per model in GB. This value determines how many models can be trained in parallel.

**When to adjust:**
- **Increase** for larger models (higher `hidden_dim`, more `num_layers`)
- **Decrease** for smaller models to fit more parallel workers
- **Use estimator** for automatic calculation based on model configuration

**Example:**
```python
results = tune_hyperparameters_parallel(
    trainset, testset, user_features, item_features,
    configurations=configs,
    model_memory_gb=2.5  # Larger models need more memory
)
```

### `gpu_memory_buffer_gb` (default: 2.0)

GPU memory to keep free in GB. This buffer accounts for PyTorch overhead and system processes.

**When to adjust:**
- **Increase** if you see OOM errors during initialization
- **Decrease** to allow more parallel workers (risky, may cause instability)
- **Typical range**: 1.5 - 3.0 GB

**Example:**
```python
results = tune_hyperparameters_parallel(
    trainset, testset, user_features, item_features,
    configurations=configs,
    gpu_memory_buffer_gb=3.0  # More conservative buffer
)
```

## Memory Estimation Functions

### `auto_estimate_memory_from_configs()`

**NEW!** Automatically analyzes all configurations and estimates memory for the largest model.

**Function signature:**
```python
def auto_estimate_memory_from_configs(
    configurations: List[Dict[str, Any]],
    num_users: int,
    num_items: int
) -> float
```

**Command-line usage:**
```bash
# Automatically estimate memory from your configurations
python gnn/param_tuning.py --mode architecture --auto-memory
```

This is the **recommended approach** when tuning models of varying sizes, as it:
- Analyzes all configurations to find the largest model
- Automatically calculates appropriate memory requirements
- Ensures all models in the tuning run will fit in memory
- No manual calculation needed!

### `estimate_model_memory()`

Estimates memory requirements for a single model configuration.

**Function signature:**
```python
def estimate_model_memory(
    hidden_dim: int = 64,
    num_layers: int = 2,
    num_users: int = 1000,
    num_items: int = 1000
) -> float
```

**Returns:** Estimated memory in GB

**Example usage:**
```python
from gnn.param_tuning import estimate_model_memory

# Estimate for a specific configuration
memory_gb = estimate_model_memory(
    hidden_dim=256,
    num_layers=4,
    num_users=943,
    num_items=1682
)

print(f"Estimated memory: {memory_gb:.1f}GB")

# Use in tuning
results = tune_hyperparameters_parallel(
    trainset, testset, user_features, item_features,
    configurations=configs,
    model_memory_gb=memory_gb
)
```

## How It Works

### Automatic Worker Calculation

For single GPU, the number of parallel workers is calculated as:

```
max_workers = (gpu_total_memory - gpu_memory_buffer_gb) / model_memory_gb
```

**Example with 16GB GPU:**
- Total memory: 16GB
- Buffer: 2GB (default)
- Available: 14GB
- Model memory: 1.5GB (default)
- Max workers: `14 / 1.5 = 9` workers

### Memory Estimation Components

The `estimate_model_memory()` function considers:

1. **Embeddings**: User and item embeddings
2. **GNN Layers**: Weight matrices and activations
3. **Output Layer**: Final prediction layer
4. **Optimizer State**: Adam optimizer uses 2x model parameters
5. **PyTorch Overhead**: Intermediate activations and buffers

## Usage Examples

### Example 1: Default Behavior

```python
# Uses default: 1.5GB per model, 2GB buffer
results = tune_hyperparameters_parallel(
    trainset, testset, user_features, item_features,
    configurations=configs
)
```

### Example 2: Small Models (More Parallelism)

```python
# Smaller models can fit more in parallel
results = tune_hyperparameters_parallel(
    trainset, testset, user_features, item_features,
    configurations=configs,
    model_memory_gb=1.0,  # Small models
    gpu_memory_buffer_gb=1.5  # Minimal buffer
)
```

### Example 3: Large Models (Less Parallelism)

```python
# Larger models need more memory
memory_gb = estimate_model_memory(
    hidden_dim=256,
    num_layers=4,
    num_users=943,
    num_items=1682
)

results = tune_hyperparameters_parallel(
    trainset, testset, user_features, item_features,
    configurations=configs,
    model_memory_gb=memory_gb,  # ~3-4GB
    gpu_memory_buffer_gb=3.0  # Extra buffer
)
```

### Example 4: Tuning Multiple Model Sizes (Auto-Estimate - Recommended)

**NEW!** When tuning models of different sizes, use auto-estimation:

```python
from gnn.param_tuning import auto_estimate_memory_from_configs

configs = [
    {'name': 'Small', 'hidden_dim': 64, 'num_layers': 2},
    {'name': 'Medium', 'hidden_dim': 128, 'num_layers': 3},
    {'name': 'Large', 'hidden_dim': 256, 'num_layers': 4},  # Largest
]

# Automatically estimate from all configs
memory_gb = auto_estimate_memory_from_configs(
    configs,
    num_users=943,
    num_items=1682
)
# Returns memory estimate for the largest model (256 hidden, 4 layers)

results = tune_hyperparameters_parallel(
    trainset, testset, user_features, item_features,
    configurations=configs,
    model_memory_gb=memory_gb  # Auto-calculated
)
```

**Or from command line:**
```bash
python gnn/param_tuning.py --mode architecture --auto-memory
```

## Memory Estimation Tool

Run the interactive tool to estimate memory for your configurations:

```bash
# View estimates for common configurations
python gnn/example_memory_estimation.py

# Interactive estimation
python gnn/example_memory_estimation.py --interactive
```

**Example output:**
```
Configuration        Hidden Dim  Layers    Est. Memory
--------------------------------------------------------------------------------
Small                        32       2            1.2GB
Baseline                     64       2            1.5GB
Deeper                       64       3            1.7GB
Wide & Deep                 128       3            2.8GB
Large                       256       2            3.5GB
Very Large                  256       4            4.2GB
```

## Troubleshooting

### OOM Errors During Training

**Problem:** Out of memory errors during training

**Solutions:**
1. Increase memory estimate:
   ```python
   model_memory_gb=2.5  # Instead of 1.5
   ```

2. Manually limit workers:
   ```python
   max_workers=2  # Force fewer parallel workers
   ```

3. Reduce batch size in configs:
   ```python
   config['batch_size'] = 256  # Instead of 512
   ```

### OOM Errors During Initialization

**Problem:** Out of memory before training starts

**Solutions:**
1. Increase buffer:
   ```python
   gpu_memory_buffer_gb=3.0  # Instead of 2.0
   ```

2. Reduce workers:
   ```python
   max_workers=1  # Sequential training
   ```

### Not Using Full GPU

**Problem:** GPU utilization is low, could run more workers

**Solutions:**
1. Decrease memory estimate:
   ```python
   model_memory_gb=1.0  # Instead of 1.5
   ```

2. Decrease buffer:
   ```python
   gpu_memory_buffer_gb=1.5  # Instead of 2.0
   ```

## Best Practices

1. **Start Conservative**: Use defaults or estimates, then adjust if needed
2. **Monitor First Run**: Watch GPU memory usage with `nvidia-smi`
3. **Use Estimator**: For varying model sizes, use `estimate_model_memory()`
4. **Test Small Batch**: Try 2-3 configs first to verify memory settings
5. **Leave Buffer**: Always keep 1.5-2GB buffer for stability

## Reference Table

### Typical Memory Requirements (MovieLens 100K)

| Configuration | Hidden Dim | Layers | Est. Memory |
|--------------|-----------|--------|-------------|
| Tiny | 32 | 2 | ~1.2GB |
| Small | 64 | 2 | ~1.5GB |
| Medium | 128 | 2 | ~2.0GB |
| Large | 256 | 2 | ~3.5GB |
| Deep | 64 | 4 | ~1.7GB |
| Wide & Deep | 128 | 3 | ~2.8GB |
| Very Large | 256 | 4 | ~4.2GB |

### GPU Capacity Guide

| GPU Memory | Recommended Workers | Model Size | Notes |
|-----------|-------------------|------------|-------|
| 8GB | 2-3 | Small-Medium | Tight fit, use conservative buffer |
| 12GB | 4-6 | Medium | Good for most use cases |
| 16GB | 6-9 | Medium-Large | Optimal for parallel tuning |
| 24GB | 10-14 | Large | Excellent for large model search |
| 32GB+ | 15+ | Very Large | Maximum parallelism |

*Assumes 2GB buffer and baseline model configurations*

## Related Documentation

- **HYPERPARAMETER_TUNING.md** - Complete tuning guide
- **QUICK_START.md** - Quick start examples
- **example_memory_estimation.py** - Memory estimation tool
- **example_tuning.py** - Basic tuning examples

## Summary

The parameterized memory system provides:
- **Flexibility**: Adjust memory parameters for your hardware and models
- **Automation**: `estimate_model_memory()` calculates requirements
- **Efficiency**: Maximize GPU utilization with appropriate worker counts
- **Stability**: Configurable buffer prevents OOM errors
- **Transparency**: Clear output shows memory allocation decisions

Use these tools to optimize parallel training for your specific use case!
