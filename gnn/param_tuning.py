"""
General-Purpose GraphSAGE Parameter Tuning Script (param_tuning.py).

Tunes hyperparameters including:
- Hidden layer dimensions
- Number of layers
- Loss functions (MSE, BPR, Combined)
- Loss weights
- Learning rates
- Batch sizes

Runs multiple GraphSAGE training configurations in parallel for faster experimentation.
"""

import sys
from pathlib import Path
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import threading
from typing import Dict, List, Tuple, Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.data_loader import (load_movielens_100k, get_train_test_split,
                                load_user_features, load_item_features,
                                get_cold_start_split)
from common.evaluation import evaluate_model
from gnn.graphsage_recommender import train_graphsage_recommender


def train_single_config(config_data):
    """
    Train a single GraphSAGE configuration.
    
    This function runs in a separate thread to enable parallel training on single GPU.
    Uses CUDA streams for better GPU utilization.
    
    Args:
        config_data: Tuple of (config_dict, trainset, testset, user_features, item_features,
                               cold_start_users, cold_start_items, device, stream_id)
            config_dict contains: name, hidden_dim, num_layers, loss_type, mse_weight, 
                                 bpr_weight, learning_rate, batch_size, num_epochs
    
    Returns:
        dict: Results for this configuration
    """
    (config_dict, trainset, testset, user_features, item_features,
     cold_start_users, cold_start_items, device, stream_id) = config_data
    
    config_name = config_dict['name']
    
    try:
        # Create CUDA stream for this training run (helps with parallel execution)
        if 'cuda' in device:
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                torch.cuda.set_device(device)
                print(f"\n[{config_name}] Starting training on {device} (stream {stream_id})...")
                result = _train_and_evaluate(
                    config_dict, trainset, testset, user_features, item_features,
                    cold_start_users, cold_start_items, device
                )
        else:
            print(f"\n[{config_name}] Starting training on {device}...")
            result = _train_and_evaluate(
                config_dict, trainset, testset, user_features, item_features,
                cold_start_users, cold_start_items, device
            )
        
        return result
        
    except Exception as e:
        print(f"[{config_name}] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'config': config_dict,
            'results': None,
            'success': False,
            'error': str(e)
        }


def _train_and_evaluate(config_dict, trainset, testset, user_features, item_features,
                       cold_start_users, cold_start_items, device):
    """
    Helper function to train and evaluate a model.
    
    Args:
        config_dict: Dictionary with training parameters
        trainset: Surprise Trainset
        testset: List of (uid, iid, rating) tuples
        user_features: DataFrame with user features
        item_features: DataFrame with item features
        cold_start_users: Optional set of cold start user IDs
        cold_start_items: Optional set of cold start item IDs
        device: Device to train on ('cuda:0', 'cpu', etc.)
    
    Returns:
        dict: Training and evaluation results
    """
    config_name = config_dict['name']
    
    # Train model with specified hyperparameters
    model = train_graphsage_recommender(
        trainset, user_features, item_features,
        hidden_dim=config_dict.get('hidden_dim', 64),
        num_layers=config_dict.get('num_layers', 2),
        num_epochs=config_dict.get('num_epochs', 20),
        batch_size=config_dict.get('batch_size', 512),
        learning_rate=config_dict.get('learning_rate', 0.001),
        loss_type=config_dict.get('loss_type', 'mse'),
        mse_weight=config_dict.get('mse_weight', 1.0),
        bpr_weight=config_dict.get('bpr_weight', 0.0),
        device=device,
        verbose=True
    )
    
    print(f"[{config_name}] Training completed, evaluating...")
    
    # Evaluate
    predictions = model.test(testset)
    eval_results = evaluate_model(
        predictions,
        k=10,
        threshold=4.0,
        verbose=False,  # Suppress verbose output to avoid cluttering parallel logs
        cold_start_users=cold_start_users,
        cold_start_items=cold_start_items
    )
    
    print(f"[{config_name}] Evaluation completed!")
    print(f"[{config_name}]   RMSE: {eval_results['rmse']:.4f}, MAE: {eval_results['mae']:.4f}")
    print(f"[{config_name}]   NDCG@10: {eval_results['ndcg@10']:.4f}")
    
    # Clean up GPU memory
    if 'cuda' in device:
        del model
        torch.cuda.empty_cache()
    
    return {
        'config': config_dict,
        'results': eval_results,
        'success': True,
        'error': None
    }


def auto_estimate_memory_from_configs(configurations: List[Dict[str, Any]],
                                     num_users: int, num_items: int) -> float:
    """
    Automatically estimate memory requirements from a list of configurations.
    Uses the largest model (max hidden_dim and num_layers) to ensure all configs fit.
    
    Args:
        configurations: List of configuration dictionaries
        num_users: Number of users in dataset
        num_items: Number of items in dataset
        
    Returns:
        Estimated memory in GB for the largest model
    """
    if not configurations:
        return 1.5  # Default fallback
    
    # Find max hidden_dim and num_layers across all configs
    max_hidden_dim = max(config.get('hidden_dim', 64) for config in configurations)
    max_num_layers = max(config.get('num_layers', 2) for config in configurations)
    
    # Estimate memory for the largest model
    memory_gb = estimate_model_memory(max_hidden_dim, max_num_layers, num_users, num_items)
    
    return memory_gb


def estimate_model_memory(hidden_dim: int = 64, num_layers: int = 2, 
                         num_users: int = 1000, num_items: int = 1000) -> float:
    """
    Estimate GPU memory requirement for a GraphSAGE model in GB.
    
    This is a rough approximation based on typical model sizes.
    Actual memory usage can vary based on batch size, optimizer state, etc.
    
    Args:
        hidden_dim: Hidden dimension size
        num_layers: Number of GraphSAGE layers
        num_users: Number of users in the dataset
        num_items: Number of items in the dataset
        
    Returns:
        Estimated memory in GB
        
    Example:
        # For a model with 128 hidden dim, 3 layers
        memory_gb = estimate_model_memory(128, 3, 943, 1682)
        print(f"Estimated memory: {memory_gb:.2f}GB")
    """
    # Base memory for embeddings (user + item)
    embedding_memory = (num_users + num_items) * hidden_dim * 4 / (1024**3)  # 4 bytes per float32
    
    # Memory for GNN layers (rough estimate)
    # Each layer has weight matrices and intermediate activations
    layer_memory = num_layers * hidden_dim * hidden_dim * 4 / (1024**3)
    
    # Output layer
    output_memory = hidden_dim * 4 / (1024**3)
    
    # Optimizer states (Adam uses 2x model parameters)
    optimizer_overhead = 2.0
    
    # PyTorch overhead and intermediate activations
    pytorch_overhead = 0.5
    
    total = (embedding_memory + layer_memory + output_memory) * optimizer_overhead + pytorch_overhead
    
    # Round up and add some buffer
    return round(total + 0.3, 1)  # Add 0.3GB buffer, round to 1 decimal


def tune_hyperparameters_parallel(trainset, testset, user_features, item_features,
                                  configurations: List[Dict[str, Any]],
                                  cold_start_users=None, cold_start_items=None,
                                  num_gpus=1, max_workers=None, 
                                  model_memory_gb: float = 1.5,
                                  gpu_memory_buffer_gb: float = 2.0,
                                  verbose=True):
    """
    Tune GraphSAGE hyperparameters in parallel.
    
    Args:
        trainset: Surprise Trainset
        testset: List of (uid, iid, rating) tuples
        user_features: DataFrame with user features
        item_features: DataFrame with item features
        configurations: List of configuration dictionaries. Each dict should contain:
            - name: Configuration name (required)
            - hidden_dim: Hidden layer dimension (default: 64)
            - num_layers: Number of GraphSAGE layers (default: 2)
            - loss_type: 'mse', 'bpr', or 'combined' (default: 'mse')
            - mse_weight: Weight for MSE loss (default: 1.0)
            - bpr_weight: Weight for BPR loss (default: 0.0)
            - learning_rate: Learning rate (default: 0.001)
            - batch_size: Batch size (default: 512)
            - num_epochs: Number of epochs (default: 20)
        cold_start_users: Optional set of cold start user IDs
        cold_start_items: Optional set of cold start item IDs
        num_gpus: Number of GPUs available (default: 1)
        max_workers: Maximum parallel workers (default: auto-detect based on GPU memory)
        model_memory_gb: Estimated GPU memory per model in GB (default: 1.5).
                        Increase for larger models (higher hidden_dim, more layers).
        gpu_memory_buffer_gb: GPU memory to keep free in GB (default: 2.0).
                             Buffer for PyTorch overhead and system processes.
        verbose: Print progress (default: True)
        
    Returns:
        dict: Results for each configuration (keyed by config name)
        
    Example configurations:
        configurations = [
            {'name': 'Baseline', 'hidden_dim': 64, 'num_layers': 2},
            {'name': 'Deeper', 'hidden_dim': 64, 'num_layers': 3},
            {'name': 'Wider', 'hidden_dim': 128, 'num_layers': 2},
            {'name': 'Combined Loss', 'hidden_dim': 64, 'num_layers': 2, 
             'loss_type': 'combined', 'bpr_weight': 0.1},
        ]
    """
    
    # Validate configurations
    if not configurations:
        raise ValueError("At least one configuration must be provided")
    
    for config in configurations:
        if 'name' not in config:
            raise ValueError("Each configuration must have a 'name' field")
    
    # Determine number of workers based on GPU memory
    if max_workers is None:
        if num_gpus > 0:
            # For single GPU, estimate how many models can fit based on memory
            if num_gpus == 1:
                gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                # Calculate workers: (total_memory - buffer) / memory_per_model
                max_workers = max(1, int((gpu_mem_gb - gpu_memory_buffer_gb) / model_memory_gb))
                max_workers = min(max_workers, len(configurations))  # Don't exceed number of configs
            else:
                max_workers = num_gpus
        else:
            max_workers = max(1, mp.cpu_count() // 2)  # Use half of CPU cores
    
    if verbose:
        print(f"\nParallel Training Configuration:")
        print(f"  Total configurations: {len(configurations)}")
        print(f"  Parallel workers: {max_workers}")
        print(f"  GPUs available: {num_gpus}")
        if num_gpus == 1:
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  Mode: Multi-threaded on single GPU (memory-efficient)")
            print(f"  GPU total memory: {gpu_mem_gb:.1f}GB")
            print(f"  GPU memory buffer: {gpu_memory_buffer_gb:.1f}GB")
            print(f"  Estimated memory per worker: {model_memory_gb:.1f}GB")
            print(f"  Total estimated GPU usage: ~{max_workers * model_memory_gb:.1f}GB")
        elif num_gpus > 1:
            print(f"  Device assignment: GPU rotation")
        else:
            print(f"  Device assignment: CPU")
    
    # Prepare training data for each configuration
    config_data_list = []
    for i, config_dict in enumerate(configurations):
        # Assign GPU in round-robin fashion if available
        if num_gpus > 0:
            if num_gpus == 1:
                # All on same GPU for single GPU setup (will use threads + streams)
                device = 'cuda:0'
                stream_id = i
            else:
                # Multiple GPUs: distribute across GPUs
                device = f'cuda:{i % num_gpus}'
                stream_id = 0
        else:
            device = 'cpu'
            stream_id = 0
        
        config_data_list.append((
            config_dict, trainset, testset, user_features, item_features,
            cold_start_users, cold_start_items, device, stream_id
        ))
    
    # Train models in parallel
    results = {}
    
    if verbose:
        print("\n" + "=" * 60)
        print("Starting Parallel Training...")
        if num_gpus == 1:
            print("Using ThreadPoolExecutor for single-GPU parallelization")
        else:
            print("Using multi-GPU or CPU parallelization")
        print("=" * 60)
    
    # Use ThreadPoolExecutor for single GPU (better memory sharing)
    # Use ProcessPoolExecutor for multi-GPU (better isolation)
    ExecutorClass = ThreadPoolExecutor if num_gpus == 1 else ThreadPoolExecutor
    
    with ExecutorClass(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_config = {
            executor.submit(train_single_config, config_data): config_data[0]['name']
            for config_data in config_data_list
        }
        
        # Collect results as they complete
        completed = 0
        total = len(future_to_config)
        for future in as_completed(future_to_config):
            config_name = future_to_config[future]
            completed += 1
            try:
                result = future.result()
                if result['success']:
                    results[result['config']['name']] = result
                    if verbose:
                        print(f"\n[Progress] {completed}/{total} configurations completed")
                else:
                    print(f"\n[WARNING] {config_name} failed: {result['error']}")
            except Exception as e:
                print(f"\n[ERROR] {config_name} raised exception: {str(e)}")
    
    if verbose:
        print("\n" + "=" * 60)
        print("All Training Completed!")
        print("=" * 60)
    
    # Print comparison
    if verbose and results:
        _print_results_summary(results, cold_start_items)
    
    return results


def save_results(results: Dict[str, Dict], filepath: str):
    """
    Save tuning results to a JSON file.
    
    Args:
        results: Results dictionary from tune_hyperparameters_parallel
        filepath: Path to save JSON file
    """
    import json
    from pathlib import Path
    
    # Prepare results for JSON serialization
    serializable_results = {}
    for config_name, result in results.items():
        serializable_results[config_name] = {
            'config': result['config'],
            'success': result['success'],
            'error': result['error'],
            'results': result['results'] if result['success'] else None
        }
    
    # Save to file
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def load_results(filepath: str) -> Dict[str, Dict]:
    """
    Load tuning results from a JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Results dictionary
    """
    import json
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results


def _print_results_summary(results: Dict[str, Dict], cold_start_items=None):
    """Print formatted results summary with configuration details."""
    print("\n" + "=" * 80)
    print("Hyperparameter Tuning Results")
    print("=" * 80)
    
    # Sort results by configuration name
    sorted_configs = sorted(results.keys())
    
    # Print configuration details
    print("\n" + "-" * 80)
    print("Configuration Details")
    print("-" * 80)
    header = f"{'Name':<20} {'Hidden':>8} {'Layers':>7} {'Loss':>10} {'LR':>8} {'Batch':>7} {'Epochs':>7}"
    print(header)
    print("-" * 80)
    
    for config_name in sorted_configs:
        config = results[config_name]['config']
        loss_str = config.get('loss_type', 'mse')
        if loss_str == 'combined':
            loss_str = f"comb({config.get('bpr_weight', 0):.2f})"
        
        print(f"{config_name:<20} "
              f"{config.get('hidden_dim', 64):>8} "
              f"{config.get('num_layers', 2):>7} "
              f"{loss_str:>10} "
              f"{config.get('learning_rate', 0.001):>8.4f} "
              f"{config.get('batch_size', 512):>7} "
              f"{config.get('num_epochs', 20):>7}")
    
    # Rating prediction metrics
    print("\n" + "-" * 80)
    print("Rating Prediction Metrics")
    print("-" * 80)
    print(f"{'Configuration':<20} {'RMSE':>10} {'MAE':>10}")
    print("-" * 80)
    
    for config_name in sorted_configs:
        res = results[config_name]['results']
        print(f"{config_name:<20} {res['rmse']:>10.4f} {res['mae']:>10.4f}")
    
    # Ranking metrics
    print("\n" + "-" * 80)
    print("Ranking Metrics (K=10)")
    print("-" * 80)
    print(f"{'Configuration':<20} {'Prec@10':>10} {'Recall@10':>10} {'NDCG@10':>10} {'Hit@10':>10}")
    print("-" * 80)
    
    for config_name in sorted_configs:
        res = results[config_name]['results']
        print(f"{config_name:<20} "
              f"{res['precision@10']:>10.4f} "
              f"{res['recall@10']:>10.4f} "
              f"{res['ndcg@10']:>10.4f} "
              f"{res['hit_rate@10']:>10.4f}")
    
    # Cold start comparison
    if cold_start_items:
        print("\n" + "-" * 80)
        print("Cold Start Item Metrics")
        print("-" * 80)
        print(f"{'Configuration':<20} {'RMSE':>10} {'MAE':>10} {'Coverage':>10}")
        print("-" * 80)
        
        for config_name in sorted_configs:
            res = results[config_name]['results']
            if res.get('cold_item_rmse') is not None:
                print(f"{config_name:<20} "
                      f"{res['cold_item_rmse']:>10.4f} "
                      f"{res['cold_item_mae']:>10.4f} "
                      f"{res['cold_item_coverage']:>9.1%}")
    
    # Best configuration analysis
    print("\n" + "=" * 80)
    print("Best Configurations")
    print("=" * 80)
    
    # Find best by different metrics
    best_rmse = min(sorted_configs, key=lambda x: results[x]['results']['rmse'])
    best_ndcg = max(sorted_configs, key=lambda x: results[x]['results']['ndcg@10'])
    best_precision = max(sorted_configs, key=lambda x: results[x]['results']['precision@10'])
    
    print(f"\nBest RMSE: {best_rmse}")
    print(f"  RMSE: {results[best_rmse]['results']['rmse']:.4f}")
    print(f"  Config: hidden_dim={results[best_rmse]['config'].get('hidden_dim', 64)}, "
          f"num_layers={results[best_rmse]['config'].get('num_layers', 2)}")
    
    print(f"\nBest NDCG@10: {best_ndcg}")
    print(f"  NDCG@10: {results[best_ndcg]['results']['ndcg@10']:.4f}")
    print(f"  Config: hidden_dim={results[best_ndcg]['config'].get('hidden_dim', 64)}, "
          f"num_layers={results[best_ndcg]['config'].get('num_layers', 2)}")
    
    print(f"\nBest Precision@10: {best_precision}")
    print(f"  Precision@10: {results[best_precision]['results']['precision@10']:.4f}")
    print(f"  Config: hidden_dim={results[best_precision]['config'].get('hidden_dim', 64)}, "
          f"num_layers={results[best_precision]['config'].get('num_layers', 2)}")
    
    print("=" * 80)


def create_grid_search_configs(
    param_grid: Dict[str, List[Any]],
    base_config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Create configurations for grid search over hyperparameters.
    
    Args:
        param_grid: Dictionary mapping parameter names to lists of values to try.
                   Example: {'hidden_dim': [32, 64, 128], 'num_layers': [2, 3]}
        base_config: Base configuration with default values for parameters not in param_grid
        
    Returns:
        List of configuration dictionaries for all combinations
        
    Example:
        configs = create_grid_search_configs(
            param_grid={
                'hidden_dim': [64, 128],
                'num_layers': [2, 3],
            },
            base_config={'loss_type': 'mse', 'learning_rate': 0.001}
        )
    """
    from itertools import product
    
    if base_config is None:
        base_config = {
            'loss_type': 'mse',
            'mse_weight': 1.0,
            'bpr_weight': 0.0,
            'learning_rate': 0.001,
            'batch_size': 512,
            'num_epochs': 20,
            'hidden_dim': 64,
            'num_layers': 2
        }
    
    configurations = []
    
    # Get all parameter names and values
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    
    # Generate all combinations
    for values in product(*param_values):
        config = base_config.copy()
        
        # Update with current parameter values
        for name, value in zip(param_names, values):
            config[name] = value
        
        # Generate descriptive name
        name_parts = []
        for name, value in zip(param_names, values):
            if name == 'hidden_dim':
                name_parts.append(f'h{value}')
            elif name == 'num_layers':
                name_parts.append(f'l{value}')
            elif name == 'loss_type':
                name_parts.append(f'{value}')
            elif name == 'bpr_weight' and value > 0:
                name_parts.append(f'bpr{value}')
            elif name == 'learning_rate':
                name_parts.append(f'lr{value}')
            elif name == 'batch_size':
                name_parts.append(f'bs{value}')
        
        config['name'] = '-'.join(name_parts) if name_parts else 'config'
        configurations.append(config)
    
    return configurations


def create_architecture_search_configs(
    hidden_dims: List[int] = [32, 64, 128, 256],
    num_layers_list: List[int] = [1, 2, 3, 4],
    base_config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Create configurations for architecture search (hidden dims and layers).
    
    Args:
        hidden_dims: List of hidden dimensions to try
        num_layers_list: List of layer counts to try
        base_config: Base configuration with other hyperparameters
        
    Returns:
        List of configuration dictionaries
    """
    return create_grid_search_configs(
        param_grid={
            'hidden_dim': hidden_dims,
            'num_layers': num_layers_list
        },
        base_config=base_config
    )


def create_loss_function_configs(
    bpr_weights: List[float] = [0.0, 0.05, 0.1, 0.15, 0.2],
    base_config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Create configurations for loss function comparison.
    
    Args:
        bpr_weights: List of BPR weights to try (0.0 = MSE only)
        base_config: Base configuration with other hyperparameters
        
    Returns:
        List of configuration dictionaries
    """
    configurations = []
    
    if base_config is None:
        base_config = {
            'hidden_dim': 64,
            'num_layers': 2,
            'learning_rate': 0.001,
            'batch_size': 512,
            'num_epochs': 20
        }
    
    for bpr_weight in bpr_weights:
        config = base_config.copy()
        if bpr_weight == 0.0:
            config['name'] = 'MSE-Only'
            config['loss_type'] = 'mse'
            config['mse_weight'] = 1.0
            config['bpr_weight'] = 0.0
        else:
            config['name'] = f'Combined-{bpr_weight}'
            config['loss_type'] = 'combined'
            config['mse_weight'] = 1.0
            config['bpr_weight'] = bpr_weight
        
        configurations.append(config)
    
    return configurations


def create_example_configurations() -> List[Dict[str, Any]]:
    """
    Create example configurations for hyperparameter tuning.
    Modify this function to define your own configurations.
    
    Returns:
        List of configuration dictionaries
    """
    configurations = []
    
    # Baseline configuration
    configurations.append({
        'name': 'Baseline',
        'hidden_dim': 64,
        'num_layers': 2,
        'loss_type': 'mse',
        'learning_rate': 0.001,
        'batch_size': 512,
        'num_epochs': 20
    })
    
    # Test different hidden dimensions
    for hidden_dim in [32, 128, 256]:
        configurations.append({
            'name': f'Hidden-{hidden_dim}',
            'hidden_dim': hidden_dim,
            'num_layers': 2,
            'loss_type': 'mse',
            'learning_rate': 0.001,
            'batch_size': 512,
            'num_epochs': 20
        })
    
    # Test different number of layers
    for num_layers in [1, 3, 4]:
        configurations.append({
            'name': f'Layers-{num_layers}',
            'hidden_dim': 64,
            'num_layers': num_layers,
            'loss_type': 'mse',
            'learning_rate': 0.001,
            'batch_size': 512,
            'num_epochs': 20
        })
    
    # Test combined loss with different weights
    for bpr_weight in [0.05, 0.1, 0.15]:
        configurations.append({
            'name': f'Combined-{bpr_weight}',
            'hidden_dim': 64,
            'num_layers': 2,
            'loss_type': 'combined',
            'mse_weight': 1.0,
            'bpr_weight': bpr_weight,
            'learning_rate': 0.001,
            'batch_size': 512,
            'num_epochs': 20
        })
    
    # Test wider + deeper model
    configurations.append({
        'name': 'Wide-Deep',
        'hidden_dim': 128,
        'num_layers': 3,
        'loss_type': 'mse',
        'learning_rate': 0.001,
        'batch_size': 512,
        'num_epochs': 20
    })
    
    return configurations


def main(tuning_mode: str = 'example', 
         model_memory_gb: float = 1.5,
         gpu_memory_buffer_gb: float = 2.0,
         auto_memory: bool = False):
    """
    Main hyperparameter tuning pipeline.
    
    Args:
        tuning_mode: Type of tuning to perform:
                    - 'example': Example configurations (default)
                    - 'architecture': Architecture search (hidden_dim x num_layers)
                    - 'loss': Loss function comparison
                    - 'grid': Full grid search
        model_memory_gb: Estimated GPU memory per model in GB (default: 1.5)
        gpu_memory_buffer_gb: GPU memory buffer in GB (default: 2.0)
        auto_memory: Automatically estimate memory from configurations (default: False)
    """
    print("=" * 80)
    print("GraphSAGE Hyperparameter Tuning (Parallel)")
    print("Tune: Hidden Dimensions, Layers, Loss Functions, and more")
    print("MovieLens 100K Dataset")
    print("=" * 80)
    
    # Load data
    print("\n[Step 1] Loading MovieLens 100K dataset...")
    data = load_movielens_100k()
    trainset, testset = get_train_test_split(data, test_size=0.2, random_state=42)
    
    print(f"  Training set: {trainset.n_ratings:,} ratings")
    print(f"  Test set: {len(testset):,} ratings")
    print(f"  Users: {trainset.n_users:,}, Items: {trainset.n_items:,}")
    
    # Load features
    print("\n[Step 2] Loading user and item features...")
    user_features = load_user_features()
    item_features = load_item_features()
    print(f"  User features: {user_features.shape}")
    print(f"  Item features: {item_features.shape}")
    
    # Create cold start split
    print("\n[Step 3] Creating cold start test sets...")
    trainset_cs, testset_cs, cold_users, cold_items = get_cold_start_split(
        data,
        test_size=0.2,
        random_state=42,
        cold_user_threshold=17,
        cold_item_threshold=10
    )
    print(f"  Cold start users: {len(cold_users)}")
    print(f"  Cold start items: {len(cold_items)}")
    
    # Detect available GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"\n[Step 4] Detecting hardware...")
    print(f"  Available GPUs: {num_gpus}")
    if num_gpus > 0:
        for i in range(num_gpus):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Create configurations based on mode
    print(f"\n[Step 5] Creating hyperparameter configurations (mode: {tuning_mode})...")
    if tuning_mode == 'architecture':
        configurations = create_architecture_search_configs(
            hidden_dims=[16, 32, 64, 128, 256, 512],
            num_layers_list=[1, 2, 3, 4, 5]
        )
    elif tuning_mode == 'loss':
        configurations = create_loss_function_configs(
            bpr_weights=[0.0, 0.05, 0.1, 0.15, 0.2]
        )
    elif tuning_mode == 'grid':
        configurations = create_grid_search_configs(
            param_grid={
                'hidden_dim': [64, 128],
                'num_layers': [2, 3],
                'loss_type': ['mse', 'combined'],
                'bpr_weight': [0.0, 0.1]
            }
        )
    else:  # example
        configurations = create_example_configurations()
    
    print(f"  Total configurations to test: {len(configurations)}")
    
    # Automatically estimate memory if requested
    if auto_memory:
        print(f"\n[Step 6] Auto-estimating memory requirements...")
        original_memory = model_memory_gb
        model_memory_gb = auto_estimate_memory_from_configs(
            configurations,
            num_users=trainset.n_users,
            num_items=trainset.n_items
        )
        # Find the largest model for reporting
        max_hidden = max(config.get('hidden_dim', 64) for config in configurations)
        max_layers = max(config.get('num_layers', 2) for config in configurations)
        print(f"  Analyzed configurations: max hidden_dim={max_hidden}, max num_layers={max_layers}")
        print(f"  Calculated memory requirement: {model_memory_gb:.1f}GB per model")
        if original_memory != 1.5:  # User provided a value
            print(f"  (Overriding user-specified {original_memory:.1f}GB with auto-calculated value)")
    
    # Display memory configuration
    print(f"\n[Step 7] Memory configuration...")
    print(f"  Model memory estimate: {model_memory_gb:.1f}GB per model")
    print(f"  GPU memory buffer: {gpu_memory_buffer_gb:.1f}GB")
    if num_gpus > 0 and num_gpus == 1:
        gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        available = gpu_total - gpu_memory_buffer_gb
        estimated_workers = max(1, int(available / model_memory_gb))
        print(f"  GPU total memory: {gpu_total:.1f}GB")
        print(f"  Estimated parallel workers: ~{estimated_workers}")
    
    # Run hyperparameter tuning in parallel
    print("\n[Step 8] Training models with different configurations (parallel)...")
    results = tune_hyperparameters_parallel(
        trainset,
        testset,
        user_features,
        item_features,
        configurations=configurations,
        cold_start_users=cold_users,
        cold_start_items=cold_items,
        num_gpus=num_gpus,
        max_workers=None,  # Auto-detect
        model_memory_gb=model_memory_gb,
        gpu_memory_buffer_gb=gpu_memory_buffer_gb,
        verbose=True
    )
    
    # Save results
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"gnn/tuning/tuning_results_{tuning_mode}_{timestamp}.json"
    save_results(results, results_file)
    
    print("\n" + "=" * 80)
    print("Hyperparameter Tuning Completed!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  - Parallel training significantly reduces experiment time")
    print("  - Systematic exploration reveals optimal architecture")
    print("  - Trade-offs between model capacity and performance")
    print("  - Combined loss can improve ranking with minimal rating loss")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='GraphSAGE Hyperparameter Tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tuning modes:
  example       - Run example configurations (diverse set)
  architecture  - Architecture search (hidden_dim x num_layers)
  loss          - Loss function comparison (MSE vs Combined)
  grid          - Full grid search over multiple parameters

Examples:
  # Basic usage
  python param_tuning.py --mode example
  python param_tuning.py --mode architecture
  
  # Auto-estimate memory from configurations (recommended)
  python param_tuning.py --mode architecture --auto-memory
  
  # With custom memory settings for larger models
  python param_tuning.py --mode architecture --model-memory 2.5
  
  # Adjust both memory parameters
  python param_tuning.py --mode loss --model-memory 2.0 --gpu-buffer 3.0
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='example',
        choices=['example', 'architecture', 'loss', 'grid'],
        help='Tuning mode to run (default: example)'
    )
    
    parser.add_argument(
        '--model-memory',
        type=float,
        default=1.5,
        metavar='GB',
        help='Estimated GPU memory per model in GB (default: 1.5). '
             'Increase for larger models (e.g., hidden_dim=256, num_layers=4)'
    )
    
    parser.add_argument(
        '--gpu-buffer',
        type=float,
        default=2.0,
        metavar='GB',
        help='GPU memory buffer in GB for PyTorch overhead (default: 2.0). '
             'Increase if you see OOM errors during initialization'
    )
    
    parser.add_argument(
        '--auto-memory',
        action='store_true',
        help='Automatically estimate memory requirements from configurations. '
             'Analyzes all configs and uses the largest model size to calculate memory needs. '
             'Overrides --model-memory if both are specified.'
    )
    
    args = parser.parse_args()
    main(tuning_mode=args.mode,
         model_memory_gb=args.model_memory,
         gpu_memory_buffer_gb=args.gpu_buffer,
         auto_memory=args.auto_memory)
