# GraphSAGE Recommender (GNN Module)

Graph Neural Network–based recommender using **GraphSAGE** on a user–item bipartite graph. Built on [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/), trained on **MovieLens 100K** with user/item features, and evaluated with rating (RMSE, MAE) and ranking (Precision@K, Recall@K, NDCG@K, Hit Rate@K) metrics.

## Overview

- **Model**: GraphSAGE with feature projection, pooling aggregation, residual connections, and an optional rating head.
- **Losses**: MSE (rating), BPR (ranking), or combined MSE+BPR.
- **Training**: Early stopping, validation split, configurable batch size and learning rate.
- **Evaluation**: Surprise-compatible `fit` / `test` interface; cold-start user/item breakdown.

## Requirements

- **Python**: 3.8+
- **Conda** (recommended) or pip

### Conda (recommended)

From the **repository root**:

```bash
conda env create -f gnn/environment.yml
conda activate gnn_recom_sys
```

### Pip

```bash
pip install torch numpy pandas scikit-learn scipy scikit-surprise
pip install torch-geometric tqdm
```

**Optional** (for FM vs GraphSAGE comparison):

```bash
pip install myfm
```

## Data

- **Dataset**: MovieLens 100K (via `scikit-surprise`; downloaded automatically on first use).
- **Features**: User (age, gender, occupation) and item (year, genres) from `common.data_loader` (`load_user_features`, `load_item_features`). Feature paths resolved via `get_movielens_data_path()`.

All scripts expect to be run from the **repository root** so that `common` and `gnn` are importable.

## Project Structure

```
gnn/
├── README.md                 # This file
├── environment.yml           # Conda environment
│
├── graph_data_loader.py      # Bipartite graph construction (PyG Data)
├── graphsage_model.py        # GraphSAGE architecture + rating head
├── train_graphsage.py        # Training loop (MSE / BPR / combined, early stopping)
├── graphsage_recommender.py  # Surprise-compatible wrapper (fit / predict / test)
│
├── compare_fm_graphsage.py   # FM vs GraphSAGE comparison (optional: myfm)
├── param_tuning.py           # Hyperparameter tuning (parallel)
├── find_threshold.py         # Cold-start user threshold helper
│
├── tuning/                   # Tuning and comparison outputs (.json)
│
├── GNN_RECOM_SYS_GUIDE.md    # Detailed GNN/recommender guide
├── GRAPHSAGE_FM_COMPARISON_ANALYSIS.md
├── HYPERPARAMETER_TUNING.md
├── LOSS_FUNCTION_ANALYSIS.md
└── MEMORY_PARAMETERS.md
```

## Quick Start

1. **Environment**

   ```bash
   conda env create -f gnn/environment.yml
   conda activate gnn_recom_sys
   ```

2. **Run from repo root**

   ```bash
   cd /path/to/recommnder_systems_guide
   python gnn/graphsage_recommender.py
   ```

   This trains a small GraphSAGE model (5 epochs), runs a few predictions, and runs evaluation on a subset of the test set.

### Docker (`docker-compose`)

With the project’s Docker setup (see root `DOCKER.md`):

```bash
# From repo root; ensure container is up: docker-compose up -d
docker-compose exec recommender-dev python gnn/graphsage_recommender.py
```

Same pattern for other scripts: prefix with `docker-compose exec recommender-dev` and use paths relative to the repo root (e.g. `gnn/compare_fm_graphsage.py`, `gnn/param_tuning.py --mode example`). The container uses the `recom_sys` conda env and `/workspace` as the project root.

## Running the Code

Use the **repository root** as the working directory for all commands below. With Docker, run them via `docker-compose exec recommender-dev ...` as above.

### 1. Graph construction (`graph_data_loader`)

Build and inspect the user–item bipartite graph:

```bash
python gnn/graph_data_loader.py
```

Prints graph stats: nodes, users, items, edges, feature dimension.

### 2. Model check (`graphsage_model`)

Test the GraphSAGE model forward pass and prediction:

```bash
python gnn/graphsage_model.py
```

### 3. Training only (`train_graphsage`)

Train a model with early stopping (no evaluation pipeline):

```bash
python gnn/train_graphsage.py
```

Uses MSE by default; trains on MovieLens 100K with user/item features. Configure device, epochs, etc. in the `__main__` block.

### 4. Recommender pipeline (`graphsage_recommender`)

Train and evaluate the Surprise-compatible recommender:

```bash
python gnn/graphsage_recommender.py
```

Or as a module:

```bash
python -m gnn.graphsage_recommender
```

Trains a short run (5 epochs), runs sample predictions, and evaluates with `evaluate_model` (RMSE, MAE, Precision@10, etc.).

### 5. FM vs GraphSAGE (`compare_fm_graphsage`)

Compare Factorization Machines and GraphSAGE (RMSE, MAE, Precision@10, Recall@10, NDCG@10, Hit@10, cold-start breakdown):

```bash
python gnn/compare_fm_graphsage.py
```

Requires `myfm`. If `myfm` is missing, only GraphSAGE is run and a warning is printed.

### 6. Hyperparameter tuning (`param_tuning`)

Run parallel tuning over different architectures, loss types, and hyperparameters:

```bash
python gnn/param_tuning.py --mode example
```

**Modes:**

| Mode           | Description                                      |
|----------------|--------------------------------------------------|
| `example`      | Diverse preset configs (default)                  |
| `architecture` | Grid over `hidden_dim` × `num_layers`            |
| `loss`         | MSE vs combined loss (varying BPR weight)        |
| `grid`         | Full grid over several parameters                |

**Options:**

```bash
# Architecture search with auto memory estimate
python gnn/param_tuning.py --mode architecture --auto-memory

# Custom memory settings (e.g. larger models)
python gnn/param_tuning.py --mode architecture --model-memory 2.5 --gpu-buffer 3.0

# Loss comparison
python gnn/param_tuning.py --mode loss
```

Results are written under `gnn/tuning/`.

### 7. Cold-start threshold (`find_threshold`)

Find a `cold_user_threshold` that gives ~50 cold-start users (default target):

```bash
python gnn/find_threshold.py
```

Edit `target_count` in the script to change the target.

## Outputs

- **`gnn/tuning/*.json`**: Tuning and comparison results (configs, metrics).
- **Console**: Training logs, validation loss, early-stopping messages, evaluation metrics.

## GPU vs CPU

- **GPU**: Default in `train_graphsage` and `graphsage_recommender` (`device='cuda'`). Falls back to CPU if CUDA is unavailable.
- **CPU**: Use `device='cpu'` in the training calls or when constructing the wrapper (e.g. in `compare_fm_graphsage` or `param_tuning` you’d need to pass it through if exposed).
- **Memory**: For large models or many parallel runs, use `--model-memory` and `--gpu-buffer` in `param_tuning` (or `--auto-memory`) to avoid OOM.

## Further Reading

- **`GNN_RECOM_SYS_GUIDE.md`**: Full GNN/recommender background and design.
- **`HYPERPARAMETER_TUNING.md`**: Tuning workflow and usage.
- **`LOSS_FUNCTION_ANALYSIS.md`**: MSE, BPR, and combined loss.
- **`MEMORY_PARAMETERS.md`**: GPU memory and scaling.
- **`GRAPHSAGE_FM_COMPARISON_ANALYSIS.md`**: FM vs GraphSAGE analysis.

## Summary of Commands

| Task              | Command                                                |
|-------------------|--------------------------------------------------------|
| Quick test        | `python gnn/graphsage_recommender.py`                  |
| Graph only        | `python gnn/graph_data_loader.py`                      |
| Model only        | `python gnn/graphsage_model.py`                        |
| Training only     | `python gnn/train_graphsage.py`                        |
| FM vs GraphSAGE   | `python gnn/compare_fm_graphsage.py`                   |
| Tuning (example)  | `python gnn/param_tuning.py --mode example`            |
| Tuning (arch)     | `python gnn/param_tuning.py --mode architecture --auto-memory` |
| Cold-start threshold | `python gnn/find_threshold.py`                      |

**Docker:** Prepend `docker-compose exec recommender-dev` to any command (e.g. `docker-compose exec recommender-dev python gnn/graphsage_recommender.py`).
