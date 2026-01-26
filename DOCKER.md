# Docker Setup Guide

This guide explains how to build and use the Docker container for developing the recommender systems code with CUDA support.

## Prerequisites

1. **Docker Desktop** installed and running on Windows
2. **NVIDIA GPU** with CUDA support (optional, but recommended for GNN training)
3. **NVIDIA Container Toolkit** (usually included with Docker Desktop on Windows)

## Building the Docker Image

### Option 1: Using Docker Compose (Recommended)

```bash
docker-compose build
```

### Option 2: Using Docker directly

```bash
docker build -t recommender-systems:latest .
```

## Running the Container

### Option 1: Using Docker Compose (Recommended)

```bash
# Start the container
docker-compose up -d

# Access the container shell
docker-compose exec recommender-dev bash

# Stop the container
docker-compose down
```

### Option 2: Using Docker directly

```bash
# Run the container with GPU support
docker run -it --gpus all -v ${PWD}:/workspace recommender-systems:latest bash

# Or on Windows PowerShell
docker run -it --gpus all -v ${PWD}:/workspace recommender-systems:latest bash
```

## Development Workflow

1. **Start the container**:
   ```bash
   docker-compose up -d
   ```

2. **Access the container**:
   ```bash
   docker-compose exec recommender-dev bash
   ```

3. **The conda environment is already activated** (recom_sys). You can verify with:
   ```bash
   conda info --envs
   python --version
   ```

4. **Run your code**:
   ```bash
   # Matrix factorization
   cd matrix_factorization
   python main.py

   # GNN training
   cd gnn
   python train_graphsage.py
   ```

5. **Install additional packages** (if needed):
   ```bash
   pip install <package-name>
   # or
   conda install -c conda-forge <package-name>
   ```

## GPU Verification

To verify CUDA/GPU support is working:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## Volume Mounts

The `docker-compose.yml` file mounts your project directory to `/workspace` in the container, so:
- Changes to your code are immediately reflected in the container
- Results and data files persist on your host machine
- No need to rebuild the image for code changes

## Troubleshooting

### GPU not detected

1. Ensure Docker Desktop has GPU support enabled:
   - Docker Desktop → Settings → Resources → WSL Integration
   - Or Docker Desktop → Settings → General → Use the WSL 2 based engine

2. Verify NVIDIA drivers are installed on Windows

3. Check if GPU is accessible:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

### Conda environment not activated

If the conda environment is not activated automatically:
```bash
conda activate recom_sys
```

### Permission issues on Windows

If you encounter permission issues with mounted volumes, ensure:
- Docker Desktop has access to the drive where your project is located
- WSL 2 integration is enabled if using WSL

## Rebuilding the Image

If you modify the Dockerfile or need to update dependencies:

```bash
docker-compose build --no-cache
```

Or with Docker directly:
```bash
docker build --no-cache -t recommender-systems:latest .
```

## Notes

- The container uses CUDA 11.8. If you need a different CUDA version, modify the base image in the Dockerfile
- The conda environment `recom_sys` combines dependencies from both `gnn/environment.yml` and `matrix_factorization/environment.yml`
- PyTorch is installed with CUDA 11.8 support
- All project files are mounted as volumes, so changes persist on your host machine
