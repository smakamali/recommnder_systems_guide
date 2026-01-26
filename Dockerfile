# Use NVIDIA CUDA 12.1 base image with conda (devel version includes full CUDA libraries)
# PyTorch 2.x works better with CUDA 12.x
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    PATH=$CONDA_DIR/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean --all -y && \
    # Accept conda Terms of Service for required channels
    $CONDA_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    $CONDA_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    ln -s $CONDA_DIR/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Make RUN commands use the new environment
SHELL ["/bin/bash", "--login", "-c"]

# Set working directory
WORKDIR /workspace

# Copy environment files
COPY gnn/environment.yml /tmp/gnn_environment.yml
COPY matrix_factorization/environment.yml /tmp/mf_environment.yml

# Create merged conda environment
# Install all packages in one go with NumPy 1.x constraint to ensure compatibility
# scikit-surprise requires NumPy 1.x, and all packages must be resolved together
RUN conda create -n recom_sys python=3.10 -y && \
    $CONDA_DIR/bin/conda run -n recom_sys conda install -y -c pytorch -c nvidia -c conda-forge \
        "numpy=1.26.4" \
        "scipy>=1.10.0,<1.14.0" \
        "pytorch>=2.0.0" \
        "pytorch-cuda=12.1" \
        "pandas>=2.0.0" \
        "scikit-learn>=1.3.0,<1.6.0" \
        "scikit-surprise>=1.1.3" \
        "implicit>=0.6.0" \
        "matplotlib>=3.7.0" \
        pip && \
    $CONDA_DIR/bin/conda run -n recom_sys pip install "torch-geometric>=2.3.0" \
                "tqdm>=4.65.0" \
                myfm && \
    # Force numpy back to 1.26.4 after pip install (pip may upgrade it)
    $CONDA_DIR/bin/conda run -n recom_sys pip install "numpy==1.26.4" && \
    $CONDA_DIR/bin/conda clean --all -y

# Copy project files - uncomment if not using docker-compose
# COPY . /workspace

# Set the default command to activate the environment
RUN echo "conda activate recom_sys" >> ~/.bashrc

# Make conda environment available in PATH by default
ENV PATH=$CONDA_DIR/envs/recom_sys/bin:$PATH \
    CONDA_DEFAULT_ENV=recom_sys

# Default command
CMD ["/bin/bash"]
