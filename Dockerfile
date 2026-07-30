# Production Dockerfile for Multi-Modal Medical Report Generation
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    WORKSPACE_DIR=/app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR ${WORKSPACE_DIR}

# Symlink python3 to python
RUN ln -s /usr/bin/python3.10 /usr/local/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy dependency requirements
COPY requirements.txt ${WORKSPACE_DIR}/requirements.txt

# Install PyTorch with CUDA 12.1 support
RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121

# Install Python requirements
RUN pip install -r requirements.txt

# Verify CUDA torch build
RUN python -c "import torch; print('torch', torch.__version__, '| CUDA build:', torch.version.cuda)"

# Copy source repository
COPY . ${WORKSPACE_DIR}

# Default verification entrypoint
CMD ["python", "scripts/smoke_test.py"]
