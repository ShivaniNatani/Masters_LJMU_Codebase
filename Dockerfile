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

# Install PyTorch with CUDA 12.1 support (version pinned to match requirements.txt
# so the later `pip install -r requirements.txt` is a no-op for torch/torchvision
# instead of silently re-resolving them from PyPI's default CPU-only index)
RUN pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu121

# Install Python requirements
RUN pip install -r requirements.txt

# Fail the build (instead of failing silently at runtime) if the two install
# steps above produced a conflicting or non-CUDA torch build
RUN pip check && python -c "import torch; assert torch.__version__.startswith('2.13.0'), torch.__version__; print('torch', torch.__version__, '| CUDA build:', torch.version.cuda)"

# Copy source repository
COPY . ${WORKSPACE_DIR}

# Default verification entrypoint
CMD ["python", "scripts/smoke_test.py"]
