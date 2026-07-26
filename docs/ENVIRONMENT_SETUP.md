# Environment Setup & Reproducibility Guide

## Quickstart Setup

### Option 1: Virtual Environment (Recommended for local dev)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option 2: Conda Environment
```bash
conda env create -f environment.yml
conda activate medical_vlm
```

### Option 3: Docker Container
```bash
docker build -t medical_vlm:latest .
docker run --gpus all -it medical_vlm:latest
```

## Running Verification Smoke Test
```bash
PYTHONPATH=. python scripts/smoke_test.py
```
Outputs JSON verification report to `results/verification_report.json`.
