# Multi-Modal Medical Report Generation Using Vision-Language Models for Automated Radiology Documentation

![Phase 1 Verification](https://img.shields.io/badge/Phase%201-Verified%20%26%20Reproducible-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0-orange)
![Transformers](https://img.shields.io/badge/Transformers-5.14.1-blue)
![License](https://img.shields.io/badge/License-Academic%20MSc-purple)

---

## 📌 Dissertation Overview
This repository contains the engineering pipeline and research environment for the MSc dissertation:
**"MULTI-MODAL MEDICAL REPORT GENERATION USING VISION-LANGUAGE MODELS FOR AUTOMATED RADIOLOGY DOCUMENTATION"**.

The goal of this research project is to design, implement, and benchmark vision-language models (VLMs) capable of generating fluent, clinically precise radiology reports (Findings and Impression) from chest X-ray modalities (MIMIC-CXR-JPG and Indiana University Chest X-Ray datasets).

---

## 📂 Repository Directory Structure

```
project/
├── configs/                  # YAML Configuration System
│   ├── datasets.yaml         # Dataset locations, split ratios, image norms
│   ├── training.yaml         # Optimizer, learning rate, hardware configs
│   ├── models.yaml           # Vision encoder & Text decoder hyperparameters
│   ├── evaluation.yaml       # BLEU, ROUGE, CIDER & CheXbert metric configs
│   └── logging.yaml          # W&B, MLflow, and file logger settings
├── data/                     # Raw, Processed & Synthetic Mock Datasets
│   ├── raw/                  # MIMIC-CXR and IU Chest X-Ray raw files
│   ├── processed/            # Patient-split metadata & cleaned CSVs
│   └── mock/                 # Synthetic X-ray images & mock reports
├── datasets/                 # PyTorch Custom Datasets & DataLoaders
│   ├── base_dataset.py       # Abstract Base Medical Dataset class
│   ├── mimic_cxr.py          # MIMIC-CXR PyTorch Dataset loader
│   ├── iu_chest_xray.py      # Indiana University Chest X-ray Dataset loader
│   └── data_loader.py        # DataLoader generator & batching logic
├── preprocessing/            # Pipeline Data Preprocessors
│   ├── patient_splitter.py   # Patient-level train/val/test splitter
│   ├── image_preprocessing.py# Image resize, normalization, augmentations
│   ├── text_preprocessing.py # Text cleaning, section extraction, tokenization
│   └── build_vocab.py        # Vocabulary builder & token JSON exporter
├── models/                   # Vision-Language Model Architectures
│   └── mock_vlm.py           # Verification stub VLM model
├── retrieval/                # Vector Retrieval Module (Phase 3 reserved)
├── label_guidance/           # Clinical Label Guidance Module (Phase 4 reserved)
├── training/                 # Model Training Pipeline (Phase 5 reserved)
├── evaluation/               # Metric Evaluation Engine (Phase 6 reserved)
├── utils/                    # System Utilities & Environment Checks
│   ├── seed.py               # Deterministic seed locking across torch/np/python
│   ├── logger.py             # File & Console Logger setup
│   └── env_check.py          # Hardware GPU/MPS & Library version verifier
├── notebooks/                # Jupyter Analysis Notebooks
│   └── 01_exploratory_data_analysis.ipynb # Interactive EDA Notebook
├── figures/                  # Publication-Quality EDA Figures
│   ├── disease_frequency.png
│   ├── image_resolution_dist.png
│   ├── report_length_dist.png
│   ├── vocabulary_dist.png
│   └── patient_stats.png
├── logs/                     # Execution & Run Log files
├── checkpoints/              # Checkpoint model weights storage
├── results/                  # CSV Summaries & Verification Reports
│   ├── dataset_summary_stats.csv
│   ├── disease_frequency_stats.csv
│   ├── report_length_stats.csv
│   ├── vocabulary_stats.csv
│   └── verification_report.json
├── scripts/                  # Executable CLI Scripts
│   ├── download_mimic_cxr.py # PhysioNet Credential Manager & MIMIC Downloader
│   ├── download_iu_cxr.py    # OpenI IU Chest X-Ray Downloader
│   ├── generate_mock_data.py # Synthetic mock X-ray & report generator
│   ├── run_eda.py            # EDA statistics & figure generation script
│   └── smoke_test.py         # End-to-end Phase 1 Smoke Test
├── docs/                     # Research Documentation & Guides
│   ├── DISSERTATION_RESEARCH_PLAN.md
│   ├── DATASET_GUIDE.md
│   └── ENVIRONMENT_SETUP.md
├── requirements.txt          # Python dependency specifications
├── environment.yml           # Conda environment definition
├── Dockerfile                # Production CUDA 12.1 Docker container
└── README.md                 # Project Documentation
```

---

## ⚡ Environment Quickstart

### 1. Installation
```bash
# Option A: Virtual Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Option B: Docker Container
docker build -t medical_vlm:latest .
docker run --gpus all -it medical_vlm:latest
```

### 2. Run Verification Smoke Test
```bash
PYTHONPATH=. python scripts/smoke_test.py
```

---

## 📊 Exploratory Data Analysis & Artifacts

Run the automated EDA generator to produce figures and CSV statistical summaries:
```bash
PYTHONPATH=. python scripts/run_eda.py
```

### Generated Figures (`figures/`)
- `disease_frequency.png`: Distribution of CheXpert medical pathology labels.
- `image_resolution_dist.png`: Scatter distribution of image dimensions (Width x Height).
- `report_length_dist.png`: Histogram & KDE of word counts across radiology reports.
- `vocabulary_dist.png`: Top 20 frequent medical vocabulary tokens.
- `patient_stats.png`: Number of studies per patient (patient-level statistics).

### CSV Summaries (`results/`)
- `dataset_summary_stats.csv`: Patient, study, and image counts.
- `disease_frequency_stats.csv`: Exact pathology frequency counts.
- `report_length_stats.csv`: Mean, median, standard deviation of report word counts.
- `vocabulary_stats.csv`: Token frequency counts.
- `verification_report.json`: Phase 1 smoke test status report.

---

## 🛡️ Reproducibility & Strict Verification

Phase 1 Verification Status: **`PASSED`** (`results/verification_report.json`)
- **Hardware & Device**: Auto-detects NVIDIA CUDA, Apple Silicon MPS, or CPU.
- **Patient Leakage Prevention**: `preprocessing/patient_splitter.py` guarantees 0 patient overlap across train/val/test splits.
- **Seed Determinism**: Fixed random seed (`42`) locked across PyTorch, NumPy, Python, and CUDA backends.

---

## 🛑 Current Phase Status: PHASE 1 COMPLETED
- Model training, evaluation, retrieval indexing, and label guidance modules are halted per methodology until Phase 2 instructions.
