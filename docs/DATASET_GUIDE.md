# Dataset Integration & Preprocessing Guide

## Datasets Supported

### 1. MIMIC-CXR-JPG (Primary Dataset)
- **Source**: PhysioNet (`https://physionet.org/content/mimic-cxr-jpg/2.0.0/`)
- **Format**: JPG chest radiology images paired with free-text radiology reports.
- **Access Control**: Credentialed Access required (CITI training completion on PhysioNet).
- **Download Automation**: `python scripts/download_mimic_cxr.py <username> <password>`

### 2. Indiana University Chest X-ray (Secondary Dataset)
- **Source**: NLM OpenI (`https://openi.nlm.nih.gov/`)
- **Format**: 7,470 Chest X-Ray DICOM images + XML radiology reports.
- **Access Control**: Open Public Access.
- **Download Automation**: `python scripts/download_iu_cxr.py`

## Synthetic Mock Dataset (Offline / Smoke Test Mode)
To ensure immediate code reproducibility without downloading 500GB raw DICOM data:
- Execute `python scripts/generate_mock_data.py`
- Creates 150 synthetic chest X-ray images + structured radiology reports at `data/mock/`
