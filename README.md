# Multi-Modal Medical Report Generation Using Vision-Language Models for Automated Radiology Documentation

MSc dissertation source code
Shivani Natani · Student ID 1187203
MSc Artificial Intelligence and Machine Learning, Liverpool John Moores University (via upGrad)
July 2026

---

## What this repository is

This is the complete implementation behind the dissertation of the same title. It builds a
vision-language pipeline that generates radiology reports from chest X-rays, and evaluates two
reliability mechanisms on top of it:

- **Retrieval-augmented generation (RAG)** - a FAISS index over training-set image embeddings
  supplies the two most similar prior cases and their reports as context.
- **Structured label guidance (SLG)** - CheXpert's fourteen-condition schema is formatted into the
  prompt so the decoder is told which findings to address.

The two mechanisms are crossed in a four-way factorial ablation (Baseline, +RAG, +SLG, Combined),
run identically on two datasets, with paired bootstrap significance testing over every comparison.

**The headline result is a negative one, and it is reported as such.** Verification of the
implementation uncovered four defects that aggregate metrics had concealed. The dissertation
reports them, quantifies their effect, and treats the diagnosis as the contribution. The evidence
is in this repository - see `docs/DIAGNOSTIC_FINDINGS.md` and `results/untrained_control/`.

---

## Results at a glance

|                                  | MIMIC-CXR                    | IU Chest X-Ray               |
|----------------------------------|------------------------------|------------------------------|
| Test cases                       | 2,922                        | 1,180                        |
| Retrieval effect on BLEU-4       | not significant (p = 0.756)  | not significant (p = 0.415)  |
| Label guidance effect on BLEU-4  | +0.0115                      | +0.0157                      |
| Reports labelled "No Finding"    | 60.3%                        | 92.0%                        |

Full numbers: `results/mimic_real_4way_metrics.json`, `results/iu_real_4way_metrics.json`,
`results/significance_tests.json`, `results/clinical_error_taxonomy.csv`.

---

## The four defects

Documented in full in `docs/DIAGNOSTIC_FINDINGS.md`. Summarised here so anyone reading the code
knows what they are looking at.

| #  | Defect                                                                       | Effect                                                            |
|----|------------------------------------------------------------------------------|-------------------------------------------------------------------|
| D1 | Encoder loaded was general-domain CLIP, not the specified biomedical model    | Visual features were never domain-adapted                         |
| D2 | Label guidance drew labels from the *reference report*, not from the image    | An oracle, not a prediction; the measured gain is an upper bound   |
| D3 | Template collapse - near-identical output text regardless of input            | Output diversity counts in `results/` make this visible            |
| D4 | Validation tracked `1/(1+loss)` rather than generated-text BLEU               | Collapse was invisible during training                            |

The archived random-initialisation control in `results/untrained_control/` is what established that
the originally reported figures could be reproduced without a trained model. It is kept deliberately.

---

## Layout

```
configs/           YAML configuration: datasets, models, training, evaluation, logging
data/mock/         150 synthetic image-report pairs, so the pipeline runs without PhysioNet access
datasets/          PyTorch Dataset classes for MIMIC-CXR and IU Chest X-Ray
preprocessing/     Patient-level splitter, image and text preprocessing, vocabulary builder
models/            baseline_vlm.py, rag_vlm.py, label_guided_vlm.py, projection.py, mock_vlm.py
retrieval/         FAISS index construction and the retriever
label_guidance/    CheXpert label encoding and prompt formatting
training/          Two-stage trainer
evaluation/        NLG metrics, clinical metrics, copy-similarity detection
scripts/           All executable entry points (see below)
results/           Metrics, significance tests, error taxonomy, EDA statistics
figures/           The twelve figures reproduced in the dissertation
docs/              Diagnostic findings, dataset guide, environment setup
tests/             Unit tests
```

---

## Reproducing the work

### 1. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Or `docker build -t medical_vlm:latest .`

### 2. Smoke test on mock data (no credentials needed)

```bash
PYTHONPATH=. python scripts/smoke_test.py
```

### 3. Obtain the datasets

MIMIC-CXR-JPG needs a credentialed PhysioNet account and may not be redistributed; IU Chest X-Ray
is open. See `docs/DATASET_GUIDE.md`, then:

```bash
PYTHONPATH=. python scripts/download_mimic_cxr.py
PYTHONPATH=. python scripts/download_iu_cxr.py
```

### 4. Build the retrieval indices

```bash
PYTHONPATH=. python scripts/build_faiss_index_mimic.py
PYTHONPATH=. python scripts/build_faiss_index_iu.py
```

### 5. Train

```bash
PYTHONPATH=. python scripts/train_mimic_real.py
PYTHONPATH=. python scripts/train_iu_real.py
```

### 6. Evaluate

The evaluation scripts **require** a trained checkpoint and abort if one is not found. This is
deliberate: the absence of that check is what allowed defect D4 to go unnoticed.

```bash
EVAL_CKPT=checkpoints/mimic_real/baseline_best_loss.pt \
  PYTHONPATH=. python scripts/evaluate_mimic_real.py
EVAL_CKPT=checkpoints/iu_real/baseline_best_loss.pt \
  PYTHONPATH=. python scripts/evaluate_iu_real.py
```

---

## What is not in this repository, and why

| Excluded                              | Size    | Regenerate with                             |
|---------------------------------------|---------|---------------------------------------------|
| `checkpoints/*.pt`                    | ~15 GB  | `scripts/train_*.py`                        |
| `retrieval/index_store/*.index`       | ~30 MB  | `scripts/build_faiss_index_*.py`            |
| `data/raw/`, `data/processed/`        | ~1.1 GB | `docs/DATASET_GUIDE.md`                     |
| Per-sample prediction CSVs            | ~23 MB  | `scripts/evaluate_*.py`                     |
| Dissertation PDF and Turnitin report  | -       | Submitted directly to upGrad-LJMU; university rules prohibit placing them on cloud services |

---

## Reproducibility

- **Seed 42** locked across PyTorch, NumPy, Python and CUDA backends (`utils/seed.py`).
- **Patient-level splitting** with zero overlap between train, validation and test
  (`preprocessing/patient_splitter.py`), verified rather than assumed.
- **Retrieval indices built over the training partition only**, so no test image can retrieve itself.
- **Significance testing** by paired bootstrap, 10,000 resamples, seed 42
  (`results/significance_tests.json`).
- Hardware auto-detection for CUDA, Apple Silicon MPS, or CPU (`utils/env_check.py`).

---

## Licence and data terms

Code is provided for academic assessment. MIMIC-CXR-JPG is used under the PhysioNet Credentialed
Health Data Use Agreement and is not redistributed here. IU Chest X-Ray is openly available from
the U.S. National Library of Medicine.
