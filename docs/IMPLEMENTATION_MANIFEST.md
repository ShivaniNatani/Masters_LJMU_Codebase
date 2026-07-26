# IMPLEMENTATION MANIFEST

**Project**: MULTI-MODAL MEDICAL REPORT GENERATION USING VISION-LANGUAGE MODELS FOR AUTOMATED RADIOLOGY DOCUMENTATION  
**Phase**: Phase 2 Baseline Vision-Language Model Implementation  
**Created**: July 26, 2026  

> [!NOTE]
> **Validation Purpose**: Phase 2 used synthetic datasets solely to verify end-to-end repository infrastructure, pipeline modularity, training convergence, and evaluation logging. Quantitative metrics do not represent real-world clinical accuracy until trained on MIMIC-CXR-JPG and IU Chest X-Ray datasets.

---

## 1. System & Model Versioning
- **Repository Phase**: Phase 2 Baseline VLM
- **Model Version Tag**: `Baseline_BioMedCLIP_FLAN_T5_v1.0`
- **Vision Backbone**: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` (OpenCLIP ViT-B/16, 768-dim patch embeddings)
- **Language Decoder**: `google/flan-t5-base` (250M parameters, $d_{\text{model}} = 768$, 12 encoder/decoder layers)
- **Vision-Language Projection**: 2-Layer MLP with GELU activation, LayerNorm, and Dropout (0.1) mapping `(B, 196, 768)` $\to$ `(B, 196, 768)`
- **LoRA Configuration**: Target modules `q`, `v`, rank $r=16$, alpha $\alpha=32$, dropout $0.05$ (2.95M trainable parameters)

---

## 2. Dataset Summary Statistics

| Metric | Value |
| :--- | :--- |
| **Total Images** | 150 |
| **Train Split Images** | 104 (52 unique patients) |
| **Validation Split Images** | 14 (7 unique patients) |
| **Test Split Images** | 32 (16 unique patients) |
| **Total Unique Patients** | 75 (0 patient leakage across splits) |
| **Average Report Length** | 28.4 words |
| **Maximum Report Length** | 64 words |
| **Vocabulary Size** | 1,000 unique medical terms |

### Top 20 Frequent Clinical Findings in Ground Truth Reports
1. `normal` (45 occurrences)
2. `clear` (38 occurrences)
3. `cardiac silhouette` (32 occurrences)
4. `cardiomegaly` (28 occurrences)
5. `lungs` (27 occurrences)
6. `pleural effusion` (24 occurrences)
7. `atelectasis` (22 occurrences)
8. `opacity` (19 occurrences)
9. `edema` (17 occurrences)
10. `pneumothorax` (15 occurrences)
11. `consolidation` (14 occurrences)
12. `pneumonia` (12 occurrences)
13. `support devices` (11 occurrences)
14. `mediastinum` (10 occurrences)
15. `hilar` (9 occurrences)
16. `pulmonary` (8 occurrences)
17. `vascular` (8 occurrences)
18. `interstitial` (7 occurrences)
19. `rib` (6 occurrences)
20. `diaphragm` (5 occurrences)

---

## 3. Configuration & Reproducibility Parameters
- **Configurations Used**:
  - `configs/models.yaml`
  - `configs/training.yaml`
  - `configs/datasets.yaml`
  - `configs/evaluation.yaml`
- **Random Seeds**: Locked globally (`42`, `101`, `2024`) across Python `random`, `numpy`, `torch.manual_seed`, and `torch.mps.manual_seed`.
- **Training Duration**: 505.71 seconds (~8.4 minutes) total over 2 stages.
- **Compute Hardware**: Apple Silicon M4 Pro GPU (MPS PyTorch backend).
- **Software Stack**:
  - Python: `3.14`
  - PyTorch: `2.13.0`
  - Transformers: `5.14.1`
  - PEFT: `0.19.1`
  - OpenCLIP: `3.3.0`
  - MLflow: `3.14.0`

---

## 4. Retained Checkpoint Deliverables & Output Locations

### Checkpoints (`checkpoints/`)
- `checkpoints/baseline_latest.pt`: Saved every epoch for training resumption.
- `checkpoints/baseline_best_loss.pt`: Lowest validation loss state (`0.0285`).
- `checkpoints/baseline_best_bleu4.pt`: Highest validation BLEU-4 state (`0.9723`).

### Figures (`figures/`)
- `figures/baseline_learning_curves.png`: Loss & BLEU-4 convergence curves across Stage 1 & Stage 2.

### Results (`results/`)
- `results/baseline_metrics.json`: Quantitative NLG & Clinical Efficacy benchmark metrics.
- `results/baseline_sample_predictions.csv`: Ground truth vs. generated baseline reports.
- `results/raw_chexbert_labels.json`: Sample-level 14 CheXpert condition extractions.
- `results/raw_radgraph_entities.json`: Sample-level RadGraph entity and relation extractions.

### Logs (`logs/`)
- `logs/mlflow.db`: SQLite database recording MLflow experiments, parameters, and metric logs.

---

## 5. Known Baseline Limitations & Next Phase Directives
1. **Hallucination Risk**: Baseline VLM without retrieval augmented context may occasionally generate generic or redundant phrases.
2. **Rare Disease Coverage**: Rare clinical findings have lower recall without structured label guidance.
3. **Future Directive**:
   - **Phase 3**: Implement Vector Retrieval Module (FAISS embedding index for RAG).
   - **Phase 4**: Implement Structured Label Guidance (CheXbert guidance integration).
