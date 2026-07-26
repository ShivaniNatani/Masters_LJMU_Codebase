# Phase 2 Implementation Plan: Baseline Vision-Language Model for Automated Radiology Documentation (Finalized)

## 1. Executive Architecture Overview
The goal of Phase 2 is to build, train, and evaluate the **Baseline Vision-Language Model (VLM)** for medical report generation without introducing auxiliary components (such as RAG or Label Guidance).

The baseline architecture couples a pre-trained **BioMedCLIP Vision Encoder** with a **FLAN-T5-Base Language Decoder** (optimized for Apple Silicon M4 Pro local development) via an explicitly designed visual projection interface.

```
┌─────────────────────────┐
│ Input Chest X-Ray Image │ (3, 224, 224)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ BioMedCLIP Vision ViT   │ [Frozen across all stages]
└────────────┬────────────┘
             │ Patch Embeddings (exclude CLS token)
             ▼
┌─────────────────────────┐
│ Vision Patch Embeddings │ (Batch, 196, 768)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Linear/MLP Projection   │ [Trainable: Stage 1 & Stage 2]
└────────────┬────────────┘
             │ Mapped to FLAN-T5 Encoder Dim (768 for Base)
             ▼
┌─────────────────────────┐
│ Visual Prefix Sequence  │ (Batch, 196, 768)
└────────────┬────────────┘
             │ Concatenated with Text Prompt Embeddings
             ▼
┌─────────────────────────┐
│ FLAN-T5-Base Encoder    │ (Batch, 196 + Prompt_Len, 768)
└────────────┬────────────┘
             │ Cross-Attention
             ▼
┌─────────────────────────┐
│ FLAN-T5-Base Decoder    │ [LoRA Fine-tuned: Stage 2]
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Generated Report Tokens │ (Batch, Target_Seq_Len, 32128)
└────────────┬────────────┘
```

---

## 2. Model Architecture & Vision-Language Interface Details

### 2.1 Vision Encoder: BioMedCLIP
- **Backbone**: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` (or OpenCLIP BioMedCLIP ViT-B/16).
- **Patch Extraction**: Input size $224 \times 224$, patch size $16 \times 16 \implies 14 \times 14 = 196$ spatial patches.
- **Embedding Dimension**: 768 dimensions per patch.
- **CLS Handling**: The global CLS embedding is discarded to preserve full 2D spatial context.
- **Encoder Status**: Frozen across both Stage 1 and Stage 2 training.

### 2.2 Projection Module (Vision-Language Interface)
- **Input Dimension**: `(Batch, 196, 768)`
- **Output Dimension**: `(Batch, 196, 768)` (matches FLAN-T5-Base hidden dimension $d_{\text{model}} = 768$).
- **Architecture**:
  - Two-layer MLP with GELU activation, LayerNorm, and Dropout (0.1).
  - Formula: $H_{\text{vis}} = \text{LayerNorm}(W_2 \cdot \text{GELU}(W_1 \cdot X_{\text{patches}} + b_1) + b_2)$.

### 2.3 Language Decoder Choice: FLAN-T5-Base
- **Backbone**: `google/flan-t5-base` ($d_{\text{model}} = 768$, 12 encoder layers, 12 decoder layers, ~250M parameters).
- **M4 Pro Justification**: FLAN-T5-Base provides an optimal trade-off for local Apple Silicon (M4 Pro GPU / MPS backend) execution:
  1. Fits entirely within unified memory while maintaining fast token generation throughput.
  2. Hidden dimension $d_{\text{model}} = 768$ aligns directly with BioMedCLIP ViT patch dimension (768), reducing parameter overhead in the projection layer.
  3. **Configurability**: Model name (e.g., `google/flan-t5-base` vs. `google/flan-t5-large`) is fully configurable via `configs/models.yaml`.
- **Prompt Structure**: `"Generate a detailed radiology findings and impression report for this chest X-ray image:"`
- **LoRA Configuration**:
  - Target Modules: `q`, `v` (in self-attention & cross-attention layers).
  - Rank ($r$): 16 (default; configurable via YAML)
  - Alpha ($\alpha$): 32 (default; configurable via YAML)
  - Dropout: 0.05

---

## 3. Detailed Tensor Dimension Flow

| Component | Tensor Name | Shape / Dimensions | Notes |
| :--- | :--- | :--- | :--- |
| Input Image | `x_img` | `(B, 3, 224, 224)` | Standard normalized X-ray tensor |
| ViT Layer Output | `vit_out` | `(B, 197, 768)` | 1 CLS token + 196 spatial patch tokens |
| Patch Extraction | `x_patches` | `(B, 196, 768)` | Sliced `vit_out[:, 1:, :]` |
| Projection Layer | `v_proj` | `(B, 196, 768)` | Projected to FLAN-T5 encoder dimension |
| Prompt Text Tokens | `p_ids` | `(B, L_prompt)` | Tokenized FLAN-T5 prompt sequence |
| Prompt Embeddings | `p_embeds` | `(B, L_prompt, 768)` | Looked up via `flan_t5.shared(p_ids)` |
| Encoder Input | `enc_inputs` | `(B, 196 + L_prompt, 768)`| Concatenation `[v_proj, p_embeds]` along seq dim |
| Target Text Tokens | `y_ids` | `(B, L_target)` | Tokenized radiology report findings & impression |
| Decoder Output Logits| `logits` | `(B, L_target, 32128)` | Vocabulary logits for Cross-Entropy loss |

---

## 4. Two-Stage Training Protocol

*(Note: All learning rates, epoch counts, batch sizes, and thresholds are initial default values configurable via `configs/training.yaml`)*

```
Stage 1: Projection Warmup (Feature Alignment)
├── Freeze: BioMedCLIP Vision Encoder (100% Frozen)
├── Freeze: FLAN-T5 Decoder (100% Frozen)
├── Train:  Projection Module ONLY (Linear / MLP)
├── Epochs: Up to 5 Epochs (with Early Stopping enabled)
├── LR:     Initial default 1e-3 (Configurable in configs/training.yaml)
└── Goal:   Align ViT visual feature space with T5 textual embedding space

Stage 2: Projection + LoRA Fine-Tuning (Frozen Vision Encoder)
├── Freeze: BioMedCLIP Vision Encoder (100% Frozen)
├── Train:  Projection Module (Trainable)
├── Train:  FLAN-T5 Decoder via LoRA Adaptors (r=16, alpha=32)
├── Epochs: Up to 30 Epochs (with Early Stopping enabled)
├── LR:     Initial default 2e-4 (Configurable in configs/training.yaml)
└── Goal:   Fine-tune multimodal report generation capabilities
```

---

## 5. Three-Tier Checkpointing Strategy

All three checkpoint states are explicitly retained in `checkpoints/`:
1. **`baseline_latest.pt`**: Saved at the end of every epoch (allows seamless training resumption).
2. **`baseline_best_loss.pt`**: Checkpoint achieving the lowest validation loss.
3. **`baseline_best_bleu4.pt`**: Checkpoint achieving the highest validation BLEU-4 score.

---

## 6. Experiment Tracking, Reproducibility & Implementation Manifest

### 6.1 MLflow Tracking
- MLflow logs all parameters, loss metrics, BLEU-4 metrics, learning rate schedules, and model checkpoints.

### 6.2 Reproducibility Guarantees
- Random seeds locked across Python, NumPy, PyTorch, CUDA, and MPS (`seeds = [42, 101, 2024]`).
- Environment freezing (`pip freeze`) and Git commit hash embedded into run metadata.

### 6.3 `IMPLEMENTATION_MANIFEST.md` Deliverable
At completion, an `IMPLEMENTATION_MANIFEST.md` document will be written to `docs/` summarizing:
- Repository & Model version
- Dataset version & Dataset Summary metrics (Train/Val/Test image count, patient count, avg & max report length, vocab size, top 20 findings)
- Configurations used & Random seeds
- Training duration & Hardware resources (M4 Pro GPU / MPS utilization)
- Software versions & Output locations
- Known limitations & future phase directives

---

## 7. Evaluation Suite & Raw Clinical Extractions

- **NLG Metrics**: BLEU (1-4), ROUGE (1, 2, L), METEOR, CIDEr, BERTScore.
- **Clinical Efficacy**: CheXbert F1 & RadGraph F1.
- **Raw Clinical Labels JSON**: Sample-level extractions saved to `results/raw_chexbert_labels.json` and `results/raw_radgraph_entities.json` for detailed false-positive/false-negative error analysis.

---

## 8. Checkpoint Schedule & Implementation Milestones

| Step | Milestone Task | Deliverables / Output Files | Estimated Effort |
| :--- | :--- | :--- | :--- |
| **M1** | **Baseline Model Architecture & Interface Code** | `models/baseline_vlm.py`, `models/projection.py` | Step 1 |
| **M2** | **Tokenizer & Dataset Pipeline** | `datasets/vlm_dataset.py`, `preprocessing/vlm_tokenizer.py` | Step 2 |
| **M3** | **Training Engine, 3-Tier Checkpointer & MLflow System** | `training/trainer.py`, `training/losses.py`, `utils/checkpoint.py`, `utils/mlflow_tracker.py` | Step 3 |
| **M4** | **Stage 1 Warmup & Stage 2 Fine-Tuning Execution** | `scripts/train_baseline.py`, `logs/baseline_training.log`, `checkpoints/baseline_best_loss.pt`, `checkpoints/baseline_best_bleu4.pt`, `checkpoints/baseline_latest.pt` | Step 4 |
| **M5** | **Evaluation Suite Engine & Raw Label Extraction** | `evaluation/nlg_metrics.py`, `evaluation/clinical_metrics.py`, `scripts/evaluate_baseline.py`, `results/raw_chexbert_labels.json` | Step 5 |
| **M6** | **Results, Learning Curves, Manifest & Final Report** | `figures/baseline_learning_curves.png`, `results/baseline_metrics.json`, `results/baseline_sample_predictions.csv`, `docs/IMPLEMENTATION_MANIFEST.md`, `docs/PHASE2_BASELINE_REPORT.md` | Step 6 |
