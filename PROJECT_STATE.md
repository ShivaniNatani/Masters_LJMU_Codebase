# PROJECT_STATE.md: Current Project State & Single Source of Truth

**Research Title**: Multi-Modal Medical Report Generation Using Vision-Language Models for Automated Radiology Documentation  
**Target Hardware / Device**: Apple Silicon M4 Pro (MPS PyTorch backend) / NVIDIA CUDA fallback  
**Tracking / Logging**: MLflow SQLite tracking (`logs/mlflow.db`) + Rotating execution logger (`logs/execution.log`)  
**Last State Synchronization**: July 26, 2026  

---

## 1. Current Architecture

The project implements a **Retrieval-Augmented Vision-Language Model (RAG VLM)** for chest X-ray radiology report generation, coupling a frozen domain-specific vision encoder with an auto-regressive text decoder via a trainable projection interface and low-rank adaptors.

- **Vision Encoder**: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` (BioMedCLIP ViT-B/16).
  - Spatial Patch Tokens: 196 tokens of 768 dimensions `(B, 196, 768)`.
  - Global Embedding: 512-dimensional projected L2-normalized vector `(B, 512)` for vector retrieval indexing.
  - Encoder Status: 100% frozen across all training stages.
- **Vision-Language Projection Interface**: 2-Layer MLP with GELU activation, Dropout ($0.1$), and LayerNorm mapping visual patch space `(B, 196, 768)` $\to$ `(B, 196, 768)`.
- **Text Decoder**: `google/flan-t5-base` (~250M parameters, 12 encoder/decoder layers, hidden dimension $d_{\text{model}} = 768$).
- **Parameter-Efficient Fine-Tuning**: LoRA (PEFT) applied to T5 attention modules (`q`, `v`), rank $r=16$, alpha $\alpha=32$, dropout $0.05$. (Total trainable parameters: 2,952,192).
- **Multimodal Retrieval Module**: FAISS `IndexFlatIP` vector index stored at `retrieval/index_store/`, indexing 512-dim BioMedCLIP image embeddings with Cosine / Inner Product similarity search.
- **Multimodal Prompt Fusion**: Concatenates projected visual patch tokens `(B, 196, 768)` with RAG context prompt embeddings `(B, L_prompt, 768)` along the encoder sequence dimension.

---

## 2. Implemented Features

- **Patient-Level Data Splitting**: `preprocessing/patient_splitter.py` guarantees 0 patient leakage across Train (70%), Val (10%), and Test (20%) splits.
- **Modular Data Loaders**: PyTorch custom datasets (`datasets/base_dataset.py`, `datasets/vlm_dataset.py`, `datasets/mimic_cxr.py`, `datasets/iu_chest_xray.py`) with multi-worker prefetching (`num_workers=min(4, cpu_count - 1)`).
- **Text & Image Preprocessing Pipeline**: Section header extraction (Findings/Impression), word normalization, vocabulary building (`preprocessing/build_vocab.py`), PIL image normalization, and data augmentations.
- **Synthetic Mock Generator**: `scripts/generate_mock_data.py` producing 150 mock DICOM-style X-rays and medical reports for end-to-end local hardware and pipeline validation.
- **2-Stage Training Engine**: `training/trainer.py` handling Stage 1 (Projection Warmup, 5 epochs) and Stage 2 (Projection + LoRA Fine-Tuning, up to 30 epochs with early stopping).
- **3-Tier Checkpoint Manager**: `utils/checkpoint.py` tracking `latest`, `best_loss`, and `best_bleu4` states.
- **MLflow Tracking & Logging**: `utils/mlflow_tracker.py` using SQLite backend (`logs/mlflow.db`) and `utils/logger.py` with `RotatingFileHandler`.
- **FAISS Vector Indexing & Retrieval**: `retrieval/faiss_index.py` and `retrieval/retriever.py` supporting similarity search ($Top-K$) and random uniform retrieval controls.
- **Evaluation Engine**: NLG metrics (`evaluation/nlg_metrics.py`), CheXbert & RadGraph clinical metrics (`evaluation/clinical_metrics.py`), and Copy-vs-Grounding similarity engine (`evaluation/copy_similarity.py`).

---

## 3. Completed Phases

- **Phase 1: Engineering Foundation** (`PASSED & VERIFIED`)
  - Project directory structure, YAML configuration system, PyTorch DataLoaders, deterministic seed locking (`seed=42`), Apple MPS acceleration verification (`scripts/smoke_test.py`), and publication-grade EDA pipeline (`scripts/run_eda.py`).
- **Phase 2: Baseline VLM** (`COMPLETED & VALIDATED`)
  - BioMedCLIP + FLAN-T5-Base architecture constructed and trained over 2 stages. Early stopping at Stage 2 Epoch 27.
  - Baseline quantitative metrics logged: BLEU-4: `0.2580`, ROUGE-L: `0.3086`, CIDEr: `0.7353`, BERTScore: `0.8932`, CheXbert Micro-F1: `0.4780`, RadGraph Entity F1: `0.4467`.
  - Archived artifacts: `checkpoints/baseline_best_loss.pt`, `checkpoints/baseline_best_bleu4.pt`, `results/baseline_metrics.json`, `docs/PHASE2_BASELINE_REPORT.md`.
- **Phase 3: Retrieval-Augmented Generation VLM** (`COMPLETED & FROZEN`)
  - FAISS `IndexFlatIP` 512-dim visual embedding retriever built and saved (`retrieval/index_store/faiss.index`).
  - 3-Way Controlled Framework benchmarked: Baseline (BLEU-4: `0.3606`, CheXbert F1: `0.4810`), Random Control (BLEU-4: `0.2090`, CheXbert F1: `0.3349`), and FAISS Similarity RAG ($K=2$, BLEU-4: `0.2477`, CheXbert F1: `0.4017`).
  - Top-K Retrieval Depth Ablation completed: $K=1$ (BLEU-4: `0.2065`, Copy Overlap: **86.68%**), $K=2$ (BLEU-4: **`0.2882`**, Copy Overlap: **63.87%**), $K=3$ (BLEU-4: `0.2012`, Copy Overlap: **49.72%**).
  - Research Validation Finding: Semantic retrieval is significantly more useful than random context (+0.0387 BLEU-4 over control), but on this synthetic validation dataset raw RAG does not outperform the un-augmented baseline model due to distractor noise and retrieval copy reliance (63.87% word overlap). This directly motivates Phase 4 (Structured Label Guidance).
  - Deliverable artifacts generated: `results/phase3_metrics.json`, `results/phase3_metrics_table.md`, `results/phase3_topk_ablation.json`, `results/phase3_sample_predictions.csv`, `results/phase3_copy_similarity.json`, `results/phase3_retrieval_logs.json`, `figures/phase3_comparative_metrics.png`, `figures/phase3_copy_vs_grounding.png`, `figures/phase3_retrieval_similarity_dist.png`, `figures/phase3_topk_ablation.png`, and `docs/PHASE3_RAG_REPORT.md`.

- **Phase 4: Structured Label Guidance VLM** (`COMPLETED & FROZEN`)
  - CheXbert 14-condition clinical pathology vector encoder and prompt builder implemented (`label_guidance/label_encoder.py`, `prompt_formatter.py`).
  - `LabelGuidedMedicalVLM` architecture trained over 2 stages (`checkpoints/label_guided_best_loss.pt`). Early stopping at Stage 2 Epoch 26.
  - 4-Way Controlled Benchmark Matrix evaluated: Baseline (BLEU-4: `0.2855`), FAISS RAG (BLEU-4: `0.2345`), SLG-Only (BLEU-4: `0.2585`), and Combined SLG-RAG (BLEU-4: `0.2067`).
  - Research Validation Finding: SLG-Only outperforms raw FAISS RAG (+0.0240 BLEU-4, +0.0684 CIDEr), proving structured clinical condition vectors provide cleaner steering signals than un-filtered text context.
  - Deliverable artifacts generated: `results/phase4_metrics.json`, `results/phase4_metrics_table.md`, `results/phase4_sample_predictions.csv`, `results/phase4_copy_similarity.json`, `figures/phase4_comparative_metrics.png`, and `docs/PHASE4_LABEL_GUIDANCE_REPORT.md`.

---

## 4. Pending Work

- **Phase 5: Real Dataset Ingestion & Scaled Training**: Ingest full MIMIC-CXR-JPG and IU Chest X-Ray datasets for large-scale training and benchmarking once PhysioNet credentials are live.
- **Phase 6: Final Dissertation Benchmarking & Error Taxonomy**: Comprehensive multi-model benchmark matrix, clinical error taxonomy, and final dissertation report.

---

## 5. Dataset Pipeline

```
[Raw X-Rays / Reports] 
        │
        ▼
[preprocessing/patient_splitter.py] ──> Train (70%) / Val (10%) / Test (20%) [0 Patient Leakage]
        │
        ├──> [preprocessing/build_vocab.py] ──> Train-split only vocabulary (vocab.json)
        ├──> [preprocessing/image_preprocessing.py] ──> Resize (224x224), Normalization
        └──> [preprocessing/text_preprocessing.py] ──> Clean text, Section Extraction
        │
        ▼
[datasets/vlm_dataset.py & data_loader.py] ──> PyTorch Tensors (image, prompt_ids, labels)
```

---

## 6. Model Pipeline

```
Image (3, 224, 224) ──> BioMedCLIP Vision Encoder (Frozen)
                             │
                             ├──> Global CLS Embed (512-dim) ──> FAISS Retriever ──> Top-K Context
                             └──> 196 Patch Embeds (196, 768)
                                       │
                                       ▼
                             Projection Layer (768 -> 768)
                                       │
                                       ▼
                              Visual Prefix (196, 768)
                                       │
                                       ├─────── Concat ───────┐
                                       │                      │
                             Prompt Embeddings (L_prompt, 768)
                                       │
                                       ▼
                       Combined Encoder Inputs (196+L_prompt, 768)
                                       │
                                       ▼
                         FLAN-T5-Base Decoder + LoRA
                                       │
                                       ▼
                           Generated Radiology Report
```

---

## 7. Evaluation Pipeline

- **Natural Language Generation (NLG) Metrics**: BLEU-1, BLEU-2, BLEU-3, BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L, METEOR, CIDEr, BERTScore F1.
- **Clinical Efficacy Metrics**:
  - CheXbert 14-condition extraction (Cardiomegaly, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Support Devices, No Finding, etc.), Micro-Precision, Micro-Recall, Micro-F1.
  - RadGraph entity overlap (Anatomy vs Observation entity F1).
- **Copy-vs-Grounding Similarity Engine**: Quantifies word overlap and ROUGE-L between generated reports and retrieved context vs. ground truth to detect verbatim copying.

---

## 8. Current Experiments

The codebase implements a **3-Way Controlled Experimental Framework** to evaluate RAG effectiveness:
1. **Condition 1 (Baseline VLM)**: Image patch tokens only, no retrieved context.
2. **Condition 2 (Random Retrieval Control)**: Retranslates image with $K=2$ uniform random database reports to isolate prompt length expansion effects.
3. **Condition 3 (FAISS Similarity RAG VLM)**: Retranslates image with $Top-K=2$ nearest-neighbor database reports indexed by BioMedCLIP visual embeddings.

---

## 9. Missing Components

1. `label_guidance/` module implementation (Phase 4).
2. Saved outputs for Phase 3 evaluation (`results/phase3_metrics.json`, `results/phase3_sample_predictions.csv`, `results/phase3_copy_similarity.json`).
3. Formal dissertation report documentation for Phase 3 (`docs/PHASE3_RAG_REPORT.md`).

---

## 10. Technical Debt & Codebase Discrepancies

- **`configs/models.yaml` Drift**: YAML configuration specifies `resnet50` and `gpt2`, whereas actual implemented models in `models/baseline_vlm.py` and `models/rag_vlm.py` hardcode `BioMedCLIP` and `flan-t5-base`.
- **`sacrebleu` API Keyword Deprecation**: `sacrebleu.corpus_bleu` in `evaluation/nlg_metrics.py` passes `max_ngram_order` which raises a `TypeError` in recent `sacrebleu` versions, triggering a fallback block.
- **NLTK `wordnet` Resource Lookup**: `evaluation/nlg_metrics.py` relies on `nltk.download('wordnet')`, which triggers a fallback warning when offline.
- **Rule-Based Clinical Extraction**: `evaluation/clinical_metrics.py` uses regex rules for CheXbert and RadGraph rather than loading full neural CheXbert/RadGraph model checkpoints for synthetic pipeline validation.
