# PHASE 3 IMPLEMENTATION PLAN: RETRIEVAL-AUGMENTED GENERATION (RAG) VLM

**Project**: MULTI-MODAL MEDICAL REPORT GENERATION USING VISION-LANGUAGE MODELS FOR AUTOMATED RADIOLOGY DOCUMENTATION  
**Phase**: Phase 3 - Multimodal FAISS Vector Retrieval & Retrieval-Augmented Report Generation  
**Target Hardware**: Apple Silicon M4 Pro (MPS PyTorch backend)  

---

## 1. Executive Summary & Objective
Phase 3 builds upon the frozen BioMedCLIP Vision Encoder and FLAN-T5-Base VLM baseline established in Phase 2. The primary objective is to implement a **Multimodal Retrieval-Augmented Generation (RAG)** pipeline using **FAISS vector indexing** to ground radiology report generation in clinical precedent.

To rigorously address the dissertation's research questions, Phase 3 establishes a 3-way controlled comparative framework:
1. **Baseline VLM** (Phase 2 - No Retrieval Context)
2. **Random Retrieval VLM Control** (Retrieves $K$ random reports to test context-length expansion effects)
3. **FAISS Similarity Retrieval VLM** (Retrieves Top-$K$ nearest neighbors via BioMedCLIP embedding distance)

---

## 2. System Architecture & Module Map

```
+-----------------------------------------------------------------------------------+
|                                    PHASE 3 RAG VLM                                |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Query Image (B, 3, 224, 224) ---> Frozen BioMedCLIP Vision Encoder ------------+ |
|                                              |                                  | |
|                                              v                                  | |
|                                   Visual Embeddings (B, 512)                    | |
|                                              |                                  | |
|                                              v                                  | |
|                        +------------------------------------------+             | |
|                        |          FAISS Vector Retriever          |             | |
|                        | (IndexFlatIP / Cosine Embedding Search)  |             | |
|                        +------------------------------------------+             | |
|                                              |                                  | |
|                                              v                                  | |
|                                Top-K Retrieved Reports (text)                   | |
|                                              |                                  | |
|                                              v                                  | |
|                         Context Prompt Formatting & Tokenization                 | |
|                           ("Context: [Report 1] Image Findings:")               | |
|                                              |                                  | |
|                                              v                                  | |
|                                Visual Projection Module (768->768)              | |
|                                              |                                  | |
|                                              v                                  | |
|                              FLAN-T5-Base Decoder + LoRA (250M)                 | |
|                                              |                                  | |
|                                              v                                  | |
|                                 Generated Radiology Report                      | |
+-----------------------------------------------------------------------------------+
```

---

## 3. Module Specifications & Implementation Steps

### Milestone 1: FAISS Vector Indexing & Multimodal Retriever (`retrieval/faiss_index.py`, `retrieval/retriever.py`)
- **FAISS Indexing**: `FAISSVectorIndex` uses normalized inner-product distance (`IndexFlatIP`) over 512-dim BioMedCLIP image embeddings extracted from the training set.
- **Retrieval Modes**:
  - `similarity`: Performs $K$-nearest neighbor similarity search ($K=2$ default).
  - `random`: Uniformly samples $K$ reports from the index as an experimental control.
- **Outputs**: Top-$K$ text reports, similarity scores (cosine metric), and source study/image identifiers.

### Milestone 2: RAG VLM Model & Prompt Formatting (`models/rag_vlm.py`)
- **Prompt Structure**:
  `"Background Context: [Retrieved Report 1] [Retrieved Report 2] Task: Write a radiology report for the chest X-ray image."`
- **Encoder Fusion**: Concatenates projected visual patch embeddings `(B, 196, 768)` with tokenized RAG prompt embeddings.
- **Decoder**: FLAN-T5-Base with LoRA ($r=16, \alpha=32$).

### Milestone 3: Detailed Logging & Copy-Similarity Engine (`evaluation/copy_similarity.py`)
To determine whether RAG improves performance via clinical grounding vs. verbatim text copying:
- **Copy Similarity**: Computes ROUGE-L, BLEU-4, and word-level overlap between the **Retrieved Report** and the **Generated Report**.
- **Grounding Similarity**: Computes ROUGE-L and BLEU-4 between the **Generated Report** and the **Ground Truth Report**.
- **Sample Retrieval Logs**: Saves per-sample JSON (`results/phase3_retrieval_logs.json`) detailing top-$k$ retrieved reports, study IDs, similarity scores, generated text, and ground truth text.

### Milestone 4: Training & Evaluation Execution (`scripts/train_rag.py`, `scripts/evaluate_rag.py`)
- Executed on MPS (Apple Silicon M4 Pro).
- Evaluates and logs results across all three conditions (Baseline vs. Random Retrieval vs. FAISS Retrieval).

---

## 4. Controlled Experiment Matrix

| Condition | Retrieval Strategy | Context Provided | Research Question Addressed |
| :--- | :--- | :--- | :--- |
| **Baseline** | None | Image Patch Tokens Only | What is the un-augmented VLM capacity? |
| **Random Control** | Random Uniform Selection | $K=2$ Random Database Reports | Does adding text context improve generation regardless of relevance? |
| **FAISS RAG** | BioMedCLIP Nearest Neighbor | Top-$K=2$ Similar Database Reports | Does semantically relevant visual-clinical context improve diagnostic accuracy? |

---

## 5. Artifacts & Deliverables
Upon completion, the following deliverables will be produced:
1. `models/rag_vlm.py` & `retrieval/faiss_index.py` & `retrieval/retriever.py`
2. `evaluation/copy_similarity.py`
3. `results/phase3_metrics.json` (Full 3-way comparative metric breakdown)
4. `results/phase3_sample_predictions.csv`
5. `results/phase3_retrieval_logs.json` (Sample-level retrieval provenance)
6. `results/phase3_copy_similarity.json` (Copying vs. Grounding analysis)
7. `docs/PHASE3_RAG_REPORT.md` (Formal technical dissertation phase report)
