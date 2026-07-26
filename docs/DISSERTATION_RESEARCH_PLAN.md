# Multi-Modal Medical Report Generation Dissertation Research Methodology

## Executive Summary
This document outlines the strict engineering methodology and architecture for the MSc Dissertation research project:
**MULTI-MODAL MEDICAL REPORT GENERATION USING VISION-LANGUAGE MODELS FOR AUTOMATED RADIOLOGY DOCUMENTATION**

## Research Objectives
1. **Automated Radiology Generation**: Generate coherent, clinically accurate radiology reports directly from chest X-ray images (MIMIC-CXR and IU Chest X-Ray).
2. **Reproducible Multimodal Pipeline**: Establish an end-to-end framework integrating computer vision encoders, language models, retrieval mechanisms, and clinical label guidance.

## Phased Implementation Roadmap
- **Phase 1 (Completed)**: Engineering Pipeline & Reproducible Research Environment. Structure repository, environment dependencies, data preprocessors, configuration system, EDA plots, and smoke test verification.
- **Phase 2 (Upcoming)**: Data Preprocessing & Patient-Level Splitting.
- **Phase 3 (Upcoming)**: Retrieval Module (FAISS embedding index).
- **Phase 4 (Upcoming)**: Label Guidance System (CheXbert / RadGraph integration).
- **Phase 5 (Upcoming)**: Vision-Language Model Fine-tuning.
- **Phase 6 (Upcoming)**: Natural Language & Clinical Efficacy Evaluation (BLEU, ROUGE, CIDER, CheXbert F1).

## Rules & Constraints
- No dataset substitutions.
- No model architecture modifications without prior specification.
- Zero patient data leakage across train/val/test splits (patient-level splitting mandatory).
