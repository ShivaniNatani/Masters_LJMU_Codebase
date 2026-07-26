# Phase 3 Quantitative Metrics Comparison Table

| Metric Category | Evaluation Metric | Baseline VLM (No Context) | Random Retrieval Control | FAISS Similarity RAG VLM | RAG Delta vs Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **NLG Metrics** | **BLEU-1** | 0.5282 | 0.4039 | 0.4413 | -0.0869 |
| | **BLEU-2** | 0.4435 | 0.3113 | 0.3498 | -0.0937 |
| | **BLEU-3** | 0.3918 | 0.2477 | 0.2859 | -0.1059 |
| | **BLEU-4** | **0.3606** | 0.2090 | **0.2477** | -0.1129 |
| | **ROUGE-1** | 0.4768 | 0.3041 | 0.3717 | -0.1051 |
| | **ROUGE-2** | 0.3073 | 0.1329 | 0.1936 | -0.1137 |
| | **ROUGE-L** | **0.4265** | 0.2736 | **0.3323** | -0.0942 |
| | **METEOR** | 0.3739 | 0.2104 | 0.2685 | -0.1054 |
| | **CIDEr** | **1.0277** | 0.5957 | **0.7059** | -0.3218 |
| | **BERTScore F1** | **0.9082** | 0.8886 | **0.8956** | -0.0126 |
| **Clinical Efficacy** | **CheXbert Micro-F1** | **0.4810** | 0.3349 | **0.4017** | -0.0793 |
| | **CheXbert Precision** | 0.4453 | 0.3396 | 0.3833 | -0.0620 |
| | **CheXbert Recall** | 0.5229 | 0.3303 | 0.4220 | -0.1009 |
| | **RadGraph Entity F1** | **0.3330** | 0.2595 | **0.1874** | -0.1456 |
