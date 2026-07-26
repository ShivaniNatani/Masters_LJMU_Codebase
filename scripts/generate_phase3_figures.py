import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.logger import setup_logger

logger = setup_logger("generate_phase3_figures")

def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load metrics
    with open("results/phase3_metrics.json", "r") as f:
        metrics = json.load(f)

    # 1. Generate Comparative Metrics Bar Plot
    models = ["Baseline (No Retrieval)", "Random Control", "FAISS Similarity RAG"]
    metric_keys = ["BLEU_4", "ROUGE_L", "CIDEr", "BERTScore_F1", "CheXbert_Micro_F1"]
    metric_names = ["BLEU-4", "ROUGE-L", "CIDEr", "BERTScore", "CheXbert F1"]

    data = {
        "Baseline (No Retrieval)": [metrics["Baseline_No_Retrieval"][k] for k in metric_keys],
        "Random Control": [metrics["Random_Retrieval_Control"][k] for k in metric_keys],
        "FAISS Similarity RAG": [metrics["FAISS_Similarity_RAG"][k] for k in metric_keys],
    }

    df_plot = pd.DataFrame(data, index=metric_names)

    plt.figure(figsize=(10, 6))
    ax = df_plot.plot(kind="bar", figsize=(10, 6), width=0.7, color=["#2b5c8f", "#d95f02", "#2ca02c"])
    plt.title("3-Way Controlled Framework: Quantitative Evaluation Metrics", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Metric Score", fontsize=12)
    plt.xlabel("Evaluation Metric", fontsize=12)
    plt.xticks(rotation=0, fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(title="Experimental Condition", fontsize=10)

    # Annotate bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, xytext=(0, 2),
                        textcoords='offset points')

    plt.tight_layout()
    fig1_path = "figures/phase3_comparative_metrics.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig1_path}")

    # 2. Copy vs Grounding Scatter Analysis Plot
    with open("results/phase3_copy_similarity.json", "r") as f:
        copy_data = json.load(f)

    samples = copy_data["Sample_Details"]
    copy_rouge = [s["copy_rouge_l"] for s in samples]
    ground_rouge = [s["grounding_rouge_l"] for s in samples]

    plt.figure(figsize=(8, 6))
    plt.scatter(copy_rouge, ground_rouge, color="#8c564b", alpha=0.8, edgecolors="k", s=70)
    plt.axline((0, 0), slope=1, color="red", linestyle="--", label="1:1 Parity Line")
    plt.title("Verbatim Copying vs Clinical Grounding Analysis", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Copy ROUGE-L (Generated vs Retrieved Context)", fontsize=12)
    plt.ylabel("Grounding ROUGE-L (Generated vs Ground Truth)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    fig2_path = "figures/phase3_copy_vs_grounding.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig2_path}")

    # 3. Retrieval Similarity Score Distribution
    with open("results/phase3_retrieval_logs.json", "r") as f:
        ret_logs = json.load(f)

    sim_scores = []
    for item in ret_logs:
        sim_scores.extend(item.get("retrieval_similarity_scores", []))

    plt.figure(figsize=(8, 5))
    sns.histplot(sim_scores, kde=True, color="#1f77b4", bins=10)
    plt.title("FAISS Cosine Similarity Score Distribution (Top-K Retrieval)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Cosine Similarity Score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig3_path = "figures/phase3_retrieval_similarity_dist.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig3_path}")

    # 4. Generate Markdown Metrics Table
    md_table = """# Phase 3 Quantitative Metrics Comparison Table

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
"""
    with open("results/phase3_metrics_table.md", "w") as f:
        f.write(md_table)
    logger.info("Saved results/phase3_metrics_table.md")

if __name__ == "__main__":
    main()
