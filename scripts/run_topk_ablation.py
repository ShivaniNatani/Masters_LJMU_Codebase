import os
import sys
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils.seed import set_seed
from utils.logger import setup_logger
from preprocessing.vlm_tokenizer import VLMTokenizerWrapper
from preprocessing.patient_splitter import patient_level_split
from datasets.vlm_dataset import VLMDataset
from datasets.data_loader import create_dataloader
from models.rag_vlm import RAGMedicalVLM
from retrieval.retriever import MultimodalRetriever
from scripts.evaluate_rag import run_evaluation_for_mode
from evaluation.copy_similarity import analyze_copy_vs_grounding

logger = setup_logger("run_topk_ablation")


def main():
    logger.info("==================================================")
    logger.info("  EXECUTING PHASE 3 TOP-K RETRIEVAL DEPTH ABLATION")
    logger.info("==================================================")

    set_seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    data_csv = "data/mock/mimic_cxr_mock.csv"
    df = pd.read_csv(data_csv)
    train_df, _, test_df = patient_level_split(df, seed=42)

    tok_wrapper = VLMTokenizerWrapper(model_name="google/flan-t5-base")
    train_ds = VLMDataset(train_df, tok_wrapper)
    test_ds = VLMDataset(test_df, tok_wrapper)

    train_loader = create_dataloader(train_ds, batch_size=4, shuffle=False)
    test_loader = create_dataloader(test_ds, batch_size=4, shuffle=False)

    model = RAGMedicalVLM(
        vision_model_name="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        text_model_name="google/flan-t5-base",
        use_lora=True,
    ).to(device)

    rag_ckpt = "checkpoints/rag_best_loss.pt"
    if os.path.exists(rag_ckpt):
        ckpt = torch.load(rag_ckpt, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)

    retriever = MultimodalRetriever(dimension=512)
    if os.path.exists("retrieval/index_store/faiss.index"):
        retriever.vector_index.load("retrieval/index_store")
    else:
        retriever.build_index_from_dataset(model, train_loader, device=device)

    ablation_results = {}

    for k in [1, 2, 3]:
        logger.info(f"--- Running FAISS Top-K = {k} Retrieval Evaluation ---")
        res_k = run_evaluation_for_mode(
            model, test_loader, tok_wrapper, retriever, device, retrieval_mode="similarity", top_k=k
        )
        retrieved_texts = [item.get("retrieved_reports", []) for item in res_k["sample_logs"]]
        
        copy_res = analyze_copy_vs_grounding(
            predictions=res_k["predictions"],
            references=res_k["references"],
            retrieved_reports=retrieved_texts,
            output_json_path=f"results/phase3_copy_similarity_topk_{k}.json",
        )

        ablation_results[f"TopK_{k}"] = {
            "NLG_Metrics": res_k["nlg_metrics"],
            "Clinical_Efficacy": res_k["clinical_metrics"],
            "Copy_Similarity": copy_res["Copy_Similarity_Metrics"],
        }

    with open("results/phase3_topk_ablation.json", "w") as f:
        json.dump(ablation_results, f, indent=2)

    logger.info("Saved results/phase3_topk_ablation.json")

    # Generate Top-K Ablation Figure
    k_vals = ["Top-K=1", "Top-K=2", "Top-K=3"]
    bleu4 = [ablation_results[f"TopK_{k}"]["NLG_Metrics"]["BLEU_4"] for k in [1, 2, 3]]
    rouge_l = [ablation_results[f"TopK_{k}"]["NLG_Metrics"]["ROUGE_L"] for k in [1, 2, 3]]
    copy_overlap = [ablation_results[f"TopK_{k}"]["Copy_Similarity"]["Mean_Copy_Word_Overlap"] for k in [1, 2, 3]]
    chexbert_f1 = [ablation_results[f"TopK_{k}"]["Clinical_Efficacy"]["CheXbert_Micro_F1"] for k in [1, 2, 3]]

    df_fig = pd.DataFrame(
        {
            "BLEU-4": bleu4,
            "ROUGE-L": rouge_l,
            "CheXbert F1": chexbert_f1,
            "Copy Word Overlap": copy_overlap,
        },
        index=k_vals,
    )

    plt.figure(figsize=(9, 5))
    ax = df_fig.plot(kind="line", marker="o", linewidth=2.5, figsize=(9, 5))
    plt.title("Effect of Retrieval Depth (Top-K) on Model Metrics & Copy Behavior", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Retrieval Depth (Top-K)", fontsize=11)
    plt.ylabel("Score / Ratio", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("figures/phase3_topk_ablation.png", dpi=300)
    plt.close()
    logger.info("Saved figures/phase3_topk_ablation.png")


if __name__ == "__main__":
    main()
