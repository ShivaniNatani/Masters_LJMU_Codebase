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
from models.label_guided_vlm import LabelGuidedMedicalVLM
from retrieval.retriever import MultimodalRetriever
from evaluation.nlg_metrics import compute_nlg_metrics
from evaluation.clinical_metrics import compute_clinical_metrics
from evaluation.copy_similarity import analyze_copy_vs_grounding

logger = setup_logger("evaluate_label_guided")


def run_slg_eval(
    model: LabelGuidedMedicalVLM,
    test_loader,
    tok_wrapper,
    retriever: MultimodalRetriever,
    device: torch.device,
    retrieval_mode: str,  # 'none' or 'similarity'
    use_slg: bool,
    top_k: int = 2,
):
    predictions = []
    references = []
    sample_logs = []

    for step, batch in enumerate(test_loader):
        images = batch["image"].to(device)
        raw_reports = batch["report_text"]
        pids = batch["patient_id"]
        sids = batch["study_id"] if "study_id" in batch else [f"S_{p}" for p in pids]

        with torch.no_grad():
            gen_ids, ret_contexts, slg_prompts = model.generate_slg_report(
                images,
                tokenizer_wrapper=tok_wrapper,
                report_texts=raw_reports,
                retriever=retriever,
                retrieval_mode=retrieval_mode,
                use_slg=use_slg,
                top_k=top_k,
                max_new_tokens=128,
                num_beams=2,
            )

        for i in range(len(gen_ids)):
            gen_text = tok_wrapper.decode_generated_ids(gen_ids[i])
            ref_text = raw_reports[i]
            pid = pids[i]
            sid = sids[i]

            ret_items = ret_contexts[i] if (ret_contexts and i < len(ret_contexts)) else []
            ret_reports = [r.get("report_text", "") for r in ret_items]
            ret_scores = [r.get("similarity_score", 0.0) for r in ret_items]

            predictions.append(gen_text)
            references.append(ref_text)

            sample_logs.append(
                {
                    "sample_id": len(predictions) - 1,
                    "patient_id": pid,
                    "study_id": sid,
                    "retrieval_mode": retrieval_mode,
                    "use_slg": use_slg,
                    "slg_prompt": slg_prompts[i],
                    "retrieved_similarity_scores": ret_scores,
                    "retrieved_reports": ret_reports,
                    "generated_report": gen_text,
                    "ground_truth_report": ref_text,
                }
            )

    tag = f"mode_{retrieval_mode}_slg_{use_slg}"
    nlg_metrics = compute_nlg_metrics(predictions, references)
    clinical_metrics = compute_clinical_metrics(
        predictions,
        references,
        raw_chexbert_path=f"results/raw_chexbert_labels_{tag}.json",
        raw_radgraph_path=f"results/raw_radgraph_entities_{tag}.json",
    )

    return {
        "predictions": predictions,
        "references": references,
        "sample_logs": sample_logs,
        "nlg_metrics": nlg_metrics,
        "clinical_metrics": clinical_metrics,
    }


def main():
    logger.info("==================================================")
    logger.info("  EVALUATING PHASE 4 SLG-RAG 4-WAY BENCHMARK      ")
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

    model = LabelGuidedMedicalVLM(
        vision_model_name="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        text_model_name="google/flan-t5-base",
        use_lora=True,
    ).to(device)

    ckpt_path = "checkpoints/label_guided_best_loss.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)

    retriever = MultimodalRetriever(dimension=512)
    if os.path.exists("retrieval/index_store/faiss.index"):
        retriever.vector_index.load("retrieval/index_store")
    else:
        retriever.build_index_from_dataset(model, train_loader, device=device)

    # 4-Way Evaluation Conditions
    logger.info("--- Condition 1: Baseline VLM (No RAG, No SLG) ---")
    res_baseline = run_slg_eval(model, test_loader, tok_wrapper, retriever, device, retrieval_mode="none", use_slg=False)

    logger.info("--- Condition 2: FAISS Similarity RAG (Top-K=2, No SLG) ---")
    res_rag = run_slg_eval(model, test_loader, tok_wrapper, retriever, device, retrieval_mode="similarity", use_slg=False, top_k=2)

    logger.info("--- Condition 3: Structured Label Guidance VLM (SLG Only, No RAG) ---")
    res_slg = run_slg_eval(model, test_loader, tok_wrapper, retriever, device, retrieval_mode="none", use_slg=True)

    logger.info("--- Condition 4: Combined SLG + FAISS RAG VLM ---")
    res_combined = run_slg_eval(model, test_loader, tok_wrapper, retriever, device, retrieval_mode="similarity", use_slg=True, top_k=2)

    # Copy similarity analysis for Combined model
    retrieved_texts = [item.get("retrieved_reports", []) for item in res_combined["sample_logs"]]
    copy_analysis = analyze_copy_vs_grounding(
        predictions=res_combined["predictions"],
        references=res_combined["references"],
        retrieved_reports=retrieved_texts,
        output_json_path="results/phase4_copy_similarity.json",
    )

    # Combined metrics JSON
    combined_metrics = {
        "Baseline_VLM": {**res_baseline["nlg_metrics"], **res_baseline["clinical_metrics"]},
        "FAISS_RAG_VLM": {**res_rag["nlg_metrics"], **res_rag["clinical_metrics"]},
        "SLG_Only_VLM": {**res_slg["nlg_metrics"], **res_slg["clinical_metrics"]},
        "Combined_SLG_RAG_VLM": {**res_combined["nlg_metrics"], **res_combined["clinical_metrics"]},
        "Copy_Similarity_Analysis": copy_analysis["Copy_Similarity_Metrics"],
    }

    with open("results/phase4_metrics.json", "w") as f:
        json.dump(combined_metrics, f, indent=2)

    logger.info("Saved results/phase4_metrics.json")

    # Export sample predictions CSV
    csv_records = []
    for i in range(len(res_combined["predictions"])):
        csv_records.append(
            {
                "patient_id": test_df.iloc[i]["patient_id"],
                "ground_truth_report": res_combined["references"][i],
                "baseline_vlm": res_baseline["predictions"][i],
                "faiss_rag_vlm": res_rag["predictions"][i],
                "slg_only_vlm": res_slg["predictions"][i],
                "combined_slg_rag_vlm": res_combined["predictions"][i],
                "slg_prompt": res_combined["sample_logs"][i]["slg_prompt"],
            }
        )

    preds_df = pd.DataFrame(csv_records)
    preds_df.to_csv("results/phase4_sample_predictions.csv", index=False)
    logger.info("Saved results/phase4_sample_predictions.csv")

    # Generate Markdown Table & Plot Figure
    conds = ["Baseline VLM", "FAISS RAG", "SLG-Only", "Combined SLG-RAG"]
    bleu4 = [combined_metrics["Baseline_VLM"]["BLEU_4"], combined_metrics["FAISS_RAG_VLM"]["BLEU_4"], combined_metrics["SLG_Only_VLM"]["BLEU_4"], combined_metrics["Combined_SLG_RAG_VLM"]["BLEU_4"]]
    rouge_l = [combined_metrics["Baseline_VLM"]["ROUGE_L"], combined_metrics["FAISS_RAG_VLM"]["ROUGE_L"], combined_metrics["SLG_Only_VLM"]["ROUGE_L"], combined_metrics["Combined_SLG_RAG_VLM"]["ROUGE_L"]]
    cider = [combined_metrics["Baseline_VLM"]["CIDEr"], combined_metrics["FAISS_RAG_VLM"]["CIDEr"], combined_metrics["SLG_Only_VLM"]["CIDEr"], combined_metrics["Combined_SLG_RAG_VLM"]["CIDEr"]]
    chexbert = [combined_metrics["Baseline_VLM"]["CheXbert_Micro_F1"], combined_metrics["FAISS_RAG_VLM"]["CheXbert_Micro_F1"], combined_metrics["SLG_Only_VLM"]["CheXbert_Micro_F1"], combined_metrics["Combined_SLG_RAG_VLM"]["CheXbert_Micro_F1"]]

    plt.figure(figsize=(10, 6))
    df_plot = pd.DataFrame({"BLEU-4": bleu4, "ROUGE-L": rouge_l, "CIDEr": cider, "CheXbert F1": chexbert}, index=conds)
    ax = df_plot.plot(kind="bar", figsize=(10, 6), width=0.7, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    plt.title("Phase 4: 4-Way Architectural Benchmark Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Metric Score", fontsize=12)
    plt.xlabel("Model Condition", fontsize=12)
    plt.xticks(rotation=0, fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(title="Metrics", fontsize=10)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points')

    plt.tight_layout()
    plt.savefig("figures/phase4_comparative_metrics.png", dpi=300)
    plt.close()
    logger.info("Saved figures/phase4_comparative_metrics.png")


if __name__ == "__main__":
    main()
