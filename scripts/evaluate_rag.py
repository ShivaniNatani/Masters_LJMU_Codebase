import os
import sys
import json
import torch
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.seed import set_seed
from utils.logger import setup_logger
from preprocessing.vlm_tokenizer import VLMTokenizerWrapper
from preprocessing.patient_splitter import patient_level_split
from datasets.vlm_dataset import VLMDataset
from datasets.data_loader import create_dataloader
from models.rag_vlm import RAGMedicalVLM
from retrieval.retriever import MultimodalRetriever
from evaluation.nlg_metrics import compute_nlg_metrics
from evaluation.clinical_metrics import compute_clinical_metrics
from evaluation.copy_similarity import analyze_copy_vs_grounding

logger = setup_logger("evaluate_rag")


def run_evaluation_for_mode(
    model: RAGMedicalVLM,
    test_loader,
    tok_wrapper,
    retriever: MultimodalRetriever,
    device: torch.device,
    retrieval_mode: str,  # 'none', 'random', or 'similarity'
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
            gen_ids, ret_contexts = model.generate_rag_report(
                images,
                tokenizer_wrapper=tok_wrapper,
                retriever=retriever,
                retrieval_mode=retrieval_mode,
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
            ret_study_ids = [r.get("study_id", "UNKNOWN") for r in ret_items]

            predictions.append(gen_text)
            references.append(ref_text)

            sample_logs.append(
                {
                    "sample_id": len(predictions) - 1,
                    "patient_id": pid,
                    "study_id": sid,
                    "retrieval_mode": retrieval_mode,
                    "retrieved_study_ids": ret_study_ids,
                    "retrieval_similarity_scores": ret_scores,
                    "retrieved_reports": ret_reports,
                    "generated_report": gen_text,
                    "ground_truth_report": ref_text,
                }
            )

    nlg_metrics = compute_nlg_metrics(predictions, references)
    clinical_metrics = compute_clinical_metrics(
        predictions,
        references,
        raw_chexbert_path=f"results/raw_chexbert_labels_{retrieval_mode}.json",
        raw_radgraph_path=f"results/raw_radgraph_entities_{retrieval_mode}.json",
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
    logger.info("  EVALUATING PHASE 3 RAG VLM & CONTROL EXPERIMENTS ")
    logger.info("==================================================")

    set_seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load test split dataset
    data_csv = "data/mock/mimic_cxr_mock.csv"
    if not os.path.exists(data_csv):
        from scripts.generate_mock_data import generate_mock_dataset
        data_csv = generate_mock_dataset(num_samples=150)

    df = pd.read_csv(data_csv)
    train_df, _, test_df = patient_level_split(df, seed=42)

    tok_wrapper = VLMTokenizerWrapper(model_name="google/flan-t5-base")
    train_ds = VLMDataset(train_df, tok_wrapper)
    test_ds = VLMDataset(test_df, tok_wrapper)

    train_loader = create_dataloader(train_ds, batch_size=4, shuffle=False)
    test_loader = create_dataloader(test_ds, batch_size=4, shuffle=False)

    # Instantiate Model
    model = RAGMedicalVLM(
        vision_model_name="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        text_model_name="google/flan-t5-base",
        use_lora=True,
    ).to(device)

    # Load weights
    rag_ckpt = "checkpoints/rag_best_loss.pt"
    if not os.path.exists(rag_ckpt):
        rag_ckpt = "checkpoints/baseline_best_loss.pt"
    if os.path.exists(rag_ckpt):
        ckpt = torch.load(rag_ckpt, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        logger.info(f"Loaded Model Checkpoint from {rag_ckpt}")

    # Build / Load Retriever
    retriever = MultimodalRetriever(dimension=512)
    if os.path.exists("retrieval/index_store/faiss.index"):
        retriever.vector_index.load("retrieval/index_store")
    else:
        retriever.build_index_from_dataset(model, train_loader, device=device)

    # Run 3 Controlled Experiments
    logger.info("--- Running Condition 1: Baseline (No Retrieval Context) ---")
    res_baseline = run_evaluation_for_mode(model, test_loader, tok_wrapper, retriever, device, retrieval_mode="none")

    logger.info("--- Running Condition 2: Random Retrieval Control (Uniform Random Database Context) ---")
    res_random = run_evaluation_for_mode(model, test_loader, tok_wrapper, retriever, device, retrieval_mode="random")

    logger.info("--- Running Condition 3: FAISS Top-K Similarity Retrieval VLM (RAG) ---")
    res_faiss = run_evaluation_for_mode(model, test_loader, tok_wrapper, retriever, device, retrieval_mode="similarity")

    # 1. Save Detailed Per-Sample Retrieval Logs
    with open("results/phase3_retrieval_logs.json", "w") as f:
        json.dump(res_faiss["sample_logs"], f, indent=2)
    logger.info("Exported per-sample retrieval logs to results/phase3_retrieval_logs.json")

    # 2. Copy-Similarity & Grounding Analysis
    retrieved_texts_faiss = [item.get("retrieved_reports", []) for item in res_faiss["sample_logs"]]
    copy_analysis = analyze_copy_vs_grounding(
        predictions=res_faiss["predictions"],
        references=res_faiss["references"],
        retrieved_reports=retrieved_texts_faiss,
        output_json_path="results/phase3_copy_similarity.json",
    )

    # 3. Export Combined Predictions CSV
    csv_records = []
    for i in range(len(res_faiss["predictions"])):
        csv_records.append(
            {
                "patient_id": test_df.iloc[i]["patient_id"],
                "ground_truth_report": res_faiss["references"][i],
                "baseline_no_retrieval": res_baseline["predictions"][i],
                "random_retrieval_control": res_random["predictions"][i],
                "faiss_similarity_rag": res_faiss["predictions"][i],
                "faiss_retrieved_context_1": retrieved_texts_faiss[i][0] if len(retrieved_texts_faiss[i]) > 0 else "",
                "faiss_retrieved_context_2": retrieved_texts_faiss[i][1] if len(retrieved_texts_faiss[i]) > 1 else "",
            }
        )

    preds_df = pd.DataFrame(csv_records)
    preds_df.to_csv("results/phase3_sample_predictions.csv", index=False)
    logger.info("Exported 3-way prediction comparison to results/phase3_sample_predictions.csv")

    # 4. Export Combined Metrics JSON
    combined_metrics = {
        "Baseline_No_Retrieval": {**res_baseline["nlg_metrics"], **res_baseline["clinical_metrics"]},
        "Random_Retrieval_Control": {**res_random["nlg_metrics"], **res_random["clinical_metrics"]},
        "FAISS_Similarity_RAG": {**res_faiss["nlg_metrics"], **res_faiss["clinical_metrics"]},
        "Copy_Similarity_Analysis": copy_analysis["Copy_Similarity_Metrics"],
    }

    with open("results/phase3_metrics.json", "w") as f:
        json.dump(combined_metrics, f, indent=2)

    logger.info("==================================================")
    logger.info("     PHASE 3 EVALUATION & CONTROL COMPLETE        ")
    logger.info("Results saved to results/phase3_metrics.json")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
