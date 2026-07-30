import os
import sys
import json
import torch
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils.seed import set_seed
from utils.logger import setup_logger
from preprocessing.vlm_tokenizer import VLMTokenizerWrapper
from datasets.vlm_dataset import VLMDataset
from datasets.data_loader import create_dataloader
from models.label_guided_vlm import LabelGuidedMedicalVLM
from retrieval.retriever import MultimodalRetriever
from evaluation.nlg_metrics import compute_nlg_metrics
from evaluation.clinical_metrics import compute_clinical_metrics

logger = setup_logger("evaluate_mimic_real")


def run_condition_eval(
    model: LabelGuidedMedicalVLM,
    test_loader,
    tok_wrapper,
    retriever: MultimodalRetriever,
    device: torch.device,
    retrieval_mode: str,
    use_slg: bool,
    top_k: int = 2,
    max_samples: int = 10**9,
):
    predictions = []
    references = []
    sample_logs = []

    count = 0
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

        if (step + 1) % 10 == 0:
            logger.info(f"[{retrieval_mode} | SLG={use_slg}] Processed {count + len(gen_ids)} samples...")

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

            count += 1
            if count >= max_samples:
                break
        if count >= max_samples:
            break

    tag = f"mimic_mode_{retrieval_mode}_slg_{use_slg}"
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
    logger.info("  EVALUATING PRIMARY DATASET: KAGGLE MIMIC-CXR    ")
    logger.info("==================================================")

    set_seed(42)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")

    splits_csv = "data/processed/mimic_cxr/mimic_cxr_splits.csv"
    if not os.path.exists(splits_csv):
        logger.error(f"Processed file missing: {splits_csv}")
        return

    df = pd.read_csv(splits_csv)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    logger.info(f"Loaded MIMIC-CXR Split: {len(train_df)} train, {len(test_df)} test samples.")

    tok_wrapper = VLMTokenizerWrapper(model_name="google/flan-t5-base")
    test_ds = VLMDataset(test_df, tok_wrapper)
    test_loader = create_dataloader(test_ds, batch_size=16, shuffle=False)

    model = LabelGuidedMedicalVLM(
        vision_model_name="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        text_model_name="google/flan-t5-base",
        use_lora=True,
    ).to(device)

    # --- FIX: load the fine-tuned checkpoint (previously evaluated an untrained model) ---
    ckpt_path = os.environ.get("EVAL_CKPT", "checkpoints/mimic_real/baseline_best_loss.pt")
    if not os.path.exists(ckpt_path):
        logger.error(f"ABORT: trained checkpoint not found at {ckpt_path}. "
                     f"Run training to completion before evaluating.")
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded trained checkpoint: {ckpt_path}")
    logger.info(f"  missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
    if len(missing) > 50:
        logger.warning("Large number of missing keys - verify checkpoint matches architecture.")
    model.eval()

    retriever = MultimodalRetriever(dimension=512)
    faiss_index_path = "retrieval/index_store/mimic_cxr_faiss.index"
    meta_path = "retrieval/index_store/mimic_cxr_metadata.json"

    if os.path.exists(faiss_index_path) and os.path.exists(meta_path):
        import faiss
        import json
        retriever.vector_index.index = faiss.read_index(faiss_index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            retriever.vector_index.metadata = json.load(f)
        logger.info(f"Loaded pre-built MIMIC FAISS index ({retriever.vector_index.index.ntotal} vectors)")

    eval_conditions = [
        {"name": "Baseline VLM (No RAG, No SLG)", "retrieval_mode": "none", "use_slg": False},
        {"name": "FAISS RAG VLM (Top-K=2)", "retrieval_mode": "similarity", "use_slg": False},
        {"name": "Structured Label Guidance (SLG-Only)", "retrieval_mode": "none", "use_slg": True},
        {"name": "Combined System (SLG + FAISS RAG)", "retrieval_mode": "similarity", "use_slg": True},
    ]

    all_results = {}
    metrics_summary = []

    for cond in eval_conditions:
        name = cond["name"]
        ret_mode = cond["retrieval_mode"]
        slg_flag = cond["use_slg"]

        logger.info(f"--- Running Evaluation Condition: {name} ---")
        res = run_condition_eval(
            model=model,
            test_loader=test_loader,
            tok_wrapper=tok_wrapper,
            retriever=retriever,
            device=device,
            retrieval_mode=ret_mode,
            use_slg=slg_flag,
            top_k=2,
            max_samples=int(os.environ.get("EVAL_MAX_SAMPLES", 10**9)),
        )

        all_results[name] = res

        row = {"Model Condition": name}
        row.update(res["nlg_metrics"])
        row.update(res["clinical_metrics"])
        metrics_summary.append(row)

    os.makedirs("results", exist_ok=True)
    summary_df = pd.DataFrame(metrics_summary)

    json_path = "results/mimic_real_4way_metrics.json"
    md_path = "results/mimic_real_metrics_table.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    markdown_table = summary_df.to_markdown(index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Kaggle MIMIC-CXR Primary Dataset 4-Way Benchmark Evaluation Matrix\n\n")
        f.write(markdown_table)
        f.write("\n")

    # --- FIX: persist per-sample predictions for error taxonomy + bootstrap testing ---
    pred_rows = []
    for cond_name, res in all_results.items():
        for log in res["sample_logs"]:
            row = dict(log)
            row["model_condition"] = cond_name
            pred_rows.append(row)
    pd.DataFrame(pred_rows).to_csv("results/mimic_real_sample_predictions.csv", index=False)
    logger.info(f"Saved {len(pred_rows)} per-sample predictions to results/mimic_real_sample_predictions.csv")

    logger.info("==================================================")
    logger.info("  MIMIC-CXR EVALUATION COMPLETE!")
    logger.info(f"  Results saved to {md_path} and {json_path}")
    logger.info("==================================================")
    print("\n" + markdown_table + "\n")


if __name__ == "__main__":
    main()
