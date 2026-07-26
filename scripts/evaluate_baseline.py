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
from models.baseline_vlm import BaselineMedicalVLM
from evaluation.nlg_metrics import compute_nlg_metrics
from evaluation.clinical_metrics import compute_clinical_metrics

logger = setup_logger("evaluate_baseline")


def main():
    logger.info("==================================================")
    logger.info("      EVALUATING BASELINE VISION-LANGUAGE MODEL    ")
    logger.info("==================================================")

    set_seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 1. Load Test Dataset Split
    data_csv = "data/mock/mimic_cxr_mock.csv"
    if not os.path.exists(data_csv):
        from scripts.generate_mock_data import generate_mock_dataset
        data_csv = generate_mock_dataset(num_samples=150)

    df = pd.read_csv(data_csv)
    _, _, test_df = patient_level_split(df, seed=42)

    tok_wrapper = VLMTokenizerWrapper(model_name="google/flan-t5-base")
    test_ds = VLMDataset(test_df, tok_wrapper)
    test_loader = create_dataloader(test_ds, batch_size=4, shuffle=False)

    # 2. Instantiate Baseline VLM Model
    model = BaselineMedicalVLM(
        vision_model_name="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        text_model_name="google/flan-t5-base",
        use_lora=True,
    ).to(device)

    # Load best checkpoint if available
    best_ckpt = "checkpoints/baseline_best_loss.pt"
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded Best Model Checkpoint from {best_ckpt}")

    model.eval()

    predictions = []
    references = []
    patient_ids = []
    sample_records = []

    logger.info("Running Beam Search Generation over Test Dataset...")
    for step, batch in enumerate(test_loader):
        images = batch["image"].to(device)
        prompt_ids = batch["prompt_ids"].to(device)
        prompt_mask = batch["prompt_mask"].to(device)
        raw_reports = batch["report_text"]
        pids = batch["patient_id"]

        with torch.no_grad():
            gen_ids = model.generate_report(
                images, prompt_ids, prompt_mask=prompt_mask, max_new_tokens=128, num_beams=2
            )

        for i in range(len(gen_ids)):
            gen_text = tok_wrapper.decode_generated_ids(gen_ids[i])
            ref_text = raw_reports[i]
            pid = pids[i]

            predictions.append(gen_text)
            references.append(ref_text)
            patient_ids.append(pid)

            sample_records.append(
                {
                    "patient_id": pid,
                    "ground_truth_report": ref_text,
                    "generated_baseline_report": gen_text,
                }
            )

    # 3. Export Sample Generated Reports CSV
    preds_df = pd.DataFrame(sample_records)
    preds_csv = "results/baseline_sample_predictions.csv"
    preds_df.to_csv(preds_csv, index=False)
    logger.info(f"Exported sample predictions to {preds_csv}")

    # 4. Compute NLG Metrics
    nlg_results = compute_nlg_metrics(predictions, references)

    # 5. Compute Clinical Efficacy Metrics & Save Raw Extractions
    clinical_results = compute_clinical_metrics(
        predictions,
        references,
        raw_chexbert_path="results/raw_chexbert_labels.json",
        raw_radgraph_path="results/raw_radgraph_entities.json",
    )

    # 6. Save Overall Evaluation Metrics JSON
    all_metrics = {"NLG_Metrics": nlg_results, "Clinical_Efficacy_Metrics": clinical_results}
    metrics_json = "results/baseline_metrics.json"
    with open(metrics_json, "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("==================================================")
    logger.info("      BASELINE EVALUATION COMPLETE                ")
    logger.info(f"Results saved to {metrics_json}")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
