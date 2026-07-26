import os
import sys
import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Add repository root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.seed import set_seed
from utils.logger import setup_logger
from utils.checkpoint import ThreeTierCheckpointManager
from utils.mlflow_tracker import MLflowTracker
from scripts.generate_mock_data import generate_mock_dataset
from preprocessing.vlm_tokenizer import VLMTokenizerWrapper
from preprocessing.patient_splitter import patient_level_split
from datasets.vlm_dataset import VLMDataset
from datasets.data_loader import create_dataloader
from models.baseline_vlm import BaselineMedicalVLM
from training.trainer import BaselineVLMTrainer

logger = setup_logger("train_baseline")


def main():
    logger.info("==================================================")
    logger.info("  EXCLUSIVELY TRAINING BASELINE VISION-LANGUAGE MODEL ")
    logger.info("==================================================")

    # 1. Lock Deterministic Random Seeds
    set_seed(42)

    # 2. Hardware Device Selection
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Active Compute Device: {device}")

    # 3. Load Configurations
    config = {}
    with open("configs/models.yaml", "r") as f:
        config["models"] = yaml.safe_load(f)
    with open("configs/training.yaml", "r") as f:
        config["training"] = yaml.safe_load(f)

    # 4. Prepare Dataset Split
    data_csv = "data/mock/mimic_cxr_mock.csv"
    if not os.path.exists(data_csv):
        data_csv = generate_mock_dataset(num_samples=150)

    df = pd.read_csv(data_csv)
    train_df, val_df, test_df = patient_level_split(df, seed=42)

    # 5. Tokenizer & Datasets
    tok_wrapper = VLMTokenizerWrapper(model_name="google/flan-t5-base")
    train_ds = VLMDataset(train_df, tok_wrapper)
    val_ds = VLMDataset(val_df, tok_wrapper)

    train_loader = create_dataloader(train_ds, batch_size=4, shuffle=True)
    val_loader = create_dataloader(val_ds, batch_size=4, shuffle=False)

    # 6. Model & Checkpointer & MLflow
    model = BaselineMedicalVLM(
        vision_model_name="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        text_model_name="google/flan-t5-base",
        use_lora=True,
    )

    checkpoint_mgr = ThreeTierCheckpointManager(checkpoint_dir="checkpoints")
    mlflow_tracker = MLflowTracker(experiment_name="Baseline_BioMedCLIP_FLAN_T5")

    mlflow_tracker.start_run(
        run_name="Baseline_Training_Run",
        params={
            "vision_encoder": "BioMedCLIP",
            "text_decoder": "FLAN-T5-Base",
            "device": str(device),
            "seed": 42,
        },
    )

    # 7. Execute 2-Stage Training
    trainer = BaselineVLMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config["training"],
        device=device,
        checkpoint_manager=checkpoint_mgr,
        mlflow_tracker=mlflow_tracker,
    )

    train_results = trainer.run_training_pipeline()

    # 8. Plot & Save Learning Curves
    history = train_results["history"]
    if history["train_loss"]:
        plt.figure(figsize=(9, 5))
        plt.plot(history["train_loss"], label="Train Loss", marker="o", color="#1f77b4")
        plt.plot(history["val_loss"], label="Val Loss", marker="s", color="#ff7f0e")
        plt.title("Baseline VLM Training & Validation Loss Curves", fontsize=14, pad=12)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Cross-Entropy Loss", fontsize=12)
        plt.legend()
        plt.tight_layout()
        curve_path = "figures/baseline_learning_curves.png"
        plt.savefig(curve_path, dpi=300)
        plt.close()
        logger.info(f"Saved Learning Curve plot to {curve_path}")
        mlflow_tracker.log_artifact(curve_path)

    mlflow_tracker.end_run()
    logger.info("Baseline Training Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()
