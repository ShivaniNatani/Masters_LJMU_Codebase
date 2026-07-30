import os
import sys
import yaml
import torch
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils.seed import set_seed
from utils.logger import setup_logger
from utils.checkpoint import ThreeTierCheckpointManager
from preprocessing.vlm_tokenizer import VLMTokenizerWrapper
from datasets.vlm_dataset import VLMDataset
from datasets.data_loader import create_dataloader
from models.label_guided_vlm import LabelGuidedMedicalVLM
from training.trainer import BaselineVLMTrainer

logger = setup_logger("train_iu_real")


def train_iu_real():
    logger.info("==================================================")
    logger.info("   TRAINING SLG-RAG VLM ON REAL IU CHEST X-RAY    ")
    logger.info("==================================================")

    set_seed(42)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Active Compute Device: {device}")

    config = {}
    with open("configs/models.yaml", "r") as f:
        config["models"] = yaml.safe_load(f)
    with open("configs/training.yaml", "r") as f:
        config["training"] = yaml.safe_load(f)

    # Load Real Processed IU Chest X-Ray Data
    data_csv = "data/processed/iu_chest_xray/iu_cxr_splits.csv"
    if not os.path.exists(data_csv):
        logger.error(f"Processed dataset missing at: {data_csv}")
        return

    df = pd.read_csv(data_csv)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    logger.info(f"Real Dataset Loaded: {len(train_df)} training samples, {len(val_df)} validation samples.")

    tok_wrapper = VLMTokenizerWrapper(model_name="google/flan-t5-base")
    train_ds = VLMDataset(train_df, tok_wrapper)
    val_ds = VLMDataset(val_df, tok_wrapper)

    train_loader = create_dataloader(train_ds, batch_size=4, shuffle=True)
    val_loader = create_dataloader(val_ds, batch_size=4, shuffle=False)

    model = LabelGuidedMedicalVLM(
        vision_model_name="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        text_model_name="google/flan-t5-base",
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
    )

    checkpoint_manager = ThreeTierCheckpointManager(checkpoint_dir="checkpoints/iu_real/")

    trainer = BaselineVLMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config["training"],
        device=device,
        checkpoint_manager=checkpoint_manager,
    )

    logger.info("Executing 2-Stage Training on Real IU Chest X-Ray Dataset...")
    results = trainer.run_training_pipeline()
    logger.info("IU Real Training Completed Successfully!")
    return results


if __name__ == "__main__":
    train_iu_real()
