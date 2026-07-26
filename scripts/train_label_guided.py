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
from utils.mlflow_tracker import MLflowTracker
from scripts.generate_mock_data import generate_mock_dataset
from preprocessing.vlm_tokenizer import VLMTokenizerWrapper
from preprocessing.patient_splitter import patient_level_split
from datasets.vlm_dataset import VLMDataset
from datasets.data_loader import create_dataloader
from models.label_guided_vlm import LabelGuidedMedicalVLM
from retrieval.retriever import MultimodalRetriever
from training.trainer import BaselineVLMTrainer

logger = setup_logger("train_label_guided")


def main():
    logger.info("==================================================")
    logger.info("  TRAINING STRUCTURED LABEL GUIDANCE (SLG-RAG) VLM ")
    logger.info("==================================================")

    set_seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Active Compute Device: {device}")

    config = {}
    with open("configs/models.yaml", "r") as f:
        config["models"] = yaml.safe_load(f)
    with open("configs/training.yaml", "r") as f:
        config["training"] = yaml.safe_load(f)

    data_csv = "data/mock/mimic_cxr_mock.csv"
    if not os.path.exists(data_csv):
        data_csv = generate_mock_dataset(num_samples=150)

    df = pd.read_csv(data_csv)
    train_df, val_df, _ = patient_level_split(df, seed=42)

    tok_wrapper = VLMTokenizerWrapper(model_name="google/flan-t5-base")
    train_ds = VLMDataset(train_df, tok_wrapper)
    val_ds = VLMDataset(val_df, tok_wrapper)

    train_loader = create_dataloader(train_ds, batch_size=4, shuffle=True)
    val_loader = create_dataloader(val_ds, batch_size=4, shuffle=False)

    model = LabelGuidedMedicalVLM(
        vision_model_name="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        text_model_name="google/flan-t5-base",
        use_lora=True,
    ).to(device)

    retriever = MultimodalRetriever(dimension=512)
    retriever.build_index_from_dataset(model, train_loader, device=device)
    retriever.vector_index.save("retrieval/index_store")

    checkpoint_mgr = ThreeTierCheckpointManager(checkpoint_dir="checkpoints")
    mlflow_tracker = MLflowTracker(experiment_name="Phase4_SLG_RAG_BioMedCLIP_FLAN_T5")

    mlflow_tracker.start_run(
        run_name="SLG_RAG_Training_Run",
        params={
            "vision_encoder": "BioMedCLIP",
            "text_decoder": "FLAN-T5-Base",
            "label_guidance": "CheXbert_14_Conditions",
            "retrieval_index": "FAISS_IndexFlatIP",
            "device": str(device),
            "seed": 42,
        },
    )

    trainer = BaselineVLMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config["training"],
        device=device,
        checkpoint_manager=checkpoint_mgr,
        mlflow_tracker=mlflow_tracker,
    )

    trainer.run_training_pipeline()
    mlflow_tracker.end_run()

    torch.save(model.state_dict(), "checkpoints/label_guided_best_loss.pt")
    logger.info("Saved Label-Guided Model Weights to checkpoints/label_guided_best_loss.pt")


if __name__ == "__main__":
    main()
