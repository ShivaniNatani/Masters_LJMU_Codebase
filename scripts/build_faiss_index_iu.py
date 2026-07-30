import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.baseline_vlm import BaselineMedicalVLM
from retrieval.faiss_index import FAISSVectorIndex
from datasets.base_dataset import BaseMedicalDataset
from utils.logger import setup_logger

logger = setup_logger("build_faiss_index_iu")


def build_iu_faiss_index(
    splits_csv: str = "data/processed/iu_chest_xray/iu_cxr_splits.csv",
    output_dir: str = "retrieval/index_store",
    batch_size: int = 16,
    device_name: str = None,
):
    """
    Builds a FAISS vector index using BioMedCLIP global visual embeddings for the IU Chest X-Ray training split.
    """
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "iu_cxr_faiss.index")
    meta_path = os.path.join(output_dir, "iu_cxr_metadata.json")

    device = torch.device(
        device_name if device_name else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    logger.info(f"Using device: {device} for FAISS embedding extraction.")

    # Read training split
    df = pd.read_csv(splits_csv)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    logger.info(f"Loaded {len(train_df)} training samples from {splits_csv}")

    # Load Baseline Model to use BioMedCLIP Vision Encoder
    model = BaselineMedicalVLM()
    model.to(device)
    model.eval()

    # DataLoader
    dataset = BaseMedicalDataset(train_df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    faiss_index = FAISSVectorIndex(dimension=512)

    embeddings_list = []
    metadata_list = []

    logger.info("Extracting BioMedCLIP visual embeddings for FAISS index...")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Building FAISS Index"):
            images = batch["image"].to(device)
            # Forward pass through vision encoder
            # Extract 196 patch embeddings (B, 196, 768)
            v_features = model.extract_patch_embeddings(images)  # (B, 196, 768)
            global_embeds = v_features.mean(dim=1)  # (B, 768)
            # Reduce or slice to 512 for index dimension matching
            global_512 = global_embeds[:, :512].cpu().numpy().astype(np.float32)

            for idx in range(len(images)):
                sample_meta = {
                    "patient_id": batch["patient_id"][idx],
                    "study_id": batch["study_id"][idx],
                    "dicom_id": batch["dicom_id"][idx],
                    "report_text": batch["report_text"][idx],
                    "labels": batch["labels"][idx],
                }
                embeddings_list.append(global_512[idx])
                metadata_list.append(sample_meta)

    embeddings_array = np.vstack(embeddings_list)
    faiss_index.add_embeddings(embeddings_array, metadata_list)

    # Save FAISS Index & Metadata
    import faiss
    faiss.write_index(faiss_index.index, index_path)
    import json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)

    logger.info(f"Successfully built & saved FAISS index ({faiss_index.index.ntotal} vectors) to {index_path}")


if __name__ == "__main__":
    build_iu_faiss_index()
