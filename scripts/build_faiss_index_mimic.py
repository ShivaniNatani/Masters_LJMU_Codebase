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

logger = setup_logger("build_faiss_index_mimic")


def build_mimic_faiss_index(
    splits_csv: str = "data/processed/mimic_cxr/mimic_cxr_splits.csv",
    output_dir: str = "retrieval/index_store",
    batch_size: int = 32,
):
    """
    Builds a FAISS vector index using BioMedCLIP visual embeddings for the MIMIC-CXR training split.
    """
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "mimic_cxr_faiss.index")
    meta_path = os.path.join(output_dir, "mimic_cxr_metadata.json")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device} for MIMIC-CXR FAISS embedding extraction.")

    df = pd.read_csv(splits_csv)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    logger.info(f"Loaded {len(train_df)} training samples from {splits_csv}")

    model = BaselineMedicalVLM()
    model.to(device)
    model.eval()

    dataset = BaseMedicalDataset(train_df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    faiss_index = FAISSVectorIndex(dimension=512)

    embeddings_list = []
    metadata_list = []

    logger.info("Extracting BioMedCLIP visual embeddings for MIMIC FAISS index...")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Building MIMIC FAISS Index"):
            images = batch["image"].to(device)
            v_features = model.extract_patch_embeddings(images)  # (B, 196, 768)
            global_embeds = v_features.mean(dim=1)  # (B, 768)
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

    import faiss
    faiss.write_index(faiss_index.index, index_path)
    import json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)

    logger.info(f"Successfully built & saved MIMIC FAISS index ({faiss_index.index.ntotal} vectors) to {index_path}")


if __name__ == "__main__":
    build_mimic_faiss_index()
