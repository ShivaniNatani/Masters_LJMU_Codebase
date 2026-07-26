import torch
import numpy as np
from typing import List, Dict, Any, Optional
from torch.utils.data import DataLoader
from retrieval.faiss_index import FAISSVectorIndex
from utils.logger import setup_logger

logger = setup_logger("multimodal_retriever")


class MultimodalRetriever:
    """
    Multimodal Retriever facilitating FAISS Cosine Similarity Search
    and Random Retrieval Control experiments.
    """

    def __init__(self, dimension: int = 512):
        self.vector_index = FAISSVectorIndex(dimension=dimension)

    def build_index_from_dataset(self, model, dataloader: DataLoader, device: torch.device):
        """
        Extracts global BioMedCLIP image embeddings across training dataset and builds FAISS index.
        """
        model.eval()
        all_embeddings = []
        all_metadata = []

        logger.info("Extracting BioMedCLIP image embeddings for FAISS index build...")
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(device)
                pids = batch["patient_id"]
                reports = batch["report_text"]

                # Extract global image embeddings via OpenCLIP visual trunk
                vis = model.vision_encoder
                if hasattr(vis, "conv1") and hasattr(vis, "transformer"):
                    x = vis.conv1(images)
                    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
                    class_token = vis.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
                    x = torch.cat([class_token, x], dim=1)
                    x = x + vis.positional_embedding.to(x.dtype)
                    x = vis.ln_pre(x)
                    x = vis.transformer(x)
                    x = vis.ln_post(x)
                    # Global CLS token output projected to 512
                    global_cls = x[:, 0, :]  # (B, 768)
                    if hasattr(vis, "proj") and vis.proj is not None:
                        img_embeds = global_cls @ vis.proj  # (B, 512)
                    else:
                        img_embeds = global_cls[:, :512]
                else:
                    out = vis(images)
                    img_embeds = out[0] if isinstance(out, tuple) else out

                embeds_np = img_embeds.cpu().numpy().astype(np.float32)
                all_embeddings.append(embeds_np)

                for i in range(len(pids)):
                    all_metadata.append(
                        {
                            "patient_id": pids[i],
                            "study_id": batch["study_id"][i] if "study_id" in batch else f"S_{pids[i]}",
                            "dicom_id": batch["dicom_id"][i] if "dicom_id" in batch else f"D_{pids[i]}",
                            "report_text": reports[i],
                        }
                    )

        stacked_embeds = np.vstack(all_embeddings)
        self.vector_index.add_embeddings(stacked_embeds, all_metadata)
        logger.info(f"FAISS Multimodal Index successfully built with {len(all_metadata)} database reports.")

    def retrieve(
        self,
        query_embeddings: torch.Tensor,
        top_k: int = 2,
        mode: str = "similarity",
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieves top_k context records.
        mode: 'similarity' (FAISS Top-K) or 'random' (Random Control)
        """
        query_np = query_embeddings.detach().cpu().numpy().astype(np.float32)
        if mode == "similarity":
            return self.vector_index.search(query_np, top_k=top_k)
        elif mode == "random":
            return self.vector_index.getRandomSamples(batch_size=len(query_np), top_k=top_k)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")
