import os
import json
import numpy as np
import faiss

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass

from typing import List, Dict, Any, Tuple
from utils.logger import setup_logger

logger = setup_logger("faiss_index")


class FAISSVectorIndex:
    """
    FAISS Index Manager supporting Inner Product (Cosine Similarity) search
    for multimodal 512-dim BioMedCLIP vision embeddings.
    """

    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict[str, Any]] = []

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Adds embeddings to the FAISS index with associated metadata.
        embeddings: (N, 512) float32 numpy array.
        """
        assert embeddings.shape[1] == self.dimension, f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
        assert len(embeddings) == len(metadata), "Length of embeddings and metadata must match"

        # L2-normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(metadata)
        logger.info(f"Added {len(embeddings)} embeddings to FAISS index. Total indexed: {self.index.ntotal}")

    def search(self, query_embeddings: np.ndarray, top_k: int = 2) -> List[List[Dict[str, Any]]]:
        """
        Performs Top-K Cosine Similarity search over the FAISS index.
        query_embeddings: (B, 512) float32 array
        Returns: List of B results, where each result is a list of top_k metadata dicts with 'similarity_score'.
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS Index is empty. Returning empty search results.")
            return [[] for _ in range(len(query_embeddings))]

        query_copy = query_embeddings.copy().astype(np.float32)
        faiss.normalize_L2(query_copy)

        scores, indices = self.index.search(query_copy, min(top_k, self.index.ntotal))

        batch_results = []
        for b in range(len(query_embeddings)):
            sample_results = []
            for k in range(len(indices[b])):
                idx = indices[b][k]
                score = float(scores[b][k])
                if idx != -1 and idx < len(self.metadata):
                    meta = self.metadata[idx].copy()
                    meta["similarity_score"] = round(score, 4)
                    sample_results.append(meta)
            batch_results.append(sample_results)

        return batch_results

    def getRandomSamples(self, batch_size: int, top_k: int = 2, seed: int = 42) -> List[List[Dict[str, Any]]]:
        """
        Random Retrieval Control: Selects K random database entries per sample.
        """
        np.random.seed(seed)
        batch_results = []
        ntotal = len(self.metadata)

        for _ in range(batch_size):
            sample_results = []
            rand_indices = np.random.choice(ntotal, size=min(top_k, ntotal), replace=False)
            for idx in rand_indices:
                meta = self.metadata[idx].copy()
                meta["similarity_score"] = 0.0
                sample_results.append(meta)
            batch_results.append(sample_results)

        return batch_results

    def save(self, dir_path: str = "retrieval/index_store"):
        os.makedirs(dir_path, exist_ok=True)
        index_path = os.path.join(dir_path, "faiss.index")
        meta_path = os.path.join(dir_path, "metadata.json")

        faiss.write_index(self.index, index_path)
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Saved FAISS index to {index_path} and metadata to {meta_path}")

    def load(self, dir_path: str = "retrieval/index_store"):
        index_path = os.path.join(dir_path, "faiss.index")
        meta_path = os.path.join(dir_path, "metadata.json")

        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded FAISS index from {dir_path}. Total items: {self.index.ntotal}")
        else:
            logger.warning(f"Index store not found at {dir_path}")
