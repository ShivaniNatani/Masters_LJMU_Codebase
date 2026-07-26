import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Tuple

from preprocessing.image_preprocessing import preprocess_image
from preprocessing.vlm_tokenizer import VLMTokenizerWrapper


class VLMDataset(Dataset):
    """
    PyTorch Dataset for Vision-Language Model baseline training & evaluation.
    Converts image files to normalized tensors and report text to tokenized targets.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer_wrapper: VLMTokenizerWrapper,
        image_size: Tuple[int, int] = (224, 224),
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer_wrapper = tokenizer_wrapper
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        # 1. Image preprocessing
        image_path = str(row.get("image_path", ""))
        image_tensor = preprocess_image(image_path, target_size=self.image_size)

        # 2. Target text encoding
        report_text = str(row.get("full_report", ""))
        labels = self.tokenizer_wrapper.encode_target_report(report_text)

        return {
            "patient_id": str(row.get("patient_id", f"P{idx}")),
            "study_id": str(row.get("study_id", f"S{idx}")),
            "image": image_tensor,
            "prompt_ids": self.tokenizer_wrapper.prompt_ids,
            "prompt_mask": self.tokenizer_wrapper.prompt_mask,
            "labels": labels,
            "report_text": report_text,
        }
