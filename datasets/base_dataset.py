import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Any
from preprocessing.image_preprocessing import preprocess_image
from preprocessing.text_preprocessing import RadiologyTextPreprocessor


class BaseMedicalDataset(Dataset):
    """
    Abstract PyTorch Dataset for Multi-Modal Medical Report Generation.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_size: tuple = (224, 224),
        preprocessor: RadiologyTextPreprocessor = None,
        max_seq_len: int = 128,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_size = image_size
        self.preprocessor = preprocessor or RadiologyTextPreprocessor()
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        image_path = str(row.get("image_path", ""))
        image_tensor = preprocess_image(image_path, target_size=self.image_size)

        report_text = str(row.get("full_report", ""))
        token_ids = self.preprocessor.encode(report_text, add_special_tokens=True)

        # Pad / truncate token sequence
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len]
        else:
            pad_id = self.preprocessor.word2idx.get(self.preprocessor.pad_token, 0)
            token_ids = token_ids + [pad_id] * (self.max_seq_len - len(token_ids))

        token_tensor = torch.tensor(token_ids, dtype=torch.long)

        return {
            "patient_id": str(row.get("patient_id", f"P{idx}")),
            "study_id": str(row.get("study_id", f"S{idx}")),
            "dicom_id": str(row.get("dicom_id", f"D{idx}")),
            "image": image_tensor,
            "input_ids": token_tensor,
            "report_text": report_text,
            "labels": str(row.get("labels", "No Finding")),
        }
