import pandas as pd
from datasets.base_dataset import BaseMedicalDataset


class IUChestXrayDataset(BaseMedicalDataset):
    """
    Indiana University Chest X-ray Specific PyTorch Dataset Loader.
    """

    def __init__(self, dataframe: pd.DataFrame, image_size: tuple = (224, 224), max_seq_len: int = 128):
        super().__init__(dataframe=dataframe, image_size=image_size, max_seq_len=max_seq_len)
