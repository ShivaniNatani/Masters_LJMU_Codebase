import pandas as pd
import numpy as np
from typing import Tuple
from utils.seed import set_seed
from utils.logger import setup_logger

logger = setup_logger("patient_splitter")


def patient_level_split(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs patient-level train/val/test split ensuring zero overlap of patient IDs
    across train, validation, and test subsets to prevent medical data leakage.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    set_seed(seed)

    unique_patients = df[patient_col].unique()
    np.random.shuffle(unique_patients)

    n_patients = len(unique_patients)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)

    train_patients = set(unique_patients[:n_train])
    val_patients = set(unique_patients[n_train : n_train + n_val])
    test_patients = set(unique_patients[n_train + n_val :])

    train_df = df[df[patient_col].isin(train_patients)].copy()
    val_df = df[df[patient_col].isin(val_patients)].copy()
    test_df = df[df[patient_col].isin(test_patients)].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    logger.info(f"Patient Splitting Complete across {n_patients} unique patients:")
    logger.info(f" - Train: {len(train_df)} samples ({len(train_patients)} patients)")
    logger.info(f" - Val:   {len(val_df)} samples ({len(val_patients)} patients)")
    logger.info(f" - Test:  {len(test_df)} samples ({len(test_patients)} patients)")

    # Assert no patient overlap
    assert len(train_patients.intersection(val_patients)) == 0
    assert len(train_patients.intersection(test_patients)) == 0
    assert len(val_patients.intersection(test_patients)) == 0

    return train_df, val_df, test_df
