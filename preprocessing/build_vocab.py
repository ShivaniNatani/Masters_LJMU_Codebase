import os
import json
import pandas as pd
from preprocessing.text_preprocessing import RadiologyTextPreprocessor
from preprocessing.patient_splitter import patient_level_split
from utils.logger import setup_logger

logger = setup_logger("build_vocab")


def build_and_save_vocab(
    csv_path: str = "data/mock/mimic_cxr_mock.csv",
    save_path: str = "data/processed/vocab.json",
    patient_col: str = "patient_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
):
    """
    Builds vocabulary from report text column and exports vocabulary mapping JSON.

    The vocabulary is fit on the TRAIN patient-split subset only. Building it on
    the full (pre-split) corpus would leak validation/test report text into the
    token space (which words clear min_word_freq, which become <unk>) that the
    model conditions on - a data leakage risk even though the corpus itself is
    unlabeled text. If `patient_col` is missing from the CSV, falls back to the
    full corpus with an explicit warning rather than failing silently.
    """
    if not os.path.exists(csv_path):
        logger.warning(f"File {csv_path} not found. Skipping vocabulary build.")
        return

    df = pd.read_csv(csv_path)
    preprocessor = RadiologyTextPreprocessor(min_word_freq=1)

    if patient_col in df.columns:
        train_df, val_df, test_df = patient_level_split(
            df,
            patient_col=patient_col,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        logger.info(
            f"Building vocabulary from TRAIN split only "
            f"({len(train_df)}/{len(df)} reports) to prevent val/test leakage."
        )
        vocab_source_df = train_df
    else:
        logger.warning(
            f"'{patient_col}' column not found in {csv_path}; cannot patient-split. "
            "Building vocabulary from the FULL corpus (val/test leakage risk)."
        )
        vocab_source_df = df

    reports = vocab_source_df["full_report"].dropna().tolist()
    vocab = preprocessor.build_vocabulary(reports)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(
            {
                "word2idx": preprocessor.word2idx,
                "idx2word": {str(k): v for k, v in preprocessor.idx2word.items()},
                "vocab_size": len(preprocessor.word2idx),
            },
            f,
            indent=2,
        )

    logger.info(f"Saved Vocabulary JSON ({len(vocab)} words) to {save_path}")


if __name__ == "__main__":
    build_and_save_vocab()
