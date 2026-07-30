import os
import ast
import re
import pandas as pd
from tqdm import tqdm
from utils.logger import setup_logger
from label_guidance.label_encoder import StructuredLabelEncoder
from preprocessing.patient_splitter import patient_level_split

logger = setup_logger("prepare_kaggle_mimic")

KAGGLE_DATASET_BASE = "/Users/shivaninatani/.cache/kagglehub/datasets/simhadrisadaram/mimic-cxr-dataset/versions/2"


def prepare_kaggle_mimic_dataset(
    kaggle_base: str = KAGGLE_DATASET_BASE,
    output_dir: str = "data/processed/mimic_cxr",
    max_records: int = 15000,  # Cap at 15k high-quality records for fast training on M4 Pro
) -> str:
    """
    Parses Kaggle MIMIC-CXR dataset, filters for existing images on disk,
    extracts patient_id, study_id, dicom_id, full_report, and CheXbert condition labels,
    and performs patient-level splitting.
    """
    os.makedirs(output_dir, exist_ok=True)
    encoder = StructuredLabelEncoder()

    train_csv = os.path.join(kaggle_base, "mimic_cxr_aug_train.csv")
    val_csv = os.path.join(kaggle_base, "mimic_cxr_aug_validate.csv")

    raw_dfs = []
    if os.path.exists(train_csv):
        raw_dfs.append(pd.read_csv(train_csv))
    if os.path.exists(val_csv):
        raw_dfs.append(pd.read_csv(val_csv))

    if not raw_dfs:
        logger.error(f"No Kaggle CSV files found in {kaggle_base}")
        return ""

    combined_raw = pd.concat(raw_dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined_raw)} raw patient records from Kaggle MIMIC-CXR CSVs.")

    extracted_records = []

    for idx, row in tqdm(combined_raw.iterrows(), total=len(combined_raw), desc="Parsing MIMIC Dataset"):
        subject_id = str(row["subject_id"])
        image_str = str(row.get("image", "[]"))
        text_str = str(row.get("text", "[]"))

        try:
            image_list = ast.literal_eval(image_str)
            text_list = ast.literal_eval(text_str)
        except Exception:
            continue

        if not image_list or not text_list:
            continue

        # Match images and reports
        for img_rel in image_list:
            full_img_path = os.path.join(kaggle_base, "official_data_iccv_final", img_rel)
            if not os.path.exists(full_img_path):
                continue

            # Parse study_id and dicom_id from relative path
            # Example: 'files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg'
            parts = img_rel.split("/")
            sid = parts[3] if len(parts) >= 4 else f"s_{subject_id}"
            did = parts[4].replace(".jpg", "") if len(parts) >= 5 else f"d_{idx}"

            # Select associated report text
            report_text = text_list[0] if text_list else ""
            if not report_text or len(report_text.strip()) < 10:
                continue

            # CheXbert pathology condition extraction
            condition_dict = encoder.extract_labels_from_text(report_text)
            positives = [cond for cond, val in condition_dict.items() if val == 1]
            labels_str = "|".join(positives) if positives else "No Finding"

            extracted_records.append(
                {
                    "patient_id": f"p{subject_id}",
                    "study_id": sid,
                    "dicom_id": did,
                    "image_path": full_img_path,
                    "full_report": report_text.strip(),
                    "labels": labels_str,
                }
            )

            if len(extracted_records) >= max_records:
                break
        if len(extracted_records) >= max_records:
            break

    df = pd.DataFrame(extracted_records)
    logger.info(f"Extracted {len(df)} valid image-report pairs with existing disk files.")

    # Patient-level split with ZERO patient data leakage across train/val/test
    train_df, val_df, test_df = patient_level_split(
        df, patient_col="patient_id", train_ratio=0.70, val_ratio=0.10, test_ratio=0.20, seed=42
    )

    combined_splits = pd.concat([train_df, val_df, test_df], ignore_index=True)
    csv_path = os.path.join(output_dir, "mimic_cxr_splits.csv")
    combined_splits.to_csv(csv_path, index=False)

    logger.info(f"MIMIC-CXR processed dataset saved to {csv_path}")
    logger.info(f"Split distribution:\n{combined_splits['split'].value_counts()}")
    return csv_path


if __name__ == "__main__":
    prepare_kaggle_mimic_dataset()
