import os
import json
import pandas as pd
from typing import Dict, List
from utils.logger import setup_logger
from label_guidance.label_encoder import StructuredLabelEncoder

logger = setup_logger("prepare_iu_cxr")


def prepare_iu_chest_xray_dataset(
    raw_dir: str = "data/raw/iu_chest_xray",
    output_dir: str = "data/processed/iu_chest_xray",
) -> str:
    """
    Processes the Indiana University (IU) Chest X-Ray dataset from jsonl & image files
    into a structured dataframe with split designations and CheXbert condition labels.
    """
    os.makedirs(output_dir, exist_ok=True)
    encoder = StructuredLabelEncoder()

    records = []

    splits = ["train", "val", "test"]

    for split_name in splits:
        jsonl_path = os.path.join(raw_dir, f"{split_name}.jsonl")
        if not os.path.exists(jsonl_path):
            logger.warning(f"Split file missing: {jsonl_path}")
            continue

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                data = json.loads(line.strip())
                report_text = data.get("response", "").strip()
                image_paths = data.get("images", [])

                if not report_text or not image_paths:
                    continue

                # CheXbert pathology condition extraction
                condition_dict = encoder.extract_labels_from_text(report_text)
                positive_labels = [cond for cond, val in condition_dict.items() if val == 1]
                labels_str = "|".join(positive_labels) if positive_labels else "No Finding"

                for img_rel_path in image_paths:
                    # Clean relative path: "/iu_xray/image/CXR2384_IM-0942/0.png" -> "CXR2384_IM-0942/0.png"
                    clean_rel = img_rel_path.replace("/iu_xray/image/", "").strip("/")
                    local_img_path = os.path.join(raw_dir, "images", clean_rel)

                    # Extract patient / study identifier
                    parts = clean_rel.split("/")
                    study_id = parts[0] if parts else f"S{line_idx}"
                    img_file = parts[1] if len(parts) > 1 else f"{line_idx}.png"
                    dicom_id = f"{study_id}_{img_file.replace('.png', '')}"
                    patient_id = study_id.split("_")[0] if "_" in study_id else study_id

                    if os.path.exists(local_img_path):
                        records.append(
                            {
                                "patient_id": patient_id,
                                "study_id": study_id,
                                "dicom_id": dicom_id,
                                "image_path": local_img_path,
                                "full_report": report_text,
                                "labels": labels_str,
                                "split": split_name,
                            }
                        )

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "iu_cxr_splits.csv")
    df.to_csv(csv_path, index=False)

    logger.info(f"IU Chest X-Ray dataset preparation complete: {len(df)} total image-report pairs saved to {csv_path}")
    logger.info(f"Split breakdown:\n{df['split'].value_counts()}")
    return csv_path


if __name__ == "__main__":
    prepare_iu_chest_xray_dataset()
