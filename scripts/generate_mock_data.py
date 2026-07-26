import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import yaml

from utils.seed import set_seed
from utils.logger import setup_logger

logger = setup_logger("generate_mock_data")


def generate_mock_dataset(num_samples: int = 150, output_dir: str = "data/mock"):
    """
    Generates synthetic DICOM-like image files and radiology reports matching
    MIMIC-CXR & Indiana University Chest X-ray schema for end-to-end testing.
    """
    set_seed(42)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    findings_templates = [
        "The cardiac silhouette is within normal limits. Lungs are clear without focal consolidation, pneumothorax, or pleural effusion. Visualized osseous structures are intact.",
        "Mild cardiomegaly is present. There is minimal bibasilar atelectasis. No overt pulmonary edema. No pneumothorax seen.",
        "Moderate right-sided pleural effusion is present with associated bibasilar opacity, representing compressive atelectasis versus pneumonia.",
        "Lungs are hyperinflated consistent with chronic obstructive pulmonary disease. No acute focal infiltrate or consolidation.",
        "Low lung volumes. Patchy opacity in the left lower lobe may represent pneumonia or atelectasis. Follow-up recommended.",
    ]

    impression_templates = [
        "No acute cardiopulmonary disease.",
        "Mild cardiomegaly with minor basilar atelectasis.",
        "Right pleural effusion and probable bibasilar atelectasis.",
        "COPD changes without acute consolidation.",
        "Left lower lobe opacity, concern for pneumonia.",
    ]

    diseases_list = [
        "Cardiomegaly",
        "Emphysema",
        "Effusion",
        "Hernia",
        "Infiltration",
        "Mass",
        "Nodule",
        "Atelectasis",
        "Pneumothorax",
        "Pleural_Thickening",
        "Pneumonia",
        "Fibrosis",
        "Edema",
        "Consolidation",
        "No Finding",
    ]

    patient_ids = [f"P{1000 + (i // 2)}" for i in range(num_samples)]
    study_ids = [f"S{5000 + i}" for i in range(num_samples)]
    dicom_ids = [f"D{9000 + i}" for i in range(num_samples)]

    records = []

    for i in range(num_samples):
        pid = patient_ids[i]
        sid = study_ids[i]
        did = dicom_ids[i]

        img_filename = f"{did}.jpg"
        img_path = os.path.join(output_dir, "images", img_filename)

        # Generate synthetic 224x224 grayscale chest X-ray mock image
        img_array = np.random.randint(40, 220, (224, 224), dtype=np.uint8)
        img = Image.fromarray(img_array).convert("RGB")
        img.save(img_path)

        f_idx = i % len(findings_templates)
        findings = findings_templates[f_idx]
        impression = impression_templates[f_idx]

        # Multi-label medical findings assignment
        assigned_diseases = random.sample(diseases_list, random.randint(1, 3))
        if "No Finding" in assigned_diseases and len(assigned_diseases) > 1:
            assigned_diseases.remove("No Finding")

        disease_str = "|".join(assigned_diseases)

        records.append(
            {
                "patient_id": pid,
                "study_id": sid,
                "dicom_id": did,
                "image_path": img_path,
                "findings": findings,
                "impression": impression,
                "full_report": f"FINDINGS: {findings} IMPRESSION: {impression}",
                "labels": disease_str,
                "width": 224,
                "height": 224,
                "view_position": "PA" if i % 2 == 0 else "LATERAL",
            }
        )

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "mimic_cxr_mock.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Generated mock dataset: {len(df)} records saved to {csv_path}")
    return csv_path


if __name__ == "__main__":
    generate_mock_dataset()
