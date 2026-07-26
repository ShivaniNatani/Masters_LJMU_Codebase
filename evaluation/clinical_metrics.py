import os
import json
import re
from typing import List, Dict, Any, Tuple
from utils.logger import setup_logger

logger = setup_logger("clinical_metrics")

CHEXPERT_CONDITIONS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]


def extract_chexbert_conditions(text: str) -> Dict[str, int]:
    """
    Regex-based Rule Extractor mapping radiology report text to 14 CheXpert conditions.
    0 = Negative/Absent, 1 = Positive/Present
    """
    text_lower = text.lower()
    labels = {}

    label_patterns = {
        "Cardiomegaly": r"(cardiomegaly|enlarged heart|heart size is enlarged)",
        "Edema": r"(edema|pulmonary edema|vascular congestion)",
        "Consolidation": r"(consolidation|airspace opacity)",
        "Pneumonia": r"(pneumonia|infiltrate)",
        "Atelectasis": r"(atelectasis|basilar collapse|volume loss)",
        "Pneumothorax": r"(pneumothorax|ptx)",
        "Pleural Effusion": r"(pleural effusion|effusion|fluid)",
        "Support Devices": r"(line|tube|pacemaker|catheter|stent)",
        "No Finding": r"(no acute|normal|unremarkable|clear)",
    }

    for cond in CHEXPERT_CONDITIONS:
        pattern = label_patterns.get(cond, rf"\b{cond.lower()}\b")
        labels[cond] = 1 if re.search(pattern, text_lower) else 0

    return labels


def extract_radgraph_entities(text: str) -> List[Dict[str, str]]:
    """
    Rule-based RadGraph entity extractor (Anatomy, Observation, Modifier).
    """
    tokens = text.lower().split()
    entities = []

    anatomy_words = {"lung", "heart", "cardiac", "pleural", "base", "mediastinum", "rib"}
    observation_words = {"opacity", "effusion", "atelectasis", "pneumonia", "edema", "normal", "clear"}

    for idx, tok in enumerate(tokens):
        tok_clean = re.sub(r"[^\w]", "", tok)
        if tok_clean in anatomy_words:
            entities.append({"entity": tok_clean, "label": "ANAT-DP", "index": idx})
        elif tok_clean in observation_words:
            entities.append({"entity": tok_clean, "label": "OBS-DP", "index": idx})

    return entities


def compute_clinical_metrics(
    predictions: List[str],
    references: List[str],
    raw_chexbert_path: str = "results/raw_chexbert_labels.json",
    raw_radgraph_path: str = "results/raw_radgraph_entities.json",
) -> Dict[str, float]:
    """
    Computes CheXbert F1 & RadGraph F1 metrics and exports sample-level raw extractions to JSON.
    """
    os.makedirs(os.path.dirname(raw_chexbert_path), exist_ok=True)
    os.makedirs(os.path.dirname(raw_radgraph_path), exist_ok=True)

    raw_chexbert_samples = []
    raw_radgraph_samples = []

    pred_matrix = []
    ref_matrix = []

    radgraph_f1_scores = []

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        # 1. CheXbert Extraction
        p_labels = extract_chexbert_conditions(pred)
        r_labels = extract_chexbert_conditions(ref)

        pred_matrix.append(list(p_labels.values()))
        ref_matrix.append(list(r_labels.values()))

        raw_chexbert_samples.append(
            {
                "sample_id": i,
                "prediction_text": pred,
                "reference_text": ref,
                "predicted_conditions": p_labels,
                "reference_conditions": r_labels,
            }
        )

        # 2. RadGraph Entity Extraction
        p_entities = extract_radgraph_entities(pred)
        r_entities = extract_radgraph_entities(ref)

        p_set = set(e["entity"] for e in p_entities)
        r_set = set(e["entity"] for e in r_entities)

        intersection = p_set.intersection(r_set)
        prec = len(intersection) / max(1, len(p_set))
        rec = len(intersection) / max(1, len(r_set))
        f1 = (2 * prec * rec) / max(1e-5, (prec + rec))
        radgraph_f1_scores.append(f1)

        raw_radgraph_samples.append(
            {
                "sample_id": i,
                "predicted_entities": p_entities,
                "reference_entities": r_entities,
                "entity_overlap_f1": round(f1, 4),
            }
        )

    # Export raw sample extractions for error analysis
    with open(raw_chexbert_path, "w") as f:
        json.dump(raw_chexbert_samples, f, indent=2)

    with open(raw_radgraph_path, "w") as f:
        json.dump(raw_radgraph_samples, f, indent=2)

    logger.info(f"Exported raw CheXbert extractions to {raw_chexbert_path}")
    logger.info(f"Exported raw RadGraph extractions to {raw_radgraph_path}")

    # Compute overall F1 scores
    import numpy as np

    p_arr = np.array(pred_matrix)
    r_arr = np.array(ref_matrix)

    tp = np.sum((p_arr == 1) & (r_arr == 1))
    fp = np.sum((p_arr == 1) & (r_arr == 0))
    fn = np.sum((p_arr == 0) & (r_arr == 1))

    micro_prec = tp / max(1, tp + fp)
    micro_rec = tp / max(1, tp + fn)
    chexbert_micro_f1 = (2 * micro_prec * micro_rec) / max(1e-5, micro_prec + micro_rec)

    mean_radgraph_f1 = float(np.mean(radgraph_f1_scores))

    return {
        "CheXbert_Micro_F1": round(float(chexbert_micro_f1), 4),
        "CheXbert_Precision": round(float(micro_prec), 4),
        "CheXbert_Recall": round(float(micro_rec), 4),
        "RadGraph_Entity_F1": round(float(mean_radgraph_f1), 4),
    }
