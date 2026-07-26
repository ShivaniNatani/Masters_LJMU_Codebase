import re
from typing import Dict, List, Any
from utils.logger import setup_logger

logger = setup_logger("label_encoder")

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

LABEL_PATTERNS = {
    "Cardiomegaly": r"(cardiomegaly|enlarged heart|heart size is enlarged|cardiac silhouette is enlarged)",
    "Edema": r"(edema|pulmonary edema|vascular congestion|interstitial edema)",
    "Consolidation": r"(consolidation|airspace opacity|focal consolidation)",
    "Pneumonia": r"(pneumonia|infiltrate|infectious process)",
    "Atelectasis": r"(atelectasis|basilar collapse|volume loss|subsegmental atelectasis)",
    "Pneumothorax": r"(pneumothorax|ptx|air in pleural space)",
    "Pleural Effusion": r"(pleural effusion|effusion|fluid in pleural space)",
    "Support Devices": r"(line|tube|pacemaker|catheter|stent|picc|endotracheal)",
    "Enlarged Cardiomediastinum": r"(enlarged cardiomediastinum|widened mediastinum)",
    "Lung Opacity": r"(opacity|opacities|opacity in lung)",
    "Lung Lesion": r"(lesion|nodule|mass|masses)",
    "Pleural Other": r"(pleural thickening|pleural scar|pleural plaque)",
    "Fracture": r"(fracture|rib fracture|fractured)",
    "No Finding": r"(no acute|normal|unremarkable|clear lungs|no acute cardiopulmonary)",
}


class StructuredLabelEncoder:
    """
    Extracts and encodes 14 CheXbert disease pathology condition vectors
    and converts them into clinical prompt representations.
    """

    def __init__(self, conditions: List[str] = CHEXPERT_CONDITIONS):
        self.conditions = conditions

    def extract_labels_from_text(self, text: str) -> Dict[str, int]:
        """
        Parses radiology text and extracts binary 0/1 indicator per condition.
        """
        text_lower = text.lower()
        labels = {}

        for cond in self.conditions:
            pattern = LABEL_PATTERNS.get(cond, rf"\b{cond.lower()}\b")
            labels[cond] = 1 if re.search(pattern, text_lower) else 0

        # Special check: If no specific pathology is positive, mark No Finding as 1
        positive_count = sum(v for k, v in labels.items() if k != "No Finding")
        if positive_count == 0:
            labels["No Finding"] = 1

        return labels

    def format_guidance_string(self, label_dict: Dict[str, int], active_only: bool = True) -> str:
        """
        Formats extracted label dictionary into structured guidance string.
        If active_only is True, includes positive findings or 'No Finding'.
        """
        if active_only:
            positives = [cond for cond, val in label_dict.items() if val == 1]
            if not positives or (len(positives) == 1 and positives[0] == "No Finding"):
                return "Clinical Pathology: No Acute Findings."
            return "Clinical Pathology: " + ", ".join([f"{p}: POSITIVE" for p in positives if p != "No Finding"]) + "."
        else:
            items = [f"{cond}: {'POSITIVE' if val == 1 else 'NEGATIVE'}" for cond, val in label_dict.items()]
            return "Clinical Pathology Guidance: [" + " | ".join(items) + "]."
