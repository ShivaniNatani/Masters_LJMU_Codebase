"""
Evaluation package containing NLG metrics (BLEU, ROUGE, METEOR, CIDEr, BERTScore)
and Clinical Efficacy metrics (CheXbert, RadGraph).
"""
from evaluation.nlg_metrics import compute_nlg_metrics
from evaluation.clinical_metrics import compute_clinical_metrics

__all__ = ["compute_nlg_metrics", "compute_clinical_metrics"]
