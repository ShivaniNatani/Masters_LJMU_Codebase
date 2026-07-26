import json
import numpy as np
from typing import List, Dict, Any
from rouge_score import rouge_scorer
import sacrebleu
from utils.logger import setup_logger

logger = setup_logger("copy_similarity")


def compute_word_overlap(text1: str, text2: str) -> float:
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    if not set1 or not set2:
        return 0.0
    intersection = set1.intersection(set2)
    return len(intersection) / max(1, len(set1.union(set2)))


def analyze_copy_vs_grounding(
    predictions: List[str],
    references: List[str],
    retrieved_reports: List[List[str]],
    output_json_path: str = "results/phase3_copy_similarity.json",
) -> Dict[str, Any]:
    """
    Quantifies:
    1. Copy Similarity: Similarity between Retrieved Context Reports and Generated Report.
    2. Grounding Accuracy: Similarity between Generated Report and Ground Truth Report.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    copy_rouge_l = []
    copy_bleu_4 = []
    copy_overlap = []

    grounding_rouge_l = []
    grounding_bleu_4 = []

    sample_analysis = []

    for i in range(len(predictions)):
        pred = predictions[i]
        ref = references[i]
        ret_list = retrieved_reports[i] if i < len(retrieved_reports) else []

        # Join retrieved reports if multiple
        ret_combined = " ".join(ret_list) if ret_list else ""

        # 1. Copying Metrics (Generated vs. Retrieved)
        if ret_combined:
            r_copy = scorer.score(ret_combined, pred)["rougeL"].fmeasure
            b_copy = sacrebleu.sentence_bleu(pred, [ret_combined]).score / 100.0
            o_copy = compute_word_overlap(pred, ret_combined)
        else:
            r_copy, b_copy, o_copy = 0.0, 0.0, 0.0

        copy_rouge_l.append(r_copy)
        copy_bleu_4.append(b_copy)
        copy_overlap.append(o_copy)

        # 2. Grounding Metrics (Generated vs. Ground Truth)
        r_ground = scorer.score(ref, pred)["rougeL"].fmeasure
        b_ground = sacrebleu.sentence_bleu(pred, [ref]).score / 100.0

        grounding_rouge_l.append(r_ground)
        grounding_bleu_4.append(b_ground)

        sample_analysis.append(
            {
                "sample_id": i,
                "copy_rouge_l": round(float(r_copy), 4),
                "copy_bleu_4": round(float(b_copy), 4),
                "copy_word_overlap": round(float(o_copy), 4),
                "grounding_rouge_l": round(float(r_ground), 4),
                "grounding_bleu_4": round(float(b_ground), 4),
            }
        )

    summary = {
        "Copy_Similarity_Metrics": {
            "Mean_Copy_ROUGE_L": round(float(np.mean(copy_rouge_l)), 4),
            "Mean_Copy_BLEU_4": round(float(np.mean(copy_bleu_4)), 4),
            "Mean_Copy_Word_Overlap": round(float(np.mean(copy_overlap)), 4),
        },
        "Grounding_Accuracy_Metrics": {
            "Mean_Grounding_ROUGE_L": round(float(np.mean(grounding_rouge_l)), 4),
            "Mean_Grounding_BLEU_4": round(float(np.mean(grounding_bleu_4)), 4),
        },
        "Sample_Details": sample_analysis,
    }

    with open(output_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved Copy Similarity Analysis to {output_json_path}")
    logger.info(f"Mean Copy Word Overlap: {summary['Copy_Similarity_Metrics']['Mean_Copy_Word_Overlap']}")
    logger.info(f"Mean Grounding ROUGE-L: {summary['Grounding_Accuracy_Metrics']['Mean_Grounding_ROUGE_L']}")

    return summary
