from typing import List, Dict, Any, Optional
from label_guidance.label_encoder import StructuredLabelEncoder
from utils.logger import setup_logger

logger = setup_logger("prompt_formatter")


class SLGPromptFormatter:
    """
    Constructs unified multimodal prompts fusing Structured Label Guidance (SLG),
    FAISS RAG retrieved reference contexts, and generation task directives.
    """

    def __init__(self, active_only: bool = True):
        self.encoder = StructuredLabelEncoder()
        self.active_only = active_only

    def construct_slg_prompt(
        self,
        report_text: Optional[str] = None,
        label_dict: Optional[Dict[str, int]] = None,
        retrieved_contexts: Optional[List[Dict[str, Any]]] = None,
        use_rag: bool = True,
    ) -> str:
        """
        Synthesizes complete SLG + RAG prompt.
        """
        # 1. Label Guidance Prefix
        if label_dict is None and report_text is not None:
            label_dict = self.encoder.extract_labels_from_text(report_text)
        elif label_dict is None:
            label_dict = {"No Finding": 1}

        slg_prefix = self.encoder.format_guidance_string(label_dict, active_only=self.active_only)

        # 2. RAG Retrieval Context
        rag_str = ""
        if use_rag and retrieved_contexts and len(retrieved_contexts) > 0:
            ctx_texts = [f"Reference Report {i+1}: {r.get('report_text', '')}" for i, r in enumerate(retrieved_contexts)]
            rag_str = " Retrieved Context: " + " ".join(ctx_texts)

        # 3. Complete Instruction Prompt
        full_prompt = f"{slg_prefix}{rag_str} Task: Write a radiology report for the chest X-ray image matching the clinical findings."
        return full_prompt
