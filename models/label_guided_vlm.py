import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple

from models.rag_vlm import RAGMedicalVLM
from label_guidance.prompt_formatter import SLGPromptFormatter
from retrieval.retriever import MultimodalRetriever
from utils.logger import setup_logger

logger = setup_logger("label_guided_vlm")


class LabelGuidedMedicalVLM(RAGMedicalVLM):
    """
    Structured Label Guidance Vision-Language Model (SLG-RAG VLM).
    Integrates CheXbert 14-condition clinical pathology vectors and FAISS multimodal RAG
    with FLAN-T5-Base decoder and LoRA adaptation.
    """

    def __init__(
        self,
        vision_model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        text_model_name: str = "google/flan-t5-base",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
    ):
        super().__init__(
            vision_model_name=vision_model_name,
            text_model_name=text_model_name,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )
        self.formatter = SLGPromptFormatter(active_only=True)

    def generate_slg_report(
        self,
        images: torch.Tensor,
        tokenizer_wrapper,
        report_texts: Optional[List[str]] = None,
        retriever: Optional[MultimodalRetriever] = None,
        retrieval_mode: str = "similarity",  # 'none', 'random', or 'similarity'
        use_slg: bool = True,
        top_k: int = 2,
        max_new_tokens: int = 128,
        num_beams: int = 2,
    ) -> Tuple[torch.Tensor, List[List[Dict[str, Any]]], List[str]]:
        """
        Executes SLG + RAG report generation pipeline:
        1. Extract global image embedding & retrieve Top-K context reports if active
        2. Extract CheXbert condition labels & format SLG prompt if active
        3. Concatenate projected visual patch tokens with SLG+RAG prompt tokens
        4. Execute beam search decoding
        """
        batch_size = images.shape[0]
        device = images.device

        # Step 1: Retrieval Context
        retrieved_contexts = []
        if retriever is not None and retrieval_mode in ["similarity", "random"]:
            global_embeds = self.extract_global_image_embedding(images)
            retrieved_contexts = retriever.retrieve(global_embeds, top_k=top_k, mode=retrieval_mode)

        # Step 2: Format SLG + RAG Prompts
        slg_prompts = []
        for b in range(batch_size):
            ref_text = report_texts[b] if (report_texts and b < len(report_texts)) else None
            ret_ctx = retrieved_contexts[b] if (retrieved_contexts and b < len(retrieved_contexts)) else None
            
            prompt = self.formatter.construct_slg_prompt(
                report_text=ref_text if use_slg else None,
                retrieved_contexts=ret_ctx,
                use_rag=(retrieval_mode != "none"),
            )
            slg_prompts.append(prompt)

        # Step 3: Tokenize Multimodal Prompts
        tok_out = tokenizer_wrapper.tokenizer(
            slg_prompts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)

        prompt_ids = tok_out["input_ids"]
        prompt_mask = tok_out["attention_mask"]

        # Step 4: Spatial Patch Embedding Extraction & Visual Projection
        patch_embeds = self.extract_patch_embeddings(images)
        v_proj = self.projection(patch_embeds)

        if hasattr(self.text_decoder, "get_input_embeddings"):
            text_embed_fn = self.text_decoder.get_input_embeddings()
        else:
            text_embed_fn = self.text_decoder.base_model.model.shared

        prompt_embeds = text_embed_fn(prompt_ids)
        inputs_embeds = torch.cat([v_proj, prompt_embeds], dim=1)

        v_mask = torch.ones((batch_size, 196), device=device)
        combined_mask = torch.cat([v_mask, prompt_mask], dim=1)

        # Step 5: Beam Search Decoding
        generated_ids = self.text_decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )

        return generated_ids, retrieved_contexts, slg_prompts
