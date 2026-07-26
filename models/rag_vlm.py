import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from models.baseline_vlm import BaselineMedicalVLM
from retrieval.retriever import MultimodalRetriever
from utils.logger import setup_logger

logger = setup_logger("rag_vlm")


class RAGMedicalVLM(BaselineMedicalVLM):
    """
    Retrieval-Augmented Vision-Language Model (RAG VLM).
    Integrates FAISS multimodal retrieval with FLAN-T5-Base decoder and LoRA adaptation.
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

    def extract_global_image_embedding(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extracts 512-dim global image embedding for vector similarity retrieval.
        """
        with torch.no_grad():
            vis = self.vision_encoder
            if hasattr(vis, "conv1") and hasattr(vis, "transformer"):
                x = vis.conv1(images)
                x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
                class_token = vis.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
                x = torch.cat([class_token, x], dim=1)
                x = x + vis.positional_embedding.to(x.dtype)
                x = vis.ln_pre(x)
                x = vis.transformer(x)
                x = vis.ln_post(x)
                cls_token = x[:, 0, :]
                if hasattr(vis, "proj") and vis.proj is not None:
                    return cls_token @ vis.proj
                return cls_token[:, :512]
            else:
                out = vis(images)
                return out[0] if isinstance(out, tuple) else out

    def generate_rag_report(
        self,
        images: torch.Tensor,
        tokenizer_wrapper,
        retriever: Optional[MultimodalRetriever] = None,
        retrieval_mode: str = "similarity",  # 'similarity', 'random', or 'none'
        top_k: int = 2,
        max_new_tokens: int = 128,
        num_beams: int = 2,
    ) -> Tuple[torch.Tensor, List[List[Dict[str, Any]]]]:
        """
        Executes RAG generation pipeline:
        1. Extract visual embedding
        2. Retrieve Top-K context reports (or random reports)
        3. Format RAG prompt
        4. Concatenate projected visual patch tokens with context prompt tokens
        5. Perform beam search decoding
        """
        batch_size = images.shape[0]
        device = images.device

        # Step 1: Retrieval
        retrieved_contexts = []
        if retriever is not None and retrieval_mode in ["similarity", "random"]:
            global_embeds = self.extract_global_image_embedding(images)
            retrieved_contexts = retriever.retrieve(global_embeds, top_k=top_k, mode=retrieval_mode)

        # Step 2: Format Prompts
        rag_prompts = []
        for b in range(batch_size):
            if retrieved_contexts and len(retrieved_contexts[b]) > 0:
                ctx_texts = [f"Reference Report {i+1}: {r['report_text']}" for i, r in enumerate(retrieved_contexts[b])]
                ctx_str = " ".join(ctx_texts)
                prompt = f"Retrieved Context: {ctx_str} Task: Write a radiology report for the image."
            else:
                prompt = "Task: Write a radiology report for the chest X-ray image."
            rag_prompts.append(prompt)

        # Step 3: Tokenize RAG Prompts
        tok_out = tokenizer_wrapper.tokenizer(
            rag_prompts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)

        prompt_ids = tok_out["input_ids"]
        prompt_mask = tok_out["attention_mask"]

        # Step 4: Extract & Project Spatial Patch Embeddings
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

        # Step 5: Beam Search Generation
        generated_ids = self.text_decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )

        return generated_ids, retrieved_contexts
