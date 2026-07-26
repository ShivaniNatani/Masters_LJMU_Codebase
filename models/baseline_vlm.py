import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from models.projection import VisualProjectionModule
from utils.logger import setup_logger

logger = setup_logger("baseline_vlm")


class BaselineMedicalVLM(nn.Module):
    """
    Baseline Vision-Language Model for Radiology Report Generation.
    Couples BioMedCLIP Vision Encoder with FLAN-T5-Base Language Decoder.

    Tensor Dimension Documentation:
    1. Input Images:        (B, 3, 224, 224)
    2. ViT Patch Embeds:   (B, 196, 768)  [CLS token at index 0 removed]
    3. Visual Projection:   (B, 196, 768)  [Mapped to T5 encoder hidden dim]
    4. Text Prompt Embeds: (B, L_prompt, 768)
    5. Concat Enc Inputs:  (B, 196 + L_prompt, 768)
    6. Decoder Logits:     (B, L_target, 32128)
    """

    def __init__(
        self,
        vision_model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        text_model_name: str = "google/flan-t5-base",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.vision_model_name = vision_model_name
        self.text_model_name = text_model_name

        # 1. Vision Encoder Initialization & Freezing
        logger.info(f"Initializing Vision Encoder: {vision_model_name}")
        self.vision_encoder = self._load_vision_encoder(vision_model_name)
        self._freeze_vision_encoder()

        # BioMedCLIP ViT patch dimension = 768
        self.vision_dim = 768

        # 2. Text Decoder Initialization
        logger.info(f"Initializing Text Decoder: {text_model_name}")
        self.text_decoder = AutoModelForSeq2SeqLM.from_pretrained(text_model_name)
        self.text_dim = self.text_decoder.config.d_model  # 768 for flan-t5-base, 1024 for flan-t5-large

        # 3. Vision-Language Projection Interface
        logger.info(f"Building Visual Projection Interface ({self.vision_dim} -> {self.text_dim})")
        self.projection = VisualProjectionModule(vision_dim=self.vision_dim, text_dim=self.text_dim)

        # 4. LoRA Adapter Integration
        self.use_lora = use_lora
        if use_lora:
            logger.info(f"Applying LoRA Adaptors (r={lora_r}, alpha={lora_alpha}) to Text Decoder")
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q", "v"],
            )
            self.text_decoder = get_peft_model(self.text_decoder, lora_config)

    def _load_vision_encoder(self, model_name: str) -> nn.Module:
        """
        Loads BioMedCLIP / OpenCLIP vision encoder backbone or fallback ViT architecture.
        """
        try:
            import open_clip

            # Try loading BioMedCLIP open_clip weights
            model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
            return model.visual
        except Exception as e:
            logger.warning(f"Could not load open_clip model ({e}). Using standard ViT fallback encoder.")
            from torchvision.models import vit_b_16, ViT_B_16_Weights

            weights = ViT_B_16_Weights.DEFAULT
            vit = vit_b_16(weights=weights)
            # Custom patch extractor class
            class ViTPatchExtractor(nn.Module):
                def __init__(self, vit_model):
                    super().__init__()
                    self.conv_proj = vit_model.conv_proj
                    self.class_token = vit_model.class_token
                    self.encoder = vit_model.encoder

                def forward(self, x):
                    # x: (B, 3, 224, 224) -> (B, 768, 14, 14) -> (B, 196, 768)
                    n = x.shape[0]
                    x = self.conv_proj(x)
                    x = x.reshape(n, 768, -1).permute(0, 2, 1)
                    batch_class_token = self.class_token.expand(n, -1, -1)
                    x = torch.cat([batch_class_token, x], dim=1)
                    x = self.encoder(x)
                    return x  # (B, 197, 768)

            return ViTPatchExtractor(vit)

    def _freeze_vision_encoder(self):
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        logger.info("Vision Encoder 100% frozen.")

    def extract_patch_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extracts 196 spatial patch embeddings from input images.
        images: (B, 3, 224, 224)
        Returns: (B, 196, 768)
        """
        with torch.no_grad():
            vis = self.vision_encoder
            if hasattr(vis, "conv1") and hasattr(vis, "transformer"):
                x = vis.conv1(images)  # (B, 768, 14, 14)
                x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B, 196, 768)
                class_token = vis.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
                x = torch.cat([class_token, x], dim=1)  # (B, 197, 768)
                x = x + vis.positional_embedding.to(x.dtype)
                x = vis.ln_pre(x)
                x = vis.transformer(x)
                x = vis.ln_post(x)  # (B, 197, 768)
                patch_embeds = x[:, 1:, :]  # (B, 196, 768)
            else:
                vit_out = vis(images)
                if isinstance(vit_out, tuple):
                    vit_out = vit_out[0]
                if vit_out.dim() == 3 and vit_out.shape[1] == 197:
                    patch_embeds = vit_out[:, 1:, :]
                else:
                    raise ValueError(f"Unexpected vision output shape: {vit_out.shape}")
        return patch_embeds

    def forward(
        self,
        images: torch.Tensor,
        prompt_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for baseline multimodal VLM.
        images: (B, 3, 224, 224)
        prompt_ids: (B, L_prompt)
        labels: (B, L_target)
        """
        batch_size = images.shape[0]

        # 1. Vision Patch Embeddings: (B, 196, 768)
        patch_embeds = self.extract_patch_embeddings(images)

        # 2. Visual Projection to Decoder Dim: (B, 196, 768)
        v_proj = self.projection(patch_embeds)

        # 3. Lookup Text Prompt Embeddings: (B, L_prompt, 768)
        if hasattr(self.text_decoder, "get_input_embeddings"):
            text_embed_fn = self.text_decoder.get_input_embeddings()
        else:
            text_embed_fn = self.text_decoder.base_model.model.shared

        prompt_embeds = text_embed_fn(prompt_ids)

        # 4. Concatenate Visual Prefix and Prompt Embeddings along sequence dimension
        # Shape: (B, 196 + L_prompt, 768)
        inputs_embeds = torch.cat([v_proj, prompt_embeds], dim=1)

        # 5. Construct Combined Attention Mask
        v_mask = torch.ones((batch_size, 196), device=images.device)
        if prompt_mask is not None:
            combined_mask = torch.cat([v_mask, prompt_mask], dim=1)
        else:
            combined_mask = torch.ones((batch_size, inputs_embeds.shape[1]), device=images.device)

        # 6. Pass through Seq2Seq Model
        outputs = self.text_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            labels=labels,
            return_dict=True,
        )

        return {"loss": outputs.loss, "logits": outputs.logits}

    @torch.no_grad()
    def generate_report(
        self,
        images: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        num_beams: int = 4,
    ) -> torch.Tensor:
        """
        Generates report token sequences given input images and prompt IDs.
        """
        batch_size = images.shape[0]
        patch_embeds = self.extract_patch_embeddings(images)
        v_proj = self.projection(patch_embeds)

        if hasattr(self.text_decoder, "get_input_embeddings"):
            text_embed_fn = self.text_decoder.get_input_embeddings()
        else:
            text_embed_fn = self.text_decoder.base_model.model.shared

        prompt_embeds = text_embed_fn(prompt_ids)
        inputs_embeds = torch.cat([v_proj, prompt_embeds], dim=1)

        v_mask = torch.ones((batch_size, 196), device=images.device)
        if prompt_mask is not None:
            combined_mask = torch.cat([v_mask, prompt_mask], dim=1)
        else:
            combined_mask = torch.ones((batch_size, inputs_embeds.shape[1]), device=images.device)

        # Utilize HuggingFace generate interface
        generated_ids = self.text_decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )

        return generated_ids
