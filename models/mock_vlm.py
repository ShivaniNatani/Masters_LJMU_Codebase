import torch
import torch.nn as nn
from typing import Dict, Any


class MockVLM(nn.Module):
    """
    Lightweight Vision-Language Model stub used for verification and smoke testing.
    Combines a Convolutional Vision Encoder with a Transformer-style Decoder head.
    """

    def __init__(self, vision_dim: int = 512, text_dim: int = 256, vocab_size: int = 1000):
        super().__init__()
        # Vision Encoder Head
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, vision_dim),
        )

        # Text Embedding & Decoder Projection
        self.token_embedding = nn.Embedding(vocab_size, text_dim)
        self.fusion_projection = nn.Linear(vision_dim + text_dim, text_dim)
        self.lm_head = nn.Linear(text_dim, vocab_size)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for smoke testing.
        images: (Batch, 3, H, W)
        input_ids: (Batch, Seq_Len)
        """
        batch_size, seq_len = input_ids.shape

        # 1. Vision Feature Extraction -> (Batch, vision_dim)
        v_feats = self.vision_encoder(images)

        # 2. Text Feature Embedding -> (Batch, Seq_Len, text_dim)
        t_embeds = self.token_embedding(input_ids)

        # 3. Multimodal Fusion (Expand vision features across sequence length)
        v_expanded = v_feats.unsqueeze(1).expand(-1, seq_len, -1)
        fused = torch.cat([v_expanded, t_embeds], dim=-1)

        # 4. Projection & Language Modeling Logits -> (Batch, Seq_Len, vocab_size)
        hidden = torch.relu(self.fusion_projection(fused))
        logits = self.lm_head(hidden)

        # Compute simple cross entropy loss if target token IDs available
        loss = None
        if input_ids is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"logits": logits, "loss": loss}
