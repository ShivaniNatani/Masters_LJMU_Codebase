import torch
import torch.nn as nn


class VisualProjectionModule(nn.Module):
    """
    Trainable Vision-Language Projection Interface.
    Maps BioMedCLIP ViT patch embeddings into the FLAN-T5 decoder embedding dimension.

    Tensor Dimensions:
        Input:  (Batch, Num_Patches=196, Vision_Dim=768)
        Hidden: (Batch, Num_Patches=196, Hidden_Dim=768)
        Output: (Batch, Num_Patches=196, Text_Dim=768)
    """

    def __init__(self, vision_dim: int = 768, text_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim

        self.net = nn.Sequential(
            nn.Linear(vision_dim, text_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim, text_dim),
            nn.LayerNorm(text_dim),
        )

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """
        patch_embeddings: (Batch, Num_Patches, Vision_Dim) e.g. (B, 196, 768)
        Returns: (Batch, Num_Patches, Text_Dim) e.g. (B, 196, 768)
        """
        assert (
            patch_embeddings.dim() == 3
        ), f"Expected 3D patch embedding tensor (B, N, D), got {patch_embeddings.shape}"
        return self.net(patch_embeddings)
