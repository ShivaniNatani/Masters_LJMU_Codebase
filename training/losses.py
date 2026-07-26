import torch
import torch.nn as nn


class SequenceCrossEntropyLoss(nn.Module):
    """
    Masked Sequence-to-Sequence Cross-Entropy Loss ignoring pad tokens (-100).
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits: (Batch, Seq_Len, Vocab_Size)
        labels: (Batch, Seq_Len)
        """
        vocab_size = logits.size(-1)
        shift_logits = logits.view(-1, vocab_size)
        shift_labels = labels.view(-1)
        return self.loss_fn(shift_logits, shift_labels)
