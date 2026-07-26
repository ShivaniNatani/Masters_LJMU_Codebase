import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """
    Constructs PyTorch DataLoader with standard batching parameters.

    `num_workers=None` (the default) auto-selects a worker count instead of the
    previous hardcoded 0, which forced fully synchronous single-process image
    decoding/tokenization and left the GPU idle between batches. Pass an
    explicit int (0 included) to override.
    """
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = min(4, max(0, cpu_count - 1))

    extra_kwargs = {}
    if num_workers > 0:
        # Keep worker processes (and their preprocessing pipeline setup) alive
        # across epochs, and let them stage batches ahead of consumption -
        # both no-ops when num_workers=0.
        extra_kwargs["persistent_workers"] = True
        extra_kwargs["prefetch_factor"] = 2

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory if torch.cuda.is_available() else False,
        drop_last=drop_last,
        **extra_kwargs,
    )
