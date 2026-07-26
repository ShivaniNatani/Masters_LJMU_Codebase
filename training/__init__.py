"""
Training package containing 2-stage trainer engine and sequence loss functions.
"""
from training.trainer import BaselineVLMTrainer
from training.losses import SequenceCrossEntropyLoss

__all__ = ["BaselineVLMTrainer", "SequenceCrossEntropyLoss"]
