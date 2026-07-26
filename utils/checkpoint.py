import os
import torch
from typing import Dict, Any, Optional
from utils.logger import setup_logger

logger = setup_logger("checkpoint_manager")


class ThreeTierCheckpointManager:
    """
    Manages saving and restoring model checkpoints across 3 tier states:
    1. latest: Saved every epoch (baseline_latest.pt)
    2. best_loss: Lowest validation loss (baseline_best_loss.pt)
    3. best_bleu4: Highest validation BLEU-4 score (baseline_best_bleu4.pt)
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.latest_path = os.path.join(checkpoint_dir, "baseline_latest.pt")
        self.best_loss_path = os.path.join(checkpoint_dir, "baseline_best_loss.pt")
        self.best_bleu4_path = os.path.join(checkpoint_dir, "baseline_best_bleu4.pt")

        self.best_val_loss = float("inf")
        self.best_val_bleu4 = -1.0

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        val_loss: float,
        val_bleu4: float = 0.0,
        stage: int = 1,
    ) -> Dict[str, bool]:
        """
        Saves checkpoints and returns flags indicating if best-loss or best-bleu4 were updated.
        """
        state = {
            "epoch": epoch,
            "stage": stage,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "val_loss": val_loss,
            "val_bleu4": val_bleu4,
        }

        updates = {"latest": True, "best_loss": False, "best_bleu4": False}

        # 1. Save Latest Checkpoint
        torch.save(state, self.latest_path)

        # 2. Save Best Loss Checkpoint
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(state, self.best_loss_path)
            updates["best_loss"] = True
            logger.info(f"New Best Val Loss Checkpoint saved! ({val_loss:.4f})")

        # 3. Save Best BLEU-4 Checkpoint
        if val_bleu4 > self.best_val_bleu4:
            self.best_val_bleu4 = val_bleu4
            torch.save(state, self.best_bleu4_path)
            updates["best_bleu4"] = True
            logger.info(f"New Best BLEU-4 Checkpoint saved! ({val_bleu4:.4f})")

        return updates

    def load(
        self,
        checkpoint_type: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> int:
        """
        Loads state dict from specified checkpoint_type ('latest', 'best_loss', 'best_bleu4').
        Returns epoch number.
        """
        path_map = {
            "latest": self.latest_path,
            "best_loss": self.best_loss_path,
            "best_bleu4": self.best_bleu4_path,
        }
        path = path_map.get(checkpoint_type, self.latest_path)

        if not os.path.exists(path):
            logger.warning(f"Checkpoint path {path} not found. Starting from scratch.")
            return 0

        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Loaded {checkpoint_type} checkpoint from {path} (Epoch {checkpoint.get('epoch', 0)})")
        return checkpoint.get("epoch", 0)
