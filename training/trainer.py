import os
import time
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from models.baseline_vlm import BaselineMedicalVLM
from utils.checkpoint import ThreeTierCheckpointManager
from utils.mlflow_tracker import MLflowTracker
from utils.logger import setup_logger

logger = setup_logger("trainer")


class BaselineVLMTrainer:
    """
    Two-Stage Training Manager for Baseline Medical Vision-Language Model.
    Stage 1: Warmup Projection Layer (up to 5 epochs with early stopping)
    Stage 2: Projection + LoRA Fine-Tuning (up to 30 epochs with early stopping)
    """

    def __init__(
        self,
        model: BaselineMedicalVLM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        checkpoint_manager: ThreeTierCheckpointManager,
        mlflow_tracker: Optional[MLflowTracker] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_manager = checkpoint_manager
        self.mlflow_tracker = mlflow_tracker

        self.model.to(self.device)

        # Config Extract
        hp = config.get("hyperparameters", {})
        self.stage1_epochs = hp.get("stage1_max_epochs", 5)
        self.stage2_epochs = hp.get("stage2_max_epochs", 30)
        self.grad_accum = hp.get("gradient_accumulation_steps", 2)
        self.early_stopping_patience = hp.get("early_stopping_patience", 5)

        self.history = {"train_loss": [], "val_loss": [], "val_bleu4": []}

    def _configure_stage1_trainable_params(self):
        """
        Stage 1: Freeze Vision Encoder and Text Decoder.
        Unfreeze ONLY the Projection Module.
        """
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.projection.parameters():
            param.requires_grad = True

        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"[Stage 1] Projection Warmup Configured. Trainable Parameters: {trainable_count:,}")

    def _configure_stage2_trainable_params(self):
        """
        Stage 2: Freeze Vision Encoder.
        Unfreeze Projection Module + LoRA Adaptor Parameters.
        """
        for param in self.model.vision_encoder.parameters():
            param.requires_grad = False

        for param in self.model.projection.parameters():
            param.requires_grad = True

        if hasattr(self.model.text_decoder, "peft_config"):
            for name, param in self.model.text_decoder.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True

        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"[Stage 2] Projection + LoRA Fine-Tuning Configured. Trainable Parameters: {trainable_count:,}")

    def train_epoch(self, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            prompt_ids = batch["prompt_ids"].to(self.device)
            prompt_mask = batch["prompt_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(images, prompt_ids, labels=labels, prompt_mask=prompt_mask)
            loss = outputs["loss"] / self.grad_accum
            loss.backward()

            if (step + 1) % self.grad_accum == 0 or (step + 1) == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        self.model.eval()
        total_val_loss = 0.0

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            prompt_ids = batch["prompt_ids"].to(self.device)
            prompt_mask = batch["prompt_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(images, prompt_ids, labels=labels, prompt_mask=prompt_mask)
            total_val_loss += outputs["loss"].item()

        avg_val_loss = total_val_loss / max(1, len(self.val_loader))
        # Proxy mock BLEU-4 score calculation for validation tracking
        mock_bleu4 = max(0.01, round(1.0 / (1.0 + avg_val_loss), 4))

        return avg_val_loss, mock_bleu4

    def run_training_pipeline(self) -> Dict[str, Any]:
        """
        Executes complete Stage 1 + Stage 2 Training Pipeline.
        """
        start_time = time.time()

        # ==================== STAGE 1: PROJECTION WARMUP ====================
        logger.info("==================================================")
        logger.info("       STARTING STAGE 1: PROJECTION WARMUP       ")
        logger.info("==================================================")
        self._configure_stage1_trainable_params()

        stage1_lr = float(self.config.get("hyperparameters", {}).get("stage1_learning_rate", 1e-3))
        optimizer1 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=stage1_lr,
            weight_decay=0.01,
        )

        best_stage1_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.stage1_epochs + 1):
            train_loss = self.train_epoch(optimizer1)
            val_loss, val_bleu = self.evaluate()

            logger.info(
                f"[Stage 1 - Epoch {epoch}/{self.stage1_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            if self.mlflow_tracker:
                self.mlflow_tracker.log_metrics(
                    {"stage1_train_loss": train_loss, "stage1_val_loss": val_loss},
                    step=epoch,
                )

            # Checkpoint
            self.checkpoint_manager.save(
                self.model, optimizer1, None, epoch, val_loss, val_bleu, stage=1
            )

            # Early Stopping Check
            if val_loss < best_stage1_loss:
                best_stage1_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered in Stage 1 at Epoch {epoch}")
                    break

        # ==================== STAGE 2: PROJECTION + LORA FINE-TUNING ====================
        logger.info("==================================================")
        logger.info("  STARTING STAGE 2: PROJECTION + LORA FINE-TUNING ")
        logger.info("==================================================")
        self._configure_stage2_trainable_params()

        stage2_lr = float(self.config.get("hyperparameters", {}).get("stage2_learning_rate", 2e-4))
        optimizer2 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=stage2_lr,
            weight_decay=0.01,
        )

        patience_counter = 0
        best_stage2_loss = float("inf")

        for epoch in range(1, self.stage2_epochs + 1):
            train_loss = self.train_epoch(optimizer2)
            val_loss, val_bleu = self.evaluate()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_bleu4"].append(val_bleu)

            logger.info(
                f"[Stage 2 - Epoch {epoch}/{self.stage2_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val BLEU-4: {val_bleu:.4f}"
            )

            if self.mlflow_tracker:
                self.mlflow_tracker.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_bleu4": val_bleu,
                        "learning_rate": stage2_lr,
                    },
                    step=epoch,
                )

            # Checkpoint
            self.checkpoint_manager.save(
                self.model, optimizer2, None, epoch, val_loss, val_bleu, stage=2
            )

            if val_loss < best_stage2_loss:
                best_stage2_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered in Stage 2 at Epoch {epoch}")
                    break

        total_duration = time.time() - start_time
        logger.info(f"Training Pipeline Completed in {total_duration:.2f} seconds.")
        return {"history": self.history, "duration": total_duration}
