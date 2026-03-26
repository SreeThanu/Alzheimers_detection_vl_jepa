"""
trainer.py
----------
Training loop with:
  - Mixed precision (torch.amp) — works on CPU / MPS / CUDA
  - Early stopping based on validation loss
  - Checkpoint saving (best model only)
  - Per-epoch metric logging
  - Combined loss: cross-entropy classification + contrastive alignment
"""

import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStopping:
    """Stop training when monitored metric stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-3, mode: str = "min"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.counter   = 0
        self.best      = float("inf") if mode == "min" else float("-inf")
        self.triggered = False

    def __call__(self, value: float) -> bool:
        improved = (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )
        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


class Trainer:
    """
    Manages the full training and validation loop.

    Args:
        model:           VLJEPAModel instance
        train_loader:    Training DataLoader
        val_loader:      Validation DataLoader
        device:          torch.device
        lr:              Learning rate
        weight_decay:    L2 penalty
        epochs:          Max training epochs
        checkpoint_dir:  Where to save best model
        use_amp:         Use automatic mixed precision
        early_stopping_patience: Epochs with no improvement before stopping
        contrastive_weight: Weight for contrastive loss (0 = disable)
        class_weights: Optional tensor [C] for CrossEntropyLoss; None = unweighted.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        checkpoint_dir: str = "experiments/checkpoints",
        checkpoint_name: str = "best_model.pt",
        use_amp: bool = True,
        early_stopping_patience: int = 7,
        contrastive_weight: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model              = model.to(device)
        self.train_loader       = train_loader
        self.val_loader         = val_loader
        self.device             = device
        self.epochs             = epochs
        self.checkpoint_dir     = Path(checkpoint_dir)
        self.checkpoint_path    = self.checkpoint_dir / checkpoint_name
        self.contrastive_weight = contrastive_weight

        # AMP — disable on CPU (no speedup), enable on MPS/CUDA if requested
        _amp_ok = device.type in ("cuda", "mps") and use_amp
        self.scaler = torch.amp.GradScaler(enabled=_amp_ok and device.type == "cuda")
        self._amp_enabled = _amp_ok
        self._amp_device_type = device.type if _amp_ok else "cpu"

        # Loss and optimiser (optional class weights from training-set distribution)
        if class_weights is not None:
            cw = class_weights.to(device).float()
            self.criterion = nn.CrossEntropyLoss(weight=cw)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer  = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=lr * 0.01
        )
        self.early_stop = EarlyStopping(patience=early_stopping_patience)

        # History for plotting
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
        }

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self) -> dict:
        """Run training for self.epochs; returns training history dict."""
        best_val_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._train_epoch()
            val_loss,   val_acc   = self._eval_epoch()

            self.scheduler.step()
            elapsed = time.time() - t0

            # Logging
            print(
                f"Epoch [{epoch:03d}/{self.epochs}]  "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}  "
                f"({elapsed:.1f}s)"
            )
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss)

            # Early stopping
            if self.early_stop(val_loss):
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        print(f"\nBest validation loss: {best_val_loss:.4f}")
        return self.history

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, tokens, labels in tqdm(self.train_loader, desc="  Train", leave=False):
            images = images.to(self.device)
            tokens = tokens.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(
                device_type=self._amp_device_type, enabled=self._amp_enabled
            ):
                out = self.model(images, tokens, labels=labels)
                cls_loss = self.criterion(out["logits"], labels)

                # Optional contrastive alignment
                if self.contrastive_weight > 0 and self.model.use_text:
                    c_loss = self.model.contrastive_loss(out["img_proj"], out["txt_proj"])
                    loss = cls_loss + self.contrastive_weight * c_loss
                else:
                    loss = cls_loss

            # Backward — use scaler only for CUDA AMP
            if self.device.type == "cuda":
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds       = out["logits"].argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def _eval_epoch(self):
        self.model.eval()
        if getattr(self.model, "cache_text_embeddings", False):
            self.model.update_text_embedding_cache()

        total_loss, correct, total = 0.0, 0, 0

        for images, tokens, labels in tqdm(self.val_loader, desc="  Val  ", leave=False):
            images = images.to(self.device)
            tokens = tokens.to(self.device)
            labels = labels.to(self.device)

            out  = self.model(images, tokens, labels=labels)
            loss = self.criterion(out["logits"], labels)

            total_loss += loss.item() * images.size(0)
            preds       = out["logits"].argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

        return total_loss / total, correct / total

    def _save_checkpoint(self, epoch: int, val_loss: float):
        torch.save(
            {
                "epoch":      epoch,
                "val_loss":   val_loss,
                "model_state": self.model.state_dict(),
                "optim_state": self.optimizer.state_dict(),
            },
            self.checkpoint_path,
        )
        print(f"  ✓ Checkpoint saved → {self.checkpoint_path}")
