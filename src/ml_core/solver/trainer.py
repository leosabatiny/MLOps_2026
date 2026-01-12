import csv
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import setup_logger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: str,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device

        self.logger = setup_logger("Trainer")

        # Classification loss (PCAM is binary classification; CrossEntropyLoss fits logits + class idx labels).
        self.criterion = nn.CrossEntropyLoss()

        self.best_val_loss = float("inf")

        self.save_dir = Path(self.config["training"]["save_dir"])
        self.ckpt_dir = Path(self.config["training"]["checkpoint_dir"])
        self.losses_csv = Path(self.config["training"]["losses_csv"])
        self.log_after_n_steps = int(self.config["training"].get("log_after_n_steps", 50))

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.losses_csv.parent.mkdir(parents=True, exist_ok=True)

        # Initialize CSV with header if it doesn't exist.
        if not self.losses_csv.exists():
            with self.losses_csv.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "split", "step", "loss", "timestamp"])

    def _append_loss_row(self, epoch: int, split: str, step: int, loss: float) -> None:
        with self.losses_csv.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, split, step, loss, time.time()])

    def train_epoch(self, dataloader: DataLoader, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.train()

        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Train E{epoch_idx}", leave=False)
        for step_idx, (images, labels) in enumerate(pbar, start=1):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            loss_val = float(loss.item())
            running_loss += loss_val
            n_batches += 1

            if step_idx % self.log_after_n_steps == 0 or step_idx == 1:
                self._append_loss_row(epoch=epoch_idx, split="train", step=step_idx, loss=loss_val)

            pbar.set_postfix(loss=f"{loss_val:.4f}")

        avg_loss = running_loss / max(1, n_batches)
        # epoch summary row (step=0)
        self._append_loss_row(epoch=epoch_idx, split="train_epoch", step=0, loss=float(avg_loss))

        # Return placeholders for (loss, acc, f1) to match your original signature.
        return float(avg_loss), 0.0, 0.0

    @torch.no_grad()
    def validate(self, dataloader: DataLoader, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.eval()

        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Val   E{epoch_idx}", leave=False)
        for step_idx, (images, labels) in enumerate(pbar, start=1):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss_val = float(loss.item())
            running_loss += loss_val
            n_batches += 1

            if step_idx % self.log_after_n_steps == 0 or step_idx == 1:
                self._append_loss_row(epoch=epoch_idx, split="val", step=step_idx, loss=loss_val)

            pbar.set_postfix(loss=f"{loss_val:.4f}")

        avg_loss = running_loss / max(1, n_batches)
        self._append_loss_row(epoch=epoch_idx, split="val_epoch", step=0, loss=float(avg_loss))

        return float(avg_loss), 0.0, 0.0

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        ckpt_path = self.ckpt_dir / f"epoch={epoch:03d}_valloss={val_loss:.4f}.pt"
        payload = {
            "epoch": epoch,
            "val_loss": float(val_loss),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        torch.save(payload, ckpt_path)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        epochs = int(self.config["training"]["epochs"])

        self.logger.info(f"Starting training for {epochs} epochs...")
        self.logger.info(f"Loss CSV: {self.losses_csv}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, _, _ = self.train_epoch(train_loader, epoch)
            val_loss, _, _ = self.validate(val_loader, epoch)

            dt = time.time() - t0
            self.logger.info(
                f"Epoch {epoch}/{epochs} done in {dt:.1f}s | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
            )

            # Always checkpoint each epoch (simple + safe).
            self.save_checkpoint(epoch=epoch, val_loss=val_loss)

            # Track "best" too (optional extra checkpoint).
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = self.ckpt_dir / "best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "val_loss": float(val_loss),
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "config": self.config,
                    },
                    best_path,
                )

