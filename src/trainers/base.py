from abc import abstractmethod
from collections import defaultdict
from typing import Optional
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from utils.logger import logger
from utils.timer import timer
from utils.tqdmm import tqdmm


class BaseTrainer:
    def __init__(
        self,
        model,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-5,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_freq: int = 10,
        device=None,
    ):
        self.model = model

        self.cur_epoch = 1
        self.total_epochs = epochs

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = None
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq

        self.device = device

    @abstractmethod
    def metrics_fn(self, y_hat, labels):
        return {}

    @abstractmethod
    def run_train_batch(self, batch):
        return 0, {}

    def load_checkpoint(self, checkpoint_path: Path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.cur_epoch = checkpoint["cur_epoch"]

    def save_checkpoint(self, checkpoint_path: Path):
        logger.info(f"Saving current checkpoint to {checkpoint_path}")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "cur_epoch": self.cur_epoch,
            },
            checkpoint_path,
        )

    @timer
    def run_epoch(self, dataloader, split="", train=False, epoch=0):
        self.model.train() if train else self.model.eval()

        all_losses = []
        all_metrics = defaultdict(list)

        with torch.set_grad_enabled(train):
            for batch in tqdmm(
                dataloader, desc=f"Epoch {epoch:02d} / {self.total_epochs:02d} [{split}]"
            ):
                if train:
                    self.optimizer.zero_grad()

                loss, metrics = self.run_batch(batch)

                if train:
                    loss.backward()
                    self.optimizer.step()

                all_losses.append(loss.item())
                for k, v in metrics.items():
                    all_metrics[k].append(v)

        return np.mean(all_losses), {k: np.mean(v) for k, v in all_metrics.items()}

    def train(self, train_dataloader, val_dataloader):
        logger.info(f"Training model for {self.total_epochs} epochs...")
        for epoch in range(self.cur_epoch, self.total_epochs + 1):
            self.cur_epoch = epoch

            logger.info(f"Epoch {epoch:02d} / {self.total_epochs:02d}")
            train_time, train_loss, train_metrics = self.run_epoch(
                train_dataloader, split="train", train=True, epoch=epoch
            )
            logger.info(
                f"Train | {train_time:7.3f}s | loss: {train_loss:.3f} | "
                + " | ".join(map(lambda p: f"{p[0]}: {p[1]:.3f}", sorted(train_metrics.items())))
            )

            val_time, val_loss, val_metrics = self.run_epoch(
                val_dataloader, split="val", epoch=epoch
            )
            logger.info(
                f"Val   | {val_time:7.3f}s | loss: {val_loss:.3f} | {val_metrics}"
                + " | ".join(map(lambda p: f"{p[0]}: {p[1]:.3f}", sorted(val_metrics.items())))
            )

            if epoch % self.checkpoint_freq == 0:
                self.save_checkpoint(self.checkpoint_dir / f"checkpoint_{epoch:03d}.pt")

    def evaluate(self, dataloader, split=""):
        logger.info(f"Evaluating model using {split} data...")
        duration, loss, metrics = self.run_epoch(dataloader, split=split, train=False, epoch=-1)
        logger.info(
            f"{split[:5]:5s} | {duration:7.3f}s | loss: {loss:.3f} | "
            + " | ".join(map(lambda p: f"{p[0]}: {p[1]:.3f}", sorted(metrics.items())))
        )

    def predict(self, dataloader):
        logger.info("Predicting")
        pass
