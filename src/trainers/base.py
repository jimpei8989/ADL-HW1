from collections import defaultdict
from typing import Optional
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from utils.logger import logger
from utils.timer import timer
from utils.tqdmm import tqdmm


class Trainer:
    def __init__(
        self,
        model,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-5,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.model = model
        self.epochs = epochs

        self.optimizer = Adam(lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = None

    @timer
    def run_epoch(self, dataloader, split="", train=False, epoch=0, device=None):
        self.model.train() if train else self.model.eval()

        losses = []
        metrics = defaultdict(list)

        with torch.set_grad_enabled(train):
            for batch in tqdmm(
                dataloader, desc=f"Epoch {epoch:02d} / {self.epochs:02d} [{split}]"
            ):
                if train:
                    self.optimizer.zero_grad()

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                y_hat = self.model(input_ids)
                loss = self.criterion(y_hat, labels)

                if train:
                    loss.backward()
                    self.optimizer.step()

                losses.append(loss.item())
                for k, v in self.metric_fn(y_hat, labels).items():
                    metrics[k].append(v)

        return np.mean(losses), {k: np.mean(v) for k, v in metrics.items()}

    @staticmethod
    def metric_fn(y_hat, labels):
        return {}

    def train(self, train_dataloader, val_dataloader, device=None):
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch:02d} / {self.epochs:02d}")
            train_time, train_loss, train_metrics = self.run_epoch(
                train_dataloader, split="train", train=True, epoch=epoch, device=device
            )
            logger.info(f"Train | {train_time:.3f} | loss: {train_loss:.3f} | {train_metrics}")

            val_time, val_loss, val_metrics = self.run_epoch(
                val_dataloader, split="val", epoch=epoch, device=device
            )
            logger.info(f"Val   | {val_time:.3f} | loss: {val_loss:.3f} | {val_metrics}")

    def evaluate(self):
        pass

    def predict(self):
        pass
