import torch
from torch.nn import CrossEntropyLoss

from trainers.base import BaseTrainer


class IntentTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.criterion = CrossEntropyLoss()

    def run_batch(self, batch):
        y_hat = self.model(batch["input_ids"].to(self.device))
        loss = self.criterion(y_hat, batch["label"].to(self.device))
        return loss, self.metrics_fn(y_hat.cpu(), batch["label"])

    def metrics_fn(self, y_hat, labels):
        return {"acc": torch.eq(y_hat.argmax(dim=1), labels).to(torch.float).mean()}
