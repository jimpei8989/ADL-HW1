from torch.nn import CrossEntropyLoss

from trainers.base import BaseTrainer


class IntentTrainer(BaseTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = CrossEntropyLoss()

    def run_batch(self, batch):
        y_hat = self.model(batch["input_ids"].to(self.device))
        labels = batch["labels"]
        loss = self.criterion(y_hat, labels.to(self.device))
        return loss, self.metrics_fn(y_hat, batch["labels"])
