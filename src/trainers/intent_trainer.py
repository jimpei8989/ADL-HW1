import torch
from torch.nn import CrossEntropyLoss
from torch.nn import functional as nnf

from trainers.base import BaseTrainer


class IntentTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.criterion = CrossEntropyLoss()

    def run_batch(self, batch):
        y_hat = self.model(batch["input_ids"].to(self.device))
        loss = self.criterion(y_hat, batch["label"].to(self.device))
        return loss, self.metrics_fn(y_hat.cpu(), batch["label"])

    def run_predict_batch(self, batch):
        y_hat = self.model(batch["input_ids"].to(self.device)).cpu()
        prediction_scores = nnf.softmax(y_hat, dim=1)
        predictions = y_hat.argmax(dim=1)
        return [
            {
                "id": batch["id"][i],
                "intent_scores": prediction_scores[i].tolist(),
                "intent": predictions[i].item(),
            }
            for i in range(batch["batch_size"])
        ]

    def metrics_fn(self, y_hat, labels):
        return {"acc": torch.eq(y_hat.argmax(dim=1), labels).to(torch.float).mean()}
