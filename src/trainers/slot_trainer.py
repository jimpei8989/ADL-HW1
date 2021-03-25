import torch
from torch.nn import CrossEntropyLoss
from torch.nn import functional as nnf

from trainers.base import BaseTrainer


class SlotTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.criterion = CrossEntropyLoss()

    def run_batch(self, batch):
        y_hat = self.model(batch["input_ids"].to(self.device))
        loss = self.criterion(
            y_hat.reshape(-1, y_hat.shape[-1]), batch["tags"].reshape(-1).to(self.device)
        )
        return loss, self.metrics_fn(y_hat.cpu(), batch["tags"])

    def run_predict_batch(self, batch):
        y_hat = self.model(batch["input_ids"].to(self.device)).cpu()
        prediction_scores = nnf.softmax(y_hat, dim=2)
        predictions = y_hat.argmax(dim=2)
        return [
            {
                "id": batch["id"][i],
                "tag_scores": prediction_scores[i].tolist(),
                "tags": predictions[i].tolist()[:l],
            }
            for i, l in enumerate(batch["lengthes"])
        ]

    def metrics_fn(self, y_hat, labels):
        predictions = y_hat.argmax(dim=2)
        empty = labels.eq(-100)

        return {
            "token_acc": labels.eq(predictions).to(torch.float).sum()
            / (~empty).to(torch.float).sum(),
            "sentence_acc": torch.all(torch.logical_or(labels.eq(predictions), empty), dim=1)
            .to(torch.float)
            .sum()
            / labels.shape[0],
        }
