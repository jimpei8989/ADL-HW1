from typing import List, Dict
from pathlib import Path

from datasets.intent_dataset import IntentDataset
from datasets.slot_dataset import SlotDataset


def to_intent_csv(predictions: List[Dict], prediction_csv: Path):
    with open(prediction_csv, "w") as f:
        print("id,intent", file=f)
        for pred in predictions:
            print(f"{pred['id']},{IntentDataset.LABELS[pred['intent']]}", file=f)


def to_slot_csv(predictions: List[Dict], prediction_csv: Path):
    with open(prediction_csv, "w") as f:
        print("id,tags", file=f)
        for pred in predictions:
            print(
                f"{pred['id']},{' '.join(map(lambda p: SlotDataset.TAGS[p], pred['tags']))}",
                file=f,
            )
