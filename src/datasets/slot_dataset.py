from pathlib import Path

import torch
from torch.utils.data import Dataset

from utils.io import json_load


class SlotDataset(Dataset):
    TAGS = (
        "O",
        "B-date",
        "I-date",
        "B-time",
        "I-time",
        "B-people",
        "I-people",
        "B-first_name",
        "B-last_name",
    )

    @classmethod
    def load(cls, json_path: Path, **kwargs):
        data = json_load(json_path)
        return cls(data, **kwargs)

    def __init__(self, data, tokenizer=None):
        self.data = data
        self.tokenizer = tokenizer
        self.tag_to_idx = {label: i for i, label in enumerate(self.TAGS)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample = self.data[index]
        ret = {
            "id": sample["id"],
            "input_ids": torch.as_tensor(
                self.tokenizer.convert_tokens_to_ids(sample["tokens"]), dtype=torch.long
            ),
        }

        if "tags" in sample:
            ret.update(
                {
                    "tags": torch.as_tensor(
                        [self.tag_to_idx[tag] for tag in sample["tags"]], dtype=torch.long
                    )
                }
            )

        return ret
