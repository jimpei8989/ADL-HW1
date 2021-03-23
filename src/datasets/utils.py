import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate

PAD_KEYS = set(["input_ids"])


def create_batch(samples):
    return {
        k: pad_sequence([s[k] for s in samples], batch_first=False)
        if k in PAD_KEYS
        else default_collate([s[k] for s in samples])
        for k in samples[0]
    }
