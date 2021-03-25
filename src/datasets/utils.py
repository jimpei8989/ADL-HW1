from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate

PADDING = {
    "input_ids": 0,
    "tags": -100,
}


def create_batch(samples):
    ret = {
        k: pad_sequence([s[k] for s in samples], batch_first=True, padding_value=PADDING[k])
        if k in PADDING
        else default_collate([s[k] for s in samples])
        for k in samples[0]
    }
    ret.update({"batch_size": len(samples)})
    return ret
