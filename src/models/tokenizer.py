from pathlib import Path
from typing import List, Optional
import numpy as np

from utils.io import json_load


class Tokenizer:
    PAD_TOKEN = "[PAD]"
    OOV_TOKEN = "[OOV]"

    @classmethod
    def from_pretrained(cls, model_dir: Path):
        word_list = json_load(model_dir / "dictionary.json")
        embeddings = np.load(model_dir / "word_vectors.npy")

        word_list = [cls.PAD_TOKEN, cls.OOV_TOKEN] + word_list
        embeddings = np.concatenate(
            [
                np.zeros((1, embeddings.shape[1])),
                embeddings.mean(axis=0).reshape(1, -1),
                embeddings,
            ]
        )

        return cls(word_list, embeddings)

    def __init__(
        self, word_list: Optional[List[str]] = None, embeddings: Optional[np.ndarray] = None
    ):
        self.word_list = word_list
        self.embeddings = embeddings
        self.token_to_idx = {token: i for i, token in enumerate(word_list)}

    def __call__(self, text):
        return self.convert_tokens_to_ids(self.tokenize(text))

    def convert_token_to_id(self, token):
        return self.token_to_idx.get(token, 1)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.convert_token_to_id(token) for token in tokens]

    def tokenize(self, text):
        return text.split()


def test():
    tokenizer = Tokenizer(["hello", "world"])
    print(tokenizer("hello world yeah"))


if __name__ == "__main__":
    test()
