from typing import Dict
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.nn import Embedding, RNN, LSTM, GRU

from utils.logger import logger


RNN_CLASS_MAPPING = {"RNN": RNN, "LSTM": LSTM, "GRU": GRU}


class BaseModel(nn.Module):
    @classmethod
    def from_saved(cls, config: Dict, saved_path: Path):
        state_dict = torch.load(saved_path)
        return cls.from_state_dict(config, state_dict)

    @classmethod
    def from_checkpoint(cls, config: Dict, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path)
        return cls.from_state_dict(config, checkpoint["model_state_dict"])

    @classmethod
    def from_state_dict(cls, config: Dict, state_dict: Dict):
        model = cls(**config)
        for key in filter(
            lambda key: key in state_dict and isinstance(model.__getattr__(key), nn.Module),
            model.keys(),
        ):
            logger.info(f"Loading {key} from state_dict")
            model.__getattr__(key).load_state_dict(state_dict[key])

    def __init__(
        self,
        num_embeddings: int = 1024,
        embedding_dim: int = 128,
        rnn_style: str = "LSTM",
        rnn_num_layers: int = 1,
        hidden_dim: int = 128,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.embedding = Embedding(num_embeddings, embedding_dim)

        self.rnn = RNN_CLASS_MAPPING[rnn_style](
            embedding_dim, hidden_dim, rnn_num_layers, bidirectional=bidirectional
        )

        num_directions = 2 if bidirectional else 1
        self.rnn_output_dim = num_directions * hidden_dim
        self.hidden_state_dim = rnn_num_layers * num_directions * hidden_dim

    def forward(self, input_ids: Tensor):
        """
        Arguments
            input_ids: torch.LongTensor of shape (BS, L)

        Returns
            output: torch.FloatTensor of shape (BS, L, num_directions*hidden_dim)
            hidden_state: torch.FloatTensor of shape (BS, rnn_num_layers*num_directions,
                hidden_size)
        """
        embedded = self.embedding(input_ids)
        output, hidden_state = self.rnn(embedded)

        output = output.permute(1, 0, 2)
        hidden_state = hidden_state.permute(1, 0, 2)
        return output, hidden_state
