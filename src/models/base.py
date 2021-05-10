from typing import Dict, Optional
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.nn import Embedding, RNN, LSTM, GRU

from utils.logger import logger


RNN_CLASS_MAPPING = {"RNN": RNN, "LSTM": LSTM, "GRU": GRU}


class BaseBaseModel:
    @classmethod
    def load_weights(cls, config: Dict, weights_path: Path):
        logger.info(f"Loading weights from {weights_path}")
        state_dict = torch.load(weights_path)
        return cls.from_state_dict(config, state_dict)

    @classmethod
    def from_checkpoint(cls, config: Dict, checkpoint_path: Path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        return cls.from_state_dict(config, checkpoint["model_state_dict"])

    @classmethod
    def from_state_dict(cls, config: Dict, state_dict: Dict):
        model = cls(**config)
        model.load_state_dict(state_dict)
        return model


class BaseModel(nn.Module, BaseBaseModel):
    def __init__(
        self,
        num_embeddings: int = 1024,
        embedding_dim: int = 128,
        embedding_initial_weights: Optional[Tensor] = None,
        freeze_embedding: bool = False,
        rnn_style: str = "LSTM",
        rnn_num_layers: int = 1,
        hidden_dim: int = 128,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.embedding = Embedding(num_embeddings, embedding_dim)
        if embedding_initial_weights is not None:
            self.embedding.load_state_dict({"weight": embedding_initial_weights})

        if freeze_embedding:
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.rnn = RNN_CLASS_MAPPING[rnn_style](
            embedding_dim,
            hidden_dim,
            rnn_num_layers,
            bidirectional=bidirectional,
            batch_first=True,
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
        hidden_state = hidden_state.permute(1, 0, 2)
        return output, hidden_state

    def save_weights(self, weights_path: Path):
        logger.info(f"Saving model weights to {weights_path}")
        torch.save(self.state_dict(), weights_path)
