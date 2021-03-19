import torch
from torch import nn, Tensor
from torch.nn import Embedding, RNN, LSTM, GRU

from utils.logger import logger


RNN_CLASS_MAPPING = {"RNN": RNN, "LSTM": LSTM, "GRU": GRU}


class BaseModel(nn.Module):
    @classmethod
    def from_checkpoint(cls, config=None, checkpoint_path=None):
        model = cls(**config)
        checkpoint = torch.load(checkpoint_path)

        logger.info(f"Loading checkpoint {checkpoint_path}")

        for key in filter(
            lambda key: key in checkpoint and isinstance(model.__getattr__(key), nn.Module),
            model.keys(),
        ):
            logger.info(f"loading {key} from checkpoint")
            model.__getattr__(key).load_state_dict(checkpoint[key])

    def __init__(
        self,
        num_embeddings: int = 1024,
        embedding_dim: int = 128,
        rnn_style: str = "LSTM",
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.embedding = Embedding(num_embeddings, embedding_dim)

        self.rnn = RNN_CLASS_MAPPING[rnn_style](
            embedding_dim, hidden_dim, num_layers, bidirectional=bidirectional
        )

        num_directions = 2 if bidirectional else 1
        self.rnn_output_dim = num_directions * hidden_dim
        self.hidden_state_dim = num_layers * num_directions * hidden_dim

    def forward(self, input_ids: Tensor):
        """
        Arguments
            input_ids: torch.LongTensor of shape (BS, L)

        Returns
            output: torch.FloatTensor of shape (BS, L, num_directions*hidden_dim)
            hidden_state: torch.FloatTensor of shape (BS, num_layers*num_directions, hidden_size)
        """
        embedded = self.embedding(input_ids)
        output, hidden_state = self.rnn(embedded)

        output = output.permute(1, 0, 2)
        hidden_state = hidden_state.permute(1, 0, 2)
        return output, hidden_state
