from torch import Tensor
from typing import List

from models.base import BaseModel
from models.utils import build_fc_layers


class SlotTagger(BaseModel):
    def __init__(
        self,
        fc_layers: List[int] = [],
        output_dim: int = 1,
        activation: str = "ReLU",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.fc = build_fc_layers(
            [self.rnn_output_dim] + fc_layers + [output_dim],
            activation=activation,
        )

    def forward(self, input_ids: Tensor):
        """
        Arguments
            input_ids: torch.LongTensor of shape (BS, L)

        Returns
            probability: torch.FloatTensor of shape (BS, C)
        """
        output, _ = super().forward(input_ids)
        return self.fc(output)


def test():
    model = SlotTagger()
    print(model)


if __name__ == "__main__":
    test()
