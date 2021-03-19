from torch import Tensor
from typing import List

from models.base import BaseModel
from models.utils import build_fc_layers


class IntentClassificationModel(BaseModel):
    def __init__(
        self,
        fc_layers: List[int] = [],
        output_dim: int = 1,
        activation: str = "ReLU",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.fc = build_fc_layers(
            [self.hidden_state_dim] + fc_layers + [output_dim], activation=activation
        )

    def forward(self, input_ids: Tensor):
        """
        Arguments
            input_ids: torch.LongTensor of shape (BS, L)

        Returns
            probability: torch.FloatTensor of shape (BS, C)
        """
        _, hidden_state = super().forward(input_ids)
        hidden_state = hidden_state.view(hidden_state.shape[0], -1)
        return self.fc(hidden_state)


def test():
    model = IntentClassificationModel()
    print(model)


if __name__ == "__main__":
    test()