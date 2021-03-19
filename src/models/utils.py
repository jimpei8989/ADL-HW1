from typing import List, Optional

from torch import nn
from torch.nn import Dropout, Linear, ReLU


ACTIVATION_MAPPING = {
    "ReLU": ReLU,
}


def build_fc_layers(fc_layers: List, activation: Optional[str] = None, dropout: float = 0.0):
    components = []

    for i, (x, y) in enumerate(zip(fc_layers[:-1], fc_layers[1:])):
        if i != 0:
            if activation:
                components.append(ACTIVATION_MAPPING[activation])
            if dropout > 0:
                components.append(Dropout(dropout))

        components.append(Linear(x, y))

    return nn.Sequential(*components)
