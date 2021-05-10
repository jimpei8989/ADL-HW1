from torch import Tensor
from typing import List

from torch import nn
from transformers import BertModel

from models.base import BaseBaseModel
from models.utils import build_fc_layers


class IntentClassifierBert(nn.Module, BaseBaseModel):
    def __init__(
        self,
        bert_name="bert-base-uncased",
        fc_layers: List[int] = [],
        output_dim: int = 1,
        activation: str = "ReLU",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.bert = BertModel.from_pretrained(bert_name)
        self.fc = build_fc_layers(
            [self.bert.config.hidden_size] + fc_layers + [output_dim], activation=activation
        )

    def forward(self, input_ids: Tensor):
        """
        Arguments
            input_ids: torch.LongTensor of shape (BS, L)

        Returns
            probability: torch.FloatTensor of shape (BS, C)
        """
        bert_output = self.bert(input_ids)
        return self.fc(bert_output.pooler_output)
