"""Discrete sequence models used by the benchmark."""

from __future__ import annotations

from torch import Tensor, nn

from benchmark.core.config import ModelConfig
from benchmark.models.base import SequenceModel


class _PoolingMixin:
    def _pool(self, x: Tensor) -> Tensor:
        return x[:, -1]


class CNN1DModel(SequenceModel, _PoolingMixin):
    def __init__(self, *, input_size:int, output_size:int, config:ModelConfig)->None:
        super().__init__(input_size=input_size, output_size=output_size)
        config.validate()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_size, config.hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_size, output_size)

    def forward(self, values:Tensor, timespans=None, observation_mask=None,
                padding_mask=None, lengths=None)->Tensor:
        del timespans, observation_mask, padding_mask, lengths
        self.validate_inputs(values)
        x = values.transpose(1,2)
        x = self.encoder(x).transpose(1,2)
        return self.head(self.dropout(self._pool(x)))


class TCNModel(SequenceModel, _PoolingMixin):
    def __init__(self, *, input_size:int, output_size:int, config:ModelConfig)->None:
        super().__init__(input_size=input_size, output_size=output_size)
        config.validate()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_size, config.hidden_size, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_size, output_size)

    def forward(self, values:Tensor, timespans=None, observation_mask=None,
                padding_mask=None, lengths=None)->Tensor:
        del timespans, observation_mask, padding_mask, lengths
        self.validate_inputs(values)
        x = values.transpose(1,2)
        x = self.encoder(x)
        x = x[..., :values.shape[1]].transpose(1,2)
        return self.head(self.dropout(self._pool(x)))


class TransformerModel(SequenceModel, _PoolingMixin):
    def __init__(self, *, input_size:int, output_size:int, config:ModelConfig)->None:
        super().__init__(input_size=input_size, output_size=output_size)
        config.validate()
        self.proj = nn.Linear(input_size, config.hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_size, output_size)

    def forward(self, values:Tensor, timespans=None, observation_mask=None,
                padding_mask=None, lengths=None)->Tensor:
        del timespans, observation_mask, lengths
        self.validate_inputs(values)
        x = self.proj(values)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.head(self.dropout(self._pool(x)))
