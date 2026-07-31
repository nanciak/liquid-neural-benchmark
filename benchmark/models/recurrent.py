"""PyTorch GRU and LSTM sequence-model wrappers."""

from __future__ import annotations

from typing import TypeAlias

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence

from benchmark.core.config import ModelConfig
from benchmark.models.base import SequenceModel


RecurrentState: TypeAlias = Tensor | tuple[Tensor, Tensor]


def _validated_lengths(
    lengths: Tensor | None,
    values: Tensor,
) -> Tensor:
    """Return validated sequence lengths on the same device as values."""

    batch_size, sequence_length, _ = values.shape

    if lengths is None:
        return torch.full(
            (batch_size,),
            sequence_length,
            dtype=torch.long,
            device=values.device,
        )

    if lengths.ndim != 1 or lengths.shape[0] != batch_size:
        raise ValueError("lengths must have shape [batch].")

    validated = lengths.to(
        device=values.device,
        dtype=torch.long,
    )

    if torch.any(validated <= 0):
        raise ValueError("All sequence lengths must be positive.")

    if torch.any(validated > sequence_length):
        raise ValueError(
            "A sequence length exceeds the padded sequence dimension."
        )

    return validated


def _last_hidden_representation(
    hidden_state: RecurrentState,
    *,
    num_layers: int,
    bidirectional: bool,
) -> Tensor:
    """Return the last-layer hidden representation."""

    hidden = (
        hidden_state[0]
        if isinstance(hidden_state, tuple)
        else hidden_state
    )

    direction_count = 2 if bidirectional else 1
    expected_first_dimension = num_layers * direction_count

    if hidden.ndim != 3:
        raise RuntimeError(
            "Recurrent hidden state must have shape "
            "[layers * directions, batch, hidden]."
        )

    if hidden.shape[0] != expected_first_dimension:
        raise RuntimeError(
            "Unexpected recurrent hidden-state layer count."
        )

    hidden = hidden.view(
        num_layers,
        direction_count,
        hidden.shape[1],
        hidden.shape[2],
    )

    last_layer = hidden[-1]

    if bidirectional:
        return torch.cat(
            (last_layer[0], last_layer[1]),
            dim=-1,
        )

    return last_layer[0]


class _RecurrentModel(SequenceModel):
    """Shared implementation for GRU and LSTM models."""

    recurrent_class: type[nn.GRU] | type[nn.LSTM]

    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        config: ModelConfig,
    ) -> None:
        super().__init__(
            input_size=input_size,
            output_size=output_size,
        )

        config.validate()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional
        self.pooling = config.pooling

        recurrent_dropout = (
            config.dropout
            if config.num_layers > 1
            else 0.0
        )

        self.encoder = self.recurrent_class(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=config.bidirectional,
        )

        representation_size = (
            config.hidden_size
            * (2 if config.bidirectional else 1)
        )

        self.output_dropout = nn.Dropout(
            config.dropout
        )

        self.output_head = nn.Linear(
            representation_size,
            output_size,
        )

    def _encode(
        self,
        values: Tensor,
        lengths: Tensor,
    ) -> RecurrentState:
        packed = pack_padded_sequence(
            values,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden_state = self.encoder(packed)

        return hidden_state

    def forward(
        self,
        values: Tensor,
        timespans: Tensor | None = None,
        observation_mask: Tensor | None = None,
        padding_mask: Tensor | None = None,
        lengths: Tensor | None = None,
    ) -> Tensor:
        del timespans, observation_mask, padding_mask

        self.validate_inputs(values, lengths)

        validated_lengths = _validated_lengths(
            lengths,
            values,
        )

        hidden_state = self._encode(
            values,
            validated_lengths,
        )

        representation = _last_hidden_representation(
            hidden_state,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
        )

        return self.output_head(
            self.output_dropout(
                representation
            )
        )


class GRUModel(_RecurrentModel):
    """Sequence classifier or regressor backed by PyTorch GRU."""

    recurrent_class = nn.GRU


class LSTMModel(_RecurrentModel):
    """Sequence classifier or regressor backed by PyTorch LSTM."""

    recurrent_class = nn.LSTM
