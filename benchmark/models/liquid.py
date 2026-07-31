"""Official ncps CfC and LTC wrappers for benchmark sequence tasks."""

from __future__ import annotations

import torch
from torch import Tensor, nn

try:
    from ncps.torch import CfC, LTC
except ImportError as error:
    raise ImportError(
        "The liquid-model dependency is missing. Install it with: pip install ncps==1.0.1"
    ) from error

from benchmark.models.base import SequenceModel


def _prepare_timespans(
    timespans: Tensor | None,
    values: Tensor,
) -> Tensor | None:
    """Return ncps-compatible elapsed times with shape [batch, time, 1]."""

    if timespans is None:
        return None

    batch_size, sequence_length, _ = values.shape

    if timespans.ndim == 2:
        timespans = timespans.unsqueeze(-1)
    elif timespans.ndim != 3 or timespans.shape[-1] != 1:
        raise ValueError(
            "timespans must have shape [batch, time] or [batch, time, 1]."
        )

    if tuple(timespans.shape) != (batch_size, sequence_length, 1):
        raise ValueError(
            "timespans must align with the batch and sequence dimensions of values."
        )

    if not torch.isfinite(timespans).all():
        raise ValueError("timespans contain NaN or infinite values.")
    if torch.any(timespans < 0):
        raise ValueError("timespans cannot contain negative values.")

    return timespans.to(device=values.device, dtype=values.dtype)


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

    lengths = lengths.to(device=values.device, dtype=torch.long)
    if torch.any(lengths <= 0):
        raise ValueError("All sequence lengths must be positive.")
    if torch.any(lengths > sequence_length):
        raise ValueError("A sequence length exceeds the padded sequence dimension.")

    return lengths


def _last_valid_output(sequence_outputs: Tensor, lengths: Tensor) -> Tensor:
    batch_indices = torch.arange(
        sequence_outputs.shape[0],
        device=sequence_outputs.device,
    )
    return sequence_outputs[batch_indices, lengths - 1]


class CfCModel(SequenceModel):
    """Sequence classifier or regressor backed by the official ncps CfC layer."""

    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        hidden_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(input_size=input_size, output_size=output_size)

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.recurrent = CfC(
            input_size,
            hidden_size,
            return_sequences=True,
            batch_first=True,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        values: Tensor,
        timespans: Tensor | None = None,
        observation_mask: Tensor | None = None,
        padding_mask: Tensor | None = None,
        lengths: Tensor | None = None,
    ) -> Tensor:
        del observation_mask, padding_mask

        self.validate_inputs(values, lengths)
        validated_lengths = _validated_lengths(lengths, values)
        validated_timespans = _prepare_timespans(timespans, values)

        sequence_outputs, _ = self.recurrent(
            values,
            timespans=validated_timespans,
        )
        representation = _last_valid_output(sequence_outputs, validated_lengths)
        return self.head(self.output_dropout(representation))


class LTCModel(SequenceModel):
    """Sequence classifier or regressor backed by the official ncps LTC layer."""

    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        hidden_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(input_size=input_size, output_size=output_size)

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.recurrent = LTC(
            input_size,
            hidden_size,
            return_sequences=True,
            batch_first=True,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        values: Tensor,
        timespans: Tensor | None = None,
        observation_mask: Tensor | None = None,
        padding_mask: Tensor | None = None,
        lengths: Tensor | None = None,
    ) -> Tensor:
        del observation_mask, padding_mask

        self.validate_inputs(values, lengths)
        validated_lengths = _validated_lengths(lengths, values)
        validated_timespans = _prepare_timespans(timespans, values)

        sequence_outputs, _ = self.recurrent(
            values,
            timespans=validated_timespans,
        )
        representation = _last_valid_output(sequence_outputs, validated_lengths)
        return self.head(self.output_dropout(representation))
