"""Official ncps CfC and LTC wrappers for benchmark sequence tasks.

The ncps classes are subclassed only to preserve the final singleton
dimension of batched timespans. The official CfC/LTC cells and their
continuous-time computations are used unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

try:
    from ncps.torch import CfC as OfficialCfC
    from ncps.torch import LTC as OfficialLTC
except ImportError as error:
    raise ImportError(
        "The liquid-model dependency is missing. "
        "Install it with: pip install ncps==1.0.1"
    ) from error

from benchmark.models.base import SequenceModel


def _prepare_timespans(
    timespans: Tensor | None,
    values: Tensor,
) -> Tensor | None:
    """Return ncps-compatible elapsed times."""

    if timespans is None:
        return None

    if values.ndim != 3:
        raise ValueError(
            "values must have shape "
            "[batch, sequence, features]."
        )

    batch_size, sequence_length, _ = values.shape

    timespans = timespans.to(
        device=values.device,
        dtype=values.dtype,
    )

    if timespans.ndim == 2:
        timespans = timespans.unsqueeze(-1)

    if (
        timespans.ndim != 3
        or timespans.shape[-1] != 1
    ):
        raise ValueError(
            "timespans must have shape "
            "[batch, sequence] or "
            "[batch, sequence, 1]."
        )

    expected_shape = (
        batch_size,
        sequence_length,
        1,
    )

    if tuple(timespans.shape) != expected_shape:
        raise ValueError(
            "timespans must align with values. "
            f"Expected {expected_shape}, "
            f"received {tuple(timespans.shape)}."
        )

    if not torch.isfinite(timespans).all():
        raise ValueError(
            "timespans contain NaN or infinite values."
        )

    if torch.any(timespans < 0):
        raise ValueError(
            "timespans cannot contain negative values."
        )

    return timespans


def _validated_lengths(
    lengths: Tensor | None,
    values: Tensor,
) -> Tensor:
    """Return validated sequence lengths."""

    batch_size, sequence_length, _ = values.shape

    if lengths is None:
        return torch.full(
            (batch_size,),
            sequence_length,
            dtype=torch.long,
            device=values.device,
        )

    if (
        lengths.ndim != 1
        or lengths.shape[0] != batch_size
    ):
        raise ValueError(
            "lengths must have shape [batch]."
        )

    lengths = lengths.to(
        device=values.device,
        dtype=torch.long,
    )

    if torch.any(lengths <= 0):
        raise ValueError(
            "All sequence lengths must be positive."
        )

    if torch.any(lengths > sequence_length):
        raise ValueError(
            "A sequence length exceeds the "
            "padded sequence dimension."
        )

    return lengths


def _last_valid_output(
    sequence_outputs: Tensor,
    lengths: Tensor,
) -> Tensor:
    """Select the final valid timestep for each sample."""

    batch_indices = torch.arange(
        sequence_outputs.shape[0],
        device=sequence_outputs.device,
    )

    return sequence_outputs[
        batch_indices,
        lengths - 1,
    ]


def _readout(
    layer: Any,
    hidden_output: Tensor,
) -> Tensor:
    """Apply the official layer readout when it exists.

    Official CfC has an ``fc`` readout module. Official LTC with integer
    units returns its cell output directly and does not expose ``fc``.
    """

    projection = getattr(
        layer,
        "fc",
        None,
    )

    if projection is None:
        return hidden_output

    return projection(hidden_output)


def _broadcast_safe_forward(
    layer: Any,
    input_tensor: Tensor,
    hx: Tensor | tuple[Tensor, Tensor] | None = None,
    timespans: Tensor | None = None,
) -> tuple[
    Tensor,
    Tensor | tuple[Tensor, Tensor],
]:
    """Mirror the official ncps forward loop with safe timespans.

    The official loop calls ``squeeze()`` on each batched timespan,
    changing [batch, 1] into [batch]. That cannot broadcast against
    [batch, hidden]. This version keeps elapsed time as [batch, 1].
    """

    device = input_tensor.device
    is_batched = input_tensor.dim() == 3

    batch_dimension = (
        0 if layer.batch_first else 1
    )
    sequence_dimension = (
        1 if layer.batch_first else 0
    )

    if not is_batched:
        input_tensor = input_tensor.unsqueeze(
            batch_dimension
        )

        if timespans is not None:
            timespans = timespans.unsqueeze(
                batch_dimension
            )

    batch_size = input_tensor.size(
        batch_dimension
    )
    sequence_length = input_tensor.size(
        sequence_dimension
    )

    if hx is None:
        hidden_state = torch.zeros(
            (batch_size, layer.state_size),
            device=device,
            dtype=input_tensor.dtype,
        )

        cell_state = (
            torch.zeros(
                (batch_size, layer.state_size),
                device=device,
                dtype=input_tensor.dtype,
            )
            if layer.use_mixed
            else None
        )

    else:
        if (
            layer.use_mixed
            and isinstance(hx, Tensor)
        ):
            raise RuntimeError(
                "mixed_memory=True requires "
                "a tuple (h0, c0)."
            )

        hidden_state, cell_state = (
            hx
            if layer.use_mixed
            else (hx, None)
        )

        if is_batched:
            if hidden_state.dim() != 2:
                raise RuntimeError(
                    "For batched input, hidden state "
                    "must be two-dimensional."
                )

        else:
            if hidden_state.dim() != 1:
                raise RuntimeError(
                    "For unbatched input, hidden state "
                    "must be one-dimensional."
                )

            hidden_state = hidden_state.unsqueeze(
                0
            )

            if cell_state is not None:
                cell_state = cell_state.unsqueeze(
                    0
                )

    output_sequence: list[Tensor] = []

    for timestep in range(
        sequence_length
    ):
        if layer.batch_first:
            inputs = input_tensor[
                :,
                timestep,
            ]

            elapsed_time = (
                1.0
                if timespans is None
                else timespans[
                    :,
                    timestep,
                ].reshape(-1, 1)
            )

        else:
            inputs = input_tensor[
                timestep
            ]

            elapsed_time = (
                1.0
                if timespans is None
                else timespans[
                    timestep
                ].reshape(-1, 1)
            )

        if layer.use_mixed:
            hidden_state, cell_state = (
                layer.lstm(
                    inputs,
                    (
                        hidden_state,
                        cell_state,
                    ),
                )
            )

        hidden_output, hidden_state = (
            layer.rnn_cell.forward(
                inputs,
                hidden_state,
                elapsed_time,
            )
        )

        if layer.return_sequences:
            output_sequence.append(
                _readout(
                    layer,
                    hidden_output,
                )
            )

    if layer.return_sequences:
        stack_dimension = (
            1 if layer.batch_first else 0
        )

        readout = torch.stack(
            output_sequence,
            dim=stack_dimension,
        )

    else:
        readout = _readout(
            layer,
            hidden_output,
        )

    final_state: (
        Tensor | tuple[Tensor, Tensor]
    ) = (
        (hidden_state, cell_state)
        if layer.use_mixed
        else hidden_state
    )

    if not is_batched:
        readout = readout.squeeze(
            batch_dimension
        )

        final_state = (
            (
                hidden_state[0],
                cell_state[0],
            )
            if layer.use_mixed
            else hidden_state[0]
        )

    return readout, final_state


class BroadcastSafeCfC(OfficialCfC):
    """Official ncps CfC with safe batched elapsed-time broadcasting."""

    def forward(
        self,
        input: Tensor,
        hx: Tensor | tuple[Tensor, Tensor] | None = None,
        timespans: Tensor | None = None,
    ) -> tuple[
        Tensor,
        Tensor | tuple[Tensor, Tensor],
    ]:
        return _broadcast_safe_forward(
            self,
            input,
            hx=hx,
            timespans=timespans,
        )


class BroadcastSafeLTC(OfficialLTC):
    """Official ncps LTC with safe batched elapsed-time broadcasting."""

    def forward(
        self,
        input: Tensor,
        hx: Tensor | tuple[Tensor, Tensor] | None = None,
        timespans: Tensor | None = None,
    ) -> tuple[
        Tensor,
        Tensor | tuple[Tensor, Tensor],
    ]:
        return _broadcast_safe_forward(
            self,
            input,
            hx=hx,
            timespans=timespans,
        )


class CfCModel(SequenceModel):
    """Sequence model backed by the official ncps CfC cell."""

    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        hidden_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            input_size=input_size,
            output_size=output_size,
        )

        if hidden_size <= 0:
            raise ValueError(
                "hidden_size must be positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)."
            )

        self.recurrent = BroadcastSafeCfC(
            input_size,
            hidden_size,
            return_sequences=True,
            batch_first=True,
        )

        self.output_dropout = nn.Dropout(
            dropout
        )

        self.head = nn.Linear(
            hidden_size,
            output_size,
        )

    def forward(
        self,
        values: Tensor,
        timespans: Tensor | None = None,
        observation_mask: Tensor | None = None,
        padding_mask: Tensor | None = None,
        lengths: Tensor | None = None,
    ) -> Tensor:
        del observation_mask
        del padding_mask

        self.validate_inputs(
            values,
            lengths,
        )

        validated_lengths = _validated_lengths(
            lengths,
            values,
        )

        validated_timespans = _prepare_timespans(
            timespans,
            values,
        )

        sequence_outputs, _ = self.recurrent(
            values,
            timespans=validated_timespans,
        )

        representation = _last_valid_output(
            sequence_outputs,
            validated_lengths,
        )

        return self.head(
            self.output_dropout(
                representation
            )
        )


class LTCModel(SequenceModel):
    """Sequence model backed by the official ncps LTC cell."""

    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        hidden_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            input_size=input_size,
            output_size=output_size,
        )

        if hidden_size <= 0:
            raise ValueError(
                "hidden_size must be positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)."
            )

        self.recurrent = BroadcastSafeLTC(
            input_size,
            hidden_size,
            return_sequences=True,
            batch_first=True,
        )

        self.output_dropout = nn.Dropout(
            dropout
        )

        self.head = nn.Linear(
            hidden_size,
            output_size,
        )

    def forward(
        self,
        values: Tensor,
        timespans: Tensor | None = None,
        observation_mask: Tensor | None = None,
        padding_mask: Tensor | None = None,
        lengths: Tensor | None = None,
    ) -> Tensor:
        del observation_mask
        del padding_mask

        self.validate_inputs(
            values,
            lengths,
        )

        validated_lengths = _validated_lengths(
            lengths,
            values,
        )

        validated_timespans = _prepare_timespans(
            timespans,
            values,
        )

        sequence_outputs, _ = self.recurrent(
            values,
            timespans=validated_timespans,
        )

        representation = _last_valid_output(
            sequence_outputs,
            validated_lengths,
        )

        return self.head(
            self.output_dropout(
                representation
            )
        )
