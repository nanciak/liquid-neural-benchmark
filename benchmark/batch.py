"""Standard batch contract for sequence datasets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True)
class SequenceBatch:
    """Container passed from data loaders to trainers and evaluators."""

    values: Tensor
    targets: Tensor
    timespans: Tensor | None = None
    observation_mask: Tensor | None = None
    padding_mask: Tensor | None = None
    lengths: Tensor | None = None
    sample_ids: tuple[str, ...] | None = None

    def validate(self) -> None:
        """Validate tensor shapes and batch alignment."""

        if self.values.ndim != 3:
            raise ValueError(
                "values must have shape [batch, sequence, features]."
            )

        batch_size, sequence_length, _ = self.values.shape

        if self.targets.shape[0] != batch_size:
            raise ValueError(
                "targets must have the same batch size as values."
            )

        if self.timespans is not None:
            valid_shapes = {
                (batch_size, sequence_length),
                (batch_size, sequence_length, 1),
            }
            if tuple(self.timespans.shape) not in valid_shapes:
                raise ValueError(
                    "timespans must have shape [batch, sequence] "
                    "or [batch, sequence, 1]."
                )

        if self.observation_mask is not None:
            if self.observation_mask.shape != self.values.shape:
                raise ValueError(
                    "observation_mask must match values shape."
                )

        if self.padding_mask is not None:
            if tuple(self.padding_mask.shape) != (
                batch_size,
                sequence_length,
            ):
                raise ValueError(
                    "padding_mask must have shape [batch, sequence]."
                )

        if self.lengths is not None:
            if tuple(self.lengths.shape) != (batch_size,):
                raise ValueError(
                    "lengths must have shape [batch]."
                )

            if torch.any(self.lengths <= 0):
                raise ValueError(
                    "All sequence lengths must be positive."
                )

            if torch.any(self.lengths > sequence_length):
                raise ValueError(
                    "A sequence length exceeds the padded sequence length."
                )

        if self.sample_ids is not None:
            if len(self.sample_ids) != batch_size:
                raise ValueError(
                    "sample_ids must contain one identifier per sample."
                )

        for tensor_name in (
            "values",
            "timespans",
        ):
            tensor = getattr(self, tensor_name)
            if tensor is not None and not torch.isfinite(tensor).all():
                raise ValueError(
                    f"{tensor_name} contains NaN or infinite values."
                )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "SequenceBatch":
        """Return a copy with all tensors moved to the requested device."""

        def move(tensor: Tensor | None) -> Tensor | None:
            if tensor is None:
                return None

            return tensor.to(
                device=device,
                non_blocking=non_blocking,
            )

        return SequenceBatch(
            values=move(self.values),
            targets=move(self.targets),
            timespans=move(self.timespans),
            observation_mask=move(self.observation_mask),
            padding_mask=move(self.padding_mask),
            lengths=move(self.lengths),
            sample_ids=self.sample_ids,
        )

    def __len__(self) -> int:
        """Return the number of samples in the batch."""

        return int(self.values.shape[0])
