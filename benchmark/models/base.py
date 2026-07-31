"""Base classes for benchmark sequence models."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class SequenceModel(nn.Module, ABC):
    """Abstract base class for all benchmark sequence models."""

    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
    ) -> None:
        super().__init__()

        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        if output_size <= 0:
            raise ValueError("output_size must be positive.")

        self.input_size = input_size
        self.output_size = output_size

    def validate_inputs(
        self,
        values: Tensor,
        lengths: Tensor | None = None,
    ) -> None:
        """Validate common input constraints."""

        if values.ndim != 3:
            raise ValueError(
                "Expected input shape [batch, sequence, features]."
            )

        if values.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, "
                f"received {values.shape[-1]}."
            )

        if lengths is not None:
            if lengths.ndim != 1:
                raise ValueError("lengths must have shape [batch].")
            if lengths.shape[0] != values.shape[0]:
                raise ValueError(
                    "Batch size of lengths does not match inputs."
                )

    @abstractmethod
    def forward(
        self,
        values: Tensor,
        timespans: Tensor | None = None,
        observation_mask: Tensor | None = None,
        padding_mask: Tensor | None = None,
        lengths: Tensor | None = None,
    ) -> Tensor:
        """Run a forward pass."""
        raise NotImplementedError
