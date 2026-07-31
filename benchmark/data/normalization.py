"""Feature-wise normalization utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True)
class TrainOnlyStandardizer:
    """Fit statistics on the training split and reuse them elsewhere."""

    epsilon: float = 1e-8
    mean_: Tensor | None = None
    scale_: Tensor | None = None
    fitted_split_: str | None = None

    def fit(
        self,
        values: Tensor,
        *,
        split_name: str,
        observation_mask: Tensor | None = None,
    ) -> "TrainOnlyStandardizer":
        if values.ndim != 3:
            raise ValueError("values must have shape [batch, sequence, features].")

        if observation_mask is None:
            flat = values.reshape(-1, values.shape[-1])
        else:
            if observation_mask.shape != values.shape:
                raise ValueError("observation_mask must match values.")
            flat = values[observation_mask.bool()].reshape(-1, values.shape[-1])

        self.mean_ = flat.mean(dim=0)
        self.scale_ = flat.std(dim=0, unbiased=False).clamp_min(self.epsilon)
        self.fitted_split_ = split_name
        return self

    def transform(
        self,
        values: Tensor,
        observation_mask: Tensor | None = None,
    ) -> Tensor:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer has not been fitted.")

        transformed = (values - self.mean_) / self.scale_

        if observation_mask is not None:
            transformed = torch.where(
                observation_mask.bool(),
                transformed,
                values,
            )

        return transformed
