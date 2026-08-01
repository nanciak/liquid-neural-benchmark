"""Optimizer factory."""

from __future__ import annotations

import torch.optim as optim
from torch import nn
from torch.optim import Optimizer


def build_optimizer(
    *,
    model: nn.Module,
    learning_rate: float,
    weight_decay: float = 0.0,
    optimizer: str = "adam",
) -> Optimizer:
    """Create an optimizer by name."""

    name = optimizer.strip().lower()

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be positive."
        )

    if weight_decay < 0:
        raise ValueError(
            "weight_decay cannot be negative."
        )

    if name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    raise ValueError(
        f"Unsupported optimizer: {optimizer}"
    )
