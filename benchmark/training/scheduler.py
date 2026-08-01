"""Learning-rate scheduler factory."""

from __future__ import annotations

from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
)


def build_scheduler(
    optimizer: Optimizer,
    *,
    scheduler: str = "none",
    **kwargs: Any,
):
    """Create a learning-rate scheduler by name."""

    name = scheduler.strip().lower()

    if name == "none":
        return None

    if name == "steplr":
        return StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 10),
            gamma=kwargs.get("gamma", 0.1),
        )

    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("t_max", 50),
        )

    if name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=kwargs.get("factor", 0.1),
            patience=kwargs.get("patience", 5),
        )

    raise ValueError(
        f"Unsupported scheduler: {scheduler}"
    )
