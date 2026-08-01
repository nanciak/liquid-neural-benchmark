"""Model checkpoint management."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


class CheckpointManager:
    """Save and restore model checkpoints."""

    def __init__(
        self,
        directory: Path,
        *,
        filename: str = "checkpoint.pt",
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / filename

    def save(
        self,
        *,
        model: nn.Module,
        epoch: int,
        validation_loss: float,
        optimizer: Optimizer | None = None,
        scheduler: Any | None = None,
        config: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        payload: dict[str, Any] = {
            "epoch": epoch,
            "validation_loss": float(validation_loss),
            "model_state_dict": model.state_dict(),
        }

        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()

        if config is not None:
            payload["config"] = (
                asdict(config)
                if is_dataclass(config)
                else config
            )

        if extra is not None:
            payload["extra"] = extra

        temporary_path = self.path.with_suffix(
            self.path.suffix + ".tmp"
        )

        torch.save(payload, temporary_path)
        temporary_path.replace(self.path)

        return self.path

    def load(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: Any | None = None,
        map_location=None,
    ) -> dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.path}"
            )

        checkpoint = torch.load(
            self.path,
            map_location=map_location,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if (
            optimizer is not None
            and "optimizer_state_dict" in checkpoint
        ):
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if (
            scheduler is not None
            and "scheduler_state_dict" in checkpoint
        ):
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        return checkpoint

    def exists(self) -> bool:
        return self.path.exists()
