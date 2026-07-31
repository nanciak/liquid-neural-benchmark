"""
Global experiment configuration.

Every experiment in the benchmark uses these dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ModelConfig:
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    bidirectional: bool = False
    pooling: str = "last_hidden"

    def validate(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.pooling not in {"last_hidden"}:
            raise ValueError(f"Unsupported pooling: {self.pooling}")


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50
    patience: int = 10
    weight_decay: float = 1e-4


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
