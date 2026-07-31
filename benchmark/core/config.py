
"""
Global experiment configuration.

Every experiment in the benchmark uses these dataclasses.
"""

from dataclasses import dataclass, field


@dataclass(slots=True)
class ModelConfig:

    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.2


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

    model: ModelConfig = field(
        default_factory=ModelConfig
    )

    training: TrainingConfig = field(
        default_factory=TrainingConfig
    )
