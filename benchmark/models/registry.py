"""Central registry for benchmark model builders."""

from collections.abc import Callable

from benchmark.models.base import SequenceModel


ModelBuilder = Callable[[int, int, object], SequenceModel]
_MODEL_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(name: str, builder: ModelBuilder) -> None:
    """Register a model builder under a normalized unique name."""

    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("Model name cannot be empty.")
    if normalized_name in _MODEL_REGISTRY:
        raise ValueError(f"Model already registered: {normalized_name}")

    _MODEL_REGISTRY[normalized_name] = builder


def get_model_builder(name: str) -> ModelBuilder:
    """Return the registered builder for a model name."""

    normalized_name = name.strip().lower()
    try:
        return _MODEL_REGISTRY[normalized_name]
    except KeyError as error:
        available = ", ".join(sorted(_MODEL_REGISTRY))
        raise KeyError(
            f"Unknown model '{name}'. Available models: {available}"
        ) from error


def available_models() -> tuple[str, ...]:
    """Return all registered model names."""

    return tuple(sorted(_MODEL_REGISTRY))
