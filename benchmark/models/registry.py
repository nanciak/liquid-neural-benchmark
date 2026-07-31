"""Central registry for benchmark model builders."""

from __future__ import annotations

from collections.abc import Callable

from benchmark.models.base import SequenceModel


ModelBuilder = Callable[[int, int, object], SequenceModel]
_MODEL_REGISTRY: dict[str, ModelBuilder] = {}


def _normalize_model_name(name: str) -> str:
    """Return a validated, normalized model name."""

    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("Model name cannot be empty.")

    return normalized_name


def register_model(name: str, builder: ModelBuilder) -> None:
    """Register a model builder under a normalized unique name."""

    normalized_name = _normalize_model_name(name)

    existing_builder = _MODEL_REGISTRY.get(normalized_name)
    if existing_builder is not None:
        if existing_builder is builder:
            return
        raise ValueError(f"Model already registered: {normalized_name}")

    _MODEL_REGISTRY[normalized_name] = builder


def get_model_builder(name: str) -> ModelBuilder:
    """Return the registered builder for a model name."""

    normalized_name = _normalize_model_name(name)

    try:
        return _MODEL_REGISTRY[normalized_name]
    except KeyError as error:
        available = ", ".join(available_models()) or "none"
        raise KeyError(
            f"Unknown model '{name}'. Available models: {available}"
        ) from error


def available_models() -> tuple[str, ...]:
    """Return all registered model names in alphabetical order."""

    return tuple(sorted(_MODEL_REGISTRY))
