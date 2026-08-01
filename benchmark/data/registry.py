"""Central registry for benchmark dataset builders."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from benchmark.data.bundle import DataBundle


DatasetBuilder = Callable[[Any], DataBundle]
_DATASET_REGISTRY: dict[str, DatasetBuilder] = {}


def _normalize_dataset_name(name: str) -> str:
    """Return a validated, normalized dataset name."""

    normalized_name = name.strip().lower()

    if not normalized_name:
        raise ValueError("Dataset name cannot be empty.")

    return normalized_name


def register_dataset(
    name: str,
    builder: DatasetBuilder,
) -> None:
    """Register a dataset builder under a normalized unique name."""

    normalized_name = _normalize_dataset_name(name)

    existing_builder = _DATASET_REGISTRY.get(normalized_name)

    if existing_builder is not None:
        if existing_builder is builder:
            return

        raise ValueError(
            f"Dataset already registered: {normalized_name}"
        )

    _DATASET_REGISTRY[normalized_name] = builder


def get_dataset_builder(
    name: str,
) -> DatasetBuilder:
    """Return the registered builder for a dataset name."""

    normalized_name = _normalize_dataset_name(name)

    try:
        return _DATASET_REGISTRY[normalized_name]

    except KeyError as error:
        available = ", ".join(available_datasets()) or "none"

        raise KeyError(
            f"Unknown dataset '{name}'. "
            f"Available datasets: {available}"
        ) from error


def available_datasets() -> tuple[str, ...]:
    """Return all registered dataset names in alphabetical order."""

    return tuple(sorted(_DATASET_REGISTRY))
