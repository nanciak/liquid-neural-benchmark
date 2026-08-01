"""Central construction of benchmark datasets."""

from __future__ import annotations

from typing import Any

from benchmark.data.bundle import DataBundle
from benchmark.data.registry import (
    available_datasets as registered_datasets,
    get_dataset_builder,
    register_dataset,
)
from benchmark.data.tiny import (
    TinySequenceAdapter,
    TinySequenceConfig,
)
from benchmark.data.uci_har import (
    UCIHARAdapter,
    UCIHARConfig,
)


def _build_tiny_sequence(
    config: TinySequenceConfig,
) -> DataBundle:
    return TinySequenceAdapter(config).prepare()


def _build_uci_har(
    config: UCIHARConfig,
) -> DataBundle:
    return UCIHARAdapter(config).prepare()


for _name, _builder in {
    "tiny_sequence": _build_tiny_sequence,
    "uci_har": _build_uci_har,
}.items():
    try:
        register_dataset(_name, _builder)
    except ValueError:
        pass


def available_datasets() -> tuple[str, ...]:
    return registered_datasets()


def build_dataset(
    *,
    name: str,
    config: Any,
) -> DataBundle:
    bundle = get_dataset_builder(name)(config)

    if not isinstance(bundle, DataBundle):
        raise TypeError(
            "Registered dataset builders must return a DataBundle."
        )

    bundle.validate()
    return bundle
