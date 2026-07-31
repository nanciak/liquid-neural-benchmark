"""Tests for the benchmark data infrastructure."""

from __future__ import annotations

import torch

from benchmark.data import (
    DataBundle,
    DatasetMetadata,
    SequenceBatch,
    SplitManifest,
    TinySequenceAdapter,
    TinySequenceConfig,
    TrainOnlyStandardizer,
)


def test_sequence_batch_validate() -> None:
    batch = SequenceBatch(
        values=torch.randn(4, 8, 3),
        targets=torch.randint(0, 2, (4,)),
        lengths=torch.full((4,), 8),
    )
    batch.validate()


def test_standardizer_fit_transform() -> None:
    values = torch.randn(10, 8, 3)
    standardizer = TrainOnlyStandardizer().fit(
        values,
        split_name="train",
    )
    transformed = standardizer.transform(values)
    assert transformed.shape == values.shape
    assert torch.isfinite(transformed).all()


def test_split_manifest_validate() -> None:
    manifest = SplitManifest(
        train_ids=("1", "2"),
        validation_ids=("3",),
        test_ids=("4",),
        strategy="deterministic",
        seed=42,
    )
    manifest.validate()


def test_tiny_dataset_prepare() -> None:
    adapter = TinySequenceAdapter(
        TinySequenceConfig()
    )
    bundle = adapter.prepare()

    assert isinstance(bundle, DataBundle)
    bundle.validate()

    assert isinstance(
        bundle.metadata,
        DatasetMetadata,
    )

    batch = next(iter(bundle.train_loader))
    batch.validate()

    assert batch.values.ndim == 3
    assert batch.values.shape[-1] == bundle.metadata.input_size
