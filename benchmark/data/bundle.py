"""Prepared dataset bundle shared by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch.utils.data import DataLoader

from benchmark.data.batch import SequenceBatch


@dataclass(slots=True)
class DataBundle:
    """Train, validation, and test loaders with dataset metadata."""

    train_loader: DataLoader[SequenceBatch]
    validation_loader: DataLoader[SequenceBatch]
    test_loader: DataLoader[SequenceBatch]
    metadata: Any
    split_manifest: Any | None = None

    def validate(self) -> None:
        """Validate that the bundle contains usable loaders and metadata."""

        if self.train_loader is None:
            raise ValueError("train_loader cannot be None.")

        if self.validation_loader is None:
            raise ValueError("validation_loader cannot be None.")

        if self.test_loader is None:
            raise ValueError("test_loader cannot be None.")

        if self.metadata is None:
            raise ValueError("metadata cannot be None.")

    @property
    def train_size(self) -> int:
        """Return the number of training samples when available."""

        return len(self.train_loader.dataset)

    @property
    def validation_size(self) -> int:
        """Return the number of validation samples when available."""

        return len(self.validation_loader.dataset)

    @property
    def test_size(self) -> int:
        """Return the number of test samples when available."""

        return len(self.test_loader.dataset)
