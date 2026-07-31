"""Abstract dataset interface used by benchmark datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

from benchmark.data.bundle import DataBundle

ConfigT = TypeVar("ConfigT")


class BaseDatasetAdapter(ABC, Generic[ConfigT]):
    """Base class implemented by every benchmark dataset."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique dataset name."""

    @abstractmethod
    def prepare(self) -> DataBundle:
        """Prepare train, validation, and test data."""

    @staticmethod
    def ensure_directory(path: Path) -> Path:
        """Create a directory if it does not already exist."""

        path.mkdir(parents=True, exist_ok=True)
        return path
