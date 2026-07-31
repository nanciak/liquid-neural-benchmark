"""Deterministic dataset splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random
from typing import Iterable, Sequence


@dataclass(slots=True, frozen=True)
class SplitManifest:
    """Serializable description of train, validation, and test membership."""

    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    strategy: str
    seed: int

    def validate(self) -> None:
        """Validate that all splits are non-empty and disjoint."""

        if not self.train_ids:
            raise ValueError("train_ids cannot be empty.")
        if not self.validation_ids:
            raise ValueError("validation_ids cannot be empty.")
        if not self.test_ids:
            raise ValueError("test_ids cannot be empty.")
        if not self.strategy.strip():
            raise ValueError("strategy cannot be empty.")
        if self.seed < 0:
            raise ValueError("seed cannot be negative.")

        train = set(self.train_ids)
        validation = set(self.validation_ids)
        test = set(self.test_ids)

        if len(train) != len(self.train_ids):
            raise ValueError("train_ids contain duplicates.")
        if len(validation) != len(self.validation_ids):
            raise ValueError("validation_ids contain duplicates.")
        if len(test) != len(self.test_ids):
            raise ValueError("test_ids contain duplicates.")

        if train & validation:
            raise ValueError("Train and validation splits overlap.")
        if train & test:
            raise ValueError("Train and test splits overlap.")
        if validation & test:
            raise ValueError("Validation and test splits overlap.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "train_ids": list(self.train_ids),
            "validation_ids": list(self.validation_ids),
            "test_ids": list(self.test_ids),
            "strategy": self.strategy,
            "seed": self.seed,
        }

    def write_json(self, path: Path) -> None:
        """Write the manifest atomically as formatted JSON."""

        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)

        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def deterministic_group_validation_split(
    train_group_ids: Sequence[str | int],
    test_group_ids: Sequence[str | int],
    *,
    validation_fraction: float,
    seed: int,
    strategy: str,
) -> SplitManifest:
    """
    Split official training groups into train and validation groups.

    The official test groups remain untouched.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if seed < 0:
        raise ValueError("seed cannot be negative.")

    train_groups = tuple(sorted({str(value) for value in train_group_ids}))
    test_groups = tuple(sorted({str(value) for value in test_group_ids}))

    if len(train_groups) < 2:
        raise ValueError(
            "At least two unique training groups are required."
        )
    if not test_groups:
        raise ValueError("At least one test group is required.")
    if set(train_groups) & set(test_groups):
        raise ValueError("Training and test groups overlap.")

    shuffled = list(train_groups)
    random.Random(seed).shuffle(shuffled)

    validation_count = max(
        1,
        round(len(shuffled) * validation_fraction),
    )
    validation_count = min(
        validation_count,
        len(shuffled) - 1,
    )

    validation_ids = tuple(sorted(shuffled[:validation_count]))
    final_train_ids = tuple(sorted(shuffled[validation_count:]))

    manifest = SplitManifest(
        train_ids=final_train_ids,
        validation_ids=validation_ids,
        test_ids=test_groups,
        strategy=strategy,
        seed=seed,
    )
    manifest.validate()
    return manifest


def filter_ids(
    identifiers: Iterable[str | int],
    allowed_ids: Iterable[str | int],
) -> list[int]:
    """Return positions whose identifier belongs to the allowed set."""

    allowed = {str(value) for value in allowed_ids}

    return [
        index
        for index, identifier in enumerate(identifiers)
        if str(identifier) in allowed
    ]
