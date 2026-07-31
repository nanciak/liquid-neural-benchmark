"""Tiny synthetic sequence dataset for end-to-end framework validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from benchmark.data.base import BaseDatasetAdapter
from benchmark.data.batch import SequenceBatch
from benchmark.data.bundle import DataBundle
from benchmark.data.metadata import DatasetMetadata
from benchmark.data.normalization import TrainOnlyStandardizer


@dataclass(slots=True, frozen=True)
class TinySequenceConfig:
    """Configuration for the synthetic sequence dataset."""

    root: Path = Path("datasets/tiny_sequence")
    batch_size: int = 16
    sequence_length: int = 12
    input_size: int = 4
    output_size: int = 3
    train_size: int = 96
    validation_size: int = 24
    test_size: int = 24
    seed: int = 42
    num_workers: int = 0

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if self.input_size <= 0:
            raise ValueError("input_size must be positive.")
        if self.output_size <= 1:
            raise ValueError("output_size must be greater than one.")

        for field_name in (
            "train_size",
            "validation_size",
            "test_size",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive.")

        if self.seed < 0:
            raise ValueError("seed cannot be negative.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")


class TinySequenceDataset(Dataset[SequenceBatch]):
    """In-memory synthetic dataset with deterministic class structure."""

    def __init__(
        self,
        values: Tensor,
        targets: Tensor,
        *,
        split_name: str,
    ) -> None:
        if values.ndim != 3:
            raise ValueError(
                "values must have shape [samples, sequence, features]."
            )
        if targets.ndim != 1:
            raise ValueError("targets must have shape [samples].")
        if values.shape[0] != targets.shape[0]:
            raise ValueError(
                "values and targets must contain the same number of samples."
            )

        self.values = values
        self.targets = targets
        self.split_name = split_name

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def __getitem__(self, index: int) -> SequenceBatch:
        sequence_length = int(self.values.shape[1])

        return SequenceBatch(
            values=self.values[index],
            targets=self.targets[index],
            timespans=torch.ones(
                sequence_length,
                dtype=self.values.dtype,
            ),
            observation_mask=torch.ones_like(
                self.values[index],
                dtype=torch.bool,
            ),
            padding_mask=torch.zeros(
                sequence_length,
                dtype=torch.bool,
            ),
            lengths=torch.tensor(
                sequence_length,
                dtype=torch.long,
            ),
            sample_ids=(
                f"{self.split_name}_{index:05d}",
            ),
        )


def collate_sequence_batches(
    samples: list[SequenceBatch],
) -> SequenceBatch:
    """Combine individual samples into one batch."""

    if not samples:
        raise ValueError("Cannot collate an empty sample list.")

    batch = SequenceBatch(
        values=torch.stack(
            [sample.values for sample in samples]
        ),
        targets=torch.stack(
            [sample.targets for sample in samples]
        ),
        timespans=torch.stack(
            [sample.timespans for sample in samples]
        ),
        observation_mask=torch.stack(
            [sample.observation_mask for sample in samples]
        ),
        padding_mask=torch.stack(
            [sample.padding_mask for sample in samples]
        ),
        lengths=torch.stack(
            [sample.lengths for sample in samples]
        ),
        sample_ids=tuple(
            sample.sample_ids[0]
            for sample in samples
        ),
    )

    batch.validate()
    return batch


def _generate_split(
    *,
    sample_count: int,
    sequence_length: int,
    input_size: int,
    output_size: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Generate a deterministic learnable multiclass sequence split."""

    generator = torch.Generator().manual_seed(seed)

    targets = torch.randint(
        low=0,
        high=output_size,
        size=(sample_count,),
        generator=generator,
    )

    values = torch.randn(
        sample_count,
        sequence_length,
        input_size,
        generator=generator,
    )

    class_offsets = torch.linspace(
        -1.5,
        1.5,
        steps=output_size,
    )

    values[:, :, 0] += class_offsets[targets].unsqueeze(1)

    time_axis = torch.linspace(
        0.0,
        1.0,
        steps=sequence_length,
    )

    if input_size > 1:
        values[:, :, 1] += (
            targets.float().unsqueeze(1)
            / max(output_size - 1, 1)
        ) * time_axis.unsqueeze(0)

    return values, targets


class TinySequenceAdapter(
    BaseDatasetAdapter[TinySequenceConfig]
):
    """Prepare a deterministic synthetic benchmark dataset."""

    @property
    def name(self) -> str:
        return "tiny_sequence_dataset"

    def prepare(self) -> DataBundle:
        self.config.validate()
        self.ensure_directory(self.config.root)

        train_values, train_targets = _generate_split(
            sample_count=self.config.train_size,
            sequence_length=self.config.sequence_length,
            input_size=self.config.input_size,
            output_size=self.config.output_size,
            seed=self.config.seed,
        )

        validation_values, validation_targets = _generate_split(
            sample_count=self.config.validation_size,
            sequence_length=self.config.sequence_length,
            input_size=self.config.input_size,
            output_size=self.config.output_size,
            seed=self.config.seed + 1,
        )

        test_values, test_targets = _generate_split(
            sample_count=self.config.test_size,
            sequence_length=self.config.sequence_length,
            input_size=self.config.input_size,
            output_size=self.config.output_size,
            seed=self.config.seed + 2,
        )

        standardizer = TrainOnlyStandardizer().fit(
            train_values,
            split_name="train",
        )

        train_values = standardizer.transform(train_values)
        validation_values = standardizer.transform(
            validation_values
        )
        test_values = standardizer.transform(test_values)

        train_dataset = TinySequenceDataset(
            train_values,
            train_targets,
            split_name="train",
        )
        validation_dataset = TinySequenceDataset(
            validation_values,
            validation_targets,
            split_name="validation",
        )
        test_dataset = TinySequenceDataset(
            test_values,
            test_targets,
            split_name="test",
        )

        loader_kwargs = {
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "collate_fn": collate_sequence_batches,
        }

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            generator=torch.Generator().manual_seed(
                self.config.seed
            ),
            **loader_kwargs,
        )

        validation_loader = DataLoader(
            validation_dataset,
            shuffle=False,
            **loader_kwargs,
        )

        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            **loader_kwargs,
        )

        metadata = DatasetMetadata(
            name=self.name,
            task_type="multiclass_classification",
            input_size=self.config.input_size,
            output_size=self.config.output_size,
            train_size=self.config.train_size,
            validation_size=self.config.validation_size,
            test_size=self.config.test_size,
            split_strategy=(
                "deterministic independently generated "
                "train/validation/test splits"
            ),
            sampling_type="regular",
            normalization_strategy=(
                "feature-wise standardization fit on training split only"
            ),
            class_names=tuple(
                f"class_{index}"
                for index in range(self.config.output_size)
            ),
            feature_names=tuple(
                f"feature_{index}"
                for index in range(self.config.input_size)
            ),
        )
        metadata.validate()

        bundle = DataBundle(
            train_loader=train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            metadata=metadata,
        )
        bundle.validate()

        return bundle
