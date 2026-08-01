"""UCI Human Activity Recognition dataset adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import urllib.request
import zipfile

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from benchmark.data.base import BaseDatasetAdapter
from benchmark.data.batch import SequenceBatch
from benchmark.data.bundle import DataBundle
from benchmark.data.metadata import DatasetMetadata
from benchmark.data.normalization import TrainOnlyStandardizer
from benchmark.data.splitting import (
    deterministic_group_validation_split,
    filter_ids,
)


_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)

_SIGNAL_NAMES = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)

_CLASS_NAMES = (
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
)


@dataclass(slots=True, frozen=True)
class UCIHARConfig:
    """Configuration for the UCI HAR dataset."""

    root: Path = Path("datasets/uci_har")
    batch_size: int = 128
    validation_subject_fraction: float = 0.20
    split_seed: int = 42
    num_workers: int = 0
    download: bool = True

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if not (
            0.0
            < self.validation_subject_fraction
            < 1.0
        ):
            raise ValueError(
                "validation_subject_fraction "
                "must be between 0 and 1."
            )

        if self.split_seed < 0:
            raise ValueError(
                "split_seed cannot be negative."
            )

        if self.num_workers < 0:
            raise ValueError(
                "num_workers cannot be negative."
            )


class UCIHARSequenceDataset(
    Dataset[SequenceBatch]
):
    """In-memory UCI HAR sequence dataset."""

    def __init__(
        self,
        values: Tensor,
        targets: Tensor,
        *,
        split_name: str,
    ) -> None:
        if values.ndim != 3:
            raise ValueError(
                "values must have shape "
                "[samples, sequence, features]."
            )

        if targets.ndim != 1:
            raise ValueError(
                "targets must have shape [samples]."
            )

        if values.shape[0] != targets.shape[0]:
            raise ValueError(
                "values and targets must have "
                "the same number of samples."
            )

        self.values = values
        self.targets = targets
        self.split_name = split_name

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def __getitem__(
        self,
        index: int,
    ) -> SequenceBatch:
        sequence_length = int(
            self.values.shape[1]
        )

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


def collate_uci_har(
    samples: list[SequenceBatch],
) -> SequenceBatch:
    """Combine individual samples into one batch."""

    if not samples:
        raise ValueError(
            "Cannot collate an empty sample list."
        )

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
            [
                sample.observation_mask
                for sample in samples
            ]
        ),
        padding_mask=torch.stack(
            [
                sample.padding_mask
                for sample in samples
            ]
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


class UCIHARAdapter(
    BaseDatasetAdapter[UCIHARConfig]
):
    """Prepare UCI HAR as 128-step, 9-channel sequences."""

    @property
    def name(self) -> str:
        return "uci_har"

    @property
    def dataset_root(self) -> Path:
        return (
            self.config.root
            / "extracted"
            / "UCI HAR Dataset"
        )

    def prepare(self) -> DataBundle:
        self.config.validate()
        self.ensure_directory(
            self.config.root
        )

        if not self.dataset_root.exists():
            if not self.config.download:
                raise FileNotFoundError(
                    "UCI HAR dataset not found at "
                    f"{self.dataset_root}."
                )

            self._download_and_extract()

        train_values = self._load_signals(
            "train"
        )

        test_values = self._load_signals(
            "test"
        )

        train_targets = self._load_targets(
            "train"
        )

        test_targets = self._load_targets(
            "test"
        )

        train_subjects = self._load_subjects(
            "train"
        )

        test_subjects = self._load_subjects(
            "test"
        )

        manifest = (
            deterministic_group_validation_split(
                train_group_ids=train_subjects,
                test_group_ids=test_subjects,
                validation_fraction=(
                    self.config
                    .validation_subject_fraction
                ),
                seed=self.config.split_seed,
                strategy=(
                    "official test split with "
                    "deterministic subject-level "
                    "validation split"
                ),
            )
        )

        manifest.write_json(
            self.config.root
            / "split_manifest.json"
        )

        train_indices = filter_ids(
            train_subjects,
            manifest.train_ids,
        )

        validation_indices = filter_ids(
            train_subjects,
            manifest.validation_ids,
        )

        training_values = torch.from_numpy(
            train_values[train_indices]
        ).float()

        validation_values = torch.from_numpy(
            train_values[validation_indices]
        ).float()

        testing_values = torch.from_numpy(
            test_values
        ).float()

        training_targets = torch.from_numpy(
            train_targets[train_indices]
        ).long()

        validation_targets = torch.from_numpy(
            train_targets[
                validation_indices
            ]
        ).long()

        testing_targets = torch.from_numpy(
            test_targets
        ).long()

        standardizer = (
            TrainOnlyStandardizer().fit(
                training_values,
                split_name="train",
            )
        )

        training_values = (
            standardizer.transform(
                training_values
            )
        )

        validation_values = (
            standardizer.transform(
                validation_values
            )
        )

        testing_values = (
            standardizer.transform(
                testing_values
            )
        )

        train_dataset = (
            UCIHARSequenceDataset(
                training_values,
                training_targets,
                split_name="train",
            )
        )

        validation_dataset = (
            UCIHARSequenceDataset(
                validation_values,
                validation_targets,
                split_name="validation",
            )
        )

        test_dataset = (
            UCIHARSequenceDataset(
                testing_values,
                testing_targets,
                split_name="test",
            )
        )

        loader_arguments = {
            "batch_size": (
                self.config.batch_size
            ),
            "num_workers": (
                self.config.num_workers
            ),
            "collate_fn": collate_uci_har,
        }

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            generator=(
                torch.Generator()
                .manual_seed(
                    self.config.split_seed
                )
            ),
            **loader_arguments,
        )

        validation_loader = DataLoader(
            validation_dataset,
            shuffle=False,
            **loader_arguments,
        )

        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            **loader_arguments,
        )

        metadata = DatasetMetadata(
            name=self.name,
            task_type=(
                "multiclass_classification"
            ),
            input_size=len(
                _SIGNAL_NAMES
            ),
            output_size=len(
                _CLASS_NAMES
            ),
            train_size=len(
                train_dataset
            ),
            validation_size=len(
                validation_dataset
            ),
            test_size=len(
                test_dataset
            ),
            split_strategy=(
                manifest.strategy
            ),
            sampling_type="regular",
            normalization_strategy=(
                "feature-wise standardization "
                "fit on training split only"
            ),
            class_names=_CLASS_NAMES,
            feature_names=_SIGNAL_NAMES,
        )

        metadata.validate()

        bundle = DataBundle(
            train_loader=train_loader,
            validation_loader=(
                validation_loader
            ),
            test_loader=test_loader,
            metadata=metadata,
            split_manifest=manifest,
        )

        bundle.validate()

        return bundle

    def _download_and_extract(
        self,
    ) -> None:
        archive_path = (
            self.config.root
            / "uci_har.zip"
        )

        extraction_root = (
            self.config.root
            / "extracted"
        )

        if not archive_path.exists():
            urllib.request.urlretrieve(
                _DATASET_URL,
                archive_path,
            )

        if extraction_root.exists():
            shutil.rmtree(
                extraction_root
            )

        extraction_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        with zipfile.ZipFile(
            archive_path,
            "r",
        ) as archive:
            archive.extractall(
                extraction_root
            )

        nested_archive = (
            extraction_root
            / "UCI HAR Dataset.zip"
        )

        if nested_archive.exists():
            with zipfile.ZipFile(
                nested_archive,
                "r",
            ) as archive:
                archive.extractall(
                    extraction_root
                )

        if not self.dataset_root.exists():
            raise RuntimeError(
                "UCI HAR extraction "
                "completed, but the dataset "
                "directory was not found."
            )

    def _load_signals(
        self,
        split: str,
    ) -> np.ndarray:
        signal_directory = (
            self.dataset_root
            / split
            / "Inertial Signals"
        )

        channels = []

        for signal_name in _SIGNAL_NAMES:
            path = (
                signal_directory
                / f"{signal_name}_{split}.txt"
            )

            channels.append(
                np.loadtxt(
                    path,
                    dtype=np.float32,
                )
            )

        values = np.stack(
            channels,
            axis=-1,
        )

        if values.ndim != 3:
            raise RuntimeError(
                "Loaded UCI HAR signals "
                "have an invalid shape."
            )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                "UCI HAR signals contain "
                "non-finite values."
            )

        return values

    def _load_targets(
        self,
        split: str,
    ) -> np.ndarray:
        path = (
            self.dataset_root
            / split
            / f"y_{split}.txt"
        )

        targets = np.loadtxt(
            path,
            dtype=np.int64,
        )

        return targets - 1

    def _load_subjects(
        self,
        split: str,
    ) -> np.ndarray:
        path = (
            self.dataset_root
            / split
            / f"subject_{split}.txt"
        )

        return np.loadtxt(
            path,
            dtype=np.int64,
        )
