"""Dataset metadata shared across benchmark datasets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class DatasetMetadata:
    """Describes a prepared dataset."""

    name: str
    task_type: str

    input_size: int
    output_size: int

    train_size: int
    validation_size: int
    test_size: int

    split_strategy: str
    sampling_type: str
    normalization_strategy: str

    class_names: tuple[str, ...]
    feature_names: tuple[str, ...]

    def validate(self) -> None:
        """Validate metadata consistency."""

        if not self.name:
            raise ValueError("Dataset name cannot be empty.")

        if self.input_size <= 0:
            raise ValueError("input_size must be positive.")

        if self.output_size <= 0:
            raise ValueError("output_size must be positive.")

        if self.train_size <= 0:
            raise ValueError("train_size must be positive.")

        if self.validation_size <= 0:
            raise ValueError("validation_size must be positive.")

        if self.test_size <= 0:
            raise ValueError("test_size must be positive.")

        if len(self.class_names) != self.output_size:
            raise ValueError(
                "Number of class names must equal output_size."
            )

        if len(self.feature_names) != self.input_size:
            raise ValueError(
                "Number of feature names must equal input_size."
            )
