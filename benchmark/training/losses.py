"""Loss function factory."""

from __future__ import annotations

from torch import nn


def build_loss(
    task_type: str,
) -> nn.Module:
    """Return the loss function for a benchmark task."""

    normalized_task = task_type.strip().lower()

    if normalized_task == "multiclass_classification":
        return nn.CrossEntropyLoss()

    if normalized_task == "binary_classification":
        return nn.BCEWithLogitsLoss()

    if normalized_task == "regression":
        return nn.MSELoss()

    raise ValueError(
        f"Unsupported task type: {task_type}"
    )
