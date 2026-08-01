"""Task-aware evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True, frozen=True)
class ClassificationMetrics:
    """Classification metrics returned by the evaluator."""

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    confusion_matrix: Tensor


def multiclass_metrics(
    logits: Tensor,
    targets: Tensor,
    *,
    class_count: int,
) -> ClassificationMetrics:
    """Compute multiclass classification metrics."""

    predictions = logits.argmax(dim=1)

    indices = (
        targets.cpu().long() * class_count
        + predictions.cpu().long()
    )

    confusion_matrix = torch.bincount(
        indices,
        minlength=class_count * class_count,
    ).reshape(class_count, class_count)

    matrix = confusion_matrix.to(torch.float64)

    true_positive = matrix.diag()
    predicted_positive = matrix.sum(dim=0)
    actual_positive = matrix.sum(dim=1)

    precision = torch.where(
        predicted_positive > 0,
        true_positive / predicted_positive,
        torch.zeros_like(true_positive),
    )

    recall = torch.where(
        actual_positive > 0,
        true_positive / actual_positive,
        torch.zeros_like(true_positive),
    )

    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )

    accuracy = (
        true_positive.sum()
        / matrix.sum().clamp_min(1.0)
    )

    return ClassificationMetrics(
        accuracy=float(accuracy),
        macro_precision=float(precision.mean()),
        macro_recall=float(recall.mean()),
        macro_f1=float(f1.mean()),
        confusion_matrix=confusion_matrix,
    )
