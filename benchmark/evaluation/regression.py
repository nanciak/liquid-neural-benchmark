"""Regression metrics for sequence prediction tasks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True, frozen=True)
class RegressionMetrics:
    """Regression evaluation metrics."""

    mse: float
    rmse: float
    mae: float
    r2: float


def regression_metrics(
    predictions: Tensor,
    targets: Tensor,
) -> RegressionMetrics:
    """
    Compute standard regression metrics.

    Parameters
    ----------
    predictions:
        Model predictions of shape (N, D)

    targets:
        Ground-truth values of shape (N, D)
    """

    if predictions.shape != targets.shape:
        raise ValueError(
            "Predictions and targets must have identical shapes."
        )

    predictions = predictions.float()
    targets = targets.float()

    errors = predictions - targets

    mse = torch.mean(errors.square())

    rmse = torch.sqrt(mse)

    mae = torch.mean(errors.abs())

    ss_res = torch.sum(errors.square())

    target_mean = torch.mean(targets)

    ss_tot = torch.sum(
        (targets - target_mean).square()
    )

    if ss_tot <= 1e-12:
        r2 = torch.tensor(
            0.0,
            device=targets.device,
        )
    else:
        r2 = 1.0 - ss_res / ss_tot

    return RegressionMetrics(
        mse=float(mse.cpu()),
        rmse=float(rmse.cpu()),
        mae=float(mae.cpu()),
        r2=float(r2.cpu()),
    )
