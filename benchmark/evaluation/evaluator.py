"""Model evaluation over benchmark data loaders."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from benchmark.data.batch import SequenceBatch
from benchmark.evaluation.metrics import (
    ClassificationMetrics,
    multiclass_metrics,
)
from benchmark.evaluation.regression import (
    RegressionMetrics,
    regression_metrics,
)


@dataclass(slots=True, frozen=True)
class EvaluationResult:
    """Result returned by the evaluator."""

    loss: float
    sample_count: int
    logits: Tensor
    targets: Tensor
    metrics: ClassificationMetrics | RegressionMetrics


class Evaluator:
    """Evaluate a trained model."""

    def __init__(
        self,
        *,
        model: nn.Module,
        loss_function: nn.Module,
        device: torch.device | str,
        task_type: str,
        output_size: int,
    ) -> None:
        self.model = model
        self.loss_function = loss_function
        self.device = torch.device(device)
        self.task_type = task_type
        self.output_size = output_size

        if self.task_type not in {
            "multiclass_classification",
            "regression",
        }:
            raise ValueError(
                f"Unsupported task type: {self.task_type}"
            )

        if self.output_size <= 0:
            raise ValueError(
                "output_size must be positive."
            )

        self.model.to(self.device)

    def evaluate(
        self,
        loader,
    ) -> EvaluationResult:
        """Evaluate a model on a data loader."""

        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        logits_parts: list[Tensor] = []
        target_parts: list[Tensor] = []

        with torch.no_grad():

            for batch in loader:

                if not isinstance(
                    batch,
                    SequenceBatch,
                ):
                    raise TypeError(
                        "DataLoader must return SequenceBatch."
                    )

                batch = batch.to(self.device)
                batch.validate()

                outputs = self.model(
                    values=batch.values,
                    timespans=batch.timespans,
                    observation_mask=(
                        batch.observation_mask
                    ),
                    padding_mask=batch.padding_mask,
                    lengths=batch.lengths,
                )

                loss = self.loss_function(
                    outputs,
                    batch.targets,
                )

                batch_size = len(batch)

                total_loss += (
                    float(loss.detach().cpu())
                    * batch_size
                )

                total_samples += batch_size

                logits_parts.append(
                    outputs.detach().cpu()
                )

                target_parts.append(
                    batch.targets.detach().cpu()
                )

        if total_samples == 0:
            raise ValueError(
                "Cannot evaluate an empty data loader."
            )

        logits = torch.cat(
            logits_parts,
            dim=0,
        )

        targets = torch.cat(
            target_parts,
            dim=0,
        )

        if self.task_type == "multiclass_classification":

            metrics = multiclass_metrics(
                logits,
                targets,
                class_count=self.output_size,
            )

        elif self.task_type == "regression":

            metrics = regression_metrics(
                logits,
                targets,
            )

        else:
            raise RuntimeError(
                f"Unsupported task type: {self.task_type}"
            )

        return EvaluationResult(
            loss=total_loss / total_samples,
            sample_count=total_samples,
            logits=logits,
            targets=targets,
            metrics=metrics,
        )
