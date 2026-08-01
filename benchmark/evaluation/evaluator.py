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


@dataclass(slots=True, frozen=True)
class EvaluationResult:
    """Result returned by the evaluator."""

    loss: float
    sample_count: int
    logits: Tensor
    targets: Tensor
    metrics: ClassificationMetrics


class Evaluator:
    """Evaluate a trained model."""

    def __init__(
        self,
        *,
        model: nn.Module,
        loss_function: nn.Module,
        device: torch.device | str,
        class_count: int,
    ) -> None:
        self.model = model
        self.loss_function = loss_function
        self.device = torch.device(device)
        self.class_count = class_count

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

                if not isinstance(batch, SequenceBatch):
                    raise TypeError(
                        "DataLoader must return SequenceBatch."
                    )

                batch = batch.to(self.device)

                outputs = self.model(
                    values=batch.values,
                    timespans=batch.timespans,
                    observation_mask=batch.observation_mask,
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

        logits = torch.cat(logits_parts)
        targets = torch.cat(target_parts)

        return EvaluationResult(
            loss=total_loss / total_samples,
            sample_count=total_samples,
            logits=logits,
            targets=targets,
            metrics=multiclass_metrics(
                logits,
                targets,
                class_count=self.class_count,
            ),
        )
