"""Generic model trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from benchmark.data.batch import SequenceBatch
from benchmark.training.early_stopping import EarlyStopping


@dataclass(slots=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float
    learning_rate: float


@dataclass(slots=True)
class TrainingResult:
    history: list[EpochMetrics] = field(default_factory=list)
    best_epoch: int = 0
    best_validation_loss: float = float("inf")
    stopped_early: bool = False


class Trainer:
    """Generic trainer for benchmark models."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        loss_function: nn.Module,
        device: torch.device | str,
        scheduler: Any | None = None,
        maximum_epochs: int = 50,
        patience: int = 10,
        minimum_delta: float = 0.0,
        gradient_clip_norm: float | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = torch.device(device)
        self.scheduler = scheduler
        self.maximum_epochs = maximum_epochs
        self.gradient_clip_norm = gradient_clip_norm

        self.early_stopping = EarlyStopping(
            patience=patience,
            min_delta=minimum_delta,
        )

        self.model.to(self.device)

    def fit(
        self,
        *,
        train_loader,
        validation_loader,
    ) -> TrainingResult:
        self.early_stopping.reset()

        result = TrainingResult()
        best_state: dict[str, Tensor] | None = None

        for epoch in range(1, self.maximum_epochs + 1):

            train_loss = self._run_epoch(
                train_loader,
                training=True,
            )

            validation_loss = self._run_epoch(
                validation_loader,
                training=False,
            )

            learning_rate = float(
                self.optimizer.param_groups[0]["lr"]
            )

            result.history.append(
                EpochMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    learning_rate=learning_rate,
                )
            )

            if validation_loss < result.best_validation_loss:
                result.best_validation_loss = validation_loss
                result.best_epoch = epoch
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.model.state_dict().items()
                }

            self._step_scheduler(validation_loss)

            if self.early_stopping.update(validation_loss):
                result.stopped_early = True
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return result

    def _run_epoch(
        self,
        loader,
        *,
        training: bool,
    ) -> float:

        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_samples = 0

        for batch in loader:

            if not isinstance(batch, SequenceBatch):
                raise TypeError(
                    "DataLoader must return SequenceBatch."
                )

            batch = batch.to(self.device)
            batch.validate()

            batch_size = len(batch)

            with torch.set_grad_enabled(training):

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

                if training:
                    self.optimizer.zero_grad(
                        set_to_none=True,
                    )

                    loss.backward()

                    if self.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip_norm,
                        )

                    self.optimizer.step()

            total_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size

        return total_loss / total_samples

    def _step_scheduler(
        self,
        validation_loss: float,
    ) -> None:

        if self.scheduler is None:
            return

        if isinstance(
            self.scheduler,
            ReduceLROnPlateau,
        ):
            self.scheduler.step(validation_loss)
        else:
            self.scheduler.step()
