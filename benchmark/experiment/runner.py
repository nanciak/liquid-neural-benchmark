"""End-to-end benchmark experiment runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import time

import numpy as np
import torch

from benchmark.core.config import ModelConfig
from benchmark.data.bundle import DataBundle
from benchmark.evaluation.evaluator import EvaluationResult, Evaluator
from benchmark.models.factory import build_model
from benchmark.training.checkpoint import CheckpointManager
from benchmark.training.losses import build_loss
from benchmark.training.optimizer import build_optimizer
from benchmark.training.scheduler import build_scheduler
from benchmark.training.trainer import Trainer, TrainingResult


@dataclass(slots=True, frozen=True)
class RunnerConfig:
    """Configuration for one benchmark experiment."""

    experiment_name: str
    model_name: str
    seed: int
    output_directory: Path
    model: ModelConfig

    optimizer_name: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_name: str = "none"

    maximum_epochs: int = 50
    patience: int = 10
    minimum_delta: float = 0.0
    gradient_clip_norm: float | None = 1.0

    def validate(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("experiment_name cannot be empty.")
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty.")
        if self.seed < 0:
            raise ValueError("seed cannot be negative.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if self.maximum_epochs <= 0:
            raise ValueError("maximum_epochs must be positive.")
        if self.patience <= 0:
            raise ValueError("patience must be positive.")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive.")

        self.model.validate()


@dataclass(slots=True)
class ExperimentResult:
    """Complete result of one benchmark experiment."""

    experiment_name: str
    model_name: str
    seed: int
    training: TrainingResult
    validation: EvaluationResult
    test: EvaluationResult
    training_seconds: float
    parameter_count: int
    checkpoint_path: Path


class ExperimentRunner:
    """Coordinate model construction, training, evaluation, and checkpointing."""

    def __init__(
        self,
        config: RunnerConfig,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def run(self, bundle: DataBundle) -> ExperimentResult:
        bundle.validate()
        self._set_seed(self.config.seed)

        model = build_model(
            name=self.config.model_name,
            input_size=bundle.metadata.input_size,
            output_size=bundle.metadata.output_size,
            config=self.config.model,
        )

        loss_function = build_loss(bundle.metadata.task_type)

        optimizer = build_optimizer(
            model=model,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            optimizer=self.config.optimizer_name,
        )

        scheduler = build_scheduler(
            optimizer,
            scheduler=self.config.scheduler_name,
            t_max=self.config.maximum_epochs,
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            scheduler=scheduler,
            device=self.device,
            maximum_epochs=self.config.maximum_epochs,
            patience=self.config.patience,
            minimum_delta=self.config.minimum_delta,
            gradient_clip_norm=self.config.gradient_clip_norm,
        )

        started_at = time.perf_counter()

        training_result = trainer.fit(
            train_loader=bundle.train_loader,
            validation_loader=bundle.validation_loader,
        )

        training_seconds = time.perf_counter() - started_at

        evaluator = Evaluator(
            model=model,
            loss_function=loss_function,
            device=self.device,
            class_count=bundle.metadata.output_size,
        )

        validation_result = evaluator.evaluate(bundle.validation_loader)
        test_result = evaluator.evaluate(bundle.test_loader)

        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        checkpoint_path = CheckpointManager(
            self.config.output_directory
        ).save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=training_result.best_epoch,
            validation_loss=training_result.best_validation_loss,
            config=self.config,
            extra={
                "dataset_name": bundle.metadata.name,
                "validation_accuracy": validation_result.metrics.accuracy,
                "validation_macro_f1": validation_result.metrics.macro_f1,
                "test_accuracy": test_result.metrics.accuracy,
                "test_macro_f1": test_result.metrics.macro_f1,
                "parameter_count": parameter_count,
                "training_seconds": training_seconds,
            },
        )

        return ExperimentResult(
            experiment_name=self.config.experiment_name,
            model_name=self.config.model_name,
            seed=self.config.seed,
            training=training_result,
            validation=validation_result,
            test=test_result,
            training_seconds=training_seconds,
            parameter_count=parameter_count,
            checkpoint_path=checkpoint_path,
        )

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
