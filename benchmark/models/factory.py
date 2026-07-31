"""Central construction of all benchmark models."""

from __future__ import annotations

from collections.abc import Callable

from benchmark.core.config import ModelConfig
from benchmark.models.base import SequenceModel
from benchmark.models.discrete import CNN1DModel, TCNModel, TransformerModel
from benchmark.models.liquid import CfCModel, LTCModel
from benchmark.models.recurrent import GRUModel, LSTMModel
from benchmark.models.registry import (
    available_models as registered_models,
    get_model_builder,
    register_model,
)

ModelBuilder = Callable[[int, int, ModelConfig], SequenceModel]


def _build_gru(input_size:int, output_size:int, config:ModelConfig)->SequenceModel:
    return GRUModel(input_size=input_size, output_size=output_size, config=config)


def _build_lstm(input_size:int, output_size:int, config:ModelConfig)->SequenceModel:
    return LSTMModel(input_size=input_size, output_size=output_size, config=config)


def _build_cnn1d(input_size:int, output_size:int, config:ModelConfig)->SequenceModel:
    return CNN1DModel(input_size=input_size, output_size=output_size, config=config)


def _build_tcn(input_size:int, output_size:int, config:ModelConfig)->SequenceModel:
    return TCNModel(input_size=input_size, output_size=output_size, config=config)


def _build_transformer(input_size:int, output_size:int, config:ModelConfig)->SequenceModel:
    return TransformerModel(input_size=input_size, output_size=output_size, config=config)


def _build_cfc(input_size:int, output_size:int, config:ModelConfig)->SequenceModel:
    return CfCModel(
        input_size=input_size,
        output_size=output_size,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
    )


def _build_ltc(input_size:int, output_size:int, config:ModelConfig)->SequenceModel:
    return LTCModel(
        input_size=input_size,
        output_size=output_size,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
    )


_DEFAULT_BUILDERS: dict[str, ModelBuilder] = {
    "gru": _build_gru,
    "lstm": _build_lstm,
    "cnn1d": _build_cnn1d,
    "tcn": _build_tcn,
    "transformer": _build_transformer,
    "cfc": _build_cfc,
    "ltc": _build_ltc,
}


def _register_default_models() -> None:
    for name, builder in _DEFAULT_BUILDERS.items():
        try:
            register_model(name, builder)
        except ValueError:
            # Already registered.
            pass


_register_default_models()


def available_models() -> tuple[str, ...]:
    """Return all registered model names."""
    return registered_models()


def build_model(
    *,
    name: str,
    input_size: int,
    output_size: int,
    config: ModelConfig,
) -> SequenceModel:
    """Construct a registered sequence model."""

    if input_size <= 0:
        raise ValueError("input_size must be positive.")
    if output_size <= 0:
        raise ValueError("output_size must be positive.")

    config.validate()

    builder = get_model_builder(name)
    model = builder(input_size, output_size, config)

    if not isinstance(model, SequenceModel):
        raise TypeError("Registered model builders must return a SequenceModel.")

    return model
