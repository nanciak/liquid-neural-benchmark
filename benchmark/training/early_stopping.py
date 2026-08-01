"""Early stopping utility."""

from __future__ import annotations


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
    ) -> None:
        if patience <= 0:
            raise ValueError("patience must be positive.")

        self.patience = patience
        self.min_delta = min_delta
        self.reset()

    def reset(self) -> None:
        """Reset the internal state."""

        self.best_score = float("inf")
        self.counter = 0
        self.should_stop = False

    def update(
        self,
        validation_loss: float,
    ) -> bool:
        """
        Update the early stopping state.

        Returns
        -------
        bool
            True if training should stop.
        """

        if validation_loss < (
            self.best_score - self.min_delta
        ):
            self.best_score = validation_loss
            self.counter = 0
            return False

        self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return self.should_stop
