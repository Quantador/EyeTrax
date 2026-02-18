from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


class  BaseModel(ABC):
    """
    Common interface every gaze-prediction model must implement
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.val_metrics: dict = {}

    @abstractmethod
    def _init_native(self, **kwargs): ...
    @abstractmethod
    def _native_train(self, X: np.ndarray, y: np.ndarray): ...
    @abstractmethod
    def _native_predict(self, X: np.ndarray) -> np.ndarray: ...

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_scaling: np.ndarray | None = None,
        val_split: float = 0.2,
        random_state: int | None = None,
    ) -> None:
        self.variable_scaling = variable_scaling
        Xs = self.scaler.fit_transform(X)
        if variable_scaling is not None:
            Xs *= variable_scaling

        # Split into train/val sets
        if val_split > 0 and len(X) > 1:
            X_train, X_val, y_train, y_val = train_test_split(
                Xs, y, test_size=val_split, random_state=random_state
            )
        else:
            X_train, X_val = Xs, Xs
            y_train, y_val = y, y

        # Train on training set
        self._native_train(X_train, y_train)

        # Compute validation metrics
        y_pred_val = self._native_predict(X_val)
        self.val_metrics["mse"] = float(mean_squared_error(y_val, y_pred_val))
        self.val_metrics["mae"] = float(mean_absolute_error(y_val, y_pred_val))
        self.val_metrics["rmse"] = float(np.sqrt(self.val_metrics["mse"]))

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        if getattr(self, "variable_scaling", None) is not None:
            Xs *= self.variable_scaling
        return self._native_predict(Xs)

    def get_validation_metrics(self) -> dict:
        """Get validation metrics from the last training run."""
        return self.val_metrics.copy()

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)
