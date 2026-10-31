"""Loss functions and their analytical gradients (SYS-13, Module M14)."""

from __future__ import annotations

from typing import cast

import numpy as np

from autogen.intelligence.neural.contracts import LossFunction


class MeanSquaredError(LossFunction):
    """Mean Squared Error loss: L(y, y_hat) = 1/N * sum (y_hat - y)^2."""

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        diff = y_pred - y_true
        return float(np.mean(np.square(diff)))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # dL/dy_pred = 2/N * (y_pred - y_true) where N is total element count
        return cast(np.ndarray, (2.0 / y_true.size) * (y_pred - y_true))


class BinaryCrossEntropy(LossFunction):
    """Binary Cross-Entropy loss: L(y, p) = -1/N * sum [y*log(p) + (1-y)*log(1-p)]."""

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        p = np.clip(y_pred, self.eps, 1.0 - self.eps)
        loss_val = - (y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
        return float(np.mean(loss_val))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        p = np.clip(y_pred, self.eps, 1.0 - self.eps)
        # dL/dp = 1/N * [(p - y) / (p * (1 - p))]
        n_elements = y_true.size
        return cast(np.ndarray, (1.0 / n_elements) * ((p - y_true) / (p * (1.0 - p))))


class CategoricalCrossEntropy(LossFunction):
    """Categorical Cross-Entropy loss for probability vectors."""

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        p = np.clip(y_pred, self.eps, 1.0)
        n_samples = y_true.shape[0] if y_true.ndim > 1 else 1
        return float(-np.sum(y_true * np.log(p)) / n_samples)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        p = np.clip(y_pred, self.eps, 1.0)
        n_samples = y_true.shape[0] if y_true.ndim > 1 else 1
        return cast(np.ndarray, -(y_true / p) / n_samples)


class HuberLoss(LossFunction):
    """Huber Loss (Smooth L1) for outlier-robust Q-learning targets."""

    def __init__(self, delta: float = 1.0) -> None:
        self.delta = delta

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        diff = y_pred - y_true
        abs_diff = np.abs(diff)
        quadratic = np.minimum(abs_diff, self.delta)
        linear = abs_diff - quadratic
        loss_val = 0.5 * np.square(quadratic) + self.delta * linear
        return float(np.mean(loss_val))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        diff = y_pred - y_true
        n_elements = y_true.size
        grad = np.where(np.abs(diff) <= self.delta, diff, self.delta * np.sign(diff))
        return cast(np.ndarray, grad / n_elements)
