"""Activation functions and their analytical derivatives (SYS-13, Module M14)."""

from __future__ import annotations

from typing import cast

import numpy as np

from autogen.intelligence.neural.contracts import Activation


class Sigmoid(Activation):
    """Sigmoid activation: sigma(z) = 1 / (1 + exp(-z))."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        z_clipped = np.clip(z, -500.0, 500.0)
        return cast(np.ndarray, 1.0 / (1.0 + np.exp(-z_clipped)))

    def derivative(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, a * (1.0 - a))

    def backward(self, grad_output: np.ndarray, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, grad_output * self.derivative(z, a))


class Tanh(Activation):
    """Hyperbolic tangent activation: tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, np.tanh(z))

    def derivative(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, 1.0 - np.square(a))

    def backward(self, grad_output: np.ndarray, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, grad_output * self.derivative(z, a))


class ReLU(Activation):
    """Rectified Linear Unit activation: ReLU(z) = max(0, z)."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, np.maximum(0.0, z))

    def derivative(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        return (z > 0.0).astype(np.float64)

    def backward(self, grad_output: np.ndarray, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, grad_output * self.derivative(z, a))


class LeakyReLU(Activation):
    """Leaky Rectified Linear Unit activation: f(z) = max(alpha * z, z)."""

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = float(alpha)

    def forward(self, z: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, np.where(z > 0.0, z, self.alpha * z))

    def derivative(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        return np.where(z > 0.0, 1.0, self.alpha).astype(np.float64)

    def backward(self, grad_output: np.ndarray, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, grad_output * self.derivative(z, a))


class Linear(Activation):
    """Identity activation for regression and Q-value outputs."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        return z

    def derivative(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        return np.ones_like(z, dtype=np.float64)

    def backward(self, grad_output: np.ndarray, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        return grad_output


class Softmax(Activation):
    """Softmax activation for categorical probability distributions."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        z_shifted = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return cast(np.ndarray, exp_z / np.sum(exp_z, axis=-1, keepdims=True))

    def derivative(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, a * (1.0 - a))

    def backward(self, grad_output: np.ndarray, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        s = np.sum(a * grad_output, axis=-1, keepdims=True)
        return cast(np.ndarray, a * (grad_output - s))
