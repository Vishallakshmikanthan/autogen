"""Data contracts and interfaces for Neural Networks and Deep RL (SYS-13, Module M14)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict


@runtime_checkable
class Activation(Protocol):
    """Protocol for element-wise activation functions."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        """Computes activation a = sigma(z)."""
        ...

    def derivative(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Computes element-wise derivative sigma'(z)."""
        ...

    def backward(self, grad_output: np.ndarray, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Computes activation gradient delta given upstream grad_output."""
        ...


@runtime_checkable
class LossFunction(Protocol):
    """Protocol for scalar loss functions with vector/matrix gradients."""

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes scalar loss L(y_true, y_pred)."""
        ...

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Computes dL / d(y_pred)."""
        ...


@runtime_checkable
class Layer(Protocol):
    """Protocol for neural network layers supporting forward and backward propagation."""

    @property
    def params(self) -> dict[str, np.ndarray]:
        """Learnable parameter tensors (e.g. weights, biases)."""
        ...

    @property
    def grads(self) -> dict[str, np.ndarray]:
        """Gradients of loss with respect to parameters."""
        ...

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through the layer."""
        ...

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backpropagates gradient through the layer and accumulates parameter gradients."""
        ...


@runtime_checkable
class Optimizer(Protocol):
    """Protocol for gradient-based parameter optimizers."""

    def step(self, layers: Sequence[Layer]) -> None:
        """Applies parameter updates using accumulated gradients."""
        ...

    def zero_grad(self, layers: Sequence[Layer]) -> None:
        """Resets accumulated parameter gradients to zero."""
        ...


class NetworkDiagnostics(BaseModel):
    """Diagnostic telemetry captured during neural network training."""

    model_config = ConfigDict(frozen=True)

    loss: float
    gradient_norm: float
    mean_q: float | None = None
    max_q: float | None = None
    entropy: float | None = None
    learning_rate: float
    step: int


@dataclass(frozen=True, slots=True)
class ExperienceTransition:
    """Single experience transition for off-policy deep reinforcement learning."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)
