"""Parameter optimizers, gradient clipping, and learning-rate schedulers (SYS-13, Module M14)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from autogen.intelligence.neural.contracts import Layer, Optimizer


def clip_grad_norm(layers: Sequence[Layer], max_norm: float) -> float:
    """Clips accumulated parameter gradients across all layers by global L2 norm."""
    total_sq_norm = 0.0
    for layer in layers:
        for grad in layer.grads.values():
            total_sq_norm += float(np.sum(np.square(grad)))
    global_norm = float(np.sqrt(total_sq_norm))
    if global_norm > max_norm and global_norm > 0.0:
        scale = max_norm / global_norm
        for layer in layers:
            for grad in layer.grads.values():
                grad *= scale
    return global_norm


def clip_grad_value(layers: Sequence[Layer], clip_value: float) -> None:
    """Clips parameter gradients element-wise to [-clip_value, clip_value]."""
    for layer in layers:
        for grad in layer.grads.values():
            np.clip(grad, -clip_value, clip_value, out=grad)


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer with optional weight decay."""

    def __init__(self, lr: float = 0.01, weight_decay: float = 0.0) -> None:
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

    def step(self, layers: Sequence[Layer]) -> None:
        for layer in layers:
            for p_name, param in layer.params.items():
                if p_name in layer.grads:
                    grad = layer.grads[p_name]
                    if self.weight_decay > 0.0:
                        grad = grad + self.weight_decay * param
                    param -= self.lr * grad

    def zero_grad(self, layers: Sequence[Layer]) -> None:
        for layer in layers:
            for grad in layer.grads.values():
                grad.fill(0.0)


class Momentum(Optimizer):
    """Momentum optimizer maintaining first-moment velocity."""

    def __init__(self, lr: float = 0.01, beta: float = 0.9, weight_decay: float = 0.0) -> None:
        self.lr = float(lr)
        self.beta = float(beta)
        self.weight_decay = float(weight_decay)
        self.velocities: dict[int, dict[str, np.ndarray]] = {}

    def step(self, layers: Sequence[Layer]) -> None:
        for idx, layer in enumerate(layers):
            if idx not in self.velocities:
                self.velocities[idx] = {k: np.zeros_like(v) for k, v in layer.params.items()}

            for p_name, param in layer.params.items():
                if p_name in layer.grads:
                    grad = layer.grads[p_name]
                    if self.weight_decay > 0.0:
                        grad = grad + self.weight_decay * param
                    v = self.velocities[idx][p_name]
                    v[:] = self.beta * v + (1.0 - self.beta) * grad
                    param -= self.lr * v

    def zero_grad(self, layers: Sequence[Layer]) -> None:
        for layer in layers:
            for grad in layer.grads.values():
                grad.fill(0.0)


class RMSProp(Optimizer):
    """Root Mean Square Propagation optimizer scaling step by exponentially decaying squared gradient."""

    def __init__(
        self,
        lr: float = 0.001,
        beta: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr = float(lr)
        self.beta = float(beta)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.sq_avgs: dict[int, dict[str, np.ndarray]] = {}

    def step(self, layers: Sequence[Layer]) -> None:
        for idx, layer in enumerate(layers):
            if idx not in self.sq_avgs:
                self.sq_avgs[idx] = {k: np.zeros_like(v) for k, v in layer.params.items()}

            for p_name, param in layer.params.items():
                if p_name in layer.grads:
                    grad = layer.grads[p_name]
                    if self.weight_decay > 0.0:
                        grad = grad + self.weight_decay * param
                    sq = self.sq_avgs[idx][p_name]
                    sq[:] = self.beta * sq + (1.0 - self.beta) * np.square(grad)
                    param -= (self.lr / (np.sqrt(sq) + self.eps)) * grad

    def zero_grad(self, layers: Sequence[Layer]) -> None:
        for layer in layers:
            for grad in layer.grads.values():
                grad.fill(0.0)


class Adam(Optimizer):
    """Adaptive Moment Estimation optimizer with first/second moment bias correction."""

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self.m: dict[int, dict[str, np.ndarray]] = {}
        self.v: dict[int, dict[str, np.ndarray]] = {}

    def step(self, layers: Sequence[Layer]) -> None:
        self.t += 1
        b1_t = 1.0 - (self.beta1 ** self.t)
        b2_t = 1.0 - (self.beta2 ** self.t)

        for idx, layer in enumerate(layers):
            if idx not in self.m:
                self.m[idx] = {k: np.zeros_like(val) for k, val in layer.params.items()}
                self.v[idx] = {k: np.zeros_like(val) for k, val in layer.params.items()}

            for p_name, param in layer.params.items():
                if p_name in layer.grads:
                    grad = layer.grads[p_name]
                    if self.weight_decay > 0.0:
                        grad = grad + self.weight_decay * param

                    m_val = self.m[idx][p_name]
                    v_val = self.v[idx][p_name]

                    # Biased first and second moments
                    m_val[:] = self.beta1 * m_val + (1.0 - self.beta1) * grad
                    v_val[:] = self.beta2 * v_val + (1.0 - self.beta2) * np.square(grad)

                    # Bias-corrected estimates
                    m_hat = m_val / b1_t
                    v_hat = v_val / b2_t

                    param -= (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat

    def zero_grad(self, layers: Sequence[Layer]) -> None:
        for layer in layers:
            for grad in layer.grads.values():
                grad.fill(0.0)


class LearningRateScheduler:
    """Base class for dynamic learning-rate schedules."""

    def __init__(self, optimizer: Optimizer, initial_lr: float) -> None:
        self.optimizer = optimizer
        self.initial_lr = float(initial_lr)
        self.current_lr = float(initial_lr)
        self.step_count = 0

    def step(self) -> float:
        self.step_count += 1
        self.current_lr = self.compute_lr(self.step_count)
        if hasattr(self.optimizer, "lr"):
            self.optimizer.lr = self.current_lr
        return self.current_lr

    def compute_lr(self, step: int) -> float:
        return self.initial_lr


class StepLR(LearningRateScheduler):
    """Decays learning rate by gamma every step_size steps."""

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1) -> None:
        super().__init__(optimizer, getattr(optimizer, "lr", 0.01))
        self.step_size = max(1, step_size)
        self.gamma = float(gamma)

    def compute_lr(self, step: int) -> float:
        factor = self.gamma ** (step // self.step_size)
        return self.initial_lr * factor


class ExponentialLR(LearningRateScheduler):
    """Exponentially decays learning rate by gamma each step."""

    def __init__(self, optimizer: Optimizer, gamma: float = 0.99) -> None:
        super().__init__(optimizer, getattr(optimizer, "lr", 0.01))
        self.gamma = float(gamma)

    def compute_lr(self, step: int) -> float:
        return self.initial_lr * (self.gamma ** step)


class CosineAnnealingLR(LearningRateScheduler):
    """Cosine annealing learning rate schedule towards eta_min."""

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0) -> None:
        super().__init__(optimizer, getattr(optimizer, "lr", 0.01))
        self.T_max = max(1, T_max)
        self.eta_min = float(eta_min)

    def compute_lr(self, step: int) -> float:
        step_clamped = min(step, self.T_max)
        cosine = np.cos(np.pi * step_clamped / self.T_max)
        return self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * (1.0 + float(cosine))
