"""Standard discrete and continuous evolutionary benchmark functions (SYS-15, Module M16)."""

from __future__ import annotations

import numpy as np


class OneMax:
    """Discrete binary maximization benchmark: sum of bits in a binary vector."""

    def __init__(self, length: int = 100) -> None:
        self.length = length
        self.optimal_value = float(length)

    def __call__(self, genome: np.ndarray) -> float:
        """Returns the count of 1 bits in the binary genome."""
        return float(np.sum(genome == 1))

    def evaluate(self, genome: np.ndarray) -> float:
        """Alias for functional call."""
        return self(genome)


class Sphere:
    """Continuous quadratic sphere minimization benchmark: f(x) = sum(x_i^2)."""

    def __init__(self, dimension: int = 10, bounds: tuple[float, float] = (-5.12, 5.12)) -> None:
        self.dimension = dimension
        self.bounds = bounds
        self.optimal_value = 0.0

    def raw_cost(self, x: np.ndarray) -> float:
        """Computes raw sphere sum of squares."""
        return float(np.sum(np.square(x)))

    def __call__(self, x: np.ndarray) -> float:
        """Returns negative raw cost as fitness (maximization convention)."""
        return -self.raw_cost(x)


class Rastrigin:
    """Highly multimodal continuous minimization benchmark with cosine modulation."""

    def __init__(
        self,
        dimension: int = 10,
        bounds: tuple[float, float] = (-5.12, 5.12),
        a: float = 10.0,
    ) -> None:
        self.dimension = dimension
        self.bounds = bounds
        self.a = a
        self.optimal_value = 0.0

    def raw_cost(self, x: np.ndarray) -> float:
        """Computes Rastrigin function value: 10*D + sum(x_i^2 - 10*cos(2*pi*x_i))."""
        d = len(x)
        return float(self.a * d + np.sum(np.square(x) - self.a * np.cos(2.0 * np.pi * x)))

    def __call__(self, x: np.ndarray) -> float:
        """Returns negative raw cost as fitness."""
        return -self.raw_cost(x)


class Rosenbrock:
    """Non-convex banana valley minimization benchmark."""

    def __init__(self, dimension: int = 10, bounds: tuple[float, float] = (-5.0, 10.0)) -> None:
        self.dimension = dimension
        self.bounds = bounds
        self.optimal_value = 0.0

    def raw_cost(self, x: np.ndarray) -> float:
        """Computes Rosenbrock sum: sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)."""
        x_curr = x[:-1]
        x_next = x[1:]
        return float(np.sum(100.0 * np.square(x_next - np.square(x_curr)) + np.square(1.0 - x_curr)))

    def __call__(self, x: np.ndarray) -> float:
        """Returns negative raw cost as fitness."""
        return -self.raw_cost(x)


class Ackley:
    """Nearly flat outer region with large hole in center: multimodal continuous benchmark."""

    def __init__(
        self,
        dimension: int = 10,
        bounds: tuple[float, float] = (-32.768, 32.768),
        a: float = 20.0,
        b: float = 0.2,
        c: float = 2.0 * np.pi,
    ) -> None:
        self.dimension = dimension
        self.bounds = bounds
        self.a = a
        self.b = b
        self.c = c
        self.optimal_value = 0.0

    def raw_cost(self, x: np.ndarray) -> float:
        """Computes Ackley function cost."""
        d = len(x)
        sum_sq = np.sum(np.square(x))
        sum_cos = np.sum(np.cos(self.c * x))
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum_sq / d))
        term2 = -np.exp(sum_cos / d)
        return float(term1 + term2 + self.a + np.e)

    def __call__(self, x: np.ndarray) -> float:
        """Returns negative raw cost as fitness."""
        return -self.raw_cost(x)

