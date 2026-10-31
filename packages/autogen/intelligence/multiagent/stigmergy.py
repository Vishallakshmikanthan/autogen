"""Stigmergy engine: chemical pheromone field with deposition, evaporation, and diffusion (SYS-14, Module M15)."""

from __future__ import annotations

import numpy as np

from autogen.simulation.state.types import Position


class PheromoneField:
    """Continuous 2D spatial pheromone field enabling indirect stigmergic coordination."""

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        evaporation_rate: float = 0.05,
        diffusion_rate: float = 0.1,
        max_concentration: float = 100.0,
    ) -> None:
        self.width = max(2, width)
        self.height = max(2, height)
        self.evaporation_rate = float(evaporation_rate)
        # Ensure diffusion obeys CFL numerical stability condition (D <= 0.2)
        self.diffusion_rate = min(0.2, float(diffusion_rate))
        self.max_concentration = float(max_concentration)

        self.grid = np.zeros((self.height, self.width), dtype=np.float64)

    def deposit(self, position: Position, amount: float = 10.0) -> None:
        """Deposits chemical concentration at specified grid position."""
        x, y = int(position.x), int(position.y)
        if 0 <= x < self.width and 0 <= y < self.height:
            new_val = self.grid[y, x] + float(amount)
            self.grid[y, x] = min(self.max_concentration, new_val)

    def step(self) -> None:
        """Applies spatial diffusion followed by exponential temporal evaporation."""
        # 1. 2D 5-point discrete Laplacian diffusion with reflective boundaries
        padded = np.pad(self.grid, pad_width=1, mode="edge")
        laplacian = (
            padded[0:-2, 1:-1]   # up
            + padded[2:, 1:-1]   # down
            + padded[1:-1, 0:-2] # left
            + padded[1:-1, 2:]   # right
            - 4.0 * self.grid    # center
        )
        self.grid += self.diffusion_rate * laplacian

        # 2. Evaporation: P <- max(0, (1 - rho) * P)
        self.grid *= (1.0 - self.evaporation_rate)
        np.clip(self.grid, 0.0, self.max_concentration, out=self.grid)

    def sense(self, position: Position) -> float:
        """Reads chemical concentration at specified coordinates."""
        x, y = int(position.x), int(position.y)
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(self.grid[y, x])
        return 0.0

    def gradient(self, position: Position) -> tuple[float, float]:
        """Computes spatial concentration gradient (dPhi/dx, dPhi/dy) pointing uphill."""
        x, y = int(position.x), int(position.y)
        x_left = max(0, x - 1)
        x_right = min(self.width - 1, x + 1)
        y_up = max(0, y - 1)
        y_down = min(self.height - 1, y + 1)

        grad_x = float(self.grid[y, x_right] - self.grid[y, x_left]) / 2.0
        grad_y = float(self.grid[y_down, x] - self.grid[y_up, x]) / 2.0
        return grad_x, grad_y

    def reset(self) -> None:
        """Clears all chemical concentrations."""
        self.grid.fill(0.0)
