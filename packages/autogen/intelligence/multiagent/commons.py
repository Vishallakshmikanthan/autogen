"""Tragedy of the Commons and non-linear renewable resource dynamics (SYS-14, Module M15)."""

from __future__ import annotations

from typing import Any


class CommonsResourcePool:
    """Shared renewable resource pool governed by logistic regrowth and over-harvest collapse dynamics."""

    def __init__(
        self,
        carrying_capacity: float = 1000.0,
        growth_rate: float = 0.1,
        collapse_threshold: float = 100.0,
        initial_stock: float | None = None,
    ) -> None:
        self.carrying_capacity = float(carrying_capacity)
        self.growth_rate = float(growth_rate)
        self.collapse_threshold = float(collapse_threshold)
        self.initial_stock = float(initial_stock if initial_stock is not None else carrying_capacity)

        self.current_stock = self.initial_stock
        self.collapsed = False
        self.step_count = 0
        self.total_harvested = 0.0
        self.depletion_step: int | None = None

    def reset(self, initial_stock: float | None = None) -> None:
        """Resets resource pool to initial state."""
        if initial_stock is not None:
            self.initial_stock = float(initial_stock)
        self.current_stock = self.initial_stock
        self.collapsed = False
        self.step_count = 0
        self.total_harvested = 0.0
        self.depletion_step = None

    def harvest(self, agent_id: str, requested_amount: float) -> float:
        """Processes agent harvest request against available stock."""
        if self.current_stock <= 0.0 or requested_amount <= 0.0:
            return 0.0

        actual_harvest = min(float(requested_amount), self.current_stock)
        self.current_stock -= actual_harvest
        self.total_harvested += actual_harvest

        if self.current_stock <= 0.0 and self.depletion_step is None:
            self.depletion_step = self.step_count

        return actual_harvest

    def step_regrowth(self) -> dict[str, Any]:
        """Applies non-linear logistic replenishment or triggers permanent collapse."""
        self.step_count += 1
        previous_stock = self.current_stock

        if self.current_stock <= self.collapse_threshold:
            self.collapsed = True
            regrowth = 0.0
            if self.depletion_step is None and self.current_stock <= 0.0:
                self.depletion_step = self.step_count
        else:
            # Logistic regrowth: dR = r * R * (1 - R / K)
            regrowth = self.growth_rate * self.current_stock * (1.0 - (self.current_stock / self.carrying_capacity))
            self.current_stock = min(self.carrying_capacity, self.current_stock + regrowth)

        return {
            "step": self.step_count,
            "previous_stock": previous_stock,
            "regrowth": regrowth,
            "current_stock": self.current_stock,
            "collapsed": self.collapsed,
            "depletion_step": self.depletion_step,
        }
