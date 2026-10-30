"""Optional stochastic weather and seasonal modifiers."""

from __future__ import annotations

import numpy as np


class WeatherEngine:
    r"""Simulates ambient weather perturbations $\xi_t$.

    Operates via an independent PRNG stream. Disabled by default.
    """

    def __init__(self, rng: np.random.Generator, enabled: bool = False) -> None:
        self._rng = rng
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def step(self, current_step: int) -> dict[str, float]:
        """Calculates environmental conditions for step $t$."""
        if not self._enabled:
            return {"temperature": 20.0, "precipitation": 0.0, "regen_modifier": 0.0}

        # Smooth periodic seasonal baseline + random perturbation
        season_phase = 2.0 * np.pi * (current_step % 365) / 365.0
        base_temp = 20.0 + 10.0 * float(np.sin(season_phase))
        noise_temp = float(self._rng.normal(0.0, 2.0))

        precip = float(max(0.0, self._rng.exponential(0.3) if self._rng.random() < 0.4 else 0.0))
        regen_modifier = float(np.clip((precip * 0.5) - (abs(base_temp - 22.0) * 0.02), -0.5, 0.5))

        return {
            "temperature": base_temp + noise_temp,
            "precipitation": precip,
            "regen_modifier": regen_modifier,
        }
