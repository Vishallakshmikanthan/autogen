"""Configuration and ablation toggles for Hybrid Cognitive Agents (SYS-15, Module M17)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class HybridAgentConfig(BaseModel):
    """Component-level configuration enabling systematic ablation of cognitive subsystems."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enable_memory: bool = True           # M04: Working / spatial memory of seen cells
    enable_astar: bool = True            # M06: A* pathfinding navigation
    enable_bayes: bool = True            # M10: Bayesian belief filtering for hidden hazards
    enable_q_learning: bool = True      # M13: Q-learning / option selection
    enable_safety_shield: bool = True    # M17: Symbolic runtime verification shield
    enable_options: bool = True          # M17: Hierarchical options framework

    hazard_threshold: float = 0.5
    grid_width: int = 32
    grid_height: int = 32
    seed: int = 42

