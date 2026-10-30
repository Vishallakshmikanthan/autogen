"""Serialization and export of exact MDP solutions for RL test oracles (SYS-10, Module M11)."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from autogen.intelligence.decision.contracts import MDPSolution


def export_exact_mdp_solution(
    solution: MDPSolution,
    filepath: Path | str,
) -> None:
    """Exports exact MDP solution (V*, Q*, pi*) to a structured JSON artifact."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "v_star": solution.v_star.tolist(),
        "q_star": solution.q_star.tolist(),
        "policy": solution.policy.tolist(),
        "iterations": int(solution.iterations),
        "max_residual": float(solution.max_residual),
        "converged": bool(solution.converged),
        "runtime_sec": float(solution.runtime_sec),
        "state_labels": list(solution.state_labels),
        "action_labels": list(solution.action_labels),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_exact_mdp_solution(filepath: Path | str) -> MDPSolution:
    """Loads and deserializes an exact MDP solution from a JSON artifact."""
    in_path = Path(filepath)
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return MDPSolution(
        v_star=np.array(data["v_star"], dtype=np.float64),
        q_star=np.array(data["q_star"], dtype=np.float64),
        policy=np.array(data["policy"], dtype=np.int32),
        iterations=int(data["iterations"]),
        max_residual=float(data["max_residual"]),
        converged=bool(data["converged"]),
        runtime_sec=float(data["runtime_sec"]),
        state_labels=tuple(data["state_labels"]),
        action_labels=tuple(data["action_labels"]),
    )
