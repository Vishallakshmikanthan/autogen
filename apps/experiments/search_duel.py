"""CLI Experiment Application: Algorithm Duel & Benchmark (SYS-06, Module M06).

Compares Uninformed, Informed, and Local search algorithms across obstacle densities and grid sizes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from autogen.intelligence.search import (
    AlgorithmDuelRunner,
    AStarSearch,
    BreadthFirstSearch,
    DepthFirstSearch,
    GreedyBestFirstSearch,
    GridSearchProblem,
    IterativeDeepeningAStar,
    IterativeDeepeningSearch,
    UniformCostSearch,
    WeightedAStarSearch,
)
from autogen.simulation.state.types import ActionType, Position


def generate_benchmark_grid(
    width: int,
    height: int,
    obstacle_ratio: float,
    seed: int,
) -> tuple[list[list[float]], Position, Position]:
    """Generates a random 2D grid matrix with obstacles and guaranteed start/goal."""
    rng = np.random.default_rng(seed)
    grid = [[1.0 for _ in range(width)] for _ in range(height)]

    # Add obstacles
    num_obstacles = int(width * height * obstacle_ratio)
    for _ in range(num_obstacles):
        ox = int(rng.integers(0, width))
        oy = int(rng.integers(0, height))
        grid[oy][ox] = float("inf")

    start = Position(0, 0)
    goal = Position(width - 1, height - 1)
    grid[start.y][start.x] = 1.0
    grid[goal.y][goal.x] = 1.0

    return grid, start, goal


def run_benchmark(
    width: int = 20,
    height: int = 20,
    obstacle_ratio: float = 0.20,
    seed: int = 42,
    output_dir: str = "results/experiments/m06",
) -> int:
    """Executes Algorithm Duel comparing search algorithms."""
    print("=== AUTOGEN Search Engine Duel (M06) ===")
    print(f"Grid: {width}x{height} | Obstacles: {obstacle_ratio * 100:.1f}% | Seed: {seed}\n")

    grid, start, goal = generate_benchmark_grid(width, height, obstacle_ratio, seed)
    problem = GridSearchProblem.from_cost_matrix(grid, start=start, goal=goal)

    algorithms = {
        "BFS": BreadthFirstSearch(),
        "DFS": DepthFirstSearch(max_depth=5000),
        "IDS": IterativeDeepeningSearch(max_depth=50),
        "UCS": UniformCostSearch(),
        "Greedy": GreedyBestFirstSearch(),
        "A*": AStarSearch(),
        "WeightedA*(1.5)": WeightedAStarSearch(weight=1.5),
        "IDA*": IterativeDeepeningAStar(max_iterations=500),
    }

    runner: AlgorithmDuelRunner[Position, ActionType] = AlgorithmDuelRunner(algorithms)
    duel_results = runner.run_duel(problem)

    print(
        f"{'Algorithm':<18} | {'Success':<7} | {'Cost':<8} | {'Expanded':<9} | {'Generated':<9} | {'Peak Open':<9} | {'Time (ms)':<9}"
    )
    print("-" * 88)

    serializable_results: dict[str, dict[str, float | int | bool]] = {}

    for name, res in duel_results.items():
        cost_str = f"{res.total_cost:.1f}" if res.success else "N/A"
        time_ms = res.runtime_seconds * 1000.0
        print(
            f"{name:<18} | {str(res.success):<7} | {cost_str:<8} | {res.nodes_expanded:<9} | {res.nodes_generated:<9} | {res.max_frontier_size:<9} | {time_ms:<9.2f}"
        )

        serializable_results[name] = {
            "success": res.success,
            "cost": res.total_cost if res.success else -1.0,
            "nodes_expanded": res.nodes_expanded,
            "nodes_generated": res.nodes_generated,
            "max_frontier_size": res.max_frontier_size,
            "runtime_ms": time_ms,
        }

    # Save to disk
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results_file = out_path / "search_duel_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "width": width,
                    "height": height,
                    "obstacle_ratio": obstacle_ratio,
                    "seed": seed,
                },
                "results": serializable_results,
            },
            f,
            indent=2,
        )

    print(f"\nResults successfully written to {results_file}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AUTOGEN Search Engine Algorithm Duel & Benchmark (M06)"
    )
    parser.add_argument("--width", type=int, default=20, help="Grid width (default: 20)")
    parser.add_argument("--height", type=int, default=20, help="Grid height (default: 20)")
    parser.add_argument(
        "--obstacles", type=float, default=0.20, help="Obstacle fraction (default: 0.20)"
    )
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed (default: 42)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/experiments/m06",
        help="Output directory for duel JSON results",
    )
    args = parser.parse_args(argv)

    return run_benchmark(
        width=args.width,
        height=args.height,
        obstacle_ratio=args.obstacles,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
