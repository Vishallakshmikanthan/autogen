"""CLI Experiment Application: Adversarial Tournaments & Depth Benchmarks (SYS-10, Module M07).

Executes:
1. Iterated Payoff-Matrix Tournaments (TitForTat, GrimTrigger, AlwaysDefect, AlwaysCooperate, Random).
2. Predator Depth vs Prey Evasion benchmark (depth sweep d=1..4 across multiple seeded episodes with 95% CIs).
3. Adversarial Deliberation: Minimax vs Expectimax vs Random Opponent.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

from autogen.intelligence.adversarial.evaluations import (
    PredatorPreyEvaluator,
    PredatorPreyState,
    order_predator_moves,
)
from autogen.intelligence.adversarial.minimax import (
    AlphaBetaSearch,
    ExpectimaxSearch,
    MinimaxSearch,
)
from autogen.intelligence.adversarial.predator_prey_game import PredatorPreySubGame
from autogen.intelligence.game_theory.iterated import (
    AlwaysCooperate,
    AlwaysDefect,
    GrimTrigger,
    RandomStrategy,
    TitForTat,
    play_iterated_match,
    run_tournament,
)
from autogen.intelligence.game_theory.matrix_games import (
    make_hawk_dove,
    make_prisoners_dilemma,
    make_stag_hunt,
)
from autogen.simulation.state.types import ActionType, Position


def compute_confidence_interval(data: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """Computes sample mean and 95% confidence interval half-width using Student-t / normal approximation."""
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(data))
    if n == 1:
        return mean, mean, mean

    std_err = float(np.std(data, ddof=1) / math.sqrt(n))
    z = 1.96 if confidence == 0.95 else 2.576
    margin = z * std_err
    return mean, mean - margin, mean + margin


def run_iterated_tournament_experiment(num_rounds: int = 100, num_seeds: int = 10) -> dict[str, dict[str, float]]:
    """Runs iterated tournaments across games (Prisoner's Dilemma, Hawk-Dove, Stag Hunt)."""
    print("--- 1. Iterated Payoff-Matrix Tournaments ---")
    games = {
        "PrisonersDilemma": make_prisoners_dilemma(),
        "HawkDove": make_hawk_dove(),
        "StagHunt": make_stag_hunt(),
    }

    tournament_summary: dict[str, dict[str, float]] = {}

    for g_name, game in games.items():
        print(f"\nGame: {g_name} ({num_rounds} rounds)")
        strats = [
            TitForTat(),
            GrimTrigger(),
            AlwaysCooperate(),
            AlwaysDefect(),
            RandomStrategy(seed=42),
        ]
        results = run_tournament(game, strats, num_rounds=num_rounds)
        tournament_summary[g_name] = results
        for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name:18s} -> Average Payoff: {score:.3f}")

    return tournament_summary


def run_predator_depth_sweep(
    depths: list[int] | None = None,
    num_episodes: int = 20,
    max_steps_per_episode: int = 50,
) -> dict[str, dict[str, float]]:
    """Evaluates predator capture rate, survival steps, and search nodes as predator lookahead depth scales."""
    if depths is None:
        depths = [1, 2, 3, 4]

    print("\n--- 2. Predator Lookahead Depth vs Prey Success ---")
    results: dict[str, dict[str, float]] = {}

    evaluator = PredatorPreyEvaluator()
    search_engine = AlphaBetaSearch[PredatorPreyState, ActionType](
        move_ordering_fn=order_predator_moves,
    )
    prey_engine = AlphaBetaSearch[PredatorPreyState, ActionType]()

    for depth in depths:
        capture_list: list[float] = []
        survival_steps: list[float] = []
        expanded_nodes: list[float] = []

        for seed in range(num_episodes):
            rng = np.random.default_rng(seed)
            pred_pos = Position(int(rng.integers(0, 5)), int(rng.integers(0, 5)))
            prey_pos = Position(int(rng.integers(10, 15)), int(rng.integers(10, 15)))

            cur_state = PredatorPreyState(
                predator_pos=pred_pos,
                prey_pos=prey_pos,
                predator_energy=100.0,
                prey_energy=100.0,
                turn=0,
                grid_width=20,
                grid_height=20,
            )

            captured = False
            step = 0
            ep_nodes = 0

            while step < max_steps_per_episode and not captured:
                step += 1
                # Predator turn
                subgame = PredatorPreySubGame(cur_state)
                pred_res = search_engine.search(subgame, cur_state, max_depth=depth, eval_fn=evaluator)
                ep_nodes += pred_res.nodes_expanded
                act = pred_res.best_action or ActionType.STAY
                cur_state = subgame.successor(cur_state, act)

                if cur_state.is_captured or cur_state.predator_pos == cur_state.prey_pos:
                    captured = True
                    break

                # Prey turn (evades with depth 2)
                subgame_prey = PredatorPreySubGame(cur_state)
                prey_res = prey_engine.search(subgame_prey, cur_state, max_depth=2, eval_fn=evaluator)
                p_act = prey_res.best_action or ActionType.STAY
                cur_state = subgame_prey.successor(cur_state, p_act)

                if cur_state.is_captured or cur_state.predator_pos == cur_state.prey_pos:
                    captured = True
                    break

            capture_list.append(1.0 if captured else 0.0)
            survival_steps.append(float(step))
            expanded_nodes.append(float(ep_nodes / max(1, step)))

        c_mean, c_low, c_high = compute_confidence_interval(capture_list)
        s_mean, s_low, s_high = compute_confidence_interval(survival_steps)
        n_mean, _, _ = compute_confidence_interval(expanded_nodes)

        results[f"depth_{depth}"] = {
            "capture_rate": round(c_mean, 3),
            "capture_ci_low": round(c_low, 3),
            "capture_ci_high": round(c_high, 3),
            "mean_survival_steps": round(s_mean, 2),
            "survival_ci_low": round(s_low, 2),
            "survival_ci_high": round(s_high, 2),
            "avg_nodes_per_step": round(n_mean, 1),
        }

        print(
            f"  Depth {depth}: Capture Rate={c_mean * 100:.1f}% [{c_low * 100:.1f}%, {c_high * 100:.1f}%] | "
            f"Prey Survival={s_mean:.1f} steps | Nodes/Step={n_mean:.1f}"
        )

    return results


def run_algorithm_comparison(num_trials: int = 20) -> dict[str, dict[str, float]]:
    """Compares Minimax vs Expectimax against deterministic and stochastic opponents."""
    print("\n--- 3. Deliberation Comparison: Minimax vs Expectimax vs Stochastic ---")
    evaluator = PredatorPreyEvaluator()
    minimax_engine = MinimaxSearch[PredatorPreyState, ActionType]()
    alphabeta_engine = AlphaBetaSearch[PredatorPreyState, ActionType](move_ordering_fn=order_predator_moves)
    expectimax_engine = ExpectimaxSearch[PredatorPreyState, ActionType]()

    comparisons: dict[str, dict[str, float]] = {}

    mb_nodes: list[float] = []
    ab_nodes: list[float] = []
    ex_nodes: list[float] = []

    for seed in range(num_trials):
        rng = np.random.default_rng(seed + 100)
        p1 = Position(int(rng.integers(0, 5)), int(rng.integers(0, 5)))
        p2 = Position(int(rng.integers(5, 10)), int(rng.integers(5, 10)))

        state = PredatorPreyState(
            predator_pos=p1,
            prey_pos=p2,
            predator_energy=100.0,
            prey_energy=100.0,
            turn=0,
            grid_width=15,
            grid_height=15,
        )
        subgame = PredatorPreySubGame(state)

        m_res = minimax_engine.search(subgame, state, max_depth=3, eval_fn=evaluator)
        a_res = alphabeta_engine.search(subgame, state, max_depth=3, eval_fn=evaluator)
        e_res = expectimax_engine.search(subgame, state, max_depth=3, eval_fn=evaluator)

        mb_nodes.append(float(m_res.nodes_expanded))
        ab_nodes.append(float(a_res.nodes_expanded))
        ex_nodes.append(float(e_res.nodes_expanded))

    m_mean, _, _ = compute_confidence_interval(mb_nodes)
    a_mean, _, _ = compute_confidence_interval(ab_nodes)
    e_mean, _, _ = compute_confidence_interval(ex_nodes)

    pruning_efficiency = (1.0 - (a_mean / max(1.0, m_mean))) * 100.0

    comparisons["metrics"] = {
        "minimax_avg_nodes": round(m_mean, 1),
        "alphabeta_avg_nodes": round(a_mean, 1),
        "expectimax_avg_nodes": round(e_mean, 1),
        "pruning_reduction_percent": round(pruning_efficiency, 2),
    }

    print(f"  Minimax Avg Nodes:    {m_mean:.1f}")
    print(f"  Alpha-Beta Avg Nodes: {a_mean:.1f} (Pruning reduction: {pruning_efficiency:.1f}%)")
    print(f"  Expectimax Avg Nodes: {e_mean:.1f}")

    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser(description="AUTOGEN M07 Adversarial Tournaments & Experiments")
    parser.add_argument("--rounds", type=int, default=100, help="Number of rounds in iterated tournament")
    parser.add_argument("--episodes", type=int, default=15, help="Number of episodes per depth")
    parser.add_argument("--output-dir", type=str, default="results/experiments/m07", help="Output directory")
    args = parser.parse_args()

    print("=== AUTOGEN ADVERSARIAL & GAME THEORY EXPERIMENTS (MODULE M07) ===\n")
    tournaments = run_iterated_tournament_experiment(num_rounds=args.rounds)
    depth_sweep = run_predator_depth_sweep(num_episodes=args.episodes)
    algo_comp = run_algorithm_comparison()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / "m07_adversarial_experiment_report.json"

    full_report = {
        "tournaments": tournaments,
        "depth_sweep": depth_sweep,
        "algorithm_comparison": algo_comp,
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nExperiment artifacts successfully exported to: {report_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

