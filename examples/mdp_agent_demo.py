"""Module M11 Decision Theory & Markov Decision Processes CLI Usage Example.

Demonstrates:
1. Multi-attribute utility evaluation and Maximum Expected Utility (MEU) selection.
2. Building an exact tabular MDP from a GridWorld.
3. Linear Policy Evaluation vs Iterative Policy Evaluation on Bellman expectation equation.
4. Value Iteration and Policy Iteration computing exact V*, Q*, and pi*.
5. Exporting exact MDP solution as ground truth artifact for M13 Reinforcement Learning.
6. MDPAgent and UtilityAgent operating in SimulationEngine with authentic DecisionTraces.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np

from autogen.agents.decision.mdp_agent import MDPAgent
from autogen.agents.decision.utility_agent import UtilityAgent
from autogen.intelligence.decision.contracts import UtilityWeights
from autogen.intelligence.decision.gridworld_mdp import (
    GridWorldMDPBuilder,
    classic_russell_norvig_grid,
)
from autogen.intelligence.decision.serialization import (
    export_exact_mdp_solution,
    load_exact_mdp_solution,
)
from autogen.intelligence.decision.solvers import (
    iterative_policy_evaluation,
    linear_policy_evaluation,
    policy_iteration,
    value_iteration,
)
from autogen.intelligence.decision.utility import (
    MultiAttributeUtility,
    argmax_expected_utility,
)
from autogen.intelligence.probabilistic.distributions import DiscreteDistribution
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.perception.contracts import PerceptionConfig
from autogen.simulation.state.types import Position


def main() -> None:
    print("=" * 75)
    print("  AUTOGEN MODULE M11: DECISION THEORY & MARKOV DECISION PROCESSES")
    print("=" * 75)

    # 1. Multi-Attribute Utility & MEU
    print("\n--- 1. Multi-Attribute Utility & Expected Utility (MEU) ---")
    weights = UtilityWeights(energy=1.0, health=2.0, food=1.5, hazard_risk=5.0)
    mau = MultiAttributeUtility(weights)
    u_safe = mau.evaluate(energy=80.0, health=100.0, food=10.0, hazard_risk=0.0)
    u_risky = mau.evaluate(energy=90.0, health=100.0, food=20.0, hazard_risk=0.8)
    print(f"Safe State Utility:  {u_safe:.1f}")
    print(f"Risky State Utility: {u_risky:.1f}")

    # 2. Russell & Norvig 4x3 GridWorld Benchmark
    print("\n--- 2. Russell & Norvig 4x3 GridWorld Exact MDP ---")
    rn_mdp = classic_russell_norvig_grid(step_cost=-0.04, slip_prob=0.2, gamma=1.0)
    print(f"MDP Compiled: States={rn_mdp.num_states}, Actions={rn_mdp.num_actions}, Gamma={rn_mdp.gamma}")

    # 3. Policy Evaluation: Linear Matrix Solve vs Iterative
    print("\n--- 3. Policy Evaluation: Linear Solve vs Iterative Backups ---")
    arbitrary_policy = np.zeros(rn_mdp.num_states, dtype=np.int32)  # All UP
    v_linear = linear_policy_evaluation(rn_mdp, arbitrary_policy)
    v_iter = iterative_policy_evaluation(rn_mdp, arbitrary_policy, theta=1e-8)
    diff = float(np.max(np.abs(v_linear - v_iter)))
    print(f"Max Absolute Discrepancy (|V_linear - V_iter|): {diff:.2e} (Agreement: {'PASS' if diff < 1e-6 else 'FAIL'})")

    # 4. Value Iteration and Policy Iteration
    print("\n--- 4. Value Iteration & Policy Iteration ---")
    sol_vi = value_iteration(rn_mdp, theta=1e-8)
    sol_pi = policy_iteration(rn_mdp)
    vi_pi_diff = float(np.max(np.abs(sol_vi.v_star - sol_pi.v_star)))
    print(f"Value Iteration Converged:  {sol_vi.converged} in {sol_vi.iterations} iterations (Runtime: {sol_vi.runtime_sec*1000:.2f} ms)")
    print(f"Policy Iteration Converged: {sol_pi.converged} in {sol_pi.iterations} iterations (Runtime: {sol_pi.runtime_sec*1000:.2f} ms)")
    print(f"Max Discrepancy between VI and PI: {vi_pi_diff:.2e}")

    state_map = {name: i for i, name in enumerate(rn_mdp.states)}
    print(f"Key Textbook Optimal Values:")
    print(f"  V*(2, 2) [near +1]:  {sol_vi.v_star[state_map['(2,2)']]:.4f} (Expected approx 0.918)")
    print(f"  V*(0, 2) [top left]: {sol_vi.v_star[state_map['(0,2)']]:.4f} (Expected approx 0.812)")
    print(f"  V*(0, 0) [start]:    {sol_vi.v_star[state_map['(0,0)']]:.4f} (Expected approx 0.705)")
    print(f"  V*(3, 0) [bottom-r]: {sol_vi.v_star[state_map['(3,0)']]:.4f} (Expected approx 0.388)")

    # 5. Export Exact Solution Artifact for M13 RL
    print("\n--- 5. Export Exact Solution Artifact (Ground Truth for M13 RL) ---")
    with tempfile.TemporaryDirectory() as tmp_dir:
        art_path = Path(tmp_dir) / "rn_4x3_exact_solution.json"
        export_exact_mdp_solution(sol_vi, art_path)
        loaded = load_exact_mdp_solution(art_path)
        print(f"Exported artifact verified and reloaded from: {art_path.name}")
        print(f"Exact match on V*: {np.allclose(sol_vi.v_star, loaded.v_star)}")

    # 6. MDPAgent in SimulationEngine
    print("\n--- 6. MDPAgent in SimulationEngine ---")
    config = EnvironmentConfig(width=6, height=6, seed=42)
    world = GridWorld(config)
    world.reset()

    mdp_6x6 = GridWorldMDPBuilder.build_grid_mdp(
        width=6,
        height=6,
        walls=set(),
        terminals={(5, 5): 10.0},
        step_cost=-0.05,
        slip_prob=0.1,
        gamma=0.95,
    )
    agent = MDPAgent(agent_id="optimal_pilot", mdp=mdp_6x6)
    world.spawn_agent("optimal_pilot")

    engine = SimulationEngine(environment=world, seed=42)
    engine.register_agent(agent, PerceptionConfig(radius=4))

    print("Stepping simulation with MDPAgent following exact optimal policy pi*...")
    for step_num in range(1, 6):
        world_state = engine.step()
        ag_state = world_state.agents.get("optimal_pilot")
        pos = ag_state.position if ag_state else "Unknown"
        trace = agent.latest_trace
        act_name = trace.rationale.get("selected_action", "None") if trace else "None"
        reason = trace.rationale.get("reason", "None") if trace else "None"
        v_star_val = trace.metrics.get("v_star", 0.0) if trace else 0.0
        print(f"  Step {step_num}: Pos={pos}, Action={act_name}, V*={v_star_val:.3f}, Reason='{reason}'")

    print("\n[SUCCESS] Module M11 Demonstration Completed.")


if __name__ == "__main__":
    main()
