"""Runnable CLI demonstration of Hybrid Cognitive Architecture & Ablation Arena (SYS-15, Module M17)."""

from __future__ import annotations

from typing import Any

from autogen.agents.hybrid.config import HybridAgentConfig
from autogen.agents.hybrid.hybrid_agent import HybridCognitiveAgent
from autogen.intelligence.hybrid.mcts import MonteCarloTreeSearch
from autogen.intelligence.hybrid.mpc import ModelPredictiveController
from autogen.intelligence.hybrid.safety_shield import SymbolicSafetyShield
from autogen.research.ablation_arena import AblationArena
from autogen.simulation.core.contracts import Observation
from autogen.simulation.state.types import ActionType, Position


class SimplePlanningModel:
    """Simple 1D line navigation forward model for MCTS and MPC demonstrations."""

    def state_key(self, state: Any) -> str:
        return str(state)

    def get_valid_actions(self, state: Any) -> list[ActionType]:
        return [ActionType.LEFT, ActionType.RIGHT, ActionType.STAY]

    def step(self, state: Any, action: ActionType) -> tuple[Any, float, bool]:
        pos = int(state)
        if action == ActionType.RIGHT:
            pos += 1
        elif action == ActionType.LEFT:
            pos -= 1

        reward = 10.0 if pos == 3 else -abs(pos - 3)
        terminal = (pos == 3)
        return pos, reward, terminal


def main() -> None:
    print("================================================================================")
    print("         AUTOGEN AI - Hybrid Cognitive Architecture & Ablation Demo (M17)       ")
    print("================================================================================\n")

    # 1. Monte Carlo Tree Search
    print("--- 1. Monte Carlo Tree Search (UCT Planning to Target x=3) ---")
    model = SimplePlanningModel()
    mcts = MonteCarloTreeSearch(model=model, exploration_constant=1.414, rollout_depth=5, seed=42)
    best_mcts_act, mcts_diag = mcts.search(root_state=0, iterations=40)
    print(f"  MCTS Selected Action from x=0: {best_mcts_act.value}")
    print(f"  Root Visits: {mcts_diag['root_visits']} | Best Action Visits: {mcts_diag['best_action_visits']}")
    print(f"  Action Q-values: {mcts_diag['child_q_values']}\n")

    # 2. Model Predictive Control
    print("--- 2. Model Predictive Control (Receding Horizon Planning) ---")
    mpc = ModelPredictiveController(model=model, horizon=4, num_samples=25, seed=42)
    mpc_act, mpc_diag = mpc.plan(current_state=1)
    print(f"  MPC Selected Action from x=1: {mpc_act.value}")
    print(f"  Planned Horizon Trajectory: {mpc_diag['planned_sequence']}")
    print(f"  Expected Trajectory Return: {mpc_diag['best_return']:.2f}\n")

    # 3. Symbolic Safety Shield
    print("--- 3. Symbolic Safety Shield: Runtime Action Interception ---")
    shield = SymbolicSafetyShield(hazard_threshold=0.5, grid_width=10, grid_height=10)
    obs = Observation(
        agent_id="agent",
        step=1,
        position=Position(x=5, y=5),
        energy=80.0,
        health=100.0,
        carried_food=0,
        visible_cells=[],
        visible_agents=[],
    )
    # Propose moving UP into high hazard cell (5, 4)
    hazards = {Position(x=5, y=4): 0.95}
    approved_act, record = shield.verify_and_shield(
        step=1,
        proposed_action=ActionType.UP,
        observation=obs,
        hazard_probabilities=hazards,
    )
    print("  Proposed Action : UP (Target: (5, 4), Hazard Risk: 0.95)")
    print(f"  Shield Intervened: {record is not None}")
    if record:
        print(f"  Violated Rule   : {record.violated_rule}")
        print(f"  Safe Action Substituted: {approved_act.value}\n")

    # 4. Hybrid Cognitive Agent Glass-Box Trace
    print("--- 4. Hybrid Cognitive Agent: Glass-Box Deliberation Trace ---")
    agent = HybridCognitiveAgent(
        agent_id="hybrid_flagship",
        config=HybridAgentConfig(grid_width=10, grid_height=10),
    )
    agent.observe(obs)
    action, trace = agent.decide()
    print(f"  Chosen Action   : {action.action_type.value}")
    print(f"  Trace Algorithm : {trace.algorithm_name}")
    print(f"  Belief Summary  : {trace.rationale['belief_summary']}")
    print(f"  Reason          : {trace.rationale['reason']}\n")

    # 5. Ablation Arena
    print("--- 5. Ablation Arena: Systematic Subsystem Ablation Study ---")
    arena = AblationArena(
        eval_seeds=[101, 202, 303, 404],
        max_steps_per_episode=15,
        grid_size=10,
    )
    print("  Evaluating Full vs Ablated Subsystems on Held-out Worlds...")
    results = arena.run_ablation_study()

    print("\n  Configuration | Mean Return | Survival Rate | Veto Count | Cohen's d vs Full")
    print("  ------------- | ----------- | ------------- | ---------- | -----------------")
    for name, res in results.items():
        d_str = "Baseline" if name == "Full" else f"{res.cohens_d_vs_full:+.3f}"
        print(
            f"  {name:13s} | {res.mean_return:11.2f} | {res.survival_rate * 100:12.1f}% | "
            f"{res.mean_shield_vetoes:10.1f} | {d_str:>17s}"
        )

    print("\n================================================================================")
    print("                    Hybrid Cognitive Architecture Demo Complete                 ")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
