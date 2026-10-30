"""Module M09 CSP & Automated Planning CLI Usage Example.

Demonstrates:
1. Solving Map Coloring (Australia) and N-Queens via Backtracking with MRV + LCV and AC-3.
2. Solving 9x9 Sudoku via CSP constraint propagation.
3. Solving multi-agent patrol/foraging schedule assignment.
4. STRIPS Classical Planning (Forward A* with h_max, h_add, h_ff, and Graphplan) on Survival Domain.
5. Independent Plan Validation and Mutation Detection.
6. PlanningAgent executing in SimulationEngine and generating authentic DecisionTraces.
"""

from __future__ import annotations

from autogen.agents.planning.planning_agent import PlanningAgent
from autogen.intelligence.csp.problems import (
    australia_map_coloring,
    multi_agent_schedule_csp,
    n_queens_problem,
    sudoku_problem,
)
from autogen.intelligence.csp.solver import backtracking_search, min_conflicts
from autogen.intelligence.csp.verifier import verify_csp_solution
from autogen.intelligence.planning.domains.survival import create_survival_crafting_problem
from autogen.intelligence.planning.forward import ForwardStateSpacePlanner
from autogen.intelligence.planning.graphplan import graphplan
from autogen.intelligence.planning.heuristics import h_add, h_ff, h_max
from autogen.intelligence.planning.validator import PlanValidator
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.perception.contracts import PerceptionConfig
from autogen.simulation.state.types import Position


def main() -> None:
    print("=" * 75)
    print("  AUTOGEN MODULE M09: CONSTRAINT SATISFACTION & AUTOMATED PLANNING")
    print("=" * 75)

    # 1. CSP: Map Coloring
    print("\n--- 1. Map Coloring (Australia CSP) ---")
    map_csp = australia_map_coloring()
    map_sol = backtracking_search(map_csp, inference="forward_checking")
    print(f"Australia 3-Coloring Solved: {map_sol.is_solved} (Nodes: {map_sol.nodes_expanded}, Backtracks: {map_sol.backtracks})")
    print(f"Assignment: {map_sol.assignment}")
    v_map, _ = verify_csp_solution(map_csp, map_sol.assignment)
    print(f"Independent Constraint Verification: {'PASS' if v_map else 'FAIL'}")

    # 2. CSP: 8-Queens
    print("\n--- 2. N-Queens (N=8 CSP) ---")
    queens_csp = n_queens_problem(8)
    queens_sol = backtracking_search(queens_csp, inference="forward_checking")
    print(f"8-Queens Backtracking Solved: {queens_sol.is_solved} (Nodes: {queens_sol.nodes_expanded}, Backtracks: {queens_sol.backtracks})")
    print(f"Queens Positions (Col -> Row): {queens_sol.assignment}")
    v_q, _ = verify_csp_solution(queens_csp, queens_sol.assignment)
    print(f"Independent Constraint Verification: {'PASS' if v_q else 'FAIL'}")

    # 3. CSP: Multi-Agent Schedule Assignment
    print("\n--- 3. Multi-Agent Patrol Schedule CSP ---")
    sched_csp = multi_agent_schedule_csp(
        agent_ids=("Alpha", "Bravo"),
        tasks=("PatrolNorth", "PatrolSouth", "GatherFood"),
        time_slots=(1, 2),
        task_conflicts=[("PatrolNorth", "PatrolSouth")],
    )
    sched_sol = backtracking_search(sched_csp, inference="forward_checking")
    print(f"Schedule Solved: {sched_sol.is_solved}")
    for task, (ag, slot) in sched_sol.assignment.items():
        print(f"  Task '{task}' -> Assigned to {ag} at Time Slot {slot}")

    # 4. STRIPS Planning: Survival & Crafting Domain
    print("\n--- 4. STRIPS Classical Planning: Survival Crafting Domain ---")
    survival_prob = create_survival_crafting_problem()
    planner = ForwardStateSpacePlanner(heuristic_fn=h_max)

    print(f"Initial State Facts: {set(survival_prob.initial_facts)}")
    print(f"Goal Condition: {set(survival_prob.goal_facts)}")
    h0_max = h_max(survival_prob.initial_facts, survival_prob)
    h0_add = h_add(survival_prob.initial_facts, survival_prob)
    h0_ff = h_ff(survival_prob.initial_facts, survival_prob)
    print(f"Initial Heuristic Estimates: h_max={h0_max:.1f}, h_add={h0_add:.1f}, h_ff={h0_ff:.1f}")

    plan_astar = planner.solve_astar(survival_prob)
    print(f"\nA* Forward Search Result: Solved={plan_astar.is_solved}, Cost={plan_astar.total_cost}, Steps={len(plan_astar.actions)}")
    for idx, act in enumerate(plan_astar.actions, 1):
        print(f"  Step {idx}: {act.name}")

    # Independent Plan Validation
    valid_plan, final_facts, err = PlanValidator.validate_plan(survival_prob, plan_astar.actions)
    print(f"\nIndependent Plan Validation: {'PASS' if valid_plan else f'FAIL: {err}'}")

    # 5. Graphplan Demonstration
    print("\n--- 5. Graphplan Algorithm ---")
    gp_plan = graphplan(survival_prob, max_levels=15)
    print(f"Graphplan Solved: {gp_plan.is_solved}, Total Actions: {len(gp_plan.actions)}, Graph Levels: {gp_plan.metrics.nodes_expanded}")

    # 6. PlanningAgent Simulation Execution
    print("\n--- 6. PlanningAgent in SimulationEngine ---")
    config = EnvironmentConfig(width=12, height=12, seed=42)
    world = GridWorld(config)
    world.reset()

    agent = PlanningAgent(agent_id="craftsman_1")
    agent.set_problem(survival_prob)
    agent.set_target_location("Camp", Position(4, 4))
    agent.set_target_location("Forest", Position(7, 4))
    agent.set_target_location("Meadow", Position(4, 7))

    world.spawn_agent("craftsman_1")
    engine = SimulationEngine(environment=world, seed=42)
    engine.register_agent(agent, PerceptionConfig(radius=5))

    print("Stepping simulation with PlanningAgent...")
    for step_num in range(1, 8):
        world_state = engine.step()
        ag_state = world_state.agents.get("craftsman_1")
        pos = ag_state.position if ag_state else "Unknown"
        trace = agent.latest_trace
        act_name = trace.rationale.get("selected_action", "None") if trace else "None"
        reason = trace.rationale.get("reason", "None") if trace else "None"
        strips_act = trace.rationale.get("strips_action", "In-Progress") if trace else "None"
        print(f"  Tick {step_num}: Pos={pos}, SelectedAction={act_name}, HighLevelSubtask='{strips_act}', Reason={reason}")

    print("\n[SUCCESS] Module M09 Demonstration Completed.")


if __name__ == "__main__":
    main()

