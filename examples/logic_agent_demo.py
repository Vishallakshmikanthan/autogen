"""Module M08 Logic Reasoning and LogicAgent CLI Usage Example.

Demonstrates:
1. Propositional parsing, CNF conversion, and DPLL SAT solving.
2. Resolution Refutation with formal proof tree reconstruction.
3. Horn clause Forward and Backward Chaining.
4. First-Order Logic Unification with occurs-check.
5. LogicAgent perception-to-knowledge-base grounding and deductive safety vetting.
"""

from __future__ import annotations

import sys
from autogen.simulation.state.types import ActionType, Position, TerrainType
from autogen.intelligence.logic.ast import PropVar
from autogen.intelligence.logic.parser import parse_formula
from autogen.intelligence.logic.cnf import to_cnf
from autogen.intelligence.logic.dpll import dpll_satisfiable
from autogen.intelligence.logic.resolution import pl_resolution
from autogen.intelligence.logic.horn import DefiniteClause, pl_fc_entails, pl_bc_entails
from autogen.intelligence.logic.fol import Constant, Function, Predicate, Variable, unify
from autogen.intelligence.logic.knowledge_base import KnowledgeBase
from autogen.agents.symbolic.safety_layer import SafetyRuleLayer
from autogen.agents.symbolic.logic_agent import LogicAgent
from autogen.simulation.core.contracts import Observation
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.perception.contracts import PerceptionConfig


def main() -> None:
    print("=" * 70)
    print("  AUTOGEN MODULE M08: KNOWLEDGE REPRESENTATION & LOGIC REASONING")
    print("=" * 70)

    # 1. DPLL SAT Solver Demonstration
    print("\n--- 1. DPLL SAT Solver ---")
    formula_str = "(P | Q) & (~P | R) & (~Q | R)"
    formula = parse_formula(formula_str)
    clauses = to_cnf(formula)
    print(f"Formula: {formula_str}")
    print(f"CNF Clauses: {clauses}")
    sat, model = dpll_satisfiable(clauses)
    print(f"Satisfiable: {sat}, Satisfying Model: {model}")

    # 2. Resolution Refutation with Proof Tree
    print("\n--- 2. Resolution Refutation Theorem Proving ---")
    kb_str = "(Rain => WetGround) & Rain"
    query_str = "WetGround"
    kb_form = parse_formula(kb_str)
    query_form = parse_formula(query_str)
    proved, proof = pl_resolution(kb_form, query_form)
    print(f"KB: {kb_str}")
    print(f"Query: {query_str}")
    print(f"Entailed (Proved by Refutation): {proved}")
    if proof:
        print(proof.pretty_print())

    # 3. Horn Chaining
    print("\n--- 3. Horn Clause Forward and Backward Chaining ---")
    clauses_horn = [
        DefiniteClause(premises=(), conclusion="Rain"),  # Fact
        DefiniteClause(premises=("Rain",), conclusion="Wet"),
        DefiniteClause(premises=("Wet",), conclusion="Slippery"),
    ]
    print(f"Horn Knowledge Base:")
    for c in clauses_horn:
        print(f"  {c}")
    fc_entailed, fc_proof = pl_fc_entails(clauses_horn, "Slippery")
    bc_entailed, bc_proof = pl_bc_entails(clauses_horn, "Slippery")
    print("Forward Chaining for 'Slippery':", fc_entailed)
    print("Backward Chaining for 'Slippery':", bc_entailed)

    # 4. FOL Unification with Occurs-Check
    print("\n--- 4. First-Order Logic Unification ---")
    x = Variable("x")
    john = Constant("john")
    p1 = Predicate("Knows", (john, x))
    p2 = Predicate("Knows", (john, Constant("jane")))
    subst = unify(p1, p2)
    print(f"Unifying {p1} with {p2} -> Substitution: {subst}")

    # Occurs-check failure demonstration
    f_x = Function("f", (x,))
    p3 = Predicate("P", (x,))
    p4 = Predicate("P", (f_x,))
    subst_occurs = unify(p3, p4)
    print(f"Unifying {p3} with {p4} (occurs check) -> Substitution: {subst_occurs} (Expected None)")

    # 5. Symbolic Safety Layer & Logic Agent in Simulation Engine
    print("\n--- 5. Deductive Safety Layer & LogicAgent in Simulation ---")
    config = EnvironmentConfig(width=10, height=10, seed=42)
    world = GridWorld(config)
    world.reset()

    agent = LogicAgent(agent_id="athena_1")
    world.spawn_agent("athena_1")

    engine = SimulationEngine(environment=world, seed=42)
    engine.register_agent(agent, PerceptionConfig(radius=4))

    print("Stepping simulation with LogicAgent...")
    for step_num in range(1, 6):
        world_state = engine.step()
        agent_state = world_state.agents.get("athena_1")
        trace = agent.latest_trace
        pos = agent_state.position if agent_state else "Unknown"
        action_name = trace.rationale.get("selected_action", "None") if trace else "None"
        reason = trace.rationale.get("reason", "None") if trace else "None"
        justification = trace.rationale.get("symbolic_justification", {}) if trace else {}
        print(f"  Step {step_num}: Position={pos}, SelectedAction={action_name}, Reason={reason}")
        if justification:
            print(f"    Proof Rejection: {justification.get('rejected_actions')}")

    print("\n[SUCCESS] Module M08 CLI Usage Example Completed.")


if __name__ == "__main__":
    main()
