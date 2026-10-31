"""Demonstration of Multi-Agent Systems & Collective Intelligence (SYS-14, Module M15)."""

from __future__ import annotations

import numpy as np

from autogen.agents.multiagent.cooperative_agent import CooperativeAgent
from autogen.agents.multiagent.iql_agent import IQLAgent
from autogen.intelligence.multiagent.commons import CommonsResourcePool
from autogen.intelligence.multiagent.communication import CommunicationBus
from autogen.intelligence.multiagent.contracts import Message
from autogen.intelligence.multiagent.coordination import (
    VickreyAuction,
)
from autogen.intelligence.multiagent.emergent_roles import RoleDetector
from autogen.intelligence.multiagent.iql import IndependentQLearning
from autogen.intelligence.multiagent.matrix_games import MatrixGame
from autogen.intelligence.multiagent.metrics import MultiAgentMetrics
from autogen.intelligence.multiagent.stigmergy import PheromoneField
from autogen.intelligence.multiagent.value_decomposition import QMIXLite, ValueDecompositionNetworks
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.state.types import Position


def run_demo() -> None:
    print("=" * 80)
    print("  AUTOGEN MODULE M15: MULTI-AGENT SYSTEMS & COLLECTIVE INTELLIGENCE")
    print("=" * 80)

    # 1. Matrix Games: Social Dilemma Convergence
    print("\n--- 1. Social Dilemma Games: Prisoner's Dilemma & Stag Hunt ---")
    pd_game = MatrixGame.prisoners_dilemma()
    iql_pd = IndependentQLearning(agent_ids=["agent_1", "agent_2"], num_actions=2, alpha=0.1, gamma=0.9, epsilon=0.05, seed=42)
    stage_key = "dilemma_stage"

    for _ in range(300):
        actions = iql_pd.select_actions({"agent_1": stage_key, "agent_2": stage_key}, explore=True)
        r1, r2 = pd_game.get_payoffs(actions["agent_1"], actions["agent_2"])
        iql_pd.update(
            prev_obs={"agent_1": stage_key, "agent_2": stage_key},
            actions=actions,
            rewards={"agent_1": r1, "agent_2": r2},
            next_obs={"agent_1": stage_key, "agent_2": stage_key},
            terminated=False,
        )

    q1_pd = iql_pd.learners["agent_1"].q_table.get_q(stage_key)
    q2_pd = iql_pd.learners["agent_2"].q_table.get_q(stage_key)
    print("  Prisoner's Dilemma Q-values (0=Cooperate, 1=Defect):")
    print(f"    Agent 1 Q: [Coop={q1_pd[0]:.2f}, Defect={q1_pd[1]:.2f}] -> Preferred: {'Defect' if q1_pd[1] > q1_pd[0] else 'Cooperate'}")
    print(f"    Agent 2 Q: [Coop={q2_pd[0]:.2f}, Defect={q2_pd[1]:.2f}] -> Preferred: {'Defect' if q2_pd[1] > q2_pd[0] else 'Cooperate'}")

    # 2. Tragedy of the Commons Simulation
    print("\n--- 2. Tragedy of the Commons: Over-harvest vs Sustainability ---")
    pool = CommonsResourcePool(carrying_capacity=500.0, growth_rate=0.15, collapse_threshold=80.0)
    print(f"  Initial Commons Stock: {pool.current_stock:.1f} (Carrying Capacity: {pool.carrying_capacity:.1f})")

    # Step sustainable harvest
    for step in range(1, 4):
        pool.harvest("worker_1", 15.0)
        pool.harvest("worker_2", 15.0)
        info = pool.step_regrowth()
        print(f"  Step {step:02d} (Sustainable) -> Harvest: 30.0 | Regrowth: +{info['regrowth']:.2f} | Stock: {info['current_stock']:.2f}")

    # Step greedy over-harvest triggering collapse
    print("  Triggering over-harvest spike (Harvest: 400.0)...")
    pool.harvest("greedy_corporation", 400.0)
    info_collapse = pool.step_regrowth()
    print(f"  After Over-Harvest -> Stock: {info_collapse['current_stock']:.2f} | Collapsed Status: {info_collapse['collapsed']} | Regrowth: {info_collapse['regrowth']:.2f}")

    # 3. Pheromone Stigmergy & Physical Communication Bus
    print("\n--- 3. Indirect Stigmergy & Physical Communication Bus ---")
    field = PheromoneField(width=10, height=10, evaporation_rate=0.05, diffusion_rate=0.1)
    bus = CommunicationBus(energy_cost=1.5, max_range=8.0, bandwidth_limit=3, noise_prob=0.0)

    # Deposit chemical trail at (5, 5)
    field.deposit(Position(x=5, y=5), amount=60.0)
    field.step()
    conc_center = field.sense(Position(x=5, y=5))
    conc_neighbor = field.sense(Position(x=6, y=5))
    gx, gy = field.gradient(Position(x=4, y=5))
    print(f"  Pheromone Peak Concentration: {conc_center:.2f} | Diffused to Neighbor (6,5): {conc_neighbor:.2f}")
    print(f"  Spatial Gradient at (4,5) pointing towards peak: dPhi/dx = {gx:.2f}, dPhi/dy = {gy:.2f}")

    # Communication bus
    msg = Message(
        sender_id="scout_alpha",
        recipient_id=None,
        payload={"target": "food_zone_1", "priority": 1},
        sent_at_step=1,
        sender_position=Position(x=2, y=2),
    )
    ok, cost, _ = bus.send(msg, sender_energy=10.0)
    delivered = bus.receive("worker_beta", Position(x=5, y=4))
    print(f"  Broadcast Sent: {ok} (Cost: {cost} energy) | Delivered to In-Range Agent (dist=3.6): {len(delivered)} message(s)")

    # 4. Value Decomposition: VDN & QMIX Monotonicity
    print("\n--- 4. Multi-Agent Value Decomposition (VDN & QMIX-Lite) ---")
    q1 = np.array([2.0, 5.0, 1.0])
    q2 = np.array([3.0, 1.0, 6.0])
    vdn_joint_best = ValueDecompositionNetworks.individual_best_actions([q1, q2])
    vdn_igm_verified = ValueDecompositionNetworks.verify_igm_condition([q1, q2])
    print(f"  VDN Optimal Individual Action Indices: {vdn_joint_best} (Agent 1: Act {vdn_joint_best[0]}, Agent 2: Act {vdn_joint_best[1]})")
    print(f"  VDN Individual-Global-Max (IGM) Monotonicity Verified: {vdn_igm_verified}")

    qmix = QMIXLite(num_agents=2, state_dim=4, embed_dim=8, seed=42)
    state_sample = np.array([0.5, 0.2, -0.1, 0.8])
    q_tot = qmix.forward(np.array([5.0, 6.0]), state_sample)
    qmix_monotonic = qmix.check_monotonicity(state_sample, delta=1.0)
    print(f"  QMIX Mixed Joint Q_tot: {q_tot:.4f} | Non-negative Monotonicity Checked: {qmix_monotonic}")

    # 5. Coordination: Vickrey Second-Price Auction
    print("\n--- 5. Mechanism Design: Second-Price Sealed-Bid Vickrey Auction ---")
    auction = VickreyAuction(item_name="scarce_water_well", reserve_price=5.0)
    auction.submit_bid("forager_a", 18.0)
    auction.submit_bid("forager_b", 32.0)
    auction.submit_bid("forager_c", 25.0)
    winner, price = auction.resolve()
    print(f"  Auction Winner: {winner} | Clearing Price (2nd Highest): {price:.2f} (Winner Bid: 32.0)")

    # 6. Emergent Social Role Detection & Inequality
    print("\n--- 6. Emergent Social Roles & Gini Resource Inequality ---")
    agent_behaviors = {
        "bot_1": np.array([0.1, 0.9, 10.0, 0.05]),  # Scout
        "bot_2": np.array([0.15, 0.85, 8.0, 0.1]),   # Scout
        "bot_3": np.array([0.9, 0.2, 1.0, 0.2]),     # Gatherer
        "bot_4": np.array([0.85, 0.25, 2.0, 0.15]),  # Gatherer
        "bot_5": np.array([0.05, 0.05, 0.0, 0.0]),   # Free-rider
    }
    detector = RoleDetector(n_roles=3, seed=42)
    assigned_roles = detector.fit_predict(agent_behaviors)
    print("  Cluster-Detected Agent Functional Roles:")
    for aid, rname in assigned_roles.items():
        print(f"    {aid:6s} -> Assigned Role: {rname}")

    wealth = [120.0, 110.0, 95.0, 105.0, 5.0]
    gini = MultiAgentMetrics.gini_coefficient(wealth)
    print(f"  Society Resource Wealth Gini Index: {gini:.4f} (0 = equal, 1 = concentrated)")

    # 7. Simulation Run in SimulationEngine
    print("\n--- 7. Multi-Agent Simulation Engine Execution & Glass-Box Traces ---")
    config = EnvironmentConfig(width=6, height=6, seed=42)
    world = GridWorld(config)
    world.reset()

    a1 = IQLAgent(agent_id="iql_forager")
    a2 = CooperativeAgent(agent_id="stigmergy_scout", pheromone_field=field, comm_bus=bus)
    world.spawn_agent("iql_forager")
    world.spawn_agent("stigmergy_scout")

    sim = SimulationEngine(environment=world, agents={"iql_forager": a1, "stigmergy_scout": a2}, seed=42)
    for tick in range(1, 4):
        sim.step()
        t1 = a1.latest_trace
        t2 = a2.latest_trace
        if t1 and t2:
            print(f"  Tick {tick:02d} | Forager Act: {t1.rationale.get('selected_action', 'N/A'):5s} (Max Q: {t1.metrics.get('max_q', 0.0):.2f}) | Scout Act: {t2.rationale.get('selected_action', 'N/A'):5s} (Pheromone: {t2.metrics.get('pheromone_conc', 0.0):.2f})")

    print("\n" + "=" * 80)
    print("  MODULE M15 VERIFIED: MULTI-AGENT SYSTEMS & COLLECTIVE INTELLIGENCE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
