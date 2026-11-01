"""Unit, Mathematical Oracle, and Integration Tests for AUTOGEN Village Society."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from autogen.agents.registry import AgentConfig, AgentRegistry
from autogen.agents.village.controller import VillageSimulationController
from autogen.agents.village.villager_agent import VillagerAgent
from autogen.intelligence.registry import AlgorithmRegistry, populate_default_catalog
from autogen.intelligence.village.persona.llm_bridge import LLMPersonaBridge, PersonaProfile
from autogen.intelligence.village.persona.symbolic_dialogue import DialogueEventType
from autogen.intelligence.village.persona.village_bus import VillageDialogueBus
from autogen.intelligence.village.social_graph import (
    PageRankStepEvent,
    TrustGraphConfig,
    VillageTrustGraph,
)
from autogen.intelligence.village.utility_ethics import UtilityEngine, VillagerUtilityWeights
from autogen.research.village_experiments import (
    run_all_village_experiments,
    run_law_enforcement_threshold_experiment,
    run_reputation_resilience_experiment,
)
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.village.commodities import (
    CommodityInventory,
    CommodityType,
    DoubleAuctionMarket,
    MarketAsk,
    MarketBid,
)
from autogen.simulation.environment.village.village_grid import (
    VillageActionType,
    VillageRole,
    VillageWorld,
    VillageWorldConfig,
)
from autogen.simulation.state.types import Position


def test_trust_boundedness_and_asymmetry_invariant() -> None:
    """Invariant 1: For all i, j, T_{j->i}(t) in [0, 1] after 1,000 arbitrary events."""
    with pytest.raises(ValueError, match="asymmetry invariant"):
        TrustGraphConfig(alpha_coop=0.40, beta_defect=0.20)

    agent_ids = [f"v_{i}" for i in range(8)]
    graph = VillageTrustGraph(agent_ids=agent_ids)
    rng = np.random.default_rng(12345)
    events = ["coop", "defect", "crime_caught"]

    for _ in range(1000):
        j_idx = int(rng.integers(0, 8))
        i_idx = int(rng.integers(0, 8))
        ev = events[int(rng.integers(0, 3))]
        val = graph.record_interaction(agent_ids[j_idx], agent_ids[i_idx], ev)
        assert 0.0 <= val <= 1.0

        # Also test random gossip propagation
        k_idx = int(rng.integers(0, 8))
        g_val = graph.propagate_gossip(agent_ids[j_idx], agent_ids[i_idx], agent_ids[k_idx])
        assert 0.0 <= g_val <= 1.0

    mat = graph.trust_matrix
    assert np.all(mat >= 0.0) and np.all(mat <= 1.0)
    assert np.allclose(np.diag(mat), 0.0)


def test_status_distribution_and_linear_algebra_oracle() -> None:
    """Invariant 2: sum(S_i) == 1 and power iteration matches exact numpy.linalg.solve oracle."""
    agent_ids = [f"villager_{i}" for i in range(6)]
    graph = VillageTrustGraph(agent_ids=agent_ids)

    # Create asymmetric trust topology including a dangling node (villager_5 trusts nobody)
    graph.record_public_event("villager_0", "coop")
    graph.record_public_event("villager_0", "coop")
    graph.record_public_event("villager_1", "defect")
    graph.record_public_event("villager_2", "crime_caught")
    for target in agent_ids:
        graph.set_trust("villager_5", target, 0.0)

    step_events: list[PageRankStepEvent] = []
    status_map = graph.compute_societal_status(step_callback=step_events.append)

    assert len(step_events) >= 1
    total_status = sum(status_map.values())
    assert abs(total_status - 1.0) < 1e-6
    assert all(0.0 <= s <= 1.0 for s in status_map.values())

    # Independent Linear Algebra Oracle: (I - d * T_tilde^T) s = ((1 - d) / N) * 1
    n = len(agent_ids)
    d = graph.config.damping_factor
    tilde_t = graph.build_row_stochastic_matrix()
    lhs = np.eye(n) - d * tilde_t.T
    rhs = np.full(n, (1.0 - d) / n, dtype=np.float64)
    oracle_s = np.linalg.solve(lhs, rhs)
    oracle_s /= oracle_s.sum()

    computed_s = np.array([status_map[aid] for aid in graph.agent_ids], dtype=np.float64)
    assert np.allclose(computed_s, oracle_s, atol=1e-6)
    # Cooperative villager_0 must outrank caught offender villager_2
    assert status_map["villager_0"] > status_map["villager_2"]


def test_detection_monotonicity_and_lighting_invariants() -> None:
    """Invariant 3: d(P_detect)/d(distance) <= 0 along unoccluded line of sight."""
    world = VillageWorld(VillageWorldConfig(width=40, height=40, seed=42))
    actor_pos = Position(20, 20)

    probs_day: list[float] = []
    for dist in range(1, 12):
        witness_pos = Position(20 + dist, 20)
        # Ensure no hut at witness_pos
        p = world.compute_witness_sight_probability(
            actor_pos=actor_pos,
            witness_pos=witness_pos,
            witness_role=VillageRole.FARMER,
            ambient_light=1.0,
        )
        probs_day.append(p)

    # Monotonically non-increasing with distance
    for k in range(len(probs_day) - 1):
        assert probs_day[k] >= probs_day[k + 1] - 1e-9
    assert probs_day[0] > probs_day[-1]

    # Nighttime lowers detection probability relative to daytime
    p_day = world.compute_witness_sight_probability(
        actor_pos, Position(22, 20), VillageRole.GUARD, ambient_light=1.0
    )
    p_night = world.compute_witness_sight_probability(
        actor_pos, Position(22, 20), VillageRole.GUARD, ambient_light=0.2
    )
    assert p_day > p_night > 0.0


def test_emergence_of_rational_starvation_crime() -> None:
    """Verifies §2.3.2: Starving villager (H_i < tau) rationally selects Granary theft
    when survival gain outweighs expected sanction penalty, whereas sated villager does not.
    """
    engine = UtilityEngine(
        VillagerUtilityWeights(
            w_phys=2.5,
            w_econ=1.0,
            w_soc=14.0,
            theta_risk=0.20,
            crisis_threshold_tau=20.0,
            crisis_penalty_kappa=18.0,
        )
    )
    prices = {CommodityType.WHEAT: 3.0, CommodityType.WATER: 1.5}
    empty_inv = CommodityInventory()

    # Case A: Well-fed villager (H_i = 85) -> Legal work beats Granary theft
    legal_fed = engine.evaluate_action(
        action_type=VillageActionType.HARVEST_FIELD,
        current_hunger=85.0,
        current_hydration=80.0,
        current_energy=80.0,
        current_coins=20.0,
        current_inventory=empty_inv,
        clearing_prices=prices,
        current_status=0.12,
        projected_hunger=85.0,
        projected_hydration=80.0,
        projected_energy=77.0,
        projected_coins=20.0,
        projected_inventory=empty_inv.with_delta(CommodityType.WHEAT, 2.8),
        projected_status=0.128,
    )
    steal_fed = engine.evaluate_action(
        action_type=VillageActionType.STEAL_GRANARY,
        current_hunger=85.0,
        current_hydration=80.0,
        current_energy=80.0,
        current_coins=20.0,
        current_inventory=empty_inv,
        clearing_prices=prices,
        current_status=0.12,
        projected_hunger=100.0,
        projected_hydration=80.0,
        projected_energy=85.0,
        projected_coins=20.0,
        projected_inventory=empty_inv.with_delta(CommodityType.WHEAT, 3.5),
        projected_status=0.12,
        detection_probability=0.45,
        coin_penalty=12.0,
        status_loss=0.10,
    )
    assert legal_fed.expected_total_utility > steal_fed.expected_total_utility

    # Case B: Imminent starvation (H_i = 4.0 < tau=20.0) -> Stealing bread lifts H_i above tau
    stay_starving = engine.evaluate_action(
        action_type=VillageActionType.STAY,
        current_hunger=4.0,
        current_hydration=80.0,
        current_energy=80.0,
        current_coins=0.0,
        current_inventory=empty_inv,
        clearing_prices=prices,
        current_status=0.12,
        projected_hunger=4.0,
        projected_hydration=80.0,
        projected_energy=80.0,
        projected_coins=0.0,
        projected_inventory=empty_inv,
        projected_status=0.12,
    )
    steal_starving = engine.evaluate_action(
        action_type=VillageActionType.STEAL_GRANARY,
        current_hunger=4.0,
        current_hydration=80.0,
        current_energy=80.0,
        current_coins=0.0,
        current_inventory=empty_inv,
        clearing_prices=prices,
        current_status=0.12,
        projected_hunger=49.0,
        projected_hydration=80.0,
        projected_energy=85.0,
        projected_coins=0.0,
        projected_inventory=empty_inv.with_delta(CommodityType.WHEAT, 3.5),
        projected_status=0.12,
        detection_probability=0.35,
        coin_penalty=12.0,
        status_loss=0.10,
    )
    assert steal_starving.expected_total_utility > stay_starving.expected_total_utility


def test_double_auction_market_and_status_discount() -> None:
    """Verifies double auction matching and high-status spread discount."""
    market = DoubleAuctionMarket()
    bids = [
        MarketBid(
            buyer_id="high_status_buyer",
            commodity=CommodityType.WHEAT,
            quantity=2.0,
            max_unit_price=4.0,
            buyer_status=0.80,
        ),
        MarketBid(
            buyer_id="low_status_buyer",
            commodity=CommodityType.WHEAT,
            quantity=2.0,
            max_unit_price=4.0,
            buyer_status=0.05,
        ),
    ]
    asks = [
        MarketAsk(
            seller_id="farmer_seller",
            commodity=CommodityType.WHEAT,
            quantity=4.0,
            min_unit_price=3.0,
        )
    ]
    trades = market.clear_market(step=1, bids=bids, asks=asks)
    assert len(trades) == 2
    assert trades[0].buyer_id == "high_status_buyer"
    assert trades[0].clearing_unit_price < trades[1].clearing_unit_price


def test_simulation_determinism_and_llm_isolation() -> None:
    """Invariants 4 & 5: Identical seed yields identical state_hash; mocked/failing LLM
    persona decorator has zero impact on physical state, balances, or decisions.
    """
    ctrl_1 = VillageSimulationController(
        world_config=VillageWorldConfig(seed=77),
        enable_llm_persona=False,
    )
    ctrl_2 = VillageSimulationController(
        world_config=VillageWorldConfig(seed=77),
        enable_llm_persona=True,
    )
    # Inject a mocked LLM completion function that returns custom dramatic text
    ctrl_2.persona_bridge = LLMPersonaBridge(
        enable_external_llm=True,
        custom_completion_fn=lambda prompt: "By the ancestors, the Granary stands tall!",
    )

    for _ in range(10):
        s1 = ctrl_1.step()
        s2 = ctrl_2.step()
        assert s1["state_hash"] == s2["state_hash"]
        assert s1["economy"] == s2["economy"]

    # Also verify graceful fallback when LLM completion raises an exception
    def failing_llm(_prompt: str) -> str:
        raise RuntimeError("Simulated API timeout")

    bridge_fail = LLMPersonaBridge(enable_external_llm=True, custom_completion_fn=failing_llm)
    sample_trace = ctrl_1.traces_by_agent["villager_arun"]
    ev = bridge_fail.decorate_decision(
        persona=PersonaProfile(agent_id="villager_arun", name="Arun", role="Farmer"),
        trace=sample_trace,
        position=(10, 10),
        hunger=80.0,
        hydration=80.0,
        coins=20.0,
        wheat_qty=2.0,
    )
    assert ev.mode == "symbolic"
    assert len(ev.utterance) > 5


def test_spatial_dialogue_bus_and_gossip_propagation() -> None:
    """Verifies spatial attenuation and third-party reputation gossip on VillageDialogueBus."""
    agent_ids = ["villager_a", "villager_b", "villager_c"]
    graph = VillageTrustGraph(agent_ids=agent_ids)
    # villager_a distrusts villager_c
    graph.set_trust("villager_a", "villager_c", 0.10)
    before_trust = graph.get_trust("villager_b", "villager_c")

    bus = VillageDialogueBus(base_hearing_radius=10.0)
    from autogen.intelligence.village.persona.symbolic_dialogue import VillageDialogueEvent

    gossip_ev = VillageDialogueEvent(
        step=1,
        speaker_id="villager_a",
        speaker_name="Arun",
        recipient_id="villager_b",
        recipient_name="Meera",
        subject_id="villager_c",
        event_type=DialogueEventType.GOSSIP,
        utterance="I saw Ravi lurking near the Granary!",
        action_chosen="STAY",
        position=(10, 10),
    )
    heard = bus.broadcast(
        event=gossip_ev,
        agent_positions={
            "villager_a": Position(10, 10),
            "villager_b": Position(12, 10),
            "villager_c": Position(35, 35),  # Out of earshot
        },
        trust_graph=graph,
    )
    assert len(heard) == 1
    assert heard[0].listener_id == "villager_b"
    after_trust = graph.get_trust("villager_b", "villager_c")
    assert after_trust < before_trust


def test_registries_and_standard_simulation_engine_compatibility() -> None:
    """Verifies registration in AlgorithmRegistry, AgentRegistry, and SimulationEngine."""
    populate_default_catalog()
    tp_meta = AlgorithmRegistry.get("trust_pagerank")
    su_meta = AlgorithmRegistry.get("social_utility_ethics")
    assert tp_meta.doc_path == "docs/algorithms/village/social_capital_and_ethics.md"
    assert su_meta.doc_path == "docs/algorithms/village/social_capital_and_ethics.md"

    agent = AgentRegistry.create(
        AgentConfig(
            agent_type="villager_agent",
            agent_id="villager_arun",
            custom_params={"name": "Arun", "role": "Farmer"},
        )
    )
    assert isinstance(agent, VillagerAgent)

    world = VillageWorld(VillageWorldConfig(width=24, height=24, seed=42))
    engine = SimulationEngine(
        environment=world,
        agents={"villager_arun": agent},
        seed=42,
    )
    final_state = engine.run(max_ticks=5)
    assert final_state.step == 5


def test_empirical_scenarios_and_api_endpoints() -> None:
    """Verifies Phase 6 empirical scenarios and FastAPI /api/v1/village endpoints."""
    law_res = run_law_enforcement_threshold_experiment()
    assert law_res.metrics["low_surveillance_theft_rate"] >= law_res.metrics["high_surveillance_theft_rate"]

    rep_res = run_reputation_resilience_experiment()
    assert rep_res.metrics["post_crime_status"] < rep_res.metrics["initial_status"]
    assert rep_res.metrics["final_recovered_status"] > rep_res.metrics["post_crime_status"]

    all_exps = run_all_village_experiments()
    assert len(all_exps) == 3

    client = TestClient(app)
    r_state = client.get("/api/v1/village/state")
    assert r_state.status_code == 200
    assert r_state.json()["data"]["step"] >= 0

    r_step = client.post("/api/v1/village/step", json={"steps": 2})
    assert r_step.status_code == 200
    assert r_step.json()["data"]["step"] >= 2

    r_drought = client.post("/api/v1/village/drought?active=true")
    assert r_drought.status_code == 200
    assert r_drought.json()["data"]["drought_active"] is True

    r_export = client.get("/api/v1/village/export")
    assert r_export.status_code == 200
    assert r_export.json()["data"]["name"] == "village_world"

