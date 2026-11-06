"""End-to-End Integration and Ablation Tests for Humanlike Villager Architecture (§5, §7)."""

from __future__ import annotations

from autogen.agents.village.controller import HumanlikeConfig, VillageSimulationController
from autogen.simulation.environment.village.village_grid import VillageWorldConfig


def test_thunderstorm_outdoor_shelter_preemption_statistical_oracle() -> None:
    """Verifies §5 Acceptance: Thunderstorm causes >= 70% of outdoor agents to switch to SHELTER within 2 ticks."""
    switch_fractions: list[float] = []

    for seed in range(42, 62):  # 20 independent seeds
        ctrl = VillageSimulationController(
            world_config=VillageWorldConfig(seed=seed, width=40, height=40),
            humanlike_config=HumanlikeConfig(
                enable_personality=True,
                enable_attention=True,
                enable_intentions=True,
                enable_contagion=True,
                enable_reflection=True,
            ),
        )

        # Warm up 2 steps so agents develop active intentions
        ctrl.step()
        ctrl.step()

        # Trigger thunderstorm
        ctrl.world.weather_process.trigger_thunderstorm(ctrl.world.step_count, intensity=0.95)

        # Step 1 tick after thunder
        ctrl.step()

        # Measure fraction of living agents whose intention is SHELTER
        living = [a for a in ctrl.agents.values() if ctrl.world.villagers[a.agent_id].is_alive]
        sheltering = sum(1 for a in living if a.intention.kind == "SHELTER")
        fraction = sheltering / max(1, len(living))
        switch_fractions.append(fraction)

    mean_fraction = sum(switch_fractions) / len(switch_fractions)
    assert mean_fraction >= 0.70, f"Expected >= 70% sheltering after thunder, got {mean_fraction:.2%}"


def test_canonical_hash_seed_determinism_invariant() -> None:
    """Invariant: Re-running simulation with identical seeds produces identical canonical hashes (§5)."""
    hashes_run1: list[str] = []
    hashes_run2: list[str] = []

    # Run 1
    ctrl1 = VillageSimulationController(
        world_config=VillageWorldConfig(seed=12345),
        humanlike_config=HumanlikeConfig(),
    )
    for _ in range(5):
        ctrl1.step()
        hashes_run1.append(ctrl1.compute_canonical_hash())

    # Run 2
    ctrl2 = VillageSimulationController(
        world_config=VillageWorldConfig(seed=12345),
        humanlike_config=HumanlikeConfig(),
    )
    for _ in range(5):
        ctrl2.step()
        hashes_run2.append(ctrl2.compute_canonical_hash())

    assert hashes_run1 == hashes_run2


def test_decision_trace_rationale_enrichment() -> None:
    """Verifies §7 Acceptance 5: DecisionTrace.rationale includes alerts, u_react, c_switch, and intention."""
    ctrl = VillageSimulationController(
        world_config=VillageWorldConfig(seed=42),
        humanlike_config=HumanlikeConfig(),
    )
    ctrl.step()

    for _aid, trace in ctrl.traces_by_agent.items():
        assert "alerts" in trace.rationale
        assert "u_react" in trace.rationale
        assert "c_switch" in trace.rationale
        assert "intention" in trace.rationale
        assert "gaze" in trace.rationale


def test_ablation_toggle_preservation() -> None:
    """Verifies that disabling humanlike layers preserves simulation stability."""
    ctrl_ablated = VillageSimulationController(
        world_config=VillageWorldConfig(seed=42),
        humanlike_config=HumanlikeConfig(
            enable_personality=False,
            enable_attention=False,
            enable_intentions=False,
            enable_contagion=False,
            enable_reflection=False,
            enable_weather=False,
        ),
    )
    for _ in range(5):
        payload = ctrl_ablated.step()
        assert payload["step"] > 0
        assert "villagers" in payload
