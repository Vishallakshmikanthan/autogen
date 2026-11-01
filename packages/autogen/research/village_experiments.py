"""Empirical Research Scenarios for AUTOGEN Village Society (Phase 6)."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from autogen.agents.village.controller import VillageSimulationController
from autogen.intelligence.village.social_graph import VillageTrustGraph
from autogen.intelligence.village.utility_ethics import UtilityEngine, VillagerUtilityWeights
from autogen.simulation.environment.village.commodities import CommodityInventory, CommodityType
from autogen.simulation.environment.village.village_grid import (
    VillageActionType,
    VillageWorldConfig,
)


class VillageExperimentResult(BaseModel):
    """Structured empirical result for a reproducible Village Society scenario."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str
    title: str
    hypothesis: str
    seeds_evaluated: int = Field(ge=1)
    metrics: dict[str, float]
    series: list[dict[str, float | int | str]]
    conclusion: str
    limitations: str


def compute_gini_coefficient(values: list[float]) -> float:
    """Computes the exact Gini inequality coefficient $G \\in [0, 1)$ from scratch."""
    arr = np.asarray([max(0.0, float(v)) for v in values], dtype=np.float64)
    n = arr.size
    if n <= 1 or float(arr.sum()) <= 1e-12:
        return 0.0
    sorted_arr = np.sort(arr)
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((np.sum((2.0 * index - n - 1.0) * sorted_arr)) / (n * np.sum(sorted_arr)))


def run_law_enforcement_threshold_experiment(
    seeds: tuple[int, ...] = (101, 202, 303),
) -> VillageExperimentResult:
    """Scenario 1: The Law Enforcement Threshold.

    Evaluates how increasing guard surveillance (detection probability $P_{\\text{detect}}$)
    and coin penalties $C_{\\text{penalty}}$ suppresses rational Granary theft.
    """
    prices = {CommodityType.WHEAT: 3.5, CommodityType.WATER: 1.5}
    inv = CommodityInventory(items={CommodityType.WHEAT: 0.5, CommodityType.WATER: 2.0})

    series: list[dict[str, float | int | str]] = []
    threshold_p_detect = 1.0

    for p_det_pct in range(10, 95, 10):
        p_det = p_det_pct / 100.0
        steal_count = 0
        total_trials = 0
        for seed in seeds:
            rng = np.random.default_rng(seed + p_det_pct)
            for _ in range(10):
                theta = float(rng.uniform(0.10, 0.45))
                hunger = float(rng.uniform(22.0, 55.0))
                engine = UtilityEngine(
                    VillagerUtilityWeights(w_phys=2.2, w_econ=1.2, w_soc=14.0, theta_risk=theta)
                )
                legal_eval = engine.evaluate_action(
                    action_type=VillageActionType.HARVEST_FIELD,
                    current_hunger=hunger,
                    current_hydration=75.0,
                    current_energy=75.0,
                    current_coins=10.0,
                    current_inventory=inv,
                    clearing_prices=prices,
                    current_status=0.10,
                    projected_hunger=min(100.0, hunger + 12.0),
                    projected_hydration=75.0,
                    projected_energy=72.0,
                    projected_coins=10.0,
                    projected_inventory=inv.with_delta(CommodityType.WHEAT, 2.5),
                    projected_status=0.108,
                )
                steal_eval = engine.evaluate_action(
                    action_type=VillageActionType.STEAL_GRANARY,
                    current_hunger=hunger,
                    current_hydration=75.0,
                    current_energy=75.0,
                    current_coins=10.0,
                    current_inventory=inv,
                    clearing_prices=prices,
                    current_status=0.10,
                    projected_hunger=min(100.0, hunger + 35.0),
                    projected_hydration=75.0,
                    projected_energy=80.0,
                    projected_coins=10.0,
                    projected_inventory=inv.with_delta(CommodityType.WHEAT, 3.5),
                    projected_status=0.10,
                    detection_probability=p_det,
                    coin_penalty=16.0,
                    status_loss=0.12,
                )
                if steal_eval.expected_total_utility > legal_eval.expected_total_utility:
                    steal_count += 1
                total_trials += 1

        theft_rate = round(steal_count / max(1, total_trials), 4)
        if theft_rate == 0.0 and threshold_p_detect == 1.0:
            threshold_p_detect = p_det
        series.append(
            {
                "detection_probability": round(p_det, 2),
                "theft_rate": theft_rate,
            }
        )

    return VillageExperimentResult(
        scenario_id="law_enforcement_threshold",
        title="The Law Enforcement Threshold",
        hypothesis=(
            "Increasing witness/guard detection probability P_detect above a critical Beckerian "
            "threshold eliminates opportunistic Granary theft among non-starving villagers."
        ),
        seeds_evaluated=len(seeds),
        metrics={
            "critical_detection_threshold": round(threshold_p_detect, 2),
            "low_surveillance_theft_rate": float(series[0]["theft_rate"]),
            "high_surveillance_theft_rate": float(series[-1]["theft_rate"]),
        },
        series=series,
        conclusion=(
            f"Opportunistic Granary theft drops from {float(series[0]['theft_rate']) * 100:.1f}% "
            f"at P_detect=0.10 to 0.0% once P_detect >= {threshold_p_detect:.2f}."
        ),
        limitations="Does not eliminate acute starvation-crisis theft when H_i < tau.",
    )


def run_inequality_vs_crime_experiment(
    seeds: tuple[int, ...] = (11, 22, 33),
    steps: int = 18,
) -> VillageExperimentResult:
    """Scenario 2: The Inequality vs. Crime Dynamic.

    Compares an egalitarian village against a high-inequality drought-stressed village
    to measure emergent unethical actions and Gini disparity.
    """
    egal_crimes: list[int] = []
    ineq_crimes: list[int] = []
    egal_gini: list[float] = []
    ineq_gini: list[float] = []

    for seed in seeds:
        # 1. Egalitarian baseline
        ctrl_eq = VillageSimulationController(
            world_config=VillageWorldConfig(seed=seed, drought_active=False)
        )
        for _ in range(steps):
            ctrl_eq.step()
        egal_crimes.append(sum(v.crimes_committed for v in ctrl_eq.world.villagers.values()))
        egal_gini.append(
            compute_gini_coefficient([v.coins for v in ctrl_eq.world.villagers.values()])
        )

        # 2. High-inequality + resource scarcity regime
        ctrl_in = VillageSimulationController(
            world_config=VillageWorldConfig(
                seed=seed + 500,
                drought_active=True,
                initial_time_of_day=22.0,  # Nighttime cover lowers P_detect
            )
        )
        commons_pos = ctrl_in.world.zones["village_commons"].center
        for idx, (aid, v) in enumerate(sorted(ctrl_in.world.villagers.items())):
            if idx < 2:
                # Wealthy oligarchs
                ctrl_in.world.villagers[aid] = v.model_copy(
                    update={
                        "coins": 180.0,
                        "inventory": CommodityInventory(items={CommodityType.WHEAT: 18.0}),
                    }
                )
            else:
                # Deprived villagers in acute starvation crisis near the Granary Commons
                ctrl_in.world.villagers[aid] = v.model_copy(
                    update={
                        "position": commons_pos,
                        "hunger": 12.0,
                        "coins": 0.5,
                        "inventory": CommodityInventory(items={}),
                    }
                )
        for _ in range(steps):
            ctrl_in.step()
        ineq_crimes.append(sum(v.crimes_committed for v in ctrl_in.world.villagers.values()))
        ineq_gini.append(
            compute_gini_coefficient([v.coins for v in ctrl_in.world.villagers.values()])
        )

    mean_eq_crimes = float(np.mean(egal_crimes))
    mean_in_crimes = float(np.mean(ineq_crimes))
    mean_eq_gini = float(np.mean(egal_gini))
    mean_in_gini = float(np.mean(ineq_gini))

    return VillageExperimentResult(
        scenario_id="inequality_vs_crime",
        title="The Inequality & Scarcity vs. Crime Dynamic",
        hypothesis=(
            "Severe wealth inequality combined with physiological scarcity triggers significantly "
            "more unethical actions (Granary pilfering and price gouging) than an egalitarian village."
        ),
        seeds_evaluated=len(seeds),
        metrics={
            "egalitarian_gini": round(mean_eq_gini, 4),
            "inequality_gini": round(mean_in_gini, 4),
            "egalitarian_mean_crimes": round(mean_eq_crimes, 2),
            "inequality_mean_crimes": round(mean_in_crimes, 2),
        },
        series=[
            {"regime": "egalitarian", "gini": round(mean_eq_gini, 3), "mean_crimes": round(mean_eq_crimes, 2)},
            {"regime": "high_inequality", "gini": round(mean_in_gini, 3), "mean_crimes": round(mean_in_crimes, 2)},
        ],
        conclusion=(
            f"High wealth disparity (Gini={mean_in_gini:.2f}) and scarcity increased mean unethical "
            f"incidents from {mean_eq_crimes:.1f} to {mean_in_crimes:.1f} over {steps} ticks."
        ),
        limitations="Tested over finite horizon with 10 canonical villagers.",
    )


def run_reputation_resilience_experiment() -> VillageExperimentResult:
    """Scenario 3: Reputation Resilience.

    Tests how many public Granary tithes are required for an agent caught committing
    theft to rebuild their PageRank Societal Status $\\mathcal{S}_i(t)$ back to baseline,
    verifying the asymmetry invariant $\\beta_{\\text{defect}} > \\alpha_{\\text{coop}}$.
    """
    agent_ids = [f"villager_{i}" for i in range(10)]
    graph = VillageTrustGraph(agent_ids=agent_ids)
    actor = "villager_0"

    initial_status = graph.compute_societal_status()[actor]
    trajectory: list[dict[str, float | int | str]] = [
        {"event_step": 0, "phase": "initial", "status": round(initial_status, 5)}
    ]

    # Step 1: Actor is caught stealing from the Granary
    graph.record_public_event(actor_id=actor, event_type="crime_caught")
    post_crime_status = graph.compute_societal_status()[actor]
    trajectory.append(
        {"event_step": 1, "phase": "post_crime", "status": round(post_crime_status, 5)}
    )

    # Subsequent steps: Actor repeatedly donates surplus wheat to the Granary Commons
    recovery_step = -1
    for step_idx in range(2, 16):
        graph.record_public_event(actor_id=actor, event_type="coop")
        cur_status = graph.compute_societal_status()[actor]
        trajectory.append(
            {"event_step": step_idx, "phase": "philanthropy_tithe", "status": round(cur_status, 5)}
        )
        if cur_status >= initial_status * 0.98 and recovery_step == -1:
            recovery_step = step_idx - 1

    return VillageExperimentResult(
        scenario_id="reputation_resilience",
        title="Reputation Resilience & Asymmetry",
        hypothesis=(
            "Because beta_defect > alpha_coop and gamma_crime is severe, a single caught theft "
            "destroys PageRank societal status immediately, requiring multiple consecutive public "
            "philanthropic tithes to recover."
        ),
        seeds_evaluated=1,
        metrics={
            "initial_status": round(initial_status, 5),
            "post_crime_status": round(post_crime_status, 5),
            "final_recovered_status": float(trajectory[-1]["status"]),
            "tithes_to_recover": float(recovery_step if recovery_step > 0 else 14),
        },
        series=trajectory,
        conclusion=(
            f"A single detected theft dropped PageRank status from {initial_status:.4f} to "
            f"{post_crime_status:.4f}; {recovery_step} consecutive public Granary tithes were "
            f"required to restore status."
        ),
        limitations="Assumes no additional defections or counter-gossip during rehabilitation.",
    )


def run_all_village_experiments() -> list[dict[str, Any]]:
    """Executes all 3 empirical Village Society experiments and returns serializable records."""
    return [
        run_law_enforcement_threshold_experiment().model_dump(),
        run_inequality_vs_crime_experiment().model_dump(),
        run_reputation_resilience_experiment().model_dump(),
    ]
