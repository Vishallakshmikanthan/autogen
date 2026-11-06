"""Unit and mathematical oracle tests for Dynamic Needs & Prepotency Hierarchy (§5.3)."""

from __future__ import annotations

from autogen.intelligence.village.needs import (
    NeedsEngine,
    compute_deficits,
    compute_dynamic_weights,
    compute_prepotency_gates,
    compute_pressures,
    update_satiation,
)
from autogen.intelligence.village.personality import Traits
from autogen.intelligence.village.utility_ethics import VillagerUtilityWeights
from autogen.simulation.environment.village.commodities import CommodityInventory, CommodityType
from autogen.simulation.environment.village.events import Modality, PerceivedEvent, WorldEvent
from autogen.simulation.environment.village.village_grid import (
    VillageActionType,
    VillageObservation,
    ZoneType,
)
from autogen.simulation.state.types import Position


def make_dummy_obs(
    hunger: float = 100.0,
    hydration: float = 100.0,
    energy: float = 100.0,
    health: float = 100.0,
    status: float = 0.50,
    food_stock: float = 6.0,
    is_raining: bool = False,
    current_zone: ZoneType = ZoneType.PRIVATE_HUTS,
) -> VillageObservation:
    """Helper to synthesize a VillageObservation with controlled physiological parameters."""
    percepts: list[PerceivedEvent] = []
    if is_raining:
        ev = WorldEvent(
            event_id="ev_rain_dummy",
            step=1,
            kind="RAIN_ONSET",
            modality=Modality.ENVIRONMENTAL,
            position=Position(0, 0),
            intensity=0.8,
            is_global=True,
        )
        percepts.append(PerceivedEvent(event=ev, distance=0.0, audible=True, visible=True))

    inv = CommodityInventory(items={CommodityType.WHEAT: food_stock})
    return VillageObservation(
        agent_id="test_villager",
        step=1,
        time_of_day=10.0,
        ambient_light=1.0,
        position=Position(10, 10),
        home_hut_position=Position(10, 10),
        hunger=hunger,
        hydration=hydration,
        energy=energy,
        health=health,
        coins=10.0,
        inventory=inv,
        home_chest_inventory=CommodityInventory(),
        societal_status=status,
        in_pillory_ticks=0,
        current_zone=current_zone,
        granary_wheat_available=0.0,
        percepts=tuple(percepts),
    )


def test_prepotency_hand_computed_oracle() -> None:
    """Verifies §5 Oracle: When d=(0.9, 0, 0, 0, 0), g3 = 0.10^0.70 ≈ 0.1995. When fed, gk = 1.0."""
    # 1. Starving condition: d1 = 0.90, others 0.0
    starving_deficits = (0.90, 0.0, 0.0, 0.0, 0.0)
    gates = compute_prepotency_gates(starving_deficits, beta=0.70)

    # g1 is always 1.0
    assert gates[0] == 1.0
    # g2 = (1 - 0.9)^0.7 = 0.1^0.7 ≈ 0.1995
    expected_g2 = round(0.10**0.70, 4)
    assert abs(gates[1] - expected_g2) < 1e-3
    assert abs(gates[1] - 0.1995) < 1e-3
    # g3 = (1 - 0.9)^0.7 * (1 - 0)^0.7 = 0.1^0.7 ≈ 0.1995
    assert abs(gates[2] - 0.1995) < 1e-3

    # 2. Fully satisfied condition: all deficits 0.0
    fed_deficits = (0.0, 0.0, 0.0, 0.0, 0.0)
    fed_gates = compute_prepotency_gates(fed_deficits, beta=0.70)
    for g in fed_gates:
        assert g == 1.0


def test_monotonicity_oracle() -> None:
    """Verifies §5 Oracle: w_phys strictly increases in d1; P1 is convex."""
    base_w = VillagerUtilityWeights(w_phys=2.0, w_soc=14.0)

    d_values = [0.0, 0.25, 0.50, 0.75, 1.0]
    p_values = [compute_pressures((d, 0, 0, 0, 0), p=2.0)[0] for d in d_values]
    w_phys_values = []

    for d, p in zip(d_values, p_values, strict=True):
        gates = compute_prepotency_gates((d, 0, 0, 0, 0))
        pressures = (p, 0.0, 0.0, 0.0, 0.0)
        dyn_w = compute_dynamic_weights(base_w, (d, 0, 0, 0, 0), gates, pressures, lambda_param=2.0)
        w_phys_values.append(dyn_w.w_phys)

    # Check strict monotonicity of w_phys
    for i in range(len(w_phys_values) - 1):
        assert w_phys_values[i + 1] > w_phys_values[i]

    # Check convexity of P: second difference is non-negative
    for i in range(len(p_values) - 2):
        delta1 = p_values[i + 1] - p_values[i]
        delta2 = p_values[i + 2] - p_values[i + 1]
        assert delta2 >= delta1


def test_loneliness_drive_growth_oracle() -> None:
    """Verifies §5 Oracle: An isolated agent's d3 is non-decreasing and reaches 1.0."""
    engine = NeedsEngine(seed=42)
    traits = Traits(extraversion=0.50)  # c* = 2.0 * (0.5 + 0.5) = 2.0
    obs = make_dummy_obs()

    deficits_d3: list[float] = []
    # 35 consecutive ticks of zero social contact (isolation)
    for _ in range(35):
        st = engine.step(obs=obs, traits=traits, contact_quality=0.0)
        deficits_d3.append(st.d_belonging)

    # Verify d3 rises monotonically towards 1.0
    assert deficits_d3[-1] == 1.0
    for i in range(len(deficits_d3) - 1):
        assert deficits_d3[i + 1] >= deficits_d3[i]

    # Now provide positive social interaction (GREET + dialogue: contact_quality = 1.5)
    st_contact = engine.step(obs=obs, traits=traits, contact_quality=1.5)
    assert st_contact.d_belonging < deficits_d3[-1]


def test_satiation_decay_oracle() -> None:
    """Verifies §5 Oracle: Marginal value of repeated action strictly decreases."""
    satiation: dict[str, float] = {}

    # Repeatedly perform GREET action 4 times
    s_history = []
    for _ in range(4):
        satiation = update_satiation(satiation, VillageActionType.GREET)
        s_history.append(satiation[VillageActionType.GREET.value])

    # Satiation strictly increases with repeated executions
    for i in range(len(s_history) - 1):
        assert s_history[i + 1] > s_history[i]
        # Marginal gain (1 - s) strictly decreases
        assert (1.0 - s_history[i + 1]) < (1.0 - s_history[i])

    # Inactivity ticks cause satiation to decay
    idle_satiation = update_satiation(satiation, VillageActionType.STAY)
    assert idle_satiation[VillageActionType.GREET.value] < s_history[-1]


def test_starving_vs_fed_social_weight_oracle() -> None:
    """Verifies §5 Oracle: A starving agent's decision weight on social terms is < 20% of a fed agent's."""
    base_w = VillagerUtilityWeights(w_phys=2.0, w_soc=14.0)

    # Fed condition: hunger = 100.0, hydration = 100.0
    fed_obs = make_dummy_obs(hunger=100.0, hydration=100.0)
    d_fed = compute_deficits(fed_obs, contact_score=1.0, mastery_score=1.0, traits=Traits())
    p_fed = compute_pressures(d_fed)
    g_fed = compute_prepotency_gates(d_fed)
    w_fed = compute_dynamic_weights(base_w, d_fed, g_fed, p_fed)

    # Starving condition: hunger = 5.0 (d1 ≈ 0.95)
    starving_obs = make_dummy_obs(hunger=5.0, hydration=5.0)
    d_starve = compute_deficits(starving_obs, contact_score=1.0, mastery_score=1.0, traits=Traits())
    p_starve = compute_pressures(d_starve)
    g_starve = compute_prepotency_gates(d_starve)
    w_starve = compute_dynamic_weights(base_w, d_starve, g_starve, p_starve)

    # Starving social weight is throttled by prepotency gate g3
    assert w_starve.w_soc < (0.20 * w_fed.w_soc)
    # Physiological weight surges under starvation pressure
    assert w_starve.w_phys > (2.0 * w_fed.w_phys)
