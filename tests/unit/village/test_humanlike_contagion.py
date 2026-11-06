"""Unit and mathematical oracle tests for Synchronous Multi-Agent Emotional Contagion (§3.4, §5)."""

from __future__ import annotations

import random

from autogen.intelligence.village.affective_pad import AffectiveStateVector
from autogen.intelligence.village.contagion import (
    CONTAGION_MAX_AXIS_DELTA,
    EmotionalContagionEngine,
)
from autogen.intelligence.village.personality import PersonalityParams, Traits, weights_from_traits
from autogen.intelligence.village.relationships import VillageRelationshipStore
from autogen.simulation.state.types import Position


def test_contagion_permutation_order_independence_oracle() -> None:
    """Invariant: Shuffled dictionary order produces bitwise-identical PAD state updates (§5)."""
    rel_store = VillageRelationshipStore()
    positions = {
        "v_0": Position(x=10, y=10),
        "v_1": Position(x=11, y=10),
        "v_2": Position(x=10, y=12),
        "v_3": Position(x=13, y=13),
    }

    personalities: dict[str, PersonalityParams] = {}
    for aid in positions:
        _, p = weights_from_traits(Traits(extraversion=0.8, agreeableness=0.7))
        personalities[aid] = p

    states = {
        "v_0": AffectiveStateVector(valence=-0.70, arousal=0.80, dominance=-0.30),
        "v_1": AffectiveStateVector(valence=0.40, arousal=0.10, dominance=0.20),
        "v_2": AffectiveStateVector(valence=0.10, arousal=0.00, dominance=0.10),
        "v_3": AffectiveStateVector(valence=-0.20, arousal=0.50, dominance=-0.10),
    }

    # Reference run in canonical order
    res_canonical = EmotionalContagionEngine.step_contagion(
        agents=states,
        positions=positions,
        personalities=personalities,
        relationships=rel_store,
    )

    # Shuffled runs
    rng = random.Random(42)
    for _ in range(5):
        keys = list(states.keys())
        rng.shuffle(keys)
        shuffled_states = {k: states[k] for k in keys}
        res_shuffled = EmotionalContagionEngine.step_contagion(
            agents=shuffled_states,
            positions=positions,
            personalities=personalities,
            relationships=rel_store,
        )
        for aid in states:
            assert res_canonical[aid].to_tuple() == res_shuffled[aid].to_tuple()


def test_contagion_per_axis_clamp_invariant() -> None:
    """Invariant: Contagion delta per tick never exceeds CONTAGION_MAX_AXIS_DELTA (0.15) per axis (§3.4)."""
    rel_store = VillageRelationshipStore()
    # 5 panicked agents surround 1 calm agent at distance 1
    pos_center = Position(x=10, y=10)
    positions = {"v_target": pos_center}
    states = {"v_target": AffectiveStateVector(valence=0.5, arousal=0.0, dominance=0.5)}
    personalities = {}
    _, p = weights_from_traits(Traits(extraversion=1.0, agreeableness=1.0))
    personalities["v_target"] = p

    for i in range(8):
        aid = f"panicked_{i}"
        positions[aid] = Position(x=10, y=11)
        states[aid] = AffectiveStateVector(valence=-1.0, arousal=1.0, dominance=-1.0)
        personalities[aid] = p

    res = EmotionalContagionEngine.step_contagion(
        agents=states,
        positions=positions,
        personalities=personalities,
        relationships=rel_store,
    )

    target_next = res["v_target"]
    diff_v = abs(target_next.valence - states["v_target"].valence)
    diff_a = abs(target_next.arousal - states["v_target"].arousal)
    diff_d = abs(target_next.dominance - states["v_target"].dominance)

    assert diff_v <= CONTAGION_MAX_AXIS_DELTA + 1e-4
    assert diff_a <= CONTAGION_MAX_AXIS_DELTA + 1e-4
    assert diff_d <= CONTAGION_MAX_AXIS_DELTA + 1e-4


def test_contagion_gamma_zero_matches_independent_behavior() -> None:
    """Invariant: When contagion_gamma is 0, agents experience zero contagion shift (§5)."""
    rel_store = VillageRelationshipStore()
    positions = {"v_0": Position(x=10, y=10), "v_1": Position(x=10, y=11)}
    states = {
        "v_0": AffectiveStateVector(valence=0.5, arousal=0.2, dominance=0.3),
        "v_1": AffectiveStateVector(valence=-0.8, arousal=0.9, dominance=-0.5),
    }
    _, base_p = weights_from_traits(Traits())
    zero_p = base_p.model_copy(update={"contagion_gamma": 0.0})
    personalities = {"v_0": zero_p, "v_1": zero_p}

    res = EmotionalContagionEngine.step_contagion(
        agents=states,
        positions=positions,
        personalities=personalities,
        relationships=rel_store,
    )

    assert res["v_0"].to_tuple() == states["v_0"].to_tuple()
    assert res["v_1"].to_tuple() == states["v_1"].to_tuple()
