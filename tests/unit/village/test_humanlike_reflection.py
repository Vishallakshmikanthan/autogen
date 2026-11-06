"""Unit and mathematical oracle tests for Nightly Reflection and Disposition Consolidation (§4.4, §5)."""

from __future__ import annotations

from autogen.intelligence.village.personality import Traits
from autogen.intelligence.village.reflection import (
    Disposition,
    MemoryEpisode,
    ReflectionEngine,
)


def test_robbed_three_times_grudge_oracle() -> None:
    """Verifies §5 Oracle: A villager robbed 3x by the same culprit ends with grudge > 0.60 and 3 evidence IDs."""
    victim_traits = Traits(neuroticism=0.6, agreeableness=0.4)
    episodes = [
        MemoryEpisode(
            episode_id="theft_1",
            step=10,
            other_id="culprit_mani",
            valence=-0.70,
            importance=0.80,
            is_harm=True,
            description="Mani stole 3.5 grain from granary.",
        ),
        MemoryEpisode(
            episode_id="theft_2",
            step=25,
            other_id="culprit_mani",
            valence=-0.70,
            importance=0.80,
            is_harm=True,
            description="Mani raided private hut chest.",
        ),
        MemoryEpisode(
            episode_id="theft_3",
            step=40,
            other_id="culprit_mani",
            valence=-0.70,
            importance=0.80,
            is_harm=True,
            description="Mani stole wheat sheaves again.",
        ),
    ]

    dispositions: dict[str, Disposition] = {}
    updated = ReflectionEngine.reflect_episodes(
        unreflected=episodes,
        current_dispositions=dispositions,
        traits=victim_traits,
    )

    mani_disp = updated["culprit_mani"]
    assert mani_disp.grudge > 0.60
    assert len(mani_disp.evidence_ids) == 3
    assert set(mani_disp.evidence_ids) == {"theft_1", "theft_2", "theft_3"}
    assert mani_disp.confidence > 0.65


def test_reflection_memory_boundedness_invariant() -> None:
    """Invariant: An agent experiencing 100 interaction episodes never stores more than 16 evidence IDs per target."""
    traits = Traits()
    episodes = [
        MemoryEpisode(
            episode_id=f"ev_{k}",
            step=k,
            other_id="neighbor_bob",
            valence=0.3,
            importance=0.5,
            is_help=True,
        )
        for k in range(100)
    ]

    dispositions = ReflectionEngine.reflect_episodes(
        unreflected=episodes,
        current_dispositions={},
        traits=traits,
    )

    bob_disp = dispositions["neighbor_bob"]
    assert len(bob_disp.evidence_ids) <= 16
    assert bob_disp.evidence_count == 8  # top-8 per batch


def test_grudge_decay_and_neuroticism_persistence() -> None:
    """Invariant: Grudge decays each tick, and higher Neuroticism retains grudges longer (§4.4)."""
    calm_traits = Traits(neuroticism=0.1, agreeableness=0.8)
    anxious_traits = Traits(neuroticism=0.9, agreeableness=0.2)

    decay_calm = ReflectionEngine.compute_grudge_decay(calm_traits)
    decay_anxious = ReflectionEngine.compute_grudge_decay(anxious_traits)

    # Anxious villager retains grudge at a strictly higher factor per tick
    assert decay_anxious > decay_calm

    disp_calm = {"rival": Disposition(target_id="rival", grudge=0.80)}
    disp_anxious = {"rival": Disposition(target_id="rival", grudge=0.80)}

    for _ in range(50):
        disp_calm = ReflectionEngine.decay_grudges(disp_calm, calm_traits)
        disp_anxious = ReflectionEngine.decay_grudges(disp_anxious, anxious_traits)

    assert disp_anxious["rival"].grudge > disp_calm["rival"].grudge


def test_subsequent_help_accelerates_reconciliation() -> None:
    """Verifies that subsequent positive actions by an offender reduce the grudge (§4.4)."""
    traits = Traits(agreeableness=0.7)
    harm_ep = [
        MemoryEpisode(
            episode_id="harm_1",
            step=1,
            other_id="culprit",
            valence=-0.8,
            importance=0.8,
            is_harm=True,
        )
    ]

    disp = ReflectionEngine.reflect_episodes(unreflected=harm_ep, current_dispositions={}, traits=traits)
    initial_grudge = disp["culprit"].grudge
    assert initial_grudge > 0.40

    help_ep = [
        MemoryEpisode(
            episode_id="help_1",
            step=20,
            other_id="culprit",
            valence=0.8,
            importance=0.9,
            is_help=True,
            description="Culprit assisted in harvest and provided grain restitution.",
        )
    ]

    reconciled_disp = ReflectionEngine.reflect_episodes(
        unreflected=help_ep,
        current_dispositions=disp,
        traits=traits,
    )

    assert reconciled_disp["culprit"].grudge < initial_grudge
    assert reconciled_disp["culprit"].fondness > 0.0
