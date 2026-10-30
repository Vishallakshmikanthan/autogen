"""Unit tests for EpisodicMemory buffer and transition queries."""

import numpy as np

from autogen.memory.contracts import EpisodicTransition
from autogen.memory.episodic_memory import EpisodicMemory
from autogen.simulation.core.contracts import Action, Observation
from autogen.simulation.state.types import ActionType, Position


def _make_transition(step: int, reward: float = 1.0, tag: str = "normal") -> EpisodicTransition:
    obs = Observation(
        agent_id="test",
        step=step,
        position=Position(step, step),
        energy=100.0,
        health=100.0,
        carried_food=0.0,
        is_alive=True,
        visible_cells=(),
        visible_agents=(),
    )
    act = Action(agent_id="test", action_type=ActionType.UP)
    return EpisodicTransition(
        step=step,
        state_observation=obs,
        action=act,
        reward=reward,
        metadata={"tag": tag},
    )


def test_episodic_memory_capacity_and_fifo():
    em = EpisodicMemory(capacity=3)
    em.push(_make_transition(1))
    em.push(_make_transition(2))
    em.push(_make_transition(3))
    assert len(em) == 3

    # 4th item evicts step 1
    em.push(_make_transition(4))
    assert len(em) == 3
    steps = [t.step for t in em.get_all()]
    assert steps == [2, 3, 4]


def test_episodic_memory_filtering_and_sampling():
    em = EpisodicMemory(capacity=10)
    for i in range(10):
        em.push(_make_transition(i, tag="special" if i % 2 == 0 else "regular"))

    specials = em.filter_by_metadata("tag", "special")
    assert len(specials) == 5
    for s in specials:
        assert s.metadata["tag"] == "special"

    rng = np.random.default_rng(42)
    batch = em.sample_batch(batch_size=4, rng=rng)
    assert len(batch) == 4
    # All sampled items must be from stored transitions
    stored_steps = {t.step for t in em.get_all()}
    for b in batch:
        assert b.step in stored_steps


def test_episodic_memory_empty_and_clear():
    em = EpisodicMemory(capacity=5)
    assert em.get_latest() is None
    assert em.sample_batch(3) == []

    em.push(_make_transition(1))
    assert len(em) == 1
    em.clear()
    assert len(em) == 0
