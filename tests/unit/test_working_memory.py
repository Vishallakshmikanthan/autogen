"""Unit tests for bounded WorkingMemory buffer."""

import pytest

from autogen.memory.contracts import EvictionPolicy
from autogen.memory.working_memory import WorkingMemory
from autogen.simulation.core.contracts import Observation
from autogen.simulation.state.types import Position


def _make_obs(step: int) -> Observation:
    return Observation(
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


def test_working_memory_fifo_capacity():
    wm = WorkingMemory(capacity=3, policy=EvictionPolicy.FIFO)
    assert len(wm) == 0

    wm.push(_make_obs(1))
    wm.push(_make_obs(2))
    wm.push(_make_obs(3))
    assert len(wm) == 3

    # Pushing 4th item evicts step 1
    wm.push(_make_obs(4))
    assert len(wm) == 3

    steps = [o.step for o in wm.get_all()]
    assert steps == [2, 3, 4]
    latest = wm.get_latest()
    assert latest is not None
    assert latest.step == 4


def test_working_memory_get_recent():
    wm = WorkingMemory(capacity=5)
    for s in range(5):
        wm.push(_make_obs(s))

    recent = wm.get_recent(k=2)
    assert len(recent) == 2
    assert recent[0].step == 3
    assert recent[1].step == 4

    # k larger than capacity returns all
    all_items = wm.get_recent(k=10)
    assert len(all_items) == 5

    # k <= 0 returns empty
    assert wm.get_recent(k=0) == []


def test_working_memory_validation_and_clear():
    with pytest.raises(ValueError, match="capacity must be >= 1"):
        WorkingMemory(capacity=0)

    wm = WorkingMemory(capacity=3)
    wm.push(_make_obs(1))
    assert len(wm) == 1
    wm.clear()
    assert len(wm) == 0
    assert wm.get_latest() is None
