"""Unit tests for physical communication channel, bandwidth, range, and noise (SYS-14, Module M15)."""

from __future__ import annotations

import pytest

from autogen.intelligence.multiagent.communication import CommunicationBus
from autogen.intelligence.multiagent.contracts import Message
from autogen.simulation.state.types import Position


@pytest.mark.unit
def test_communication_energy_cost_and_bandwidth():
    """Verifies that sending messages costs energy and rejects oversized payloads."""
    bus = CommunicationBus(energy_cost=2.0, bandwidth_limit=3, max_range=10.0, noise_prob=0.0)

    # 1. Insufficient energy
    msg = Message(
        sender_id="a1",
        recipient_id="a2",
        payload={"k1": 1},
        sent_at_step=1,
        sender_position=Position(x=0, y=0),
    )
    ok, cost, reason = bus.send(msg, sender_energy=1.0)
    assert not ok
    assert cost == 0.0

    # 2. Exceeds bandwidth
    oversized_msg = Message(
        sender_id="a1",
        recipient_id="a2",
        payload={"k1": 1, "k2": 2, "k3": 3, "k4": 4},
        sent_at_step=1,
        sender_position=Position(x=0, y=0),
    )
    ok, cost, reason = bus.send(oversized_msg, sender_energy=10.0)
    assert not ok
    assert "bandwidth limit" in reason

    # 3. Valid transmission
    ok, cost, reason = bus.send(msg, sender_energy=10.0)
    assert ok
    assert cost == 2.0


@pytest.mark.unit
def test_communication_spatial_range_filtering():
    """Verifies Euclidean distance range filtering on reception."""
    bus = CommunicationBus(energy_cost=1.0, max_range=5.0, noise_prob=0.0)

    msg = Message(
        sender_id="sender",
        recipient_id=None,  # Broadcast
        payload={"alert": "hazard"},
        sent_at_step=2,
        sender_position=Position(x=0, y=0),
    )
    bus.send(msg, sender_energy=5.0)

    # Recipient near (dist = 3.0 <= 5.0) receives message
    near_inbox = bus.receive("near_agent", Position(x=3, y=0))
    assert len(near_inbox) == 1
    assert near_inbox[0].payload["alert"] == "hazard"

    # Recipient far (dist = 8.0 > 5.0) does not receive message
    far_inbox = bus.receive("far_agent", Position(x=8, y=0))
    assert len(far_inbox) == 0

