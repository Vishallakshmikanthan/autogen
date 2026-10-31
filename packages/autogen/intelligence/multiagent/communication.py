"""Physical communication channel with energy costs, range attenuation, and transmission noise (SYS-14, Module M15)."""

from __future__ import annotations

import numpy as np

from autogen.intelligence.multiagent.contracts import Message
from autogen.simulation.state.types import Position


class CommunicationBus:
    """Simulated physical communication medium enforcing bandwidth, range, noise, and energy constraints."""

    def __init__(
        self,
        energy_cost: float = 1.0,
        max_range: float = 10.0,
        bandwidth_limit: int = 5,
        noise_prob: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.energy_cost = float(energy_cost)
        self.max_range = float(max_range)
        self.bandwidth_limit = int(bandwidth_limit)
        self.noise_prob = float(noise_prob)
        self.rng = np.random.default_rng(seed)

        self._inbox: list[Message] = []
        self.total_messages_sent = 0
        self.total_messages_delivered = 0
        self.total_energy_consumed = 0.0

    def send(
        self,
        message: Message,
        sender_energy: float,
    ) -> tuple[bool, float, str]:
        """Validates bandwidth and energy, deducts cost, and puts message on the wire."""
        if sender_energy < self.energy_cost:
            return False, 0.0, "Insufficient energy to transmit message"

        if len(message.payload) > self.bandwidth_limit:
            return False, 0.0, f"Payload size {len(message.payload)} exceeds bandwidth limit {self.bandwidth_limit}"

        # Apply transmission noise if triggered
        final_payload = dict(message.payload)
        if self.rng.random() < self.noise_prob:
            final_payload["_noise_corrupted"] = True

        clean_or_noisy_msg = Message(
            sender_id=message.sender_id,
            recipient_id=message.recipient_id,
            payload=final_payload,
            sent_at_step=message.sent_at_step,
            sender_position=message.sender_position,
        )

        self._inbox.append(clean_or_noisy_msg)
        self.total_messages_sent += 1
        self.total_energy_consumed += self.energy_cost

        return True, self.energy_cost, "Transmitted"

    def receive(self, agent_id: str, agent_position: Position) -> list[Message]:
        """Delivers in-range messages addressed to agent_id or broadcast."""
        delivered: list[Message] = []

        for msg in self._inbox:
            # Check recipient
            if msg.recipient_id is not None and msg.recipient_id != agent_id:
                continue

            # Check Euclidean range
            dx = float(msg.sender_position.x - agent_position.x)
            dy = float(msg.sender_position.y - agent_position.y)
            dist = float(np.hypot(dx, dy))

            if dist <= self.max_range:
                delivered.append(msg)
                self.total_messages_delivered += 1

        return delivered

    def step(self) -> None:
        """Clears packets transmitted in previous tick."""
        self._inbox.clear()

    def reset(self) -> None:
        """Resets communication bus telemetry."""
        self._inbox.clear()
        self.total_messages_sent = 0
        self.total_messages_delivered = 0
        self.total_energy_consumed = 0.0
