"""Spatial speech broadcasting bus with distance attenuation and eavesdropping."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

from autogen.intelligence.village.persona.symbolic_dialogue import (
    DialogueEventType,
    VillageDialogueEvent,
)
from autogen.intelligence.village.social_graph import VillageTrustGraph
from autogen.simulation.state.types import Position


class HeardUtterance(BaseModel):
    """Record of an utterance heard by a nearby listener with distance attenuation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    listener_id: str
    event: VillageDialogueEvent
    distance: float = Field(ge=0.0)
    audibility: float = Field(ge=0.0, le=1.0)
    is_eavesdropped: bool = False


class VillageDialogueBus:
    """Spatial communication bus managing speech attenuation and gossip propagation.

    Private diary entries (`DIARY_ENTRY`) are never broadcast spatially.
    Public declarations (`PUBLIC_DECLARATION`) have double hearing radius.
    """

    def __init__(self, base_hearing_radius: float = 10.0, max_history: int = 200) -> None:
        self.base_hearing_radius = base_hearing_radius
        self.max_history = max_history
        self.history: list[VillageDialogueEvent] = []
        self.inbox_by_agent: dict[str, list[HeardUtterance]] = {}

    def compute_audibility(
        self,
        speaker_pos: Position,
        listener_pos: Position,
        event_type: DialogueEventType = DialogueEventType.SPEECH,
    ) -> float:
        """Computes spatial sound attenuation $A(d) = \\max(0, 1 - d / R_{\\text{eff}})$."""
        if event_type == DialogueEventType.DIARY_ENTRY:
            return 0.0
        dist = math.hypot(speaker_pos.x - listener_pos.x, speaker_pos.y - listener_pos.y)
        eff_radius = (
            self.base_hearing_radius * 1.8
            if event_type == DialogueEventType.PUBLIC_DECLARATION
            else self.base_hearing_radius
        )
        if dist > eff_radius:
            return 0.0
        return float(round(max(0.0, 1.0 - dist / (eff_radius + 0.001)), 4))

    def broadcast(
        self,
        event: VillageDialogueEvent,
        agent_positions: dict[str, Position],
        trust_graph: VillageTrustGraph | None = None,
    ) -> list[HeardUtterance]:
        """Broadcasts an utterance across the village and triggers gossip updates."""
        self.history.append(event)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

        if event.event_type == DialogueEventType.DIARY_ENTRY:
            return []

        speaker_pos = Position(event.position[0], event.position[1])
        deliveries: list[HeardUtterance] = []

        for listener_id, l_pos in sorted(agent_positions.items()):
            if listener_id == event.speaker_id:
                continue
            aud = self.compute_audibility(speaker_pos, l_pos, event.event_type)
            if aud <= 0.0:
                continue

            dist = math.hypot(speaker_pos.x - l_pos.x, speaker_pos.y - l_pos.y)
            eavesdropped = (
                event.recipient_id is not None and event.recipient_id != listener_id
            )
            heard = HeardUtterance(
                listener_id=listener_id,
                event=event,
                distance=round(dist, 3),
                audibility=aud,
                is_eavesdropped=eavesdropped,
            )
            deliveries.append(heard)
            self.inbox_by_agent.setdefault(listener_id, []).append(heard)
            if len(self.inbox_by_agent[listener_id]) > 25:
                self.inbox_by_agent[listener_id] = self.inbox_by_agent[listener_id][-25:]

            # If this is a GossipEvent about a subject, propagate reputation in trust graph
            if (
                trust_graph is not None
                and event.event_type == DialogueEventType.GOSSIP
                and event.subject_id is not None
            ):
                trust_graph.propagate_gossip(
                    speaker_id=event.speaker_id,
                    listener_id=listener_id,
                    subject_id=event.subject_id,
                )

        return deliveries

    def get_recent_events(self, limit: int = 25) -> list[VillageDialogueEvent]:
        """Returns the most recent dialogue events."""
        return self.history[-limit:]
