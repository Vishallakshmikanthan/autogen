"""World Event Bus, Modalities, Percepts, and Weather Process for AUTOGEN Village.

Emits world events (cries, arrivals, weather/thunder, crimes), filters them into
percepts per agent sensing range and modalities, and maintains backend weather state.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from autogen.simulation.state.types import Position


class Modality(str, Enum):
    """Sensory modality of a world event (§2.2)."""

    VISUAL = "visual"
    AUDITORY = "auditory"
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"


class WorldEvent(BaseModel):
    """An objective event occurring in the village world during a simulation tick."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    step: int
    kind: str  # "CRY_OUT", "THUNDER", "CHASE", "ARRIVAL", "THEFT_SEEN", "GREETING", "RAIN_ONSET"
    modality: Modality
    source_id: str | None = None
    target_id: str | None = None
    position: Position
    intensity: float = Field(ge=0.0, le=1.0, description="Loudness, brightness, or severity in [0, 1]")
    valence_hint: float = Field(default=0.0, ge=-1.0, le=1.0, description="-1 (threat/harm) to +1 (pleasant)")
    is_global: bool = False
    payload: dict[str, Any] = Field(default_factory=dict)


class PerceivedEvent(BaseModel):
    """A subjective perceived event filtered through an agent's sensing modalities and range."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event: WorldEvent
    distance: float = Field(ge=0.0, description="Euclidean distance to the event source (0.0 for global)")
    audible: bool = True
    visible: bool = True


class WorldEventBus:
    """Central bus for collecting tick world events and delivering filtered percepts to agents."""

    def __init__(self, max_audible_radius: float = 18.0) -> None:
        self.max_audible_radius = max_audible_radius
        self._events_by_step: dict[int, list[WorldEvent]] = {}
        self._event_counter = 0

    def emit(self, event: WorldEvent) -> None:
        """Appends a new event to the tick's event log."""
        step_events = self._events_by_step.setdefault(event.step, [])
        step_events.append(event)

    def create_and_emit(
        self,
        *,
        step: int,
        kind: str,
        modality: Modality,
        position: Position,
        intensity: float = 0.5,
        valence_hint: float = 0.0,
        source_id: str | None = None,
        target_id: str | None = None,
        is_global: bool = False,
        payload: dict[str, Any] | None = None,
    ) -> WorldEvent:
        """Helper to create, register, and return a unique world event."""
        self._event_counter += 1
        ev_id = f"ev_{step}_{self._event_counter}_{kind.lower()}"
        event = WorldEvent(
            event_id=ev_id,
            step=step,
            kind=kind,
            modality=modality,
            source_id=source_id,
            target_id=target_id,
            position=position,
            intensity=min(1.0, max(0.0, intensity)),
            valence_hint=min(1.0, max(-1.0, valence_hint)),
            is_global=is_global,
            payload=payload or {},
        )
        self.emit(event)
        return event

    def get_events(self, step: int | None = None) -> list[WorldEvent]:
        """Returns all events for a given step, or all accumulated events if step is None."""
        if step is not None:
            return list(self._events_by_step.get(step, []))
        all_ev: list[WorldEvent] = []
        for s in sorted(self._events_by_step.keys()):
            all_ev.extend(self._events_by_step[s])
        return all_ev

    def clear_step(self, step: int) -> None:
        """Prunes historical events prior to the specified step horizon to bound memory."""
        self._events_by_step = {s: evs for s, evs in self._events_by_step.items() if s >= step}

    def perceive(
        self,
        agent_pos: Position,
        vision_radius: int = 8,
        is_indoor: bool = False,
        step: int | None = None,
    ) -> tuple[PerceivedEvent, ...]:
        """Filters world events into perceived events according to physical sensing range and modality."""
        events = self.get_events(step)
        percepts: list[PerceivedEvent] = []

        for ev in events:
            if ev.is_global:
                dist = 0.0
                is_audible = True
                is_visible = not is_indoor or ev.modality == Modality.ENVIRONMENTAL
                percepts.append(
                    PerceivedEvent(event=ev, distance=dist, audible=is_audible, visible=is_visible)
                )
                continue

            dx = float(agent_pos.x - ev.position.x)
            dy = float(agent_pos.y - ev.position.y)
            dist = math.hypot(dx, dy)

            is_audible = False
            is_visible = False

            if ev.modality in (Modality.AUDITORY, Modality.SOCIAL, Modality.ENVIRONMENTAL):
                if dist <= self.max_audible_radius:
                    is_audible = True

            if ev.modality in (Modality.VISUAL, Modality.SOCIAL, Modality.ENVIRONMENTAL):
                if dist <= float(vision_radius):
                    is_visible = True

            if is_audible or is_visible:
                percepts.append(
                    PerceivedEvent(
                        event=ev,
                        distance=round(dist, 2),
                        audible=is_audible,
                        visible=is_visible,
                    )
                )

        return tuple(percepts)


class WeatherProcess:
    """Backend-owned deterministic/stochastic weather process (§4, §2.2).

    Simulates village climate, generating RAIN_ONSET and THUNDER world events.
    """

    def __init__(
        self,
        event_bus: WorldEventBus,
        storm_probability: float = 0.02,
        thunder_in_storm_probability: float = 0.40,
    ) -> None:
        self.bus = event_bus
        self.storm_probability = storm_probability
        self.thunder_in_storm_probability = thunder_in_storm_probability
        self.is_raining: bool = False
        self.rain_duration_ticks: int = 0

    def trigger_thunderstorm(self, step: int, intensity: float = 0.90) -> list[WorldEvent]:
        """Manually triggers an immediate thunderstorm event."""
        self.is_raining = True
        self.rain_duration_ticks = 10
        thunder_ev = self.bus.create_and_emit(
            step=step,
            kind="THUNDER",
            modality=Modality.ENVIRONMENTAL,
            position=Position(x=0, y=0),
            intensity=min(1.0, max(0.5, intensity)),
            valence_hint=-0.80,
            is_global=True,
            payload={"storm_type": "thunderstorm"},
        )
        return [thunder_ev]

    def step(self, step: int, rng: np.random.Generator) -> list[WorldEvent]:
        """Advances weather process by one tick and returns newly emitted weather events."""
        emitted: list[WorldEvent] = []

        if self.is_raining:
            self.rain_duration_ticks -= 1
            if self.rain_duration_ticks <= 0:
                self.is_raining = False
            elif rng.random() < self.thunder_in_storm_probability:
                thunder_ev = self.bus.create_and_emit(
                    step=step,
                    kind="THUNDER",
                    modality=Modality.ENVIRONMENTAL,
                    position=Position(x=0, y=0),
                    intensity=float(rng.uniform(0.85, 1.0)),
                    valence_hint=-0.80,
                    is_global=True,
                    payload={"storm_type": "thunder"},
                )
                emitted.append(thunder_ev)
        else:
            if rng.random() < self.storm_probability:
                self.is_raining = True
                self.rain_duration_ticks = int(rng.integers(8, 20))
                rain_ev = self.bus.create_and_emit(
                    step=step,
                    kind="RAIN_ONSET",
                    modality=Modality.ENVIRONMENTAL,
                    position=Position(x=0, y=0),
                    intensity=0.60,
                    valence_hint=-0.30,
                    is_global=True,
                    payload={"duration": self.rain_duration_ticks},
                )
                emitted.append(rain_ev)

        return emitted
