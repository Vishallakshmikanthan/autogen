"""Synchronous, typed event bus supporting the Observer pattern."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any, TypeVar

from autogen.simulation.core.contracts import SimulationEvent

E = TypeVar("E", bound=SimulationEvent)
EventHandler = Callable[[Any], None]


class EventBus:
    """Synchronous in-memory event dispatcher for simulation telemetry."""

    def __init__(self) -> None:
        self._subscribers: dict[type[SimulationEvent], list[Callable[[Any], None]]] = defaultdict(
            list
        )
        self._all_subscribers: list[Callable[[SimulationEvent], None]] = []

    def subscribe(self, event_class: type[E], handler: Callable[[E], None]) -> None:
        """Registers a handler for a specific event type."""
        self._subscribers[event_class].append(handler)

    def subscribe_all(self, handler: Callable[[SimulationEvent], None]) -> None:
        """Registers a handler for all simulation events."""
        self._all_subscribers.append(handler)

    def publish(self, event: SimulationEvent) -> None:
        """Dispatches an event synchronously to all registered subscribers."""
        event_cls = type(event)
        for handler in self._subscribers[event_cls]:
            handler(event)
        for general_handler in self._all_subscribers:
            general_handler(event)

    def clear(self) -> None:
        """Clears all registered subscribers."""
        self._subscribers.clear()
        self._all_subscribers.clear()
