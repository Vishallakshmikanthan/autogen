"""Pluggable memory retrieval strategies (recency, spatial proximity, and cosine similarity)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol

from autogen.simulation.state.types import Position


class RetrievalStrategy(Protocol):
    """Protocol for memory query and retrieval engines."""

    def retrieve(
        self,
        items: Sequence[Any],
        query: dict[str, Any],
        k: int = 5,
    ) -> list[Any]:
        """Queries and ranks items according to strategy criteria."""
        ...


class RecencyRetrieval:
    """Retrieves items ranked by most recent observation timestamp."""

    def retrieve(
        self,
        items: Sequence[Any],
        query: dict[str, Any] | None = None,
        k: int = 5,
    ) -> list[Any]:
        if not items or k <= 0:
            return []

        def get_timestamp(item: Any) -> int:
            if hasattr(item, "step"):
                return int(item.step)
            if hasattr(item, "last_observed_tick"):
                return int(item.last_observed_tick)
            return 0

        sorted_items = sorted(items, key=get_timestamp, reverse=True)
        return sorted_items[:k]


class ProximityRetrieval:
    """Retrieves spatial items ranked by spatial proximity (Manhattan distance) to a query point."""

    def retrieve(
        self,
        items: Sequence[Any],
        query: dict[str, Any],
        k: int = 5,
    ) -> list[Any]:
        if not items or k <= 0:
            return []

        target_pos = query.get("position")
        if target_pos is None:
            raise ValueError("ProximityRetrieval query requires 'position' key.")

        tx = target_pos.x if isinstance(target_pos, Position) else target_pos[0]
        ty = target_pos.y if isinstance(target_pos, Position) else target_pos[1]

        def dist(item: Any) -> float:
            pos = getattr(item, "position", None)
            if pos is None:
                return float("inf")
            px = pos.x if isinstance(pos, Position) else pos[0]
            py = pos.y if isinstance(pos, Position) else pos[1]
            return float(abs(px - tx) + abs(py - ty))

        sorted_items = sorted(items, key=dist)
        return sorted_items[:k]


class CosineSimilarityRetrieval:
    """Retrieves items ranked by vector cosine similarity against a query vector."""

    def retrieve(
        self,
        items: Sequence[Any],
        query: dict[str, Any],
        k: int = 5,
    ) -> list[Any]:
        if not items or k <= 0:
            return []

        query_vec = query.get("vector")
        if query_vec is None:
            raise ValueError("CosineSimilarityRetrieval query requires 'vector' key.")

        q_norm = math.sqrt(sum(float(x) ** 2 for x in query_vec))
        if q_norm == 0.0:
            return list(items[:k])

        scored: list[tuple[Any, float]] = []
        for item in items:
            vec = getattr(item, "feature_vector", None)
            if vec is None or len(vec) != len(query_vec):
                scored.append((item, -1.0))
                continue

            dot = sum(float(a) * float(b) for a, b in zip(vec, query_vec, strict=False))
            i_norm = math.sqrt(sum(float(x) ** 2 for x in vec))
            if i_norm > 0.0:
                similarity = dot / (q_norm * i_norm)
            else:
                similarity = 0.0
            scored.append((item, similarity))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [item for item, _ in scored[:k]]


class CompositeRetrieval:
    """Combines recency, proximity, and confidence into a multi-criteria score."""

    def __init__(
        self,
        weight_proximity: float = 0.5,
        weight_confidence: float = 0.3,
        weight_recency: float = 0.2,
    ) -> None:
        self.w_prox = weight_proximity
        self.w_conf = weight_confidence
        self.w_rec = weight_recency

    def retrieve(
        self,
        items: Sequence[Any],
        query: dict[str, Any],
        k: int = 5,
    ) -> list[Any]:
        if not items or k <= 0:
            return []

        target_pos = query.get("position")
        current_step = query.get("current_step", 0)

        tx = (
            (target_pos.x if isinstance(target_pos, Position) else target_pos[0])
            if target_pos
            else 0
        )
        ty = (
            (target_pos.y if isinstance(target_pos, Position) else target_pos[1])
            if target_pos
            else 0
        )

        def score(item: Any) -> float:
            # Proximity score (inverse distance normalized)
            pos = getattr(item, "position", None)
            if pos is not None:
                px = pos.x if isinstance(pos, Position) else pos[0]
                py = pos.y if isinstance(pos, Position) else pos[1]
                distance = abs(px - tx) + abs(py - ty)
                prox_score = 1.0 / (1.0 + distance)
            else:
                prox_score = 0.0

            # Confidence score
            conf_score = float(getattr(item, "confidence", 1.0))

            # Recency score
            tick = getattr(item, "last_observed_tick", getattr(item, "step", 0))
            delta = max(0, current_step - tick)
            rec_score = 1.0 / (1.0 + 0.05 * delta)

            return float(
                (self.w_prox * prox_score) + (self.w_conf * conf_score) + (self.w_rec * rec_score)
            )

        sorted_items = sorted(items, key=score, reverse=True)
        return sorted_items[:k]
