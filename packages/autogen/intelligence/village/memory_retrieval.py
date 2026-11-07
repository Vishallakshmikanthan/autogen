"""Multimodal Memory Retrieval Engine (§4.2).

Ranks and retrieves episodic memories based on recency, static importance,
conceptual/actor/location relevance (Jaccard similarity), and Bower's mood congruence.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from autogen.intelligence.village.memory_stream import MemoryItem, MemoryStream

W_RECENCY: float = 1.0
W_IMPORTANCE: float = 1.0
W_RELEVANCE: float = 1.5
W_MOOD: float = 0.4
TAU_RECENCY: float = 24.0  # half day decay horizon (24 simulation ticks)
MAX_PAD_DISTANCE: float = 2.0 * math.sqrt(3.0)  # max Euclidean distance in [-1, 1]^3


class ScoreBreakdown(BaseModel):
    """Component-level attribution of the composite memory retrieval score (§4.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mid: str
    total_score: float
    recency_score: float
    importance_score: float
    relevance_score: float
    mood_score: float


class MemoryQuery(BaseModel):
    """Retrieval query context synthesized from the agent's current situation (§4.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(ge=0, description="Current simulation step")
    actors: tuple[str, ...] = Field(default=(), description="Nearby or interaction target villager IDs")
    location: str = Field(default="", description="Current zone ID or landmark name")
    tags: tuple[str, ...] = Field(default=(), description="Active keywords, intention goals, or alert themes")
    pad: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="Agent's current continuous PAD affective coordinates",
    )


def score_memory(
    memory: MemoryItem,
    query: MemoryQuery,
    *,
    w_r: float = W_RECENCY,
    w_i: float = W_IMPORTANCE,
    w_q: float = W_RELEVANCE,
    w_m: float = W_MOOD,
    tau_r: float = TAU_RECENCY,
) -> ScoreBreakdown:
    """Calculates composite retrieval score $\\text{score}(m \\mid q)$ (§4.2):

    $$\\mathrm{score}(m \\mid q) = w_r e^{-(t - t^{acc})/\\tau_r} + w_i I_m + w_q J(m, q) + w_m \\left(1 - \\frac{\\lVert \\mathbf{pad}_m - \\mathbf{pad}_q \\rVert_2}{2\\sqrt{3}}\\right)$$
    """
    # 1. Recency component: exponential decay from last access step
    delta_t = max(0, query.step - memory.last_access_step)
    recency = math.exp(-float(delta_t) / max(1.0, tau_r))

    # 2. Importance component: intrinsic encoding importance in [0, 1]
    importance = memory.importance

    # 3. Relevance component: Jaccard overlap across actors, tags, and location
    mem_set: set[str] = set(memory.actors) | set(memory.tags)
    if memory.location:
        mem_set.add(memory.location)

    query_set: set[str] = set(query.actors) | set(query.tags)
    if query.location:
        query_set.add(query.location)

    union = mem_set | query_set
    if union:
        intersection = mem_set & query_set
        relevance = float(len(intersection)) / float(len(union))
    else:
        relevance = 0.0

    # 4. Mood-congruence component: inverse normalized PAD Euclidean distance
    mv, ma, md = memory.pad
    qv, qa, qd = query.pad
    dist = math.sqrt((mv - qv) ** 2 + (ma - qa) ** 2 + (md - qd) ** 2)
    mood = max(0.0, 1.0 - (dist / MAX_PAD_DISTANCE))

    total = (w_r * recency) + (w_i * importance) + (w_q * relevance) + (w_m * mood)

    return ScoreBreakdown(
        mid=memory.mid,
        total_score=float(round(total, 4)),
        recency_score=float(round(recency, 4)),
        importance_score=float(round(importance, 4)),
        relevance_score=float(round(relevance, 4)),
        mood_score=float(round(mood, 4)),
    )


def retrieve_top_k(
    stream: MemoryStream,
    query: MemoryQuery,
    k: int = 5,
    rehearse: bool = True,
    on_step: Callable[[ScoreBreakdown], None] | None = None,
    w_r: float = W_RECENCY,
    w_i: float = W_IMPORTANCE,
    w_q: float = W_RELEVANCE,
    w_m: float = W_MOOD,
    tau_r: float = TAU_RECENCY,
) -> list[tuple[MemoryItem, ScoreBreakdown]]:
    """Retrieves top-k scored memories, optionally rehearsing them (§4.2, §4.3).

    Ties in composite score are broken deterministically by memory ID.
    """
    if k <= 0 or len(stream) == 0:
        return []

    scored: list[tuple[MemoryItem, ScoreBreakdown]] = []
    for item in stream.all_items():
        breakdown = score_memory(
            item,
            query,
            w_r=w_r,
            w_i=w_i,
            w_q=w_q,
            w_m=w_m,
            tau_r=tau_r,
        )
        if on_step is not None:
            on_step(breakdown)
        scored.append((item, breakdown))

    # Rank by total_score descending, tie-break by mid ascending (deterministic)
    scored.sort(key=lambda pair: (-pair[1].total_score, pair[0].mid))
    top_results = scored[:k]

    if rehearse:
        # Rehearse top-k items in the stream
        for idx, (item, breakdown) in enumerate(top_results):
            updated_item = item.with_access(query.step)
            stream.add(updated_item)
            top_results[idx] = (updated_item, breakdown)

    return top_results

