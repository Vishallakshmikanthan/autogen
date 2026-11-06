"""Structured Episodic Memory Stream, Importance Scoring, and Forgetting (§4.1, §4.3).

Provides bounded, deterministic episodic memory storage with mathematical importance
attribution, Ebbinghaus forgetting decay, rumination rehearsal, and pruning.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

MemoryKind = Literal["observation", "action", "conversation", "feeling", "reflection", "belief"]

S_0_DEFAULT: float = 96.0  # 2 full days in simulation ticks
EPSILON_PRUNE: float = 0.05
MAX_STREAM_CAPACITY: int = 300


class MemoryItem(BaseModel):
    """A structured, explainable episodic memory unit (§4.1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mid: str = Field(description="Deterministic memory identifier: f'{agent_id}:{step}:{seq}'")
    step: int = Field(ge=0, description="Simulation step at which the memory was encoded")
    kind: MemoryKind = Field(description="Taxonomic category of the experience")
    actors: tuple[str, ...] = Field(default=(), description="Sorted tuple of participating agent IDs")
    location: str = Field(default="", description="Zone ID or spatial region name")
    summary: str = Field(default="", description="Human-readable symbolic event summary")
    tags: tuple[str, ...] = Field(default=(), description="Sorted conceptual keywords for semantic matching")
    valence: float = Field(default=0.0, ge=-1.0, le=1.0, description="Affective valence in [-1, 1]")
    pad: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="Agent emotional PAD vector at encoding time",
    )
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Static importance score computed at encoding")
    access_count: int = Field(default=0, ge=0, description="Total number of retrieval rehearsals")
    last_access_step: int = Field(ge=0, description="Simulation step of most recent rehearsal access")
    evidence_of: tuple[str, ...] = Field(
        default=(),
        description="For reflections/beliefs: sequence of source memory IDs supporting this synthesis",
    )

    def with_access(self, access_step: int) -> MemoryItem:
        """Returns a copy with incremented access count and updated access step."""
        return self.model_copy(
            update={
                "access_count": self.access_count + 1,
                "last_access_step": max(self.last_access_step, access_step),
            }
        )


def compute_importance(
    *,
    valence: float,
    arousal: float,
    self_involvement: float,
    rarity: float,
    consequence: float,
) -> float:
    """Computes fixed importance score $I \\in [0, 1]$ at memory write time (§4.1):

    $$I = \\mathrm{clip}(0.35 |v| + 0.25 a^{+} + 0.20 s + 0.10 r + 0.10 c, \\ 0.0, \\ 1.0)$$

    where $a^{+} = \\max(0, a)$, $s \\in \\{1.0, 0.5, 0.0\\}$, $r = \\frac{1}{1 + n}$,
    and $c$ is consequence scaled from self-observed economic and trust shifts.
    """
    v_term = 0.35 * abs(float(valence))
    a_term = 0.25 * max(0.0, float(arousal))
    s_term = 0.20 * float(np.clip(self_involvement, 0.0, 1.0))
    r_term = 0.10 * float(np.clip(rarity, 0.0, 1.0))
    c_term = 0.10 * float(np.clip(consequence, 0.0, 1.0))

    raw = v_term + a_term + s_term + r_term + c_term
    return float(round(float(np.clip(raw, 0.0, 1.0)), 4))


def compute_strength(
    *,
    importance: float,
    step: int,
    last_access_step: int,
    access_count: int,
    is_reflection: bool = False,
    s_0: float = S_0_DEFAULT,
) -> float:
    """Computes memory retention strength $S(m, t)$ under Ebbinghaus decay and rehearsal (§4.3):

    $$S(m, t) = I_m \\cdot \\exp\\left(-\\frac{t - t^{acc}}{S_0}\\right) \\cdot (1 + 0.3 \\ln(1 + n^{acc}))$$
    """
    effective_s0 = s_0 * 3.0 if is_reflection else s_0
    delta_t = max(0, step - last_access_step)
    decay = math.exp(-float(delta_t) / max(1.0, effective_s0))
    rehearsal_boost = 1.0 + 0.30 * math.log1p(max(0, access_count))
    return float(round(importance * decay * rehearsal_boost, 5))


class MemoryStream:
    """Bounded, ordered, and searchable episodic memory stream for an individual villager (§4.1, §4.3)."""

    def __init__(self, agent_id: str, capacity: int = MAX_STREAM_CAPACITY) -> None:
        self.agent_id = agent_id
        self.capacity = max(10, capacity)
        self._items: dict[str, MemoryItem] = {}
        self._seq: int = 0
        self._reflected_mids: set[str] = set()
        self._recent_tag_history: dict[tuple[str, str], list[int]] = {}

    def __len__(self) -> int:
        return len(self._items)

    def get(self, mid: str) -> MemoryItem | None:
        """Retrieves a single memory item by identifier."""
        return self._items.get(mid)

    def all_items(self) -> list[MemoryItem]:
        """Returns all memory items sorted deterministically by (step, mid)."""
        return sorted(self._items.values(), key=lambda m: (m.step, m.mid))

    def unreflected(self) -> list[MemoryItem]:
        """Returns all memory items that have not yet undergone reflection consolidation."""
        return [
            m for m in sorted(self._items.values(), key=lambda m: (m.step, m.mid))
            if m.mid not in self._reflected_mids and m.kind != "reflection" and m.kind != "belief"
        ]

    def mark_reflected(self, mids: Iterable[str]) -> None:
        """Marks the given memory IDs as consolidated by reflection."""
        for mid in mids:
            if mid in self._items:
                self._reflected_mids.add(mid)

    def calculate_rarity(self, kind: str, primary_tag: str, current_step: int) -> float:
        """Computes rarity $r = 1 / (1 + n)$ based on occurrences of (kind, primary_tag) in past 48 ticks."""
        key = (kind, primary_tag)
        timestamps = self._recent_tag_history.get(key, [])
        # Filter to last 48 ticks
        valid = [t for t in timestamps if (current_step - t) <= 48]
        self._recent_tag_history[key] = valid
        return 1.0 / (1.0 + len(valid))

    def add(self, item: MemoryItem) -> MemoryItem:
        """Adds a verified MemoryItem to the stream and tracks frequency history."""
        self._items[item.mid] = item
        primary_tag = item.tags[0] if item.tags else "general"
        key = (item.kind, primary_tag)
        self._recent_tag_history.setdefault(key, []).append(item.step)
        return item

    def create_and_add(
        self,
        *,
        step: int,
        kind: MemoryKind,
        actors: Sequence[str] = (),
        location: str = "",
        summary: str = "",
        tags: Sequence[str] = (),
        valence: float = 0.0,
        pad: tuple[float, float, float] = (0.0, 0.0, 0.0),
        self_involvement: float = 0.5,
        consequence: float = 0.0,
        evidence_of: Sequence[str] = (),
        fixed_importance: float | None = None,
    ) -> MemoryItem:
        """Constructs, scores, and stores a new MemoryItem with deterministic ID."""
        self._seq += 1
        mid = f"{self.agent_id}:{step}:{self._seq}"
        sorted_actors = tuple(sorted(set(actors)))
        sorted_tags = tuple(sorted(set(tags)))

        primary_tag = sorted_tags[0] if sorted_tags else "general"
        rarity = self.calculate_rarity(kind, primary_tag, step)

        if fixed_importance is not None:
            importance = float(np.clip(fixed_importance, 0.0, 1.0))
        else:
            arousal = pad[1]
            importance = compute_importance(
                valence=valence,
                arousal=arousal,
                self_involvement=self_involvement,
                rarity=rarity,
                consequence=consequence,
            )

        item = MemoryItem(
            mid=mid,
            step=step,
            kind=kind,
            actors=sorted_actors,
            location=location,
            summary=summary,
            tags=sorted_tags,
            valence=float(round(np.clip(valence, -1.0, 1.0), 4)),
            pad=pad,
            importance=importance,
            access_count=0,
            last_access_step=step,
            evidence_of=tuple(evidence_of),
        )
        return self.add(item)

    def prune(self, current_step: int, epsilon: float = EPSILON_PRUNE) -> list[str]:
        """Evicts forgotten memories ($S < \\epsilon$) or items beyond capacity (§4.3).

        Guarantees that unreflected items are never pruned before consolidation.
        Ties are broken deterministically by oldest mid.
        """
        evicted: list[str] = []

        # 1. Prune expired items below epsilon threshold (excluding unreflected)
        to_delete: list[str] = []
        for mid, item in self._items.items():
            if mid not in self._reflected_mids and item.kind not in ("reflection", "belief"):
                continue  # Never prune unreflected experiences before reflection pass
            str_val = compute_strength(
                importance=item.importance,
                step=current_step,
                last_access_step=item.last_access_step,
                access_count=item.access_count,
                is_reflection=(item.kind in ("reflection", "belief")),
            )
            if str_val < epsilon:
                to_delete.append(mid)

        for mid in to_delete:
            del self._items[mid]
            self._reflected_mids.discard(mid)
            evicted.append(mid)

        # 2. Enforce stream capacity cap
        if len(self._items) > self.capacity:
            excess = len(self._items) - self.capacity
            # Candidate candidates for eviction: prefer reflected items first
            candidates: list[tuple[float, int, str]] = []
            for mid, item in self._items.items():
                is_unref = mid not in self._reflected_mids and item.kind not in ("reflection", "belief")
                # Give huge artificial strength boost to unreflected items so they are evicted last
                unref_boost = 1000.0 if is_unref else 0.0
                str_val = compute_strength(
                    importance=item.importance,
                    step=current_step,
                    last_access_step=item.last_access_step,
                    access_count=item.access_count,
                    is_reflection=(item.kind in ("reflection", "belief")),
                )
                candidates.append((str_val + unref_boost, item.step, mid))

            # Sort ascending by strength, then oldest step, then mid
            candidates.sort(key=lambda t: (t[0], t[1], t[2]))
            for _, _, evict_mid in candidates[:excess]:
                del self._items[evict_mid]
                self._reflected_mids.discard(evict_mid)
                evicted.append(evict_mid)

        return evicted

    def digest(self) -> str:
        """Returns deterministic SHA-256 digest of stored items for simulation hashing."""
        payload = [
            (
                m.mid,
                m.step,
                m.kind,
                m.actors,
                m.tags,
                m.importance,
                m.access_count,
                m.last_access_step,
            )
            for m in sorted(self._items.values(), key=lambda x: x.mid)
        ]
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
