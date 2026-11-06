"""Synchronous Multi-Agent Emotional Contagion Engine (§3.4).

Couples neighboring agents' affective PAD vectors through physical proximity,
dyadic affinity, mutual attention, and personality-modulated susceptibility (gamma).
Ensures strict order-independence and caps per-tick contagion deltas.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from autogen.intelligence.village.affective_pad import AffectiveStateVector
from autogen.intelligence.village.personality import PersonalityParams
from autogen.intelligence.village.relationships import VillageRelationshipStore
from autogen.simulation.state.types import Position

CONTAGION_LENGTH_TILES: float = 6.0
CONTAGION_MAX_AXIS_DELTA: float = 0.15
NEGATIVITY_BIAS_FACTOR: float = 1.30


def compute_contagion_weight(
    pos_i: Position,
    pos_j: Position,
    affinity_ij: float,
    source_valence: float,
    max_radius: float = 8.0,
) -> float:
    """Computes directed contagion coupling weight $w_{ij}$ (§3.4):

    $$w_{ij} = e^{-d / L} \\cdot (0.5 + 0.5 \\cdot \\text{aff}_{ij}) \\cdot \\text{attn}_{ij}$$
    Boosted by 1.3x if source valence is negative (< -0.30) (empirical negativity bias).
    """
    dx = float(pos_i.x - pos_j.x)
    dy = float(pos_i.y - pos_j.y)
    dist = math.hypot(dx, dy)

    if dist > max_radius:
        return 0.0

    prox = math.exp(-dist / CONTAGION_LENGTH_TILES)
    rel_factor = 0.50 + 0.50 * float(np.clip(affinity_ij, 0.0, 1.0))
    weight = prox * rel_factor

    if source_valence < -0.30:
        weight *= NEGATIVITY_BIAS_FACTOR

    return float(round(weight, 5))


class EmotionalContagionEngine:
    """Executes order-independent synchronous emotional contagion across all active villagers."""

    @staticmethod
    def step_contagion(
        agents: Mapping[str, AffectiveStateVector],
        positions: Mapping[str, Position],
        personalities: Mapping[str, PersonalityParams],
        relationships: VillageRelationshipStore,
        sensing_radius: float = 8.0,
    ) -> dict[str, AffectiveStateVector]:
        """Calculates synchronous PAD adjustments:

        $$\\Delta P_i = \\text{clip}\\left( \\gamma_i \\sum_j w_{ij} (P_j - P_i), -0.15, 0.15 \\right)$$
        """
        agent_ids = sorted(agents.keys())
        deltas: dict[str, tuple[float, float, float]] = {}

        for aid_i in agent_ids:
            state_i = agents[aid_i]
            pos_i = positions.get(aid_i, Position(x=0, y=0))
            gamma_i = personalities[aid_i].contagion_gamma if aid_i in personalities else 0.08

            sum_dv = 0.0
            sum_da = 0.0
            sum_dd = 0.0

            for aid_j in agent_ids:
                if aid_i == aid_j:
                    continue

                state_j = agents[aid_j]
                pos_j = positions.get(aid_j, Position(x=0, y=0))
                aff_ij = relationships.get_affinity(aid_i, aid_j)

                w_ij = compute_contagion_weight(
                    pos_i,
                    pos_j,
                    aff_ij,
                    state_j.valence,
                    max_radius=sensing_radius,
                )

                if w_ij <= 1e-6:
                    continue

                diff_v = state_j.valence - state_i.valence
                diff_a = state_j.arousal - state_i.arousal
                diff_d = state_j.dominance - state_i.dominance

                sum_dv += w_ij * diff_v
                sum_da += w_ij * diff_a
                sum_dd += w_ij * diff_d

            # Apply gamma and cap per axis at [-0.15, 0.15]
            dv_capped = float(
                np.clip(
                    gamma_i * sum_dv,
                    -CONTAGION_MAX_AXIS_DELTA,
                    CONTAGION_MAX_AXIS_DELTA,
                )
            )
            da_capped = float(
                np.clip(
                    gamma_i * sum_da,
                    -CONTAGION_MAX_AXIS_DELTA,
                    CONTAGION_MAX_AXIS_DELTA,
                )
            )
            dd_capped = float(
                np.clip(
                    gamma_i * sum_dd,
                    -CONTAGION_MAX_AXIS_DELTA,
                    CONTAGION_MAX_AXIS_DELTA,
                )
            )

            deltas[aid_i] = (dv_capped, da_capped, dd_capped)

        # Synchronously apply deltas
        next_states: dict[str, AffectiveStateVector] = {}
        for aid, st in agents.items():
            dv, da, dd = deltas.get(aid, (0.0, 0.0, 0.0))
            v_new = float(np.clip(st.valence + dv, -1.0, 1.0))
            a_new = float(np.clip(st.arousal + da, -1.0, 1.0))
            d_new = float(np.clip(st.dominance + dd, -1.0, 1.0))

            next_states[aid] = st.model_copy(
                update={
                    "valence": round(v_new, 4),
                    "arousal": round(a_new, 4),
                    "dominance": round(d_new, 4),
                }
            )

        return next_states
