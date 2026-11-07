"""Lossy Gossip Transmission, Epistemic Provenance, and Corroboration Dynamics (§3.3).

Provides first-class belief objects, speaker spin exaggeration, listener credibility/skepticism
filtering, rumor echo loop suppression, and noisy-OR multi-source corroboration.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Belief(BaseModel):
    """Structured, provenance-tracked subjective belief about an agent or village state (§3.3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    belief_id: str = Field(description="Unique belief identifier")
    subject: str = Field(description="Agent ID or village entity the belief is about")
    claim: str = Field(description="Categorical proposition (e.g. 'stole_grain', 'generous', 'ill')")
    valence: float = Field(default=0.0, ge=-1.0, le=1.0, description="Affective appraisal in [-1, 1]")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Epistemic certainty in [0, 1]")
    source_chain: tuple[str, ...] = Field(
        default=(),
        description="Chronological provenance tuple of transmitters: (originator, hop_1, ..., immediate_source)",
    )
    first_hand: bool = Field(default=True, description="True if self-witnessed, False if received via gossip")
    heard_at_step: int = Field(ge=0, description="Simulation step when belief was formed or heard")
    times_heard: int = Field(default=1, ge=1, description="Cumulative independent corroboration count")

    @property
    def hop_count(self) -> int:
        """Returns social distance (number of hops from original observer)."""
        return max(0, len(self.source_chain) - 1) if not self.first_hand else 0


def compute_speaker_spin(
    *,
    arousal: float,
    affinity_to_subject: float,
) -> float:
    """Computes speaker emotional distortion / exaggeration multiplier (§3.3):

    $$\\text{spin} = 1.0 + 0.25 \\max(0, a_s) + 0.20 \\cdot \\text{dislike}$$
    where $\\text{dislike} = \\max(0.0, (0.50 - \\text{affinity}) \\times 2.0) \\in [0, 1]$.
    """
    a_pos = max(0.0, float(arousal))
    dislike = max(0.0, (0.50 - float(affinity_to_subject)) * 2.0)
    spin = 1.0 + 0.25 * a_pos + 0.20 * dislike
    return float(round(spin, 4))


def compute_listener_credibility(
    *,
    trust_in_speaker: float,
    speaker_status: float,
    affinity_for_speaker: float,
) -> float:
    """Evaluates listener appraisal of speaker source credibility (§3.3):

    $$\\text{cred} = 0.50 \\cdot \\text{trust} + 0.30 \\cdot \\text{status} + 0.20 \\cdot \\text{affinity}$$
    """
    t = float(np.clip(trust_in_speaker, 0.0, 1.0))
    s = float(np.clip(speaker_status, 0.0, 1.0))
    a = float(np.clip(affinity_for_speaker, 0.0, 1.0))
    cred = 0.50 * t + 0.30 * s + 0.20 * a
    return float(round(float(np.clip(cred, 0.0, 1.0)), 4))


def compute_acceptance_probability(
    *,
    credibility: float,
    consistency: float,
    openness: float,
) -> float:
    """Computes sigmoid acceptance probability of communicated claim (§3.3):

    $$\\text{skepticism} = 0.50 + 0.50 \\cdot (1.0 - O)$$
    $$x = 4.0 \\cdot \\left(\\frac{\\text{credibility} \\cdot \\text{consistency}}{\\text{skepticism}} - 0.50\\right)$$
    $$P_{\\text{accept}} = \\sigma(x)$$
    """
    skepticism = 0.50 + 0.50 * (1.0 - float(np.clip(openness, 0.0, 1.0)))
    ratio = (float(credibility) * float(consistency)) / max(0.05, skepticism)
    x = 4.0 * (ratio - 0.50)
    # Numerically stable sigmoid
    if x >= 0:
        p = 1.0 / (1.0 + math.exp(-x))
    else:
        z = math.exp(x)
        p = z / (1.0 + z)
    return float(round(float(np.clip(p, 0.0, 1.0)), 4))


def merge_corroborated_confidence(
    prior_confidence: float,
    incoming_confidence: float,
    credibility: float,
) -> float:
    """Combines independent confirmations via Noisy-OR formula (§3.3):

    $$c_{\\text{merged}} = 1.0 - (1.0 - c_{\\text{prior}})(1.0 - c_{\\text{incoming}} \\cdot \\text{cred})$$
    """
    c_p = float(np.clip(prior_confidence, 0.0, 1.0))
    effective_inc = float(np.clip(incoming_confidence * credibility, 0.0, 1.0))
    merged = 1.0 - (1.0 - c_p) * (1.0 - effective_inc)
    return float(round(float(np.clip(merged, 0.0, 1.0)), 4))


class GossipEngine:
    """Stateless algorithmic engine for lossy belief transmission and rumor dynamics (§3.3)."""

    @classmethod
    def transmit(
        cls,
        *,
        speaker_id: str,
        listener_id: str,
        belief: Belief,
        speaker_arousal: float,
        speaker_affinity_to_subject: float,
        listener_trust_in_speaker: float,
        listener_status_of_speaker: float,
        listener_affinity_for_speaker: float,
        listener_openness: float,
        listener_prior_belief: Belief | None,
        rng: np.random.Generator,
        step: int,
    ) -> tuple[Belief | None, bool, str]:
        """Attempts transmission of belief from speaker to listener.

        Returns (new_belief_if_accepted, accepted_bool, reason_string).
        """
        # 1. Echo Loop Suppression: If listener is already in source chain, reject immediately
        if listener_id in belief.source_chain:
            return None, False, "loop_echo_suppression"

        # 2. Speaker Distortion (spin & hop decay)
        spin = compute_speaker_spin(
            arousal=speaker_arousal,
            affinity_to_subject=speaker_affinity_to_subject,
        )
        sent_val = float(round(np.clip(belief.valence * spin, -1.0, 1.0), 4))
        # Non-first-hand information loses 10% confidence per hop
        decay = 1.0 if belief.first_hand else 0.90
        sent_conf = float(round(np.clip(belief.confidence * decay, 0.0, 1.0), 4))

        # 3. Listener Source Credibility & Consistency
        cred = compute_listener_credibility(
            trust_in_speaker=listener_trust_in_speaker,
            speaker_status=listener_status_of_speaker,
            affinity_for_speaker=listener_affinity_for_speaker,
        )

        if listener_prior_belief is not None:
            # Prior consistency: penalize if incoming valence contradicts firsthand knowledge
            consistency = 1.0 - abs(listener_prior_belief.valence - sent_val) / 2.0
        else:
            consistency = 1.0

        p_accept = compute_acceptance_probability(
            credibility=cred,
            consistency=consistency,
            openness=listener_openness,
        )

        # 4. Stochastic Acceptance Roll
        if float(rng.random()) >= p_accept:
            return None, False, f"rejected_by_skepticism (p={p_accept:.3f})"

        # 5. Formulate Accepted Belief with Provenance
        new_source_chain = belief.source_chain + (speaker_id,)
        if listener_prior_belief is not None:
            # Check if source chains are disjoint (independent corroboration)
            prior_sources = set(listener_prior_belief.source_chain)
            new_sources = set(new_source_chain)
            is_disjoint = len(prior_sources & new_sources) == 0

            if is_disjoint:
                merged_conf = merge_corroborated_confidence(
                    prior_confidence=listener_prior_belief.confidence,
                    incoming_confidence=sent_conf,
                    credibility=cred,
                )
                times = listener_prior_belief.times_heard + 1
            else:
                # Shared lineage: take max confidence, don't double count
                merged_conf = max(listener_prior_belief.confidence, sent_conf * cred)
                times = listener_prior_belief.times_heard

            # Blend valence weighted by confidence
            w_prior = listener_prior_belief.confidence
            w_inc = sent_conf * cred
            denom = max(1e-6, w_prior + w_inc)
            merged_val = (w_prior * listener_prior_belief.valence + w_inc * sent_val) / denom

            accepted_belief = Belief(
                belief_id=listener_prior_belief.belief_id,
                subject=belief.subject,
                claim=belief.claim,
                valence=float(round(np.clip(merged_val, -1.0, 1.0), 4)),
                confidence=float(round(np.clip(merged_conf, 0.0, 1.0), 4)),
                source_chain=new_source_chain,
                first_hand=False,
                heard_at_step=step,
                times_heard=times,
            )
        else:
            accepted_belief = Belief(
                belief_id=f"belief:{listener_id}:{belief.subject}:{step}",
                subject=belief.subject,
                claim=belief.claim,
                valence=sent_val,
                confidence=float(round(np.clip(sent_conf * cred, 0.0, 1.0), 4)),
                source_chain=new_source_chain,
                first_hand=False,
                heard_at_step=step,
                times_heard=1,
            )

        return accepted_belief, True, "accepted"


class BeliefStore:
    """Manages an agent's internal catalog of beliefs indexed by (subject, claim)."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._beliefs: dict[tuple[str, str], Belief] = {}

    def __len__(self) -> int:
        return len(self._beliefs)

    def get(self, subject: str, claim: str) -> Belief | None:
        """Retrieves belief about specific subject and claim."""
        return self._beliefs.get((subject, claim))

    def get_all_for_subject(self, subject: str) -> list[Belief]:
        """Returns all beliefs concerning target subject."""
        return [b for (s, _), b in sorted(self._beliefs.items()) if s == subject]

    def all_beliefs(self) -> list[Belief]:
        """Returns all stored beliefs sorted deterministically."""
        return sorted(self._beliefs.values(), key=lambda b: (b.subject, b.claim))

    def put(self, belief: Belief) -> None:
        """Inserts or replaces a belief in the store."""
        self._beliefs[(belief.subject, belief.claim)] = belief

    def digest(self) -> str:
        """Produces deterministic SHA-256 state digest of stored beliefs."""
        payload = [
            (
                b.belief_id,
                b.subject,
                b.claim,
                b.valence,
                b.confidence,
                b.source_chain,
                b.times_heard,
            )
            for b in self.all_beliefs()
        ]
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
