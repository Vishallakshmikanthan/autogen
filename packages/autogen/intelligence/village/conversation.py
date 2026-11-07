"""Multi-Tick Conversation State Machine, Engagement Dynamics & Utility Dialogue Acts (§3.1, §3.2, §3.5).

Coordinates multi-step social encounters, dialogue act evaluation, emotional coupling,
engagement decay, snub penalties, and post-conversation relationship settlement.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from autogen.intelligence.village.attention import Alert
    from autogen.intelligence.village.personality import Traits
    from autogen.intelligence.village.relationships import VillageRelationshipStore


class ConvState(str, Enum):
    """Lifecycle states of an active inter-villager conversation (§3.1)."""

    IDLE = "IDLE"
    APPROACHING = "APPROACHING"
    GREETING = "GREETING"
    TALKING = "TALKING"
    CLOSING = "CLOSING"
    INTERRUPTED = "INTERRUPTED"
    ABANDONED = "ABANDONED"


class DialogueActType(str, Enum):
    """Categorical speech acts chosen via utility arbitration (§3.2)."""

    GREET = "GREET"
    SMALL_TALK = "SMALL_TALK"
    SHARE_NEWS = "SHARE_NEWS"
    GOSSIP = "GOSSIP"
    ASK_FAVOR = "ASK_FAVOR"
    COMFORT = "COMFORT"
    COMPLAIN = "COMPLAIN"
    ACCUSE = "ACCUSE"
    APOLOGIZE = "APOLOGIZE"
    FAREWELL = "FAREWELL"


class DialogueAct(BaseModel):
    """A structured, intentional conversational utterance unit (§3.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    act_type: DialogueActType
    speaker_id: str
    target_id: str
    subject_id: str | None = None
    claim: str | None = None
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    topic: str = "general"
    utterance_hint: str = ""
    is_risky: bool = False


class Conversation(BaseModel):
    """Multi-tick shared conversation state machine between villagers (§3.1)."""

    model_config = ConfigDict(extra="forbid")

    conv_id: str
    participants: tuple[str, str]
    state: ConvState = ConvState.APPROACHING
    topic_stack: list[str] = Field(default_factory=lambda: ["greeting"])
    turn_owner: str
    turns: int = 0
    engagement: dict[str, float] = Field(default_factory=dict)
    history: list[DialogueAct] = Field(default_factory=list)
    started_step: int
    interrupted_by_alert: bool = False
    abandoned_by: str | None = None

    def other_participant(self, agent_id: str) -> str:
        """Returns the other participant ID in dyad."""
        return self.participants[1] if self.participants[0] == agent_id else self.participants[0]


def compute_engagement(
    current_engagement: float,
    *,
    topic_relevance: float = 0.50,
    affinity: float = 0.50,
    turns: int = 0,
) -> float:
    """Computes engagement update $E_i(t+1)$ for conversation participant (§3.1):

    $$E_i(t+1) = \\mathrm{clip}(E_i(t) + 0.15 \\cdot r_{\\text{topic}} + 0.10 \\cdot \\text{affinity} - 0.08 - 0.10 \\cdot b_i, \\ 0.0, \\ 1.0)$$
    where $b_i = \\min(1.0, \\text{turns} \\times 0.15)$ models cumulative boredom.
    """
    boredom = min(1.0, turns * 0.15)
    r = float(np.clip(topic_relevance, 0.0, 1.0))
    aff = float(np.clip(affinity, 0.0, 1.0))
    delta = 0.15 * r + 0.10 * aff - 0.08 - 0.10 * boredom
    return float(round(float(np.clip(current_engagement + delta, 0.0, 1.0)), 4))


def compute_dialogue_act_utility(
    act: DialogueAct,
    *,
    traits: Traits,
    w_soc: float,
    w_rel: float,
    current_affinity: float,
    speaker_pad: tuple[float, float, float],
    p_overheard: float = 0.0,
    is_subject_high_status: bool = False,
) -> float:
    """Calculates multi-attribute utility $U_{\\text{act}}(d)$ for candidate dialogue act (§3.2):

    $$U_{\\text{act}}(d) = w_{\\text{soc}} \\Delta \\text{status} + w_{\\text{rel}} \\Delta \\text{affinity} + w_{\\text{info}} \\text{info} + w_{\\text{mood}} \\text{mood} - \\text{cost} - \\mathbb{1}_{\\text{risky}} \\theta \\cdot P_{\\text{overheard}} \\cdot \\kappa$$
    """
    w_info = 1.0 + 0.50 * traits.openness
    w_mood = 0.8 + 0.50 * traits.neuroticism
    theta_risk = 0.35

    # Conscientiousness/Openness risk sensitivity scalar kappa (§5.2)
    kappa = 20.0 * (0.50 + 0.50 * traits.conscientiousness) * (1.0 - 0.50 * traits.openness)

    # Base attributes per act type
    d_status = 0.0
    d_affinity = 0.0
    info_val = 0.0
    cost = 0.05
    act_valence = act.valence

    if act.act_type == DialogueActType.GREET:
        d_affinity = 0.04 * (1.0 - current_affinity)
        d_status = 0.01
        cost = 0.01

    elif act.act_type == DialogueActType.SMALL_TALK:
        d_affinity = 0.06 * (1.0 - current_affinity) * (0.5 + 0.5 * traits.extraversion)
        cost = 0.03

    elif act.act_type == DialogueActType.SHARE_NEWS:
        info_val = 0.40 * (1.0 + traits.extraversion)
        d_status = 0.03

    elif act.act_type == DialogueActType.GOSSIP:
        info_val = 0.35
        # Gossip bonds speaker and listener against third party, but is socially risky
        d_affinity = 0.05 * (1.0 - current_affinity)
        d_status = 0.02

    elif act.act_type == DialogueActType.COMFORT:
        d_affinity = 0.12 * traits.agreeableness * (1.0 - current_affinity)
        cost = 0.04

    elif act.act_type == DialogueActType.COMPLAIN:
        d_status = -0.02
        d_affinity = -0.03 if current_affinity < 0.4 else 0.02
        act_valence = -0.40

    elif act.act_type == DialogueActType.ACCUSE:
        d_status = 0.05 if traits.conscientiousness > 0.6 else -0.04
        d_affinity = -0.25
        cost = 0.10
        act_valence = -0.80

    elif act.act_type == DialogueActType.APOLOGIZE:
        d_affinity = 0.15 * traits.agreeableness
        d_status = -0.03
        cost = 0.05
        act_valence = 0.30

    elif act.act_type == DialogueActType.FAREWELL:
        d_affinity = 0.02
        cost = 0.01

    # Mood congruence: correlation between act valence and speaker pleasure
    v_speaker = speaker_pad[0]
    mood_cong = 1.0 - abs(v_speaker - act_valence) / 2.0

    u_base = (
        (w_soc * 0.10 * d_status)
        + (w_rel * 1.50 * d_affinity)
        + (w_info * info_val)
        + (w_mood * mood_cong)
        - cost
    )

    # Risk penalty from eavesdropping on defamatory or accusatory gossip
    is_risky = act.is_risky or (
        act.act_type in (DialogueActType.GOSSIP, DialogueActType.ACCUSE)
        and (is_subject_high_status or act.subject_id is not None)
    )
    if is_risky:
        risk_penalty = theta_risk * float(np.clip(p_overheard, 0.0, 1.0)) * (kappa * 0.05)
        u_base -= risk_penalty

    return float(round(u_base, 4))


class ConversationEngine:
    """Stateful orchestrator for conversational progression and settlements (§3.1, §3.5)."""

    THETA_LEAVE: float = 0.20
    MAX_TURNS_DEFAULT: int = 6

    @classmethod
    def start_conversation(
        cls,
        conv_id: str,
        initiator_id: str,
        recipient_id: str,
        step: int,
        initial_topic: str = "greeting",
    ) -> Conversation:
        """Instantiates a new Conversation FSM in APPROACHING state."""
        return Conversation(
            conv_id=conv_id,
            participants=(initiator_id, recipient_id),
            state=ConvState.APPROACHING,
            topic_stack=[initial_topic],
            turn_owner=initiator_id,
            turns=0,
            engagement={initiator_id: 0.85, recipient_id: 0.70},
            history=[],
            started_step=step,
        )

    @classmethod
    def step_conversation(
        cls,
        conv: Conversation,
        active_alerts: Sequence[Alert],
        relationships: VillageRelationshipStore,
        chosen_act: DialogueAct | None = None,
        max_turns: int = MAX_TURNS_DEFAULT,
    ) -> tuple[ConvState, DialogueAct | None, str]:
        """Advances conversational state machine by one turn (§3.1).

        Returns (new_state, emitted_act, transition_reason).
        """
        # 1. Emergency Interruption Check: High-threat alert overrides conversation immediately
        for alert in active_alerts:
            if alert.threat > 0.40 or alert.percept.event.kind in ("THUNDER", "RAIN_ONSET"):
                conv.state = ConvState.INTERRUPTED
                conv.interrupted_by_alert = True
                return ConvState.INTERRUPTED, None, f"interrupted_by_{alert.percept.event.kind}"

        # 2. State Machine Transitions
        if conv.state == ConvState.APPROACHING:
            conv.state = ConvState.GREETING
            greet_act = DialogueAct(
                act_type=DialogueActType.GREET,
                speaker_id=conv.turn_owner,
                target_id=conv.other_participant(conv.turn_owner),
                topic="greeting",
                utterance_hint="Warm village greeting",
            )
            conv.history.append(greet_act)
            conv.turns += 1
            conv.turn_owner = conv.other_participant(conv.turn_owner)
            return ConvState.GREETING, greet_act, "approach_to_greeting"

        if conv.state == ConvState.GREETING:
            conv.state = ConvState.TALKING
            # Acknowledge greeting with small talk or reciprocal greet
            reply_act = DialogueAct(
                act_type=DialogueActType.SMALL_TALK,
                speaker_id=conv.turn_owner,
                target_id=conv.other_participant(conv.turn_owner),
                topic="small_talk",
                utterance_hint="Friendly greeting reply",
            )
            conv.history.append(reply_act)
            conv.turns += 1
            conv.turn_owner = conv.other_participant(conv.turn_owner)
            return ConvState.TALKING, reply_act, "greeting_acknowledged"

        if conv.state == ConvState.TALKING:
            # Update engagement for both participants
            speaker = conv.turn_owner
            listener = conv.other_participant(speaker)

            aff_spk = relationships.get_affinity(speaker, listener)
            aff_lis = relationships.get_affinity(listener, speaker)

            e_spk = compute_engagement(conv.engagement.get(speaker, 0.5), affinity=aff_spk, turns=conv.turns)
            e_lis = compute_engagement(conv.engagement.get(listener, 0.5), affinity=aff_lis, turns=conv.turns)

            conv.engagement[speaker] = e_spk
            conv.engagement[listener] = e_lis

            # Check for sudden conversational abandonment (snubbing)
            if e_spk < cls.THETA_LEAVE:
                conv.state = ConvState.ABANDONED
                conv.abandoned_by = speaker
                return ConvState.ABANDONED, None, "abandoned_due_to_boredom"
            if e_lis < cls.THETA_LEAVE:
                conv.state = ConvState.ABANDONED
                conv.abandoned_by = listener
                return ConvState.ABANDONED, None, "abandoned_due_to_boredom"

            # Execute turn speech act
            act = chosen_act or DialogueAct(
                act_type=DialogueActType.SMALL_TALK,
                speaker_id=speaker,
                target_id=listener,
                topic="small_talk",
            )
            conv.history.append(act)
            conv.turns += 1

            # Termination condition
            if conv.turns >= max_turns or act.act_type == DialogueActType.FAREWELL:
                conv.state = ConvState.CLOSING
                return ConvState.CLOSING, act, "max_turns_or_farewell"

            conv.turn_owner = listener
            return ConvState.TALKING, act, "turn_stepped"

        if conv.state in (ConvState.CLOSING, ConvState.INTERRUPTED, ConvState.ABANDONED):
            conv.state = ConvState.IDLE
            return ConvState.IDLE, None, "settled_to_idle"

        return conv.state, None, "no_op"

    @classmethod
    def settle_conversation(
        cls,
        conv: Conversation,
        relationships: VillageRelationshipStore,
        step: int,
    ) -> dict[str, Any]:
        """Settles relational updates and episodic memories post-conversation (§3.5)."""
        p1, p2 = conv.participants

        # Interrupted by emergency has NO relationship penalty
        if conv.state == ConvState.INTERRUPTED or conv.interrupted_by_alert:
            return {
                "outcome": "interrupted_emergency",
                "affinity_delta": 0.0,
                "familiarity_boost": 0.02,
            }

        # Snubbed / Abandoned mid-sentence produces relational penalty
        if conv.state == ConvState.ABANDONED:
            leaver = conv.abandoned_by or p1
            stayed = conv.other_participant(leaver)
            # Stayed villager resents the sudden snub
            relationships.record_interaction(stayed, leaver, quality=-0.35, trust_delta=-0.05)
            return {
                "outcome": "abandoned_snub",
                "leaver": leaver,
                "victim": stayed,
                "affinity_penalty": -0.10,
            }

        # Normal clean conversation closure (CLOSING or completed)
        mean_valence = (
            sum(act.valence for act in conv.history) / max(1, len(conv.history))
        )
        avg_eng = sum(conv.engagement.values()) / max(1, len(conv.engagement))
        quality = float(round(0.60 * mean_valence + 0.40 * (avg_eng - 0.50), 4))

        relationships.record_dyadic_interaction(
            p1, p2, quality=quality, trust_delta=0.02, respect_delta=0.01
        )
        return {
            "outcome": "clean_closure",
            "quality": quality,
            "turns": conv.turns,
            "history_count": len(conv.history),
        }

