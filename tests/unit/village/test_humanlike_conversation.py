"""Unit and oracle tests for Multi-Tick Conversation FSM and Engagement Dynamics."""

import pytest

from autogen.intelligence.village.attention import Alert
from autogen.intelligence.village.conversation import (
    Conversation,
    ConversationEngine,
    ConvState,
    DialogueAct,
    DialogueActType,
    compute_engagement,
)
from autogen.intelligence.village.relationships import VillageRelationshipStore
from autogen.simulation.environment.village.events import Modality, PerceivedEvent, WorldEvent
from autogen.simulation.state.types import Position


class TestConversationFSM:
    """Verification of conversation lifecycle transitions and turn mechanics."""

    def test_full_conversation_lifecycle(self) -> None:
        """Verify sequential transition: APPROACHING -> GREETING -> TALKING -> CLOSING -> IDLE."""
        rel_store = VillageRelationshipStore()
        conv = ConversationEngine.start_conversation(
            conv_id="conv_1",
            initiator_id="arun",
            recipient_id="bheem",
            step=10,
        )
        assert conv.state == ConvState.APPROACHING
        assert conv.turn_owner == "arun"

        # Turn 1: APPROACHING -> GREETING (Arun greets Bheem)
        state1, act1, reason1 = ConversationEngine.step_conversation(conv, [], rel_store)
        assert state1 == ConvState.GREETING
        assert act1 is not None and act1.act_type == DialogueActType.GREET
        assert act1.speaker_id == "arun"
        assert conv.turn_owner == "bheem"

        # Turn 2: GREETING -> TALKING (Bheem replies)
        state2, act2, reason2 = ConversationEngine.step_conversation(conv, [], rel_store)
        assert state2 == ConvState.TALKING
        assert act2 is not None and act2.speaker_id == "bheem"
        assert conv.turn_owner == "arun"

        # Turn 3: TALKING (Arun shares news)
        talk_act = DialogueAct(
            act_type=DialogueActType.SHARE_NEWS,
            speaker_id="arun",
            target_id="bheem",
            topic="harvest",
            valence=0.4,
        )
        state3, act3, reason3 = ConversationEngine.step_conversation(
            conv, [], rel_store, chosen_act=talk_act, max_turns=4
        )
        assert state3 == ConvState.TALKING
        assert act3 == talk_act
        assert conv.turn_owner == "bheem"

        # Turn 4: TALKING -> CLOSING (Bheem bids FAREWELL reaching max turns)
        farewell_act = DialogueAct(
            act_type=DialogueActType.FAREWELL,
            speaker_id="bheem",
            target_id="arun",
            topic="closing",
        )
        state4, act4, reason4 = ConversationEngine.step_conversation(
            conv, [], rel_store, chosen_act=farewell_act, max_turns=4
        )
        assert state4 == ConvState.CLOSING

        # Turn 5: CLOSING -> IDLE
        state5, act5, reason5 = ConversationEngine.step_conversation(conv, [], rel_store)
        assert state5 == ConvState.IDLE


class TestEngagementDynamics:
    """Mathematical verification of engagement updates and social snub penalties."""

    def test_engagement_formula_hand_fixtures(self) -> None:
        """Verify hand-computed engagement dynamics:

        E(t+1) = clip(E(t) + 0.15*r + 0.10*aff - 0.08 - 0.10*boredom, 0, 1)
        """
        # Baseline: E=0.70, r=0.6, aff=0.8, turns=0 -> boredom=0
        # delta = 0.15(0.6) + 0.10(0.8) - 0.08 - 0.0 = 0.09 + 0.08 - 0.08 = +0.09
        e_next = compute_engagement(0.70, topic_relevance=0.6, affinity=0.8, turns=0)
        assert e_next == pytest.approx(0.79, abs=1e-4)

        # High boredom: turns=5 -> boredom = min(1.0, 5*0.15) = 0.75
        # delta = 0.15(0.5) + 0.10(0.5) - 0.08 - 0.10(0.75) = 0.075 + 0.050 - 0.080 - 0.075 = -0.030
        e_bored = compute_engagement(0.30, topic_relevance=0.5, affinity=0.5, turns=5)
        assert e_bored == pytest.approx(0.27, abs=1e-4)

    def test_snub_abandonment_penalty(self) -> None:
        """When an agent walks away mid-conversation (engagement < 0.20), a relationship penalty is applied."""
        rel_store = VillageRelationshipStore()
        rel_store.record_interaction("arun", "bheem", quality=0.0)
        initial_aff = rel_store.get_affinity("bheem", "arun")
        initial_trust = rel_store.get_trust("bheem", "arun")

        conv = Conversation(
            conv_id="conv_snub",
            participants=("arun", "bheem"),
            state=ConvState.TALKING,
            turn_owner="arun",
            turns=2,
            engagement={"arun": 0.10, "bheem": 0.60},  # Arun is completely disengaged
            started_step=5,
        )

        state, act, reason = ConversationEngine.step_conversation(conv, [], rel_store)
        assert state == ConvState.ABANDONED
        assert conv.abandoned_by == "arun"

        # Settle abandonment
        settlement = ConversationEngine.settle_conversation(conv, rel_store, step=6)
        assert settlement["outcome"] == "abandoned_snub"
        assert settlement["victim"] == "bheem"

        # Bheem's affinity and trust toward Arun dropped due to the snub
        assert rel_store.get_affinity("bheem", "arun") < initial_aff
        assert rel_store.get_trust("bheem", "arun") < initial_trust


class TestEmergencyInterruption:
    """Verification of alert-driven conversational interruption without snub penalty."""

    def test_storm_hazard_emergency_bypass(self) -> None:
        """A sudden thunderstorm alert interrupts conversation immediately with zero snub penalty."""
        rel_store = VillageRelationshipStore()
        conv = Conversation(
            conv_id="conv_storm",
            participants=("arun", "bheem"),
            state=ConvState.TALKING,
            turn_owner="arun",
            turns=2,
            engagement={"arun": 0.80, "bheem": 0.80},
            started_step=5,
        )

        ev = WorldEvent(
            event_id="ev_thunder",
            step=6,
            kind="THUNDER",
            modality=Modality.ENVIRONMENTAL,
            position=Position(x=0, y=0),
            intensity=1.0,
            is_global=True,
        )
        alert = Alert(
            percept=PerceivedEvent(event=ev, distance=0.0),
            salience=1.0,
            threat=0.90,
            eligible_step=6,
        )

        state, act, reason = ConversationEngine.step_conversation(conv, [alert], rel_store)
        assert state == ConvState.INTERRUPTED
        assert conv.interrupted_by_alert is True

        # Settle emergency interruption
        settlement = ConversationEngine.settle_conversation(conv, rel_store, step=6)
        assert settlement["outcome"] == "interrupted_emergency"
        assert settlement["affinity_delta"] == 0.0
