"""Unit and oracle tests for Utility-Grounded Dialogue Act Selection."""

import pytest

from autogen.intelligence.village.conversation import (
    DialogueAct,
    DialogueActType,
    compute_dialogue_act_utility,
)
from autogen.intelligence.village.personality import Traits


class TestDialogueActUtility:
    """Mathematical verification of dialogue act utility components and risk penalties."""

    def test_hand_computed_greet_utility(self) -> None:
        """Verify hand-computed utility score for GREET act:

        u = w_soc * 0.10 * d_status + w_rel * 1.50 * d_affinity + w_info * info + w_mood * mood - cost
        """
        traits = Traits(openness=0.40, conscientiousness=0.60, extraversion=0.50, agreeableness=0.80, neuroticism=0.20)
        # For GREET: d_status = 0.01, d_affinity = 0.04 * (1 - 0.5) = 0.02, info = 0.0, cost = 0.01, valence = 0.0
        # w_info = 1.0 + 0.5(0.4) = 1.2
        # w_mood = 0.8 + 0.5(0.2) = 0.9
        # speaker_pad = (0.2, 0.0, 0.0) -> mood_cong = 1.0 - |0.2 - 0.0| / 2 = 0.9
        # u = (10.0 * 0.10 * 0.01) + (10.0 * 1.50 * 0.02) + (1.2 * 0.0) + (0.9 * 0.9) - 0.01
        #   = 0.01 + 0.30 + 0.0 + 0.81 - 0.01 = 1.11
        u_greet = compute_dialogue_act_utility(
            DialogueAct(
                act_type=DialogueActType.GREET,
                speaker_id="arun",
                target_id="bheem",
                valence=0.0,
            ),
            traits=traits,
            w_soc=10.0,
            w_rel=10.0,
            current_affinity=0.50,
            speaker_pad=(0.20, 0.0, 0.0),
            p_overheard=0.0,
        )
        assert u_greet == pytest.approx(1.11, abs=1e-4)

    def test_eavesdropping_risk_penalty(self) -> None:
        """Risky gossip near an eavesdropper incurs penalty proportional to P_overheard * kappa."""
        traits = Traits(openness=0.20, conscientiousness=0.80)
        # kappa = 20 * (0.5 + 0.5*0.8) * (1 - 0.5*0.2) = 20 * 0.9 * 0.9 = 16.2
        # risk_penalty = 0.35 * P_overheard * (16.2 * 0.05) = 0.35 * P_overheard * 0.81 = 0.2835 * P_overheard

        gossip_act = DialogueAct(
            act_type=DialogueActType.GOSSIP,
            speaker_id="arun",
            target_id="bheem",
            subject_id="captain_tara",
            claim="embezzled_granary",
            valence=-0.50,
            is_risky=True,
        )

        u_safe = compute_dialogue_act_utility(
            gossip_act,
            traits=traits,
            w_soc=10.0,
            w_rel=10.0,
            current_affinity=0.50,
            speaker_pad=(0.0, 0.0, 0.0),
            p_overheard=0.0,
            is_subject_high_status=True,
        )

        u_overheard = compute_dialogue_act_utility(
            gossip_act,
            traits=traits,
            w_soc=10.0,
            w_rel=10.0,
            current_affinity=0.50,
            speaker_pad=(0.0, 0.0, 0.0),
            p_overheard=1.0,
            is_subject_high_status=True,
        )

        delta = u_safe - u_overheard
        # Expected drop is exactly 0.2835
        assert delta == pytest.approx(0.2835, abs=1e-4)

    def test_mood_congruence_preference(self) -> None:
        """A distressed speaker prefers venting/complaining over cheerful small talk."""
        traits = Traits(neuroticism=0.80)
        # Distressed speaker: pleasure = -0.70
        speaker_pad = (-0.70, 0.50, -0.40)

        complain_act = DialogueAct(
            act_type=DialogueActType.COMPLAIN,
            speaker_id="meera",
            target_id="arun",
            valence=-0.40,
        )
        small_talk_act = DialogueAct(
            act_type=DialogueActType.SMALL_TALK,
            speaker_id="meera",
            target_id="arun",
            valence=0.30,
        )

        u_complain = compute_dialogue_act_utility(
            complain_act,
            traits=traits,
            w_soc=5.0,
            w_rel=5.0,
            current_affinity=0.50,
            speaker_pad=speaker_pad,
        )
        u_small_talk = compute_dialogue_act_utility(
            small_talk_act,
            traits=traits,
            w_soc=5.0,
            w_rel=5.0,
            current_affinity=0.50,
            speaker_pad=speaker_pad,
        )

        # Due to strong negative mood congruence penalty on small talk, complain has higher mood score
        assert u_complain > u_small_talk
