"""Mode 1: Deterministic Symbolic Dialogue & Grammar-Tree Narrative Engine (§5.1)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from autogen.core.types import DecisionTrace


class DialogueEventType(str, Enum):
    """Categorical types of narrative and social utterances in AUTOGEN Village."""

    SPEECH = "SpeechEvent"
    GOSSIP = "GossipEvent"
    PUBLIC_DECLARATION = "PublicDeclaration"
    DIARY_ENTRY = "DiaryEntry"


class VillageDialogueEvent(BaseModel):
    """Immutable record of a spoken utterance, gossip, or personal diary entry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(ge=0)
    speaker_id: str
    speaker_name: str
    speaker_role: str = "Villager"
    recipient_id: str | None = None
    recipient_name: str | None = None
    subject_id: str | None = None
    event_type: DialogueEventType = DialogueEventType.SPEECH
    utterance: str
    action_chosen: str
    is_unethical: bool = False
    mode: str = "symbolic"
    position: tuple[int, int] = (20, 20)
    trust_score: float = Field(default=0.5, ge=0.0, le=1.0)
    emotion: str = "Calm / Composed"
    icon: str = "💬"


class SymbolicDialogueEngine:
    """100% offline deterministic grammar-tree dialogue generator (§5.1).

    Translates an agent's mathematical DecisionTrace, internal physiological/inventory
    state, and trust graph relationships into structured natural-language utterances.
    """

    def generate_utterance(
        self,
        trace: DecisionTrace,
        speaker_name: str,
        speaker_role: str,
        position: tuple[int, int],
        hunger: float,
        hydration: float,
        coins: float,
        wheat_qty: float,
        recipient_id: str | None = None,
        recipient_name: str | None = None,
        trust_toward_recipient: float = 0.5,
        witnessed_offender_id: str | None = None,
        witnessed_offender_name: str | None = None,
    ) -> VillageDialogueEvent:
        """Generates a deterministic symbolic utterance grounded in actual agent state."""
        rationale: dict[str, Any] = trace.rationale
        v_act = str(rationale.get("village_action", "STAY"))
        is_unethical = bool(rationale.get("is_unethical", False))
        p_detect = float(rationale.get("detection_probability", 0.0))
        u_total = float(rationale.get("expected_total_utility", 0.0))
        emotion = str(rationale.get("dominant_emotion", "Calm / Composed"))

        # 1. If the agent recently witnessed a crime or defection, emit GossipEvent
        if witnessed_offender_id and witnessed_offender_id != trace.agent_id and recipient_id:
            off_name = witnessed_offender_name or witnessed_offender_id
            utterance = (
                f"⚔️ I saw {off_name} lurking near the Granary and acting dishonestly. "
                f"Guard your chests and alert the Gram Sabha!"
            )
            return VillageDialogueEvent(
                step=trace.step_id,
                speaker_id=trace.agent_id,
                speaker_name=speaker_name,
                speaker_role=speaker_role,
                recipient_id=recipient_id,
                recipient_name=recipient_name,
                subject_id=witnessed_offender_id,
                event_type=DialogueEventType.GOSSIP,
                utterance=utterance,
                action_chosen=v_act,
                is_unethical=is_unethical,
                mode="symbolic",
                position=position,
                trust_score=round(trust_toward_recipient, 3),
                emotion=emotion,
                icon="⚔️",
            )

        # 2. Deceptive / Hoarding speech
        if v_act in ("HOARD_AND_GOUGE", "FREE_RIDE_COMMONS"):
            utterance = (
                "⚔️ My shelves are bare, friend. I cannot spare any grain at normal prices."
                if wheat_qty >= 2.0
                else "⚔️ Times are harsh; everyone must look after their own storehouse first."
            )
            return VillageDialogueEvent(
                step=trace.step_id,
                speaker_id=trace.agent_id,
                speaker_name=speaker_name,
                speaker_role=speaker_role,
                recipient_id=recipient_id,
                recipient_name=recipient_name,
                event_type=DialogueEventType.SPEECH,
                utterance=utterance,
                action_chosen=v_act,
                is_unethical=True,
                mode="symbolic",
                position=position,
                trust_score=round(trust_toward_recipient, 3),
                emotion=emotion,
                icon="⚔️",
            )

        # 3. Covert theft / sabotage inner diary entry
        if v_act in ("STEAL_GRANARY", "STEAL_HUT_CHEST", "SABOTAGE_GRANARY", "OFFER_BRIBE"):
            utterance = (
                f"⚔️ [Diary] Hunger={hunger:.0f}, P_detect={p_detect:.2f}, E[U]={u_total:.2f}. "
                f"Survival outweighs the sanction risk—moving in the shadows."
            )
            return VillageDialogueEvent(
                step=trace.step_id,
                speaker_id=trace.agent_id,
                speaker_name=speaker_name,
                speaker_role=speaker_role,
                event_type=DialogueEventType.DIARY_ENTRY,
                utterance=utterance,
                action_chosen=v_act,
                is_unethical=True,
                mode="symbolic",
                position=position,
                trust_score=round(trust_toward_recipient, 3),
                emotion=emotion,
                icon="⚔️",
            )

        # 4. Public philanthropy / Granary tithe declaration
        if v_act == "TITHE_GRANARY":
            utterance = (
                "🌾 Fellow villagers, I tithe my surplus wheat to the Granary Commons "
                "so none among us shall starve."
            )
            return VillageDialogueEvent(
                step=trace.step_id,
                speaker_id=trace.agent_id,
                speaker_name=speaker_name,
                speaker_role=speaker_role,
                recipient_id=recipient_id,
                recipient_name=recipient_name,
                event_type=DialogueEventType.PUBLIC_DECLARATION,
                utterance=utterance,
                action_chosen=v_act,
                is_unethical=False,
                mode="symbolic",
                position=position,
                trust_score=round(trust_toward_recipient, 3),
                emotion=emotion,
                icon="🌾",
            )

        # 5. Friendly Market Trade
        if v_act == "TRADE_MARKET":
            item_offered = "wheat" if wheat_qty > 2.0 else "coin"
            item_sought = "water and tools" if wheat_qty > 2.0 else "fresh grain"
            tone = "trusted friend" if trust_toward_recipient >= 0.6 else "neighbor"
            utterance = (
                f"🌾 Hail, {tone}! I have surplus {item_offered}. "
                f"Would you trade for {item_sought} at fair clearing value?"
            )
            return VillageDialogueEvent(
                step=trace.step_id,
                speaker_id=trace.agent_id,
                speaker_name=speaker_name,
                speaker_role=speaker_role,
                recipient_id=recipient_id,
                recipient_name=recipient_name,
                event_type=DialogueEventType.SPEECH,
                utterance=utterance,
                action_chosen=v_act,
                is_unethical=False,
                mode="symbolic",
                position=position,
                trust_score=round(trust_toward_recipient, 3),
                emotion=emotion,
                icon="🌾",
            )

        # 6. Guard patrol or investigation declaration
        if v_act in ("PATROL_GUARD", "INVESTIGATE_CRIME"):
            utterance = (
                "⚖️ Inspecting footprints near the Granary lock—no crime escapes the Watch!"
                if v_act == "INVESTIGATE_CRIME"
                else "⚖️ Keeping watch over the square and Granary. Honest folk have nothing to fear."
            )
            return VillageDialogueEvent(
                step=trace.step_id,
                speaker_id=trace.agent_id,
                speaker_name=speaker_name,
                speaker_role=speaker_role,
                recipient_id=recipient_id,
                recipient_name=recipient_name,
                event_type=DialogueEventType.PUBLIC_DECLARATION,
                utterance=utterance,
                action_chosen=v_act,
                is_unethical=False,
                mode="symbolic",
                position=position,
                trust_score=round(trust_toward_recipient, 3),
                emotion=emotion,
                icon="⚖️",
            )

        # 7. Gram Sabha & specialized guild actions
        recip_greeting = f"{recipient_name}, " if recipient_name and recipient_name != "Self" else ""
        icon_sym = "📜"
        if v_act == "PRESIDE_SABHA":
            utterance = "⚖️ Citizens of the Gram Sabha! Hear the State of the Village and cast your trust-weighted votes!"
            icon_sym = "⚖️"
        elif v_act == "ATTEND_SABHA":
            utterance = f"⚖️ {recip_greeting}I stand in the Town Square council circle to vote on our village laws."
            icon_sym = "⚖️"
        elif v_act == "MILL_WHEAT":
            utterance = f"🌾 {recip_greeting}grinding golden wheat into fine flour and warm loaves at the Old Windmill!"
            icon_sym = "🌾"
        elif v_act == "FORGE_TOOLS":
            utterance = f"🔨 {recip_greeting}striking hot iron on the Smithy anvil to forge tools for our farmers!"
            icon_sym = "🔨"
        elif v_act == "BREW_HEALING_BROTH":
            utterance = f"🌿 {recip_greeting}steeping wild forest herbs into restorative broth for the weary."
            icon_sym = "🌿"
        elif v_act == "SING_BALLAD":
            utterance = f"🎶 {recip_greeting}hark! My lute sings of the harvest, the Watch, and the honors of our village!"
            icon_sym = "🎶"
        elif v_act == "COMMUNITY_SERVICE":
            utterance = f"🤝 {recip_greeting}I labor for the Granary Commons to make amends and earn back your trust."
            icon_sym = "🤝"
        elif v_act == "COURTSHIP_BOND":
            utterance = f"❤️ {recip_greeting}our household bond grows strong—may our lineage bless this village!"
            icon_sym = "❤️"
        elif v_act == "MOURN_AT_CHAPEL":
            utterance = "🕯️ Lighting a quiet candle at St. Aurelia Chapel in memory of our ancestors."
            icon_sym = "🕯️"
        elif v_act == "DRAW_WATER":
            utterance = f"💧 {recip_greeting}drawing fresh, cool water from the town well."
        elif v_act == "HARVEST_FIELD":
            utterance = f"🌾 {recip_greeting}the golden wheat stalks are ripe and ready for harvest."
            icon_sym = "🌾"
        elif v_act == "FORAGE_FOREST":
            utterance = f"🌲 {recip_greeting}gathering wild herbs and fallen firewood along the tree line."
        elif v_act == "CLAIM_FAMINE_RELIEF":
            utterance = "🌾 Emergency times—requesting grain relief from the Village Granary."
            icon_sym = "🌾"
        elif v_act == "REST_HUT":
            utterance = f"🏠 {recip_greeting}resting my weary limbs and locking my chest for the night."
        else:
            goal_raw = str(rationale.get("target_zone", "town_square"))
            goal_clean = goal_raw.replace("_", " ").title()
            if recip_greeting:
                utterance = f"📜 {recip_greeting}on my way toward {goal_clean}—peace be upon your day!"
            else:
                utterance = f"📜 Walking toward {goal_clean} to see to the village work."

        return VillageDialogueEvent(
            step=trace.step_id,
            speaker_id=trace.agent_id,
            speaker_name=speaker_name,
            speaker_role=speaker_role,
            recipient_id=recipient_id,
            recipient_name=recipient_name,
            event_type=DialogueEventType.SPEECH,
            utterance=utterance,
            action_chosen=v_act,
            is_unethical=is_unethical,
            mode="symbolic",
            position=position,
            trust_score=round(trust_toward_recipient, 3),
            emotion=emotion,
            icon=icon_sym,
        )
