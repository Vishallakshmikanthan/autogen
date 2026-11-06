"""Affective Emotional Dynamics: 3D Valence-Arousal-Dominance (PAD) + OCC Appraisal Engine (§2).

Implements the continuous Affective State Vector $\\mathbf{E}_i(t) \\in [-1, 1]^3$ and discrete
emergent emotions (Joy, Grief, Jealousy, Malice, Guilt, Righteous Anger, Calm) from scratch.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class EmotionCategory(str, Enum):
    """Discrete emergent emotions derived from PAD coordinates and OCC cognitive appraisal (§2.1)."""

    JOY = "Joy / Happiness"
    GRIEF = "Sadness / Grief"
    JEALOUSY = "Jealousy / Envy"
    MALICE = "Evilness / Malice"
    GUILT = "Guilt / Remorse"
    RIGHTEOUS_ANGER = "Righteous Anger"
    CALM = "Calm / Composed"
    FEAR = "Fear / Distress"


# Canonical PAD coordinates (v, a, d) from Blueprint Table §2.1
CANONICAL_EMOTION_COORDS: dict[EmotionCategory, tuple[float, float, float]] = {
    EmotionCategory.JOY: (0.80, 0.50, 0.60),
    EmotionCategory.GRIEF: (-0.80, -0.60, -0.70),
    EmotionCategory.JEALOUSY: (-0.50, 0.60, -0.30),
    EmotionCategory.MALICE: (-0.40, 0.70, 0.80),
    EmotionCategory.GUILT: (-0.60, 0.40, -0.60),
    EmotionCategory.RIGHTEOUS_ANGER: (-0.60, 0.90, 0.70),
    EmotionCategory.CALM: (0.05, 0.00, 0.10),
    EmotionCategory.FEAR: (-0.64, 0.60, -0.43),
}

EMOTION_BEHAVIORAL_IMPACT: dict[EmotionCategory, str] = {
    EmotionCategory.JOY: "Generous tithing, friendly greetings, open market bids, bouncy stride",
    EmotionCategory.GRIEF: "Slow movement (0.5x), weeping posture, visiting chapel, withdrawn speech",
    EmotionCategory.JEALOUSY: "Rumor-spreading, boycotting trades, covert theft motivation",
    EmotionCategory.MALICE: "Hoarding commons, price gouging, sabotage temptation",
    EmotionCategory.GUILT: "Covert restitution, community service, heightened detection anxiety",
    EmotionCategory.RIGHTEOUS_ANGER: "Sounding alarm bell, escorting offender to pillory, public denunciation",
    EmotionCategory.CALM: "Balanced deliberation and steady village craftsmanship",
    EmotionCategory.FEAR: "Seeking shelter, rapid scanning, seeking protective companions, trembling posture",
}


class AffectiveStimulus(BaseModel):
    """Cognitive or environmental stimulus $\\text{Stimulus}_k(t)$ affecting PAD state (§2.2)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    stimulus_type: str = "general"
    stimulus_name: str = ""
    valence_delta: float = Field(default=0.0, ge=-2.0, le=2.0)
    arousal_delta: float = Field(default=0.0, ge=-2.0, le=2.0)
    dominance_delta: float = Field(default=0.0, ge=-2.0, le=2.0)
    delta_v: float = Field(default=0.0, ge=-2.0, le=2.0)
    delta_a: float = Field(default=0.0, ge=-2.0, le=2.0)
    delta_d: float = Field(default=0.0, ge=-2.0, le=2.0)
    weight: float = Field(default=0.35, gt=0.0, le=2.0)
    intensity: float = Field(default=1.0, gt=0.0, le=2.0)
    description: str = ""


class AffectiveStateVector(BaseModel):
    """Continuous 3D Affective State Vector $\\mathbf{E}_i(t) = [v_i(t), a_i(t), d_i(t)]^T \\in [-1, 1]^3$."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    agent_id: str = ""
    valence: float = Field(default=0.10, ge=-1.0, le=1.0, description="Pleasure vs. Pain")
    arousal: float = Field(default=0.05, ge=-1.0, le=1.0, description="Alertness vs. Lethargy")
    dominance: float = Field(default=0.10, ge=-1.0, le=1.0, description="Empowerment vs. Helplessness")
    dominant_emotion: EmotionCategory = EmotionCategory.CALM
    emotion_intensity: float = Field(default=0.15, ge=0.0, le=1.0)
    behavioral_impact: str = EMOTION_BEHAVIORAL_IMPACT[EmotionCategory.CALM]
    movement_speed_multiplier: float = Field(default=1.0, gt=0.1, le=2.0)
    aura_hex: str = "#38bdf8"

    @property
    def aura_color_hex(self) -> str:
        return self.aura_hex

    def to_tuple(self) -> tuple[float, float, float]:
        """Returns $(v_i, a_i, d_i)$ tuple."""
        return (self.valence, self.arousal, self.dominance)

    def to_vector(self) -> tuple[float, float, float]:
        """Returns $(v_i, a_i, d_i)$ tuple."""
        return (self.valence, self.arousal, self.dominance)


class AffectiveStepEvent(BaseModel):
    """Step-traceable event emitted when updating an agent's PAD emotional vector."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    step: int
    previous_pad: tuple[float, float, float]
    new_pad: tuple[float, float, float]
    dominant_emotion: EmotionCategory
    stimuli_applied: tuple[str, ...]


class AffectivePADConfig(BaseModel):
    """Hyperparameters for the PAD + OCC emotional transition model (§2.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lambda_decay: float = Field(
        default=0.88,
        ge=0.85,
        le=0.95,
        description="Emotional inertia factor in [0.85, 0.95]",
    )
    envy_wealth_gap_tau: float = Field(
        default=25.0,
        gt=0.0,
        description="Neighbor wealth gap threshold triggering Jealousy when hungry",
    )
    guilt_trust_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Trust threshold T_{j->i} > 0.7 above which betrayal triggers Guilt",
    )
    calm_radius: float = Field(
        default=0.26,
        gt=0.0,
        le=0.6,
        description="Euclidean norm below which the agent is classified as Calm",
    )
    contagion_gamma: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        description="Base emotional contagion coupling factor (§3.4)",
    )


class AffectivePADEngine:
    """From-scratch OCC Appraisal and 3D PAD Emotional Transition Engine (§2)."""

    def __init__(
        self,
        config: AffectivePADConfig | None = None,
        baseline: tuple[float, float, float] = (0.12, 0.05, 0.10),
    ) -> None:
        self.config = config or AffectivePADConfig()
        self.baseline = (
            float(np.clip(baseline[0], -1.0, 1.0)),
            float(np.clip(baseline[1], -1.0, 1.0)),
            float(np.clip(baseline[2], -1.0, 1.0)),
        )

    def classify_emotion(
        self,
        valence: float,
        arousal: float,
        dominance: float,
        override_category: EmotionCategory | None = None,
    ) -> tuple[EmotionCategory, float, str, float, str]:
        """Maps $(v, a, d) \\in [-1, 1]^3$ to the nearest discrete OCC emotion prototype."""
        norm = math.sqrt(valence * valence + arousal * arousal + dominance * dominance) / math.sqrt(3.0)
        intensity = float(min(1.0, max(0.0, round(norm, 4))))

        if override_category is not None:
            cat = override_category
        elif norm < self.config.calm_radius:
            cat = EmotionCategory.CALM
        else:
            best_cat = EmotionCategory.CALM
            best_dist = 999.0
            for candidate, (cv, ca, cd) in CANONICAL_EMOTION_COORDS.items():
                if candidate == EmotionCategory.CALM:
                    continue
                dist = math.sqrt(
                    (valence - cv) ** 2 + (arousal - ca) ** 2 + (dominance - cd) ** 2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_cat = candidate
            cat = best_cat

        speed_mult = 1.0
        aura_hex = "#38bdf8"
        if cat == EmotionCategory.JOY:
            speed_mult = 1.15
            aura_hex = "#34d399"
        elif cat == EmotionCategory.GRIEF:
            speed_mult = 0.50
            aura_hex = "#60a5fa"
        elif cat == EmotionCategory.JEALOUSY:
            speed_mult = 0.95
            aura_hex = "#a855f7"
        elif cat == EmotionCategory.MALICE:
            speed_mult = 1.10
            aura_hex = "#ef4444"
        elif cat == EmotionCategory.GUILT:
            speed_mult = 0.75
            aura_hex = "#f59e0b"
        elif cat == EmotionCategory.RIGHTEOUS_ANGER:
            speed_mult = 1.25
            aura_hex = "#f97316"
        elif cat == EmotionCategory.FEAR:
            speed_mult = 1.30
            aura_hex = "#fbbf24"

        return cat, intensity, EMOTION_BEHAVIORAL_IMPACT[cat], speed_mult, aura_hex

    def appraise_stimuli(
        self,
        *,
        hunger: float,
        hydration: float,
        coins: float,
        societal_status: float,
        theta_risk: float,
        max_neighbor_coins: float = 25.0,
        granary_wheat: float = 60.0,
        recent_fair_trade: bool = False,
        recent_tithe: bool = False,
        recent_child_or_wedding: bool = False,
        grief_active: bool = False,
        committed_crime_against_trusted: bool = False,
        witnessed_crime_as_warden: bool = False,
        heard_bard_ballad: bool = False,
        performed_redemption: bool = False,
    ) -> tuple[list[AffectiveStimulus], EmotionCategory | None]:
        """OCC cognitive appraisal mapping physiological, economic, and relational events to PAD stimuli."""
        stimuli: list[AffectiveStimulus] = []
        override: EmotionCategory | None = None

        # 1. Acute Grief from death of kinship/partner or severe status collapse (§2.1, §3.1)
        if grief_active or societal_status < 0.03:
            gv, ga, gd = CANONICAL_EMOTION_COORDS[EmotionCategory.GRIEF]
            stimuli.append(
                AffectiveStimulus(
                    stimulus_type="kin_loss_or_status_collapse",
                    valence_delta=gv,
                    arousal_delta=ga,
                    dominance_delta=gd,
                    weight=0.75 if grief_active else 0.45,
                    description="Mourning loss of kinship or acute societal standing collapse.",
                )
            )
            if grief_active:
                override = EmotionCategory.GRIEF

        # 2. Righteous Anger when catching a thief red-handed or breaking granary seal (§2.1)
        if witnessed_crime_as_warden:
            rv, ra, rd = CANONICAL_EMOTION_COORDS[EmotionCategory.RIGHTEOUS_ANGER]
            stimuli.append(
                AffectiveStimulus(
                    stimulus_type="caught_thief_red_handed",
                    valence_delta=rv,
                    arousal_delta=ra,
                    dominance_delta=rd,
                    weight=0.70,
                    description="Righteous indignation upon witnessing a granary or hut crime.",
                )
            )
            override = EmotionCategory.RIGHTEOUS_ANGER

        # 3. Guilt / Remorse after committing theft/deceit against a trusted friend (§2.1)
        if committed_crime_against_trusted:
            uv, ua, ud = CANONICAL_EMOTION_COORDS[EmotionCategory.GUILT]
            stimuli.append(
                AffectiveStimulus(
                    stimulus_type="betrayed_trusted_friend",
                    valence_delta=uv,
                    arousal_delta=ua,
                    dominance_delta=ud,
                    weight=0.68,
                    description="Acute remorse after wronging a trusted villager (T > 0.7).",
                )
            )
            override = EmotionCategory.GUILT

        # 4. Jealousy / Envy when neighbor wealth exceeds self while hungry (§2.1)
        wealth_gap = max_neighbor_coins - coins
        if hunger < 55.0 and wealth_gap > self.config.envy_wealth_gap_tau:
            jv, ja, jd = CANONICAL_EMOTION_COORDS[EmotionCategory.JEALOUSY]
            stimuli.append(
                AffectiveStimulus(
                    stimulus_type="neighbor_wealth_envy",
                    valence_delta=jv,
                    arousal_delta=ja,
                    dominance_delta=jd,
                    weight=0.55,
                    description=f"Envy triggered by wealth disparity (gap={wealth_gap:.1f}c) while hungry.",
                )
            )
            if override is None:
                override = EmotionCategory.JEALOUSY

        # 5. Evilness / Malice under sustained low trust + high risk tolerance (§2.1)
        if societal_status < 0.065 and theta_risk >= 0.42 and not committed_crime_against_trusted:
            mv, ma, md = CANONICAL_EMOTION_COORDS[EmotionCategory.MALICE]
            stimuli.append(
                AffectiveStimulus(
                    stimulus_type="opportunistic_malice",
                    valence_delta=mv,
                    arousal_delta=ma,
                    dominance_delta=md,
                    weight=0.52,
                    description="Sustained low trust and high risk tolerance fuel malicious opportunism.",
                )
            )
            if override is None:
                override = EmotionCategory.MALICE

        # 6. Joy / Happiness from fair trade, granary abundance, birth/wedding, or bard ballad (§2.1)
        if (
            recent_fair_trade
            or recent_tithe
            or recent_child_or_wedding
            or heard_bard_ballad
            or performed_redemption
            or (granary_wheat >= 70.0 and hunger >= 75.0 and hydration >= 75.0)
        ):
            jv, ja, jd = CANONICAL_EMOTION_COORDS[EmotionCategory.JOY]
            w_joy = 0.60 if recent_child_or_wedding else 0.38
            stimuli.append(
                AffectiveStimulus(
                    stimulus_type="prosperity_or_communal_joy",
                    valence_delta=jv,
                    arousal_delta=ja,
                    dominance_delta=jd,
                    weight=w_joy,
                    description="Positive affect from fair trade, communal tithing, ballad, or family milestone.",
                )
            )

        return stimuli, override

    def step_emotion(
        self,
        agent_id: str,
        step: int,
        current_state: AffectiveStateVector,
        stimuli: list[AffectiveStimulus],
        override_category: EmotionCategory | None = None,
        step_callback: Callable[[AffectiveStepEvent], None] | None = None,
    ) -> AffectiveStateVector:
        """Executes the exact PAD mathematical transition equation (§2.2):

        $$\\mathbf{E}_i(t+1) = \\lambda_{\\text{decay}} \\mathbf{E}_i(t) + (1 - \\lambda_{\\text{decay}}) \\mathbf{E}_{\\text{baseline}} + \\sum_{k} \\mathbf{w}_k \\cdot \\text{Stimulus}_k(t)$$
        """
        lam = self.config.lambda_decay
        prev = current_state.to_tuple()

        v_next = lam * prev[0] + (1.0 - lam) * self.baseline[0]
        a_next = lam * prev[1] + (1.0 - lam) * self.baseline[1]
        d_next = lam * prev[2] + (1.0 - lam) * self.baseline[2]

        applied_names: list[str] = []
        for stim in stimuli:
            dv = stim.delta_v if stim.delta_v != 0.0 else stim.valence_delta
            da = stim.delta_a if stim.delta_a != 0.0 else stim.arousal_delta
            dd = stim.delta_d if stim.delta_d != 0.0 else stim.dominance_delta
            w = stim.weight if stim.weight != 0.35 else stim.intensity
            v_next += w * dv
            a_next += w * da
            d_next += w * dd
            applied_names.append(stim.stimulus_name or stim.stimulus_type)

        v_clamped = float(np.clip(v_next, -1.0, 1.0))
        a_clamped = float(np.clip(a_next, -1.0, 1.0))
        d_clamped = float(np.clip(d_next, -1.0, 1.0))

        cat, intensity, impact, speed_mult, aura_hex = self.classify_emotion(
            v_clamped, a_clamped, d_clamped, override_category=override_category
        )

        next_vec = AffectiveStateVector(
            agent_id=agent_id,
            valence=round(v_clamped, 4),
            arousal=round(a_clamped, 4),
            dominance=round(d_clamped, 4),
            dominant_emotion=cat,
            emotion_intensity=round(intensity, 4),
            behavioral_impact=impact,
            movement_speed_multiplier=round(speed_mult, 3),
            aura_hex=aura_hex,
        )

        if step_callback is not None:
            step_callback(
                AffectiveStepEvent(
                    agent_id=agent_id,
                    step=step,
                    previous_pad=prev,
                    new_pad=next_vec.to_tuple(),
                    dominant_emotion=cat,
                    stimuli_applied=tuple(applied_names),
                )
            )

        return next_vec

    def appraise_observation(
        self,
        *,
        agent_id: str,
        hunger: float,
        hydration: float,
        energy: float,
        coins: float,
        societal_status: float,
        in_pillory: bool = False,
        witnessed_categories: list[tuple[str, str, str | None, bool]] | None = None,
        max_visible_peer_status: float = 0.1,
    ) -> list[AffectiveStimulus]:
        """Appraises sensory observations into AffectiveStimulus instances."""
        stimuli: list[AffectiveStimulus] = []
        if in_pillory:
            stimuli.append(
                AffectiveStimulus(
                    stimulus_type="pillory_humiliation",
                    valence_delta=-0.80,
                    arousal_delta=-0.50,
                    dominance_delta=-0.80,
                    weight=0.70,
                    description="Severe public humiliation and remorse while bound to pillory.",
                )
            )
        if hunger < 30.0 or hydration < 30.0:
            stimuli.append(
                AffectiveStimulus(
                    stimulus_type="physiological_distress",
                    valence_delta=-0.50,
                    arousal_delta=0.40,
                    dominance_delta=-0.40,
                    weight=0.45,
                    description="Acute hunger and thirst impair emotional composure.",
                )
            )
        elif hunger > 75.0 and hydration > 75.0:
            stimuli.append(
                AffectiveStimulus(
                    stimulus_type="physiological_contentment",
                    valence_delta=0.40,
                    arousal_delta=0.15,
                    dominance_delta=0.25,
                    weight=0.35,
                    description="Nourished state produces calm satisfaction and social warmth.",
                )
            )

        if max_visible_peer_status - societal_status > 0.35 and hunger < 60.0:
            stimuli.append(
                AffectiveStimulus(
                    stimulus_type="status_gap_jealousy",
                    valence_delta=-0.45,
                    arousal_delta=0.55,
                    dominance_delta=-0.25,
                    weight=0.40,
                    description="Observing highly esteemed peers stirs bitter envy when struggling.",
                )
            )

        if witnessed_categories:
            for cat, _actor, _target, _detected in witnessed_categories:
                if cat == "coop":
                    stimuli.append(
                        AffectiveStimulus(
                            stimulus_type="witnessed_cooperation",
                            valence_delta=0.35,
                            arousal_delta=0.20,
                            dominance_delta=0.25,
                            weight=0.30,
                            description="Witnessing prosocial cooperation elevates community trust and joy.",
                        )
                    )
                elif cat in ("defect", "crime_caught"):
                    stimuli.append(
                        AffectiveStimulus(
                            stimulus_type="witnessed_crime",
                            valence_delta=-0.35,
                            arousal_delta=0.50,
                            dominance_delta=0.30,
                            weight=0.40,
                            description="Witnessing crime triggers righteous indignation and alarm.",
                        )
                    )
        return stimuli

    def step_pad_state(
        self,
        current_state: AffectiveStateVector | None = None,
        stimuli: list[AffectiveStimulus] | None = None,
        step: int = 0,
        agent_id: str = "agent",
        current: AffectiveStateVector | None = None,
    ) -> AffectiveStateVector:
        """Alias for step_emotion convenience."""
        st = current if current is not None else (current_state or AffectiveStateVector())
        stim_list = stimuli if stimuli is not None else []
        return self.step_emotion(
            agent_id=agent_id,
            step=step,
            current_state=st,
            stimuli=stim_list,
        )

    def compute_utility_modifiers(
        self, affective_state: AffectiveStateVector
    ) -> dict[str, float]:
        """Returns multiplicative or additive utility modifiers driven by $\\mathbf{E}_i(t)$."""
        cat = affective_state.dominant_emotion
        v, a, d = affective_state.valence, affective_state.arousal, affective_state.dominance

        coop_bonus = max(0.0, 0.35 * v)
        theft_temptation = 0.0
        redemption_drive = 0.0
        enforcement_drive = 0.0

        if cat == EmotionCategory.JOY:
            coop_bonus += 0.45
        elif cat == EmotionCategory.JEALOUSY:
            theft_temptation += 0.65 * max(0.2, a)
        elif cat == EmotionCategory.MALICE:
            theft_temptation += 0.85 * max(0.2, d)
        elif cat == EmotionCategory.GUILT:
            redemption_drive += 1.25
            theft_temptation -= 0.80
        elif cat == EmotionCategory.RIGHTEOUS_ANGER:
            enforcement_drive += 1.40
        elif cat == EmotionCategory.GRIEF:
            coop_bonus += 0.25  # Altruistic bonding during mourning (§3.1)
        elif cat == EmotionCategory.FEAR:
            theft_temptation -= 0.50

        return {
            "coop_bonus": round(coop_bonus, 4),
            "theft_temptation": round(theft_temptation, 4),
            "redemption_drive": round(redemption_drive, 4),
            "enforcement_drive": round(enforcement_drive, 4),
        }

