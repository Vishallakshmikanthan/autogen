"""Big Five Personality Model and Parameter Modulation for AUTOGEN Villagers.

Maps Big Five personality traits (OCEAN in [0, 1]) to utility weights,
affective baseline, salience gains, switching cost scale, and social propensities
as defined in Module M22-H.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from autogen.intelligence.village.utility_ethics import VillagerUtilityWeights
from autogen.simulation.environment.village.village_grid import VillageRole


class Traits(BaseModel):
    """Big Five (OCEAN) personality trait dimensions, normalized to [0.0, 1.0]."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    openness: float = Field(default=0.50, ge=0.0, le=1.0, description="Curiosity and receptivity to novelty")
    conscientiousness: float = Field(default=0.50, ge=0.0, le=1.0, description="Commitment, routine adherence, impulse control")
    extraversion: float = Field(default=0.50, ge=0.0, le=1.0, description="Social orientation, enthusiasm, expressiveness")
    agreeableness: float = Field(default=0.50, ge=0.0, le=1.0, description="Cooperation, empathy, trust, and forgiveness")
    neuroticism: float = Field(default=0.50, ge=0.0, le=1.0, description="Emotional reactivity, threat sensitivity, anxiety")

    @property
    def openness_inverse(self) -> float:
        """Inverse openness (1 - O), used in rumor skepticism."""
        return float(np.clip(1.0 - self.openness, 0.0, 1.0))


class PersonalityParams(BaseModel):
    """Derived cognitive, affective, and behavioral parameters modulated by Big Five traits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    salience_threat_gain: float = Field(ge=0.5, le=2.5, description="Threat salience multiplier (1 + 0.5 * N)")
    salience_novel_gain: float = Field(ge=0.5, le=2.5, description="Novelty salience multiplier (1 + 0.5 * O)")
    empathy_gain: float = Field(ge=0.5, le=2.5, description="Empathy/distress salience multiplier (1 + 0.5 * A)")
    switching_cost_scale: float = Field(ge=0.5, le=2.0, description="Commitment stickiness multiplier (1 + 0.4 * C)")
    contagion_gamma: float = Field(ge=0.01, le=0.5, description="Emotional contagion susceptibility (gamma)")
    grudge_decay: float = Field(ge=0.90, le=0.9999, description="Per-tick retention factor for grudges")
    talkativeness: float = Field(ge=0.0, le=1.5, description="Propensity to initiate conversation (0.2 + 0.7 * E)")
    pad_baseline: tuple[float, float, float] = Field(description="Trait-derived emotional baseline (v0, a0, d0)")


ROLE_ECON_MULTIPLIERS: dict[VillageRole, float] = {
    VillageRole.MERCHANT: 1.5,
    VillageRole.FARMER: 1.2,
    VillageRole.MILLER: 1.2,
    VillageRole.BLACKSMITH: 1.1,
    VillageRole.WATER_CARRIER: 1.0,
    VillageRole.WOODCUTTER: 1.0,
    VillageRole.HERBALIST: 1.0,
    VillageRole.MAGISTRATE: 0.9,
    VillageRole.GUARD: 0.8,
    VillageRole.BARD: 0.8,
    VillageRole.WANDERER: 0.7,
    VillageRole.OUTCAST: 0.6,
}


def pad_baseline_from(t: Traits) -> tuple[float, float, float]:
    """Computes trait-derived PAD emotional baseline using Mehrabian-style mapping:

    v_0 = clip(0.12 + 0.15 * (E - 0.5) - 0.15 * (N - 0.5), -1.0, 1.0)
    a_0 = clip(0.05 + 0.10 * (O - 0.5) + 0.10 * (N - 0.5), -1.0, 1.0)
    d_0 = clip(0.10 + 0.15 * (E - 0.5) - 0.10 * (N - 0.5), -1.0, 1.0)
    """
    v = float(np.clip(0.12 + 0.15 * (t.extraversion - 0.5) - 0.15 * (t.neuroticism - 0.5), -1.0, 1.0))
    a = float(np.clip(0.05 + 0.10 * (t.openness - 0.5) + 0.10 * (t.neuroticism - 0.5), -1.0, 1.0))
    d = float(np.clip(0.10 + 0.15 * (t.extraversion - 0.5) - 0.10 * (t.neuroticism - 0.5), -1.0, 1.0))
    return (round(v, 4), round(a, 4), round(d, 4))


def weights_from_traits(
    t: Traits,
    role: VillageRole = VillageRole.FARMER,
) -> tuple[VillagerUtilityWeights, PersonalityParams]:
    """Generates utility weights and personality params from Big Five traits (§5.2, §3).

    Math specifications:
    - w_soc = 14.0 * (0.7 + 0.6 * E)
    - w_econ = 1.0 * (0.8 + 0.4 * C) * m_role
    - kappa = 18.0 * (0.8 + 0.5 * N)
    - theta_risk = clip(0.25 + 0.35*(O-.5) - 0.30*(C-.5) - 0.25*(A-.5) - 0.20*(N-.5), 0.02, 0.9)
    - w_phys = 2.2
    - gamma = 0.08 * (0.6 + 0.4 * E + 0.4 * A)
    - grudge_decay = min(0.999, max(0.95, 0.995 + 0.004 * N - 0.003 * A))
    """
    m_role = ROLE_ECON_MULTIPLIERS.get(role, 1.0)

    w_soc = float(round(14.0 * (0.7 + 0.6 * t.extraversion), 4))
    w_econ = float(round(1.0 * (0.8 + 0.4 * t.conscientiousness) * m_role, 4))
    kappa = float(round(18.0 * (0.8 + 0.5 * t.neuroticism), 4))
    theta_risk = float(
        np.clip(
            0.25
            + 0.35 * (t.openness - 0.5)
            - 0.30 * (t.conscientiousness - 0.5)
            - 0.25 * (t.agreeableness - 0.5)
            - 0.20 * (t.neuroticism - 0.5),
            0.02,
            0.90,
        )
    )

    weights = VillagerUtilityWeights(
        w_phys=2.2,
        w_econ=w_econ,
        w_soc=w_soc,
        theta_risk=round(theta_risk, 4),
        crisis_threshold_tau=20.0,
        crisis_penalty_kappa=kappa,
    )

    salience_threat_gain = float(round(1.0 + 0.5 * t.neuroticism, 4))
    salience_novel_gain = float(round(1.0 + 0.5 * t.openness, 4))
    empathy_gain = float(round(1.0 + 0.5 * t.agreeableness, 4))
    switching_cost_scale = float(round(1.0 + 0.4 * t.conscientiousness, 4))
    contagion_gamma = float(round(0.08 * (0.6 + 0.4 * t.extraversion + 0.4 * t.agreeableness), 4))
    grudge_decay = float(
        round(
            min(0.999, max(0.95, 0.995 + 0.004 * t.neuroticism - 0.003 * t.agreeableness)),
            5,
        )
    )
    talkativeness = float(round(0.2 + 0.7 * t.extraversion, 4))
    pad_baseline = pad_baseline_from(t)

    params = PersonalityParams(
        salience_threat_gain=salience_threat_gain,
        salience_novel_gain=salience_novel_gain,
        empathy_gain=empathy_gain,
        switching_cost_scale=switching_cost_scale,
        contagion_gamma=contagion_gamma,
        grudge_decay=grudge_decay,
        talkativeness=talkativeness,
        pad_baseline=pad_baseline,
    )

    return weights, params


# The 10 canonical Big Five archetype trait vectors
ARCHETYPE_TRAIT_CATALOG: dict[str, Traits] = {
    "Diligent Farmer": Traits(openness=0.30, conscientiousness=0.90, extraversion=0.40, agreeableness=0.60, neuroticism=0.30),
    "Anxious Farmer": Traits(openness=0.40, conscientiousness=0.60, extraversion=0.30, agreeableness=0.50, neuroticism=0.80),
    "Gossipy Water-Carrier": Traits(openness=0.50, conscientiousness=0.50, extraversion=0.90, agreeableness=0.50, neuroticism=0.50),
    "Stoic Water-Carrier": Traits(openness=0.30, conscientiousness=0.80, extraversion=0.20, agreeableness=0.60, neuroticism=0.20),
    "Shrewd Merchant": Traits(openness=0.70, conscientiousness=0.70, extraversion=0.70, agreeableness=0.30, neuroticism=0.40),
    "Generous Merchant": Traits(openness=0.60, conscientiousness=0.50, extraversion=0.80, agreeableness=0.90, neuroticism=0.40),
    "Stern Guard": Traits(openness=0.20, conscientiousness=0.90, extraversion=0.40, agreeableness=0.30, neuroticism=0.40),
    "Lenient Guard": Traits(openness=0.50, conscientiousness=0.60, extraversion=0.60, agreeableness=0.80, neuroticism=0.30),
    "Curious Wanderer": Traits(openness=0.95, conscientiousness=0.30, extraversion=0.70, agreeableness=0.60, neuroticism=0.30),
    "Drifter Wanderer": Traits(openness=0.60, conscientiousness=0.20, extraversion=0.30, agreeableness=0.30, neuroticism=0.70),
}

# Mapping canonical agent IDs to personality traits
CANONICAL_AGENT_TRAITS: dict[str, Traits] = {
    "villager_arun": ARCHETYPE_TRAIT_CATALOG["Diligent Farmer"],
    "villager_meera": ARCHETYPE_TRAIT_CATALOG["Anxious Farmer"],
    "villager_ravi": ARCHETYPE_TRAIT_CATALOG["Stoic Water-Carrier"],
    "villager_kavya": ARCHETYPE_TRAIT_CATALOG["Gossipy Water-Carrier"],
    "villager_imran": ARCHETYPE_TRAIT_CATALOG["Generous Merchant"],
    "villager_lata": ARCHETYPE_TRAIT_CATALOG["Lenient Guard"],
    "villager_bheem": ARCHETYPE_TRAIT_CATALOG["Shrewd Merchant"],
    "villager_tara": ARCHETYPE_TRAIT_CATALOG["Stern Guard"],
    "villager_dev": ARCHETYPE_TRAIT_CATALOG["Lenient Guard"],
    "villager_nila": ARCHETYPE_TRAIT_CATALOG["Curious Wanderer"],
}

