"""Demographics & Lifecycle (§3), Town Assemblies / Gram Sabha (§5), and Justice & Redemption (§6).

Implements generational genetics, mortality/mourning, trust-weighted council trials,
forensic evidence investigations, bribery dynamics, and PageRank status rehabilitation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from autogen.intelligence.village.social_graph import VillageTrustGraph
from autogen.intelligence.village.utility_ethics import VillagerUtilityWeights


class LifeStage(str, Enum):
    """Biological lifecycle stages of a villager (§3)."""

    CHILD = "Child"
    ADULT = "Adult"
    ELDER = "Elder"
    DECEASED = "Deceased"


class LineageRecord(BaseModel):
    """Demographic, kinship, and aging state of a villager (§3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    model_config = ConfigDict(frozen=True, extra="ignore")

    agent_id: str
    name: str = ""
    generation: int = Field(default=1, ge=1)
    age_ticks: int = Field(default=120, ge=0)
    life_stage: LifeStage = LifeStage.ADULT
    max_energy_cap: float = Field(default=100.0, ge=10.0, le=100.0)
    starvation_streak: int = Field(default=0, ge=0)
    parent_ids: tuple[str, ...] = ()
    partner_id: str | None = None
    children_ids: tuple[str, ...] = ()
    grief_ticks_remaining: int = Field(default=0, ge=0)
    inherited_traditions: tuple[str, ...] = ()

    @property
    def spouse_id(self) -> str | None:
        return self.partner_id


class HouseholdPartnership(BaseModel):
    """Record of a married/bonded household partnership (§3.2)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    partnership_id: str
    partner_a: str = ""
    partner_b: str = ""
    partner_a_id: str = ""
    partner_b_id: str = ""
    hut_id: str = ""
    shared_hut_id: str = ""
    formed_step: int = 0
    formed_at_step: int = 0
    gestation_ticks: int = Field(default=0, ge=0)
    children_born: tuple[str, ...] = ()
    mutual_trust: float = 0.80
    combined_grain_reserve: float = 20.0


class ChildBirthRecord(BaseModel):
    """Record of a newly born villager with genetically inherited utility weights (§3.2)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    step: int
    child_id: str
    child_name: str
    parent_a: str
    parent_b: str
    hut_id: str = ""
    generation: int = 1
    inherited_weights: VillagerUtilityWeights = Field(default_factory=VillagerUtilityWeights)
    inherited_trust_score: float = 0.50
    inherited_traditions: tuple[str, ...] = ()
    inherited_theta_risk: float = 0.25
    inherited_w_soc: float = 12.0
    inherited_w_econ: float = 1.0


class FuneralRecord(BaseModel):
    """Record of a villager's death, inheritance distribution, and Chapel mourning (§3.1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int
    deceased_id: str
    deceased_name: str
    cause_of_death: str
    age_at_death: int
    heir_id: str | None = None
    coins_inherited: float = 0.0
    moved_to_commons: bool = False
    mourners: tuple[str, ...] = ()


class LifecycleConfig(BaseModel):
    """Parameters governing aging, mortality, courtship, and genetic inheritance (§3)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    seed: int = 42
    child_maturation_ticks: int = Field(default=40, ge=5)
    elder_threshold_ticks: int = Field(default=1000, ge=100, description="T_elder in §3.1")
    max_lifespan_span_ticks: int = Field(default=1200, ge=200, description="T_max in §3.1")
    fatal_starvation_ticks: int = Field(default=8, ge=2, description="N_fatal consecutive zero-vital ticks")
    grief_duration_ticks: int = Field(default=48, ge=4, description="48 ticks of acute Grief (§3.1)")
    courtship_trust_threshold: float = Field(default=0.75, ge=0.5, le=1.0, description="T_{i<->j} > 0.75")
    courtship_grain_reserve_min: float = Field(default=20.0, ge=1.0, description="Combined grain > 20 units")
    gestation_interval_ticks: int = Field(default=8, ge=2)
    genetic_mutation_sigma: float = Field(default=0.04, ge=0.0, le=0.25, description="sigma in §3.2")
    max_village_population: int = Field(default=16, ge=10, le=30)


class LifecycleEngine:
    """From-scratch Biological Lifecycle, Aging, Mortality, and Genetic Inheritance Engine (§3)."""

    CHILD_NAMES: tuple[str, ...] = (
        "Asha",
        "Kiran",
        "Veda",
        "Rohan",
        "Surya",
        "Diya",
        "Kavi",
        "Nayan",
    )

    def __init__(self, config: LifecycleConfig | None = None, seed: int | None = None) -> None:
        if config is not None:
            self.config = config
        elif seed is not None:
            self.config = LifecycleConfig(seed=seed)
        else:
            self.config = LifecycleConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.lineages: dict[str, LineageRecord] = {}
        self.partnerships: dict[str, HouseholdPartnership] = {}
        self.births: list[ChildBirthRecord] = []
        self.funerals: list[FuneralRecord] = []

    def initialize_canonical_lineage(
        self, villagers: list[tuple[str, str]]
    ) -> dict[str, LineageRecord]:
        """Initializes canonical lineage records for a cohort of villagers."""
        for aid, name in villagers:
            rec = self.init_villager_lineage(agent_id=aid)
            self.lineages[aid] = rec.model_copy(update={"name": name})
        return dict(self.lineages)

    def init_villager_lineage(
        self,
        agent_id: str,
        initial_age: int = 160,
        generation: int = 1,
        parent_ids: tuple[str, ...] = (),
        traditions: tuple[str, ...] = (),
    ) -> LineageRecord:
        """Initializes or resets the demographic lineage record for an agent."""
        stage = self.compute_life_stage(initial_age, is_alive=True)
        max_e = self.compute_max_energy(initial_age)
        rec = LineageRecord(
            agent_id=agent_id,
            generation=generation,
            age_ticks=initial_age,
            life_stage=stage,
            max_energy_cap=max_e,
            parent_ids=parent_ids,
            inherited_traditions=traditions,
        )
        self.lineages[agent_id] = rec
        return rec

    def compute_max_energy(self, age_ticks: int, base_energy: float = 100.0) -> float:
        """Computes natural stamina decay beyond $T_{\\text{elder}}$ (§3.1):

        $$\\text{Energy}_{\\text{max}}(age) = \\text{Base} \\times \\left(1 - \\frac{age - T_{\\text{elder}}}{T_{\\text{max}}}\\right)$$
        """
        if age_ticks <= self.config.elder_threshold_ticks:
            return float(base_energy)
        excess = float(age_ticks - self.config.elder_threshold_ticks)
        ratio = excess / float(self.config.max_lifespan_span_ticks)
        decayed = base_energy * (1.0 - ratio)
        return float(max(20.0, min(base_energy, round(decayed, 4))))

    def compute_life_stage(self, age_ticks: int, is_alive: bool = True) -> LifeStage:
        """Determines biological life stage from age in ticks."""
        if not is_alive:
            return LifeStage.DECEASED
        if age_ticks < self.config.child_maturation_ticks:
            return LifeStage.CHILD
        if age_ticks >= self.config.elder_threshold_ticks:
            return LifeStage.ELDER
        return LifeStage.ADULT

    def synthesize_child_genetics(
        self,
        weights_a: VillagerUtilityWeights,
        weights_b: VillagerUtilityWeights,
        rng: np.random.Generator,
    ) -> VillagerUtilityWeights:
        """Computes genetically inherited utility weights and risk tolerance (§3.2):

        $$\\theta_{\\text{risk}}^{(\\text{child})} = \\frac{\\theta_{\\text{risk}}^{(A)} + \\theta_{\\text{risk}}^{(B)}}{2} + \\mathcal{N}(0, \\sigma^2)$$
        """
        sigma = self.config.genetic_mutation_sigma
        noise_theta = float(rng.normal(0.0, sigma))
        noise_phys = float(rng.normal(0.0, sigma * 2.0))
        noise_econ = float(rng.normal(0.0, sigma * 1.5))
        noise_soc = float(rng.normal(0.0, sigma * 8.0))

        child_theta = float(
            np.clip(0.5 * (weights_a.theta_risk + weights_b.theta_risk) + noise_theta, 0.05, 0.95)
        )
        child_phys = float(
            np.clip(0.5 * (weights_a.w_phys + weights_b.w_phys) + noise_phys, 1.2, 4.0)
        )
        child_econ = float(
            np.clip(0.5 * (weights_a.w_econ + weights_b.w_econ) + noise_econ, 0.5, 3.0)
        )
        child_soc = float(
            np.clip(0.5 * (weights_a.w_soc + weights_b.w_soc) + noise_soc, 5.0, 24.0)
        )

        return VillagerUtilityWeights(
            w_phys=round(child_phys, 3),
            w_econ=round(child_econ, 3),
            w_soc=round(child_soc, 3),
            theta_risk=round(child_theta, 3),
            crisis_threshold_tau=weights_a.crisis_threshold_tau,
            crisis_penalty_kappa=weights_a.crisis_penalty_kappa,
            epsilon=weights_a.epsilon,
        )

    def spawn_child_record(
        self,
        *,
        step: int,
        child_id: str,
        child_name: str,
        parent_a: LineageRecord,
        parent_b: LineageRecord,
        weights_a: tuple[float, float, float] | VillagerUtilityWeights,
        weights_b: tuple[float, float, float] | VillagerUtilityWeights,
        rng: np.random.Generator | None = None,
    ) -> tuple[ChildBirthRecord, LineageRecord]:
        """Directly creates a child birth record and initialized lineage record from two parents."""
        r = rng if rng is not None else self.rng
        if isinstance(weights_a, tuple):
            wa = VillagerUtilityWeights(theta_risk=weights_a[0], w_soc=weights_a[1], w_econ=weights_a[2])
        else:
            wa = weights_a
        if isinstance(weights_b, tuple):
            wb = VillagerUtilityWeights(theta_risk=weights_b[0], w_soc=weights_b[1], w_econ=weights_b[2])
        else:
            wb = weights_b

        child_weights = self.synthesize_child_genetics(wa, wb, r)
        gen = max(parent_a.generation, parent_b.generation) + 1
        traditions = (
            f"Born in step {step} to household {parent_a.agent_id} and {parent_b.agent_id}.",
        )
        child_lin = self.init_villager_lineage(
            agent_id=child_id,
            initial_age=0,
            generation=gen,
            parent_ids=(parent_a.agent_id, parent_b.agent_id),
            traditions=traditions,
        )
        birth_rec = ChildBirthRecord(
            step=step,
            child_id=child_id,
            child_name=child_name,
            parent_a=parent_a.agent_id,
            parent_b=parent_b.agent_id,
            hut_id="",
            generation=gen,
            inherited_weights=child_weights,
            inherited_trust_score=0.55,
            inherited_traditions=traditions,
            inherited_theta_risk=child_weights.theta_risk,
            inherited_w_soc=child_weights.w_soc,
            inherited_w_econ=child_weights.w_econ,
        )
        self.births.append(birth_rec)
        return birth_rec, child_lin

    def evaluate_courtship_and_births(
        self,
        *,
        step: int,
        alive_agent_ids: list[str],
        trust_graph: VillageTrustGraph,
        grain_reserves_by_agent: dict[str, float],
        hut_by_agent: dict[str, str],
        weights_by_agent: dict[str, VillagerUtilityWeights],
        memories_by_agent: dict[str, tuple[str, ...]],
        rng: np.random.Generator,
    ) -> tuple[list[HouseholdPartnership], list[ChildBirthRecord]]:
        """Evaluates household pairing ($T_{i \\leftrightarrow j} > 0.75$, grain > 20) and child gestation (§3.2)."""
        new_partnerships: list[HouseholdPartnership] = []
        new_births: list[ChildBirthRecord] = []

        # 1. Check unmarried fertile adults for new household partnerships
        unmarried = [
            aid
            for aid in sorted(alive_agent_ids)
            if aid in self.lineages
            and self.lineages[aid].partner_id is None
            and self.lineages[aid].life_stage == LifeStage.ADULT
        ]

        paired_this_step: set[str] = set()
        for idx, aid_a in enumerate(unmarried):
            if aid_a in paired_this_step:
                continue
            for aid_b in unmarried[idx + 1 :]:
                if aid_b in paired_this_step:
                    continue
                t_ab = trust_graph.get_trust(aid_a, aid_b)
                t_ba = trust_graph.get_trust(aid_b, aid_a)
                combined_grain = grain_reserves_by_agent.get(aid_a, 0.0) + grain_reserves_by_agent.get(aid_b, 0.0)

                if (
                    min(t_ab, t_ba) >= self.config.courtship_trust_threshold
                    and combined_grain >= self.config.courtship_grain_reserve_min
                ):
                    pid = f"household_{aid_a}_{aid_b}"
                    hut_id = hut_by_agent.get(aid_a, "hut_0")
                    part = HouseholdPartnership(
                        partnership_id=pid,
                        partner_a=aid_a,
                        partner_b=aid_b,
                        hut_id=hut_id,
                        formed_step=step,
                        gestation_ticks=1,
                    )
                    self.partnerships[pid] = part
                    new_partnerships.append(part)
                    self.lineages[aid_a] = self.lineages[aid_a].model_copy(update={"partner_id": aid_b})
                    self.lineages[aid_b] = self.lineages[aid_b].model_copy(update={"partner_id": aid_a})
                    paired_this_step.add(aid_a)
                    paired_this_step.add(aid_b)
                    break

        # 2. Advance gestation and instantiate child villagers when interval completes
        for pid, part in list(self.partnerships.items()):
            if (
                part.partner_a not in alive_agent_ids
                or part.partner_b not in alive_agent_ids
                or len(alive_agent_ids) + len(new_births) >= self.config.max_village_population
                or len(part.children_born) >= 2
            ):
                continue

            next_gest = part.gestation_ticks + 1
            if next_gest >= self.config.gestation_interval_ticks:
                birth_idx = len(self.births)
                cname = self.CHILD_NAMES[birth_idx % len(self.CHILD_NAMES)]
                child_id = f"villager_child_{cname.lower()}_{birth_idx + 1}"

                wa = weights_by_agent.get(part.partner_a, VillagerUtilityWeights())
                wb = weights_by_agent.get(part.partner_b, VillagerUtilityWeights())
                child_weights = self.synthesize_child_genetics(wa, wb, rng)

                # Inherit family trust score and episodic traditions (§3.2)
                sa = trust_graph.get_status(part.partner_a)
                sb = trust_graph.get_status(part.partner_b)
                inherited_trust = float(np.clip(0.5 * (sa + sb) * len(alive_agent_ids) * 0.5 + 0.35, 0.35, 0.85))

                mem_a = memories_by_agent.get(part.partner_a, ())
                mem_b = memories_by_agent.get(part.partner_b, ())
                traditions = tuple(
                    list(mem_a[-1:])
                    + list(mem_b[-1:])
                    + [f"Born in step {step} to household {part.partner_a} & {part.partner_b}."]
                )

                gen = max(
                    self.lineages[part.partner_a].generation,
                    self.lineages[part.partner_b].generation,
                ) + 1

                self.init_villager_lineage(
                    agent_id=child_id,
                    initial_age=0,
                    generation=gen,
                    parent_ids=(part.partner_a, part.partner_b),
                    traditions=traditions,
                )

                # Update parents' children list
                for parent_id in (part.partner_a, part.partner_b):
                    lin = self.lineages[parent_id]
                    self.lineages[parent_id] = lin.model_copy(
                        update={"children_ids": (*lin.children_ids, child_id)}
                    )

                self.partnerships[pid] = part.model_copy(
                    update={
                        "gestation_ticks": 0,
                        "children_born": (*part.children_born, child_id),
                    }
                )

                birth_rec = ChildBirthRecord(
                    step=step,
                    child_id=child_id,
                    child_name=cname,
                    parent_a=part.partner_a,
                    parent_b=part.partner_b,
                    hut_id=part.hut_id,
                    generation=gen,
                    inherited_weights=child_weights,
                    inherited_trust_score=round(inherited_trust, 4),
                    inherited_traditions=traditions,
                )
                self.births.append(birth_rec)
                new_births.append(birth_rec)
            else:
                self.partnerships[pid] = part.model_copy(update={"gestation_ticks": next_gest})

        return new_partnerships, new_births

    def record_death_and_mourning(
        self,
        *,
        step: int,
        deceased_id: str,
        deceased_name: str,
        cause_of_death: str,
        coins: float,
        alive_agent_ids: list[str],
        trust_graph: VillageTrustGraph,
    ) -> FuneralRecord:
        """Processes mortality inheritance and places close friends ($T_{j \\to i} > 0.6$) into 48-tick Grief (§3.1)."""
        lin = self.lineages.get(deceased_id, LineageRecord(agent_id=deceased_id))
        self.lineages[deceased_id] = lin.model_copy(update={"life_stage": LifeStage.DECEASED})

        # Determine heir: partner first, then first alive child, otherwise Village Commons
        heir_id: str | None = None
        if lin.partner_id and lin.partner_id in alive_agent_ids:
            heir_id = lin.partner_id
        else:
            for cid in lin.children_ids:
                if cid in alive_agent_ids:
                    heir_id = cid
                    break

        mourners: list[str] = []
        for other_id in alive_agent_ids:
            if other_id == deceased_id:
                continue
            t_val = trust_graph.get_trust(other_id, deceased_id)
            is_kin = (
                other_id == lin.partner_id
                or other_id in lin.children_ids
                or other_id in lin.parent_ids
            )
            if t_val > 0.60 or is_kin:
                mourners.append(other_id)
                if other_id in self.lineages:
                    self.lineages[other_id] = self.lineages[other_id].model_copy(
                        update={"grief_ticks_remaining": self.config.grief_duration_ticks}
                    )

        rec = FuneralRecord(
            step=step,
            deceased_id=deceased_id,
            deceased_name=deceased_name,
            cause_of_death=cause_of_death,
            age_at_death=lin.age_ticks,
            heir_id=heir_id,
            coins_inherited=round(coins, 2),
            moved_to_commons=(heir_id is None),
            mourners=tuple(mourners),
        )
        self.funerals.append(rec)
        return rec


# ==============================================================================
# SECTION 5: VILLAGE ASSEMBLIES (GRAM SABHA) & POLITICAL INSTITUTIONS
# ==============================================================================


class SabhaAccusation(BaseModel):
    """Public grievance or crime accusation presented at the Gram Sabha (§5.1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accuser_id: str
    accuser_name: str
    accused_id: str
    accused_name: str
    crime_type: str
    evidence_score: float = Field(ge=0.0, le=1.0)
    accusation_speech: str
    defense_speech: str


class SabhaVoteRecord(BaseModel):
    """Trust-weighted community vote tally for an accused offender (§5.1.3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accused_id: str
    accused_name: str
    guilty_weight: float = Field(ge=0.0)
    innocent_weight: float = Field(ge=0.0)
    convicted: bool
    sanction_applied: str
    pillory_ticks_assigned: int = 0
    fine_coins_assigned: float = 0.0


class VillageAssemblyRecord(BaseModel):
    """Complete record of a convoked Town Meeting / Gram Sabha (§5.1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    model_config = ConfigDict(frozen=True, extra="ignore")

    assembly_id: str
    step: int
    magistrate_id: str
    magistrate_name: str
    trigger_reason: str
    granary_wheat: float
    water_reserves: float
    public_crime_count: int
    state_address: str
    accusations: tuple[SabhaAccusation, ...] = ()
    votes: tuple[SabhaVoteRecord, ...] = ()
    policy_enacted: str = "Normal Market Clearing"
    mandatory_tithe_rate: float = Field(default=0.0, ge=0.0, le=0.5)
    attendees: tuple[str, ...] = ()
    attendance_count: int = 0
    quorum_achieved: bool = True
    passed: bool = True


class GramSabhaEngine:
    """Emergent Political Institution Engine governing Town Assemblies and Trust-Weighted Trials (§5)."""

    def __init__(
        self,
        periodic_interval_steps: int = 12,
        emergency_granary_threshold: float = 32.0,
        seed: int = 42,
    ) -> None:
        self.periodic_interval_steps = periodic_interval_steps
        self.emergency_granary_threshold = emergency_granary_threshold
        self.seed = seed
        self.assemblies: list[VillageAssemblyRecord] = []
        self.active_assembly: VillageAssemblyRecord | None = None
        self.assembly_ticks_remaining: int = 0
        self.active_tithe_mandate_rate: float = 0.0

    @property
    def mandatory_tithe_rate(self) -> float:
        return self.active_tithe_mandate_rate

    @property
    def enacted_policies(self) -> list[str]:
        return [a.policy_enacted for a in self.assemblies if a.policy_enacted]

    def should_convoke(
        self,
        step: int,
        granary_wheat: float,
        uninvestigated_crimes: int,
        force_convoke: bool = False,
    ) -> tuple[bool, str]:
        """Determines if the Town Hall bell tolls to summon a Gram Sabha (§5.1)."""
        if force_convoke:
            return True, "Magistrate Special Summons"
        if step > 0 and step % self.periodic_interval_steps == 0:
            return True, "Periodic Sabbath Assembly"
        if granary_wheat < self.emergency_granary_threshold and self.assembly_ticks_remaining == 0:
            if not self.assemblies or (step - self.assemblies[-1].step) >= 6:
                return True, f"Emergency Famine Crisis (Granary={granary_wheat:.1f})"
        if uninvestigated_crimes >= 2 and self.assembly_ticks_remaining == 0:
            if not self.assemblies or (step - self.assemblies[-1].step) >= 5:
                return True, f"Serial Theft Tribunal ({uninvestigated_crimes} offenses)"
        return False, ""

    def conduct_assembly(
        self,
        *,
        step: int,
        magistrate_id: str,
        magistrate_name: str,
        trigger_reason: str,
        granary_wheat: float,
        water_reserves: float,
        total_crimes: int,
        suspect_records: list[dict[str, Any]],
        alive_villagers: dict[str, str],
        trust_graph: VillageTrustGraph,
    ) -> VillageAssemblyRecord:
        """Executes the 4-stage Gram Sabha protocol (§5.1):
        1. Assembly Gathering
        2. Magistrate State-of-the-Village Address
        3. Open Floor Grievances & Trust-Weighted Voting
        4. Policy Enactment
        """
        status_map = trust_graph.get_status_map()
        attendees = tuple(sorted(alive_villagers.keys()))

        # Stage 2: Magistrate State-of-the-Village Address
        state_address = (
            f"Hear ye! Granary holds {granary_wheat:.1f} wheat, Well holds {water_reserves:.0f} water, "
            f"and {total_crimes} public offenses stand on the docket."
        )

        # Stage 3: Open Floor Grievances & Trust-Weighted Voting
        accusations: list[SabhaAccusation] = []
        votes: list[SabhaVoteRecord] = []

        for susp in suspect_records[:2]:
            accuser_id = str(susp.get("accuser_id", magistrate_id))
            accuser_name = alive_villagers.get(accuser_id, magistrate_name)
            accused_id = str(susp["accused_id"])
            accused_name = alive_villagers.get(accused_id, accused_id)
            crime_type = str(susp.get("crime_type", "STEAL_GRANARY"))
            ev_score = float(susp.get("evidence_score", 0.72))

            acc_speech = (
                f"I accuse {accused_name} of {crime_type}! Forensic and witness evidence stands at {ev_score:.2f}."
            )
            def_speech = (
                "Forgive me, council! Hunger pressed my hand—I pledge restitution to the Commons!"
                if ev_score >= 0.65
                else "I stand innocent before the Sabha; the shadows deceived the watch!"
            )
            accusations.append(
                SabhaAccusation(
                    accuser_id=accuser_id,
                    accuser_name=accuser_name,
                    accused_id=accused_id,
                    accused_name=accused_name,
                    crime_type=crime_type,
                    evidence_score=round(ev_score, 4),
                    accusation_speech=acc_speech,
                    defense_speech=def_speech,
                )
            )

            # Tally PageRank trust-weighted community votes (§5.1.3)
            guilty_w = 0.0
            innocent_w = 0.0
            for voter_id in attendees:
                if voter_id == accused_id:
                    continue
                voter_status = status_map.get(voter_id, 0.1)
                trust_in_accused = trust_graph.get_trust(voter_id, accused_id)
                trust_in_accuser = trust_graph.get_trust(voter_id, accuser_id)

                # Voter convicts if evidence weighted by accuser credibility exceeds trust in accused
                conviction_belief = ev_score * (0.5 + 0.5 * trust_in_accuser)
                if conviction_belief > trust_in_accused * 0.85:
                    guilty_w += voter_status
                else:
                    innocent_w += voter_status

            convicted = guilty_w > innocent_w
            sanction = (
                "4-Tick Pillory & 10c Restitution Fine"
                if convicted
                else "Acquitted with Probation Warning"
            )
            votes.append(
                SabhaVoteRecord(
                    accused_id=accused_id,
                    accused_name=accused_name,
                    guilty_weight=round(guilty_w, 4),
                    innocent_weight=round(innocent_w, 4),
                    convicted=convicted,
                    sanction_applied=sanction,
                    pillory_ticks_assigned=4 if convicted else 0,
                    fine_coins_assigned=10.0 if convicted else 0.0,
                )
            )

        # Stage 4: Policy Enactment (§5.1.4)
        if granary_wheat < self.emergency_granary_threshold:
            policy = "Emergency Famine Decree: Mandatory 10% Grain Tithe & Rationing"
            policy = "Emergency Famine Decree: MANDATORY_GRAIN_TITHE_10 (10% grain tithe)"
            tithe_rate = 0.10
        elif votes and any(v.convicted for v in votes):
            policy = "Heightened Watch Decree: Double Guard Patrols & Restitution Service"
            tithe_rate = 0.05
        else:
            policy = "Prosperity Charter: Free Market Trade & Craft Guild Bounty"
            tithe_rate = 0.0

        self.active_tithe_mandate_rate = tithe_rate
        rec = VillageAssemblyRecord(
            assembly_id=f"sabha_step_{step}",
            step=step,
            magistrate_id=magistrate_id,
            magistrate_name=magistrate_name,
            trigger_reason=trigger_reason,
            granary_wheat=round(granary_wheat, 2),
            water_reserves=round(water_reserves, 2),
            public_crime_count=total_crimes,
            state_address=state_address,
            accusations=tuple(accusations),
            votes=tuple(votes),
            policy_enacted=policy,
            mandatory_tithe_rate=tithe_rate,
            attendees=attendees,
            attendance_count=len(attendees),
            quorum_achieved=True,
            passed=True,
        )
        self.assemblies.append(rec)
        self.active_assembly = rec
        self.assembly_ticks_remaining = 2
        return rec

    def convene_sabha(
        self,
        *,
        step: int,
        granary_wheat: float,
        well_water: float,
        drought_active: bool,
        agent_statuses: dict[str, float],
        agent_names: dict[str, str],
        trust_lookup: Any = None,
        known_offenders: list[tuple[str, str, int]] | None = None,
        emergency_reason: str | None = None,
    ) -> VillageAssemblyRecord:
        """High-level convene entry point for controller and tests (§5.1)."""
        trigger = emergency_reason or ("Emergency Famine Crisis" if drought_active else "Periodic Sabbath Assembly")
        mag_id = "villager_lata" if "villager_lata" in agent_names else next(iter(agent_names.keys()), "magistrate")
        mag_name = agent_names.get(mag_id, "Magistrate")

        suspects: list[dict[str, Any]] = []
        if known_offenders:
            for off in known_offenders:
                suspects.append(
                    {
                        "accused_id": off[0],
                        "accuser_id": mag_id,
                        "crime_type": off[1],
                        "evidence_score": 0.82,
                    }
                )

        tg = VillageTrustGraph(agent_ids=list(agent_names.keys()))
        if callable(trust_lookup):
            for a in agent_names.keys():
                for b in agent_names.keys():
                    try:
                        tg.set_trust(a, b, float(trust_lookup(a, b)))
                    except Exception:
                        pass
        return self.conduct_assembly(
            step=step,
            magistrate_id=mag_id,
            magistrate_name=mag_name,
            trigger_reason=trigger,
            granary_wheat=granary_wheat,
            water_reserves=well_water,
            total_crimes=len(known_offenders or []),
            suspect_records=suspects,
            alive_villagers=agent_names,
            trust_graph=tg,
        )


# ==============================================================================
# SECTION 6: CRIME, CORRUPTION, FORENSICS, AND REDEMPTION LOOP
# ==============================================================================


class ForensicTrace(BaseModel):
    """Physical evidence left at a crime scene for Guard investigation (§6.2.1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: str
    step: int
    culprit_id: str
    culprit_name: str
    crime_action: str
    zone_id: str
    base_evidence_prob: float = Field(ge=0.0, le=1.0)
    witness_ids: tuple[str, ...] = ()
    investigated: bool = False
    bribe_suppressed: bool = False


class RedemptionServiceRecord(BaseModel):
    """Record of community service performed by a repentant offender (§6.2.4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int
    actor_id: str
    actor_name: str
    service_type: str
    status_before: float
    status_after: float


class JusticeRedemptionEngine:
    """Manages forensic crime traces, guard investigations, bribery resistance, and redemption (§6)."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.forensic_traces: list[ForensicTrace] = []
        self.redemption_records: list[RedemptionServiceRecord] = []
        self.bribes_offered: int = 0
        self.bribes_rejected: int = 0

    def record_forensic_trace(
        self,
        *,
        step: int,
        culprit_id: str,
        culprit_name: str,
        crime_action: str,
        zone_id: str,
        detection_prob: float,
        witness_ids: tuple[str, ...] = (),
    ) -> ForensicTrace:
        """Creates a forensic trace when an unseen theft or sabotage occurs (§6.2.1)."""
        ev_prob = float(np.clip(0.45 + 0.50 * detection_prob, 0.25, 0.96))
        tr = ForensicTrace(
            trace_id=f"forensic_{step}_{culprit_id}",
            step=step,
            culprit_id=culprit_id,
            culprit_name=culprit_name,
            crime_action=crime_action,
            zone_id=zone_id,
            base_evidence_prob=round(ev_prob, 4),
            witness_ids=witness_ids,
        )
        self.forensic_traces.append(tr)
        return tr

    def investigate_pending_traces(
        self,
        *,
        step: int,
        guard_id: str,
        guard_w_soc: float,
        guard_theta_risk: float,
        culprit_coins_map: dict[str, float],
        trust_graph: VillageTrustGraph,
    ) -> list[dict[str, Any]]:
        """Guard investigates recent forensic traces and interrogates neighbors weighted by PageRank (§6.2.2).

        Also models Bribery & Corruption (§6.1): wealthy culprit offers coins to turn a blind eye;
        principled guards with high $w_{\\text{soc}}$ and low $\\theta_{\\text{risk}}$ reject the bribe.
        """
        status_map = trust_graph.get_status_map()
        findings: list[dict[str, Any]] = []
        updated_traces: list[ForensicTrace] = []

        for tr in self.forensic_traces:
            if tr.investigated or (step - tr.step) > 8:
                updated_traces.append(tr)
                continue

            # Compute eyewitness credibility boost from PageRank status S_j
            witness_cred = sum(status_map.get(wid, 0.08) for wid in tr.witness_ids)
            total_evidence = float(np.clip(tr.base_evidence_prob + 0.65 * witness_cred, 0.0, 0.99))

            # Check if wealthy culprit attempts to bribe the investigating guard (§6.1)
            culprit_coins = culprit_coins_map.get(tr.culprit_id, 0.0)
            bribe_accepted = False
            if culprit_coins >= 35.0:
                self.bribes_offered += 1
                bribe_offer = 10.0
                # Guard compares bribe utility vs righteous duty (w_soc * S_guard)
                guard_duty_value = guard_w_soc * status_map.get(guard_id, 0.1) * (1.0 - guard_theta_risk)
                if bribe_offer * guard_theta_risk > guard_duty_value:
                    bribe_accepted = True
                else:
                    self.bribes_rejected += 1

            updated_traces.append(
                tr.model_copy(
                    update={
                        "investigated": True,
                        "bribe_suppressed": bribe_accepted,
                    }
                )
            )

            if not bribe_accepted and total_evidence >= 0.52:
                findings.append(
                    {
                        "accuser_id": guard_id,
                        "accused_id": tr.culprit_id,
                        "crime_type": tr.crime_action,
                        "evidence_score": round(total_evidence, 4),
                    }
                )

        self.forensic_traces = updated_traces[-40:]
        return findings

    def perform_redemption_service(
        self,
        *,
        step: int,
        actor_id: str,
        actor_name: str,
        service_type: str,
        trust_graph: VillageTrustGraph,
    ) -> RedemptionServiceRecord:
        """Executes community service rehabilitation, restoring trust edges and PageRank status (§6.2.4)."""
        before_s = trust_graph.get_status(actor_id)
        # Community service applies two rounds of cooperative trust repair across all villagers
        trust_graph.record_public_event(actor_id=actor_id, event_type="coop")
        trust_graph.record_public_event(actor_id=actor_id, event_type="coop")
        new_map = trust_graph.compute_societal_status()
        after_s = new_map.get(actor_id, before_s)

        rec = RedemptionServiceRecord(
            step=step,
            actor_id=actor_id,
            actor_name=actor_name,
            service_type=service_type,
            status_before=round(before_s, 5),
            status_after=round(after_s, 5),
        )
        self.redemption_records.append(rec)
        return rec

