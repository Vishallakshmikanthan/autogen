"""Unit and oracle tests for Reflection Consolidation, Attribution, and Forgiveness."""

import math

import numpy as np
import pytest

from autogen.intelligence.village.memory_stream import MemoryItem
from autogen.intelligence.village.reflection import Disposition, ReflectionEngine, Traits
from autogen.simulation.environment.village.village_grid import VillageActionType


class TestReflectionTriggers:
    """Verification of condition triggers for reflection passes."""

    def test_sleep_consolidation_trigger(self) -> None:
        """Completing REST_HUT immediately triggers reflection."""
        items = [
            MemoryItem(
                mid="v1:1:1",
                step=1,
                last_access_step=1,
                importance=0.30,
                kind="observation",
            )
        ]
        should_ref, reason = ReflectionEngine.should_reflect(
            completed_action=VillageActionType.REST_HUT,
            unreflected=items,
        )
        assert should_ref is True
        assert reason == "sleep_consolidation"

    def test_importance_accumulation_trigger(self) -> None:
        """Accumulated unreflected importance >= 2.50 triggers reflection."""
        # 5 items with 0.55 importance each -> sum = 2.75 >= 2.50
        items = [
            MemoryItem(
                mid=f"v1:1:{i}",
                step=1,
                last_access_step=1,
                importance=0.55,
                kind="observation",
            )
            for i in range(5)
        ]
        should_ref, reason = ReflectionEngine.should_reflect(
            completed_action=VillageActionType.HARVEST_FIELD,
            unreflected=items,
        )
        assert should_ref is True
        assert reason == "importance_threshold"

    def test_acute_event_trigger(self) -> None:
        """A single acute event with importance >= 0.85 triggers immediate reflection."""
        items = [
            MemoryItem(
                mid="v1:1:1",
                step=1,
                last_access_step=1,
                importance=0.90,
                kind="observation",
                tags=("crime_caught",),
            )
        ]
        should_ref, reason = ReflectionEngine.should_reflect(
            completed_action=VillageActionType.DRAW_WATER,
            unreflected=items,
        )
        assert should_ref is True
        assert reason == "acute_event"

    def test_sub_threshold_no_trigger(self) -> None:
        """Low importance routine observations without REST_HUT do not trigger reflection."""
        items = [
            MemoryItem(
                mid="v1:1:1",
                step=1,
                last_access_step=1,
                importance=0.40,
                kind="observation",
            )
        ]
        should_ref, reason = ReflectionEngine.should_reflect(
            completed_action=VillageActionType.HARVEST_FIELD,
            unreflected=items,
        )
        assert should_ref is False
        assert reason == "none"


class TestSituationalAttribution:
    """Mathematical verification of situational moral blame discounting."""

    def test_severe_drought_blame_discount(self) -> None:
        """Under severe drought & zero granary reserves, highly agreeable observers discount blame by 50%."""
        agreeable_traits = Traits(agreeableness=1.0)
        blame_agreeable = ReflectionEngine.compute_situational_blame(
            agreeable_traits, is_drought=True, granary_wheat=0.0
        )
        # sigma_pub = 0.50 (drought) + 0.50 (zero wheat) = 1.00
        # blame = 1.0 - 0.50 * 1.0 * 1.0 = 0.50
        assert blame_agreeable == pytest.approx(0.50, abs=1e-4)

        hostile_traits = Traits(agreeableness=0.0)
        blame_hostile = ReflectionEngine.compute_situational_blame(
            hostile_traits, is_drought=True, granary_wheat=0.0
        )
        # blame = 1.0 - 0.50 * 0.0 * 1.0 = 1.00
        assert blame_hostile == pytest.approx(1.00, abs=1e-4)

    def test_abundance_full_blame(self) -> None:
        """In times of abundance (no drought, 100 wheat), full blame (1.00) is assigned regardless of traits."""
        agreeable_traits = Traits(agreeableness=1.0)
        blame = ReflectionEngine.compute_situational_blame(
            agreeable_traits, is_drought=False, granary_wheat=100.0
        )
        # sigma_pub = 0.0 -> blame = 1.00
        assert blame == pytest.approx(1.00, abs=1e-4)

    def test_partial_scarcity(self) -> None:
        """Granary at 50 wheat with no drought produces intermediate blame discount."""
        traits = Traits(agreeableness=0.80)
        blame = ReflectionEngine.compute_situational_blame(
            traits, is_drought=False, granary_wheat=50.0
        )
        # scarcity = (100 - 50)/100 * 0.50 = 0.25
        # blame = 1.0 - 0.50 * 0.80 * 0.25 = 1.0 - 0.10 = 0.90
        assert blame == pytest.approx(0.90, abs=1e-4)


class TestForgivenessDynamics:
    """Verification of probabilistic forgiveness dynamics and grudge reduction."""

    def test_zero_grudge_invariant(self) -> None:
        """An agent with zero grudge does not perform or need forgiveness."""
        disp = Disposition(target_id="peer", grudge=0.0)
        traits = Traits(agreeableness=0.90)
        rng = np.random.default_rng(42)
        new_disp, forgiven = ReflectionEngine.check_forgiveness(
            disposition=disp,
            traits=traits,
            witnessed_community_service=True,
            steps_since_harm=48,
            rng=rng,
        )
        assert forgiven is False
        assert new_disp.grudge == 0.0

    def test_grudge_halving_on_forgiveness(self) -> None:
        """When forgiveness succeeds, the existing grudge is cut in half."""
        disp = Disposition(target_id="peer", grudge=0.80)
        traits = Traits(agreeableness=1.0)
        # Force forgiveness by high elapsed time and high agreeableness
        rng = np.random.default_rng(12345)
        # Try until forgiven
        forg = False
        new_disp = disp
        for _ in range(50):
            new_disp, forg = ReflectionEngine.check_forgiveness(
                disposition=disp,
                traits=traits,
                witnessed_community_service=True,
                steps_since_harm=200,
                rng=rng,
            )
            if forg:
                break
        assert forg is True
        assert new_disp.grudge == pytest.approx(0.40, abs=1e-4)

    def test_statistical_forgiveness_rate(self) -> None:
        """Test empirical forgiveness probability over 1,000 independent seeds matches analytical expectation:

        P_f = clip(0.20 + 0.50 A + 0.30 * sinc, 0.0, 1.0) * (1 - exp(-dt / 48))
        For A = 0.60, sinc = 1.0 (witnessed), dt = 48:
        term1 = 0.20 + 0.30 + 0.30 = 0.80
        term2 = 1 - exp(-1) ~= 0.63212
        P_expected = 0.80 * 0.63212 = 0.505696
        """
        traits = Traits(agreeableness=0.60)
        disp = Disposition(target_id="peer", grudge=0.70)
        dt = 48

        p_expected = 0.80 * (1.0 - math.exp(-1.0))

        num_trials = 1000
        forgiveness_count = 0

        for seed in range(num_trials):
            rng = np.random.default_rng(seed)
            _, forgiven = ReflectionEngine.check_forgiveness(
                disposition=disp,
                traits=traits,
                witnessed_community_service=True,
                steps_since_harm=dt,
                rng=rng,
            )
            if forgiven:
                forgiveness_count += 1

        empirical_rate = forgiveness_count / float(num_trials)
        # Binomial confidence interval for N=1000 at p=0.5: std = sqrt(p(1-p)/1000) ~= 0.0158
        # 3-sigma tolerance = ~0.048
        assert empirical_rate == pytest.approx(p_expected, abs=0.048)


class TestCrimePatternDetection:
    """Verification of spatial-temporal crime cluster synthesis."""

    def test_crime_cluster_triggers_belief(self) -> None:
        """>= 3 crimes in (location, is_night) bin creates a high-level belief."""
        # 3 theft events at 'granary' at step 4 (step 4 * 0.5 = hour 2.0 -> night)
        items = [
            MemoryItem(
                mid=f"m_crime_{i}",
                step=4,
                last_access_step=4,
                importance=0.85,
                kind="observation",
                location="granary",
                tags=("theft", "granary"),
            )
            for i in range(3)
        ]
        beliefs = ReflectionEngine.detect_crime_patterns(items, agent_id="guard", step=10)
        assert len(beliefs) == 1
        b = beliefs[0]
        assert b["kind"] == "belief"
        assert b["location"] == "granary"
        assert "night" in b["summary"]
        assert len(b["evidence_of"]) == 3
        assert b["importance"] >= 0.90

    def test_sub_cluster_no_belief(self) -> None:
        """< 3 crimes does not synthesize a belief."""
        items = [
            MemoryItem(
                mid=f"m_crime_{i}",
                step=4,
                last_access_step=4,
                importance=0.85,
                kind="observation",
                location="granary",
                tags=("theft", "granary"),
            )
            for i in range(2)
        ]
        beliefs = ReflectionEngine.detect_crime_patterns(items, agent_id="guard", step=10)
        assert len(beliefs) == 0
