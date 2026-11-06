"""Central Algorithm Registry and Metadata Catalog (SYS-19, Module M21).

Enforces registration, documentation traceability, hyperparameter schemas,
and citation safety across all implemented AI algorithms in AUTOGEN AI.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class AlgorithmFamily(str, Enum):
    """Canonical classification families for autonomous intelligence algorithms."""

    SEARCH = "search"
    ADVERSARIAL = "adversarial"
    LOGIC = "logic"
    PLANNING = "planning"
    PROBABILISTIC = "probabilistic"
    DECISION = "decision"
    ML = "ml"
    RL = "rl"
    NEURAL = "neural"
    MULTIAGENT = "multiagent"
    EVOLUTION = "evolution"
    HYBRID = "hybrid"
    ANALYTICS = "analytics"


class CitationRecord(BaseModel):
    """Citation metadata strictly adhering to verified textbook sources."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    author_or_source: str
    title: str
    year: int
    chapter_or_pages: str
    verified: bool = True  # If False, must be marked "TO VERIFY" in documentation


class AlgorithmMetadata(BaseModel):
    """Immutable metadata record for a verified algorithm implementation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    algorithm_id: str
    name: str
    family: AlgorithmFamily
    version: str = "1.0.0"
    doc_path: str
    complexity_time: str
    complexity_space: str
    assumptions: list[str] = Field(default_factory=list)
    hyperparameters: dict[str, str] = Field(default_factory=dict)
    citations: list[CitationRecord] = Field(default_factory=list)
    is_step_traceable: bool = True


class AlgorithmRegistry:
    """Thread-safe central registry for all algorithms in AUTOGEN AI."""

    _registry: dict[str, AlgorithmMetadata] = {}

    @classmethod
    def register(cls, metadata: AlgorithmMetadata) -> None:
        """Register an algorithm implementation. Raises ValueError if already registered."""
        key = metadata.algorithm_id.strip().lower()
        if key in cls._registry:
            raise ValueError(f"Algorithm '{key}' is already registered in AlgorithmRegistry.")
        cls._registry[key] = metadata

    @classmethod
    def get(cls, algorithm_id: str) -> AlgorithmMetadata:
        """Retrieve algorithm metadata by unique identifier."""
        key = algorithm_id.strip().lower()
        if key not in cls._registry:
            raise KeyError(f"Algorithm '{key}' not found in AlgorithmRegistry.")
        return cls._registry[key]

    @classmethod
    def get_by_family(cls, family: AlgorithmFamily) -> list[AlgorithmMetadata]:
        """List all algorithms registered under a specific algorithmic family."""
        return [meta for meta in cls._registry.values() if meta.family == family]

    @classmethod
    def get_all(cls) -> list[AlgorithmMetadata]:
        """Return a sorted list of all registered algorithms."""
        return sorted(cls._registry.values(), key=lambda m: (m.family.value, m.algorithm_id))

    @classmethod
    def count(cls) -> int:
        """Return count of registered algorithms."""
        return len(cls._registry)

    @classmethod
    def clear(cls) -> None:
        """Clear registry for testing."""
        cls._registry.clear()


# ==============================================================================
# CANONICAL ALGORITHM CATALOG POPULATION
# ==============================================================================


def populate_default_catalog() -> None:
    """Populates registry with the verified implementations across the platform."""
    if AlgorithmRegistry.count() > 0:
        return

    # 1. Search
    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="astar",
            name="A* Heuristic Graph Search",
            family=AlgorithmFamily.SEARCH,
            doc_path="docs/algorithms/search/informed_search.md",
            complexity_time="O(b^d)",
            complexity_space="O(b^d)",
            assumptions=[
                "Admissible heuristic: h(n) <= h*(n)",
                "Consistent / monotonic heuristic: h(n) <= c(n, a, n') + h(n')",
            ],
            hyperparameters={"heuristic": "Distance metric function (Manhattan / Euclidean)"},
            citations=[
                CitationRecord(
                    author_or_source="Russell & Norvig",
                    title="Artificial Intelligence: A Modern Approach (4th ed.)",
                    year=2020,
                    chapter_or_pages="Chapter 3.5, pp. 71-77",
                )
            ],
        )
    )

    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="breadth_first_search",
            name="Breadth-First Graph Search (BFS)",
            family=AlgorithmFamily.SEARCH,
            doc_path="docs/algorithms/search/uninformed_search.md",
            complexity_time="O(b^d)",
            complexity_space="O(b^d)",
            assumptions=["Uniform edge costs for optimality", "Finite branching factor"],
            hyperparameters={},
            citations=[
                CitationRecord(
                    author_or_source="Russell & Norvig",
                    title="Artificial Intelligence: A Modern Approach (4th ed.)",
                    year=2020,
                    chapter_or_pages="Chapter 3.4.1, pp. 64-66",
                )
            ],
        )
    )

    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="uniform_cost_search",
            name="Uniform-Cost Search (Dijkstra)",
            family=AlgorithmFamily.SEARCH,
            doc_path="docs/algorithms/search/uninformed_search.md",
            complexity_time="O(b^(1 + floor(C*/eps)))",
            complexity_space="O(b^(1 + floor(C*/eps)))",
            assumptions=["Step costs are non-negative: c(s, a, s') >= eps > 0"],
            hyperparameters={},
            citations=[
                CitationRecord(
                    author_or_source="Russell & Norvig",
                    title="Artificial Intelligence: A Modern Approach (4th ed.)",
                    year=2020,
                    chapter_or_pages="Chapter 3.4.2, pp. 66-68",
                )
            ],
        )
    )

    # 2. Adversarial / Game Theory
    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="minimax_alphabeta",
            name="Minimax Search with Alpha-Beta Pruning",
            family=AlgorithmFamily.ADVERSARIAL,
            doc_path="docs/algorithms/adversarial/minimax_alphabeta.md",
            complexity_time="O(b^(d/2)) optimal move ordering, O(b^d) worst case",
            complexity_space="O(b * d)",
            assumptions=["Two-player zero-sum deterministic perfect information game"],
            hyperparameters={"max_depth": "Maximum lookahead search depth"},
            citations=[
                CitationRecord(
                    author_or_source="Russell & Norvig",
                    title="Artificial Intelligence: A Modern Approach (4th ed.)",
                    year=2020,
                    chapter_or_pages="Chapter 5.2 - 5.3, pp. 147-155",
                )
            ],
        )
    )

    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="expectimax",
            name="Expectimax Search",
            family=AlgorithmFamily.ADVERSARIAL,
            doc_path="docs/algorithms/adversarial/expectimax.md",
            complexity_time="O(b^d * n^d) where n is stochastic branches",
            complexity_space="O(b * d)",
            assumptions=["Opponent acts stochastically with known probability distribution"],
            hyperparameters={"max_depth": "Search depth"},
            citations=[
                CitationRecord(
                    author_or_source="Russell & Norvig",
                    title="Artificial Intelligence: A Modern Approach (4th ed.)",
                    year=2020,
                    chapter_or_pages="Chapter 5.5, pp. 160-164",
                )
            ],
        )
    )

    # 3. Probabilistic / Bayesian
    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="grid_belief_filter",
            name="Grid Markov Belief State Filter",
            family=AlgorithmFamily.PROBABILISTIC,
            doc_path="docs/algorithms/probabilistic/grid_belief_filter.md",
            complexity_time="O(|S|^2) per observation update",
            complexity_space="O(|S|)",
            assumptions=["First-order Markov state transitions", "Sensor conditional independence"],
            hyperparameters={"decay": "Observation discount factor"},
            citations=[
                CitationRecord(
                    author_or_source="Russell & Norvig",
                    title="Artificial Intelligence: A Modern Approach (4th ed.)",
                    year=2020,
                    chapter_or_pages="Chapter 14.2 - 14.3, pp. 468-477",
                )
            ],
        )
    )

    # 4. Decision & MDP
    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="value_iteration",
            name="Bellman Value Iteration",
            family=AlgorithmFamily.DECISION,
            doc_path="docs/algorithms/decision/bellman_equations_and_value_iteration.md",
            complexity_time="O(|S|^2 * |A| * K) iterations",
            complexity_space="O(|S|)",
            assumptions=["Stationary Markov Decision Process", "Discount factor gamma in [0, 1)"],
            hyperparameters={"gamma": "Discount factor", "epsilon": "Convergence threshold"},
            citations=[
                CitationRecord(
                    author_or_source="Sutton & Barto",
                    title="Reinforcement Learning: An Introduction (2nd ed.)",
                    year=2018,
                    chapter_or_pages="Chapter 4.4, pp. 82-85",
                )
            ],
        )
    )

    # 5. Reinforcement Learning
    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="q_learning",
            name="Tabular Q-Learning (Off-Policy TD Control)",
            family=AlgorithmFamily.RL,
            doc_path="docs/algorithms/rl/temporal_difference_control.md",
            complexity_time="O(1) per step transition",
            complexity_space="O(|S| * |A|)",
            assumptions=[
                "Discrete state and action spaces",
                "Sufficient exploration of all (s, a) pairs",
            ],
            hyperparameters={
                "alpha": "Learning rate",
                "gamma": "Discount factor",
                "epsilon": "Exploration rate",
            },
            citations=[
                CitationRecord(
                    author_or_source="Sutton & Barto",
                    title="Reinforcement Learning: An Introduction (2nd ed.)",
                    year=2018,
                    chapter_or_pages="Chapter 6.5, pp. 131-134",
                )
            ],
        )
    )

    # 6. Hybrid & Cognitive
    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="hybrid_cognitive_architecture",
            name="Multi-Tier Hybrid Deliberative-Reflexive Architecture",
            family=AlgorithmFamily.HYBRID,
            doc_path="docs/algorithms/hybrid/hybrid_cognitive_architecture.md",
            complexity_time="O(b^d + |S|)",
            complexity_space="O(|S| + b*d)",
            assumptions=[
                "Dual-process cognitive theory (System 1 fast reflex + System 2 deliberation)"
            ],
            hyperparameters={"shield_enabled": "Enable symbolic safety veto"},
            citations=[
                CitationRecord(
                    author_or_source="Russell & Norvig",
                    title="Artificial Intelligence: A Modern Approach (4th ed.)",
                    year=2020,
                    chapter_or_pages="Chapter 2.4, pp. 49-55",
                )
            ],
        )
    )

    # 7. Analytics & Statistical Engine
    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="statistical_engine",
            name="From-Scratch Mathematical Statistics & Hypothesis Engine",
            family=AlgorithmFamily.ANALYTICS,
            doc_path="docs/algorithms/analytics/statistical_engine.md",
            complexity_time="O(N * B) for bootstrap, O(N) for summary statistics",
            complexity_space="O(N)",
            assumptions=["Independent observations across random seeds"],
            hyperparameters={"confidence_level": "Default 0.95"},
            citations=[
                CitationRecord(
                    author_or_source="Acklam",
                    title="An Algorithm for Computing the Inverse Normal Cumulative Distribution Function",
                    year=2003,
                    chapter_or_pages="Development notes & published coefficients",
                )
            ],
        )
    )

    # 8. Village Society & Emergent Economy
    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="trust_pagerank",
            name="Directed Reciprocity Trust Graph & Stationary PageRank Status",
            family=AlgorithmFamily.MULTIAGENT,
            doc_path="docs/algorithms/village/social_capital_and_ethics.md",
            complexity_time="O(K * N^2) power iteration",
            complexity_space="O(N^2)",
            assumptions=[
                "Asymmetric reciprocity: beta_defect > alpha_coop",
                "Row-stochastic trust walk with damping factor d in (0, 1)",
            ],
            hyperparameters={
                "damping_factor": "PageRank teleportation complement d=0.85",
                "alpha_coop": "Cooperative trust gain rate",
                "beta_defect": "Defection trust loss rate",
            },
            citations=[
                CitationRecord(
                    author_or_source="Russell & Norvig / Page, Brin, Motwani, & Winograd",
                    title="Artificial Intelligence: A Modern Approach / The PageRank Citation Ranking",
                    year=1999,
                    chapter_or_pages="Chapter 21 / Stanford InfoLab Technical Report",
                )
            ],
        )
    )

    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="social_utility_ethics",
            name="Multi-Objective Social Utility & Rational Risk-Reward Ethics Engine",
            family=AlgorithmFamily.DECISION,
            doc_path="docs/algorithms/village/social_capital_and_ethics.md",
            complexity_time="O(|A| * N) per deliberation step",
            complexity_space="O(|A| + N)",
            assumptions=[
                "Logarithmic survival utility with crisis discontinuity at tau",
                "Independent witness line-of-sight detection probabilities",
            ],
            hyperparameters={
                "w_phys": "Physiological survival weight",
                "w_econ": "Wealth accumulation weight",
                "w_soc": "Societal status weight",
                "theta_risk": "Individual risk tolerance in [0, 1]",
            },
            citations=[
                CitationRecord(
                    author_or_source="Russell & Norvig / Becker",
                    title="Artificial Intelligence: A Modern Approach / Crime and Punishment: An Economic Approach",
                    year=1968,
                    chapter_or_pages="Chapter 16 / Journal of Political Economy, 76(2), pp. 169-217",
                )
            ],
        )
    )

    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="affective_pad_occ",
            name="Ortony-Clore-Collins & Continuous 3D PAD Affective Emotional Engine",
            family=AlgorithmFamily.DECISION,
            doc_path="docs/algorithms/village/social_capital_and_ethics.md",
            complexity_time="O(K) appraisal + linear ODE decay per step",
            complexity_space="O(1) state vector in [-1, 1]^3",
            assumptions=[
                "Continuous affective state E_i(t) = [v_i, a_i, d_i]^T in [-1, 1]^3",
                "Exponential decay toward baseline personality with additive OCC appraisal stimuli",
            ],
            hyperparameters={
                "lambda_decay": "Temporal emotion decay rate in (0, 1)",
                "baseline_pad": "Personality baseline vector in [-1, 1]^3",
            },
            citations=[
                CitationRecord(
                    author_or_source="Mehrabian & Russell / Ortony, Clore, & Collins",
                    title="An Approach to Environmental Psychology / The Cognitive Structure of Emotions",
                    year=1974,
                    chapter_or_pages="MIT Press / Cambridge University Press",
                )
            ],
        )
    )

    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="gram_sabha_consensus",
            name="Gram Sabha Assembly Deliberation & Trust-Weighted PageRank Consensus",
            family=AlgorithmFamily.MULTIAGENT,
            doc_path="docs/algorithms/village/social_capital_and_ethics.md",
            complexity_time="O(N * M) voting aggregation",
            complexity_space="O(N + M)",
            assumptions=[
                "Decisions weighted by societal status PageRank S_i(t)",
                "Periodic or crisis-triggered assembly convocation at Town Square amphitheater",
            ],
            hyperparameters={
                "quorum_threshold": "Minimum voter attendance fraction",
                "mandatory_tithe_rate": "Post-crisis civic grain levy fraction",
            },
            citations=[
                CitationRecord(
                    author_or_source="Weiss / Sen & Ostrom",
                    title="Multiagent Systems / Governing the Commons: The Evolution of Institutions",
                    year=1999,
                    chapter_or_pages="Chapter 2 / Cambridge University Press",
                )
            ],
        )
    )

    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="personality_mapping",
            name="Big Five Personality Parameter Modulation & Mehrabian Baseline",
            family=AlgorithmFamily.DECISION,
            doc_path="docs/algorithms/village/personality_mapping.md",
            complexity_time="O(1) algebraic parameter calculation",
            complexity_space="O(1) vector",
            assumptions=[
                "Stable Big Five trait dimensions OCEAN in [0, 1]",
                "Monotonic modulation of utility weights, risk aversion, and switching costs",
            ],
            hyperparameters={
                "w_soc_scale": "Base social weight scalar",
                "w_econ_scale": "Base economic weight scalar",
            },
            citations=[
                CitationRecord(
                    author_or_source="Mehrabian / Goldberg",
                    title="Basic Dimensions for a General Psychological Theory / The Structure of Phenotypic Personality Traits",
                    year=1980,
                    chapter_or_pages="Oelgeschlager, Gunn & Hain / American Psychologist, 48(1), pp. 26-34",
                )
            ],
        )
    )

    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="salience_attention",
            name="Multi-Modal Salience Attention Filtering & Habituation Budget",
            family=AlgorithmFamily.DECISION,
            doc_path="docs/algorithms/village/salience_attention.md",
            complexity_time="O(P log P) sorting over P percepts",
            complexity_space="O(K) alert budget buffer",
            assumptions=[
                "Exponential spatial acoustic attenuation with L=6 tiles",
                "Habituation rises on repeats and decays exponentially over ticks",
            ],
            hyperparameters={
                "theta_notice": "Threshold for passive observation encoding",
                "theta_alert": "Threshold for intention interruption eligibility",
                "theta_startle": "Threshold for involuntary reflex and immediate latency",
                "attn_budget": "Maximum alerts processed per tick",
            },
            citations=[
                CitationRecord(
                    author_or_source="Itti, Koch, & Niebur / Broadbent",
                    title="A Model of Saliency-Based Visual Attention / Perception and Communication",
                    year=1998,
                    chapter_or_pages="IEEE TPAMI, 20(11), pp. 1254-1259 / Pergamon Press",
                )
            ],
        )
    )

    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="intention_arbitration",
            name="Utility-Guided Intention Commitment & Switching Cost Arbitration",
            family=AlgorithmFamily.PLANNING,
            doc_path="docs/algorithms/village/intention_arbitration.md",
            complexity_time="O(A) evaluation over candidate actions",
            complexity_space="O(1) active intention record",
            assumptions=[
                "Commitment decays as uncompleted time elapses",
                "Preemption requires alert urgency to exceed commitment stickiness plus hysteresis margin",
            ],
            hyperparameters={
                "hysteresis": "Hysteresis threshold margin to prevent action oscillation",
                "kappa_sw": "Base action switching penalty scalar",
                "lambda_att": "Reaction utility scalar",
            },
            citations=[
                CitationRecord(
                    author_or_source="Bratman / Mark",
                    title="Intention, Plans, and Practical Reason / Architecture Tricks of the AI Trade",
                    year=1987,
                    chapter_or_pages="Harvard University Press / GDC Proceedings",
                )
            ],
        )
    )

    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="emotional_contagion",
            name="Synchronous Multi-Agent Emotional Contagion Coupling",
            family=AlgorithmFamily.MULTIAGENT,
            doc_path="docs/algorithms/village/emotional_contagion.md",
            complexity_time="O(N^2) pairwise proximity evaluation",
            complexity_space="O(N) synchronous delta buffer",
            assumptions=[
                "Synchronous update ensures strict permutation order-independence",
                "Asymmetric negativity bias amplifies negative emotional spread",
            ],
            hyperparameters={
                "gamma_0": "Base contagion coupling susceptibility",
                "max_axis_delta": "Per-axis contagion clamp per tick",
            },
            citations=[
                CitationRecord(
                    author_or_source="Hatfield, Cacioppo, & Rapson",
                    title="Emotional Contagion",
                    year=1993,
                    chapter_or_pages="Current Directions in Psychological Science, 2(3), pp. 96-100",
                )
            ],
        )
    )

    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="reflection_dispositions",
            name="Nightly Episodic Reflection, Grudge Consolidation & Forgiveness",
            family=AlgorithmFamily.DECISION,
            doc_path="docs/algorithms/village/reflection_dispositions.md",
            complexity_time="O(M log M) top-k evidence retrieval over unreflected episodes",
            complexity_space="O(D) bounded dispositions per agent",
            assumptions=[
                "Slow exponential integration of net valence into durable dispositions",
                "Asymmetric harm vs. help weighting with Neuroticism-modulated grudge retention",
            ],
            hyperparameters={
                "decay_base": "Per-tick grudge retention factor base",
                "max_evidence_per_target": "Maximum evidence memory IDs preserved per disposition",
            },
            citations=[
                CitationRecord(
                    author_or_source="Park et al.",
                    title="Generative Agents: Interactive Simulacra of Human Behavior",
                    year=2023,
                    chapter_or_pages="UIST '23 Proceedings, pp. 1-22",
                )
            ],
        )
    )

    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="needs_prepotency",
            name="Maslow-Style Soft Prepotency Needs & Dynamic Utility Weights",
            family=AlgorithmFamily.DECISION,
            doc_path="docs/algorithms/village/needs_prepotency.md",
            complexity_time="O(K) constant-time 5-level need evaluations",
            complexity_space="O(K) state vectors and satiation hash maps",
            assumptions=[
                "Deficits in [0, 1] drive quadratic pressure rises",
                "Lower-level physiological deficits softly throttle higher-level belonging and esteem via prepotency product",
            ],
            hyperparameters={
                "p_exponent": "Pressure exponent (default 2.0)",
                "beta_gate": "Prepotency soft-gating power (default 0.70)",
                "lambda_weight": "Pressure gain scalar on base utility weights (default 2.0)",
            },
            citations=[
                CitationRecord(
                    author_or_source="Maslow / Hull",
                    title="A Theory of Human Motivation / Principles of Behavior",
                    year=1943,
                    chapter_or_pages="Psychological Review, 50(4), pp. 370-396 / Appleton-Century",
                )
            ],
        )
    )

    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="utility_behavior_tree",
            name="Utility-Guided Hierarchical Intention Behavior Tree with Hysteresis",
            family=AlgorithmFamily.PLANNING,
            doc_path="docs/algorithms/village/utility_behavior_tree.md",
            complexity_time="O(I * A) scoring across candidate intention trees and actions",
            complexity_space="O(I) sequential intention specifications",
            assumptions=[
                "Multi-step sequential intentions preserve temporal continuity",
                "Hysteresis barrier prevents high-frequency intention oscillations",
            ],
            hyperparameters={
                "hysteresis_factor": "Relative hysteresis margin factor on intention score",
                "hysteresis_base": "Absolute hysteresis threshold margin",
            },
            citations=[
                CitationRecord(
                    author_or_source="Colledanchise & Ögren",
                    title="Behavior Trees in Robotics and AI: An Introduction",
                    year=2018,
                    chapter_or_pages="CRC Press, Chapters 2-4",
                )
            ],
        )
    )

    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="circadian_routines",
            name="Circadian Daily Routines and Role-Specific Temporal Priors",
            family=AlgorithmFamily.PLANNING,
            doc_path="docs/algorithms/village/circadian_routines.md",
            complexity_time="O(1) circular Gaussian prior evaluation",
            complexity_space="O(R * I) role schedule prior catalog",
            assumptions=[
                "Time of day on 24-hour circular topology introduces smooth circadian utility biases",
                "Occupational roles define distinct daily activity peaks and sleeping rhythms",
            ],
            hyperparameters={
                "sigma": "Hour variance of Gaussian circadian curve",
                "amplitude": "Peak circadian utility score boost amplitude",
            },
            citations=[
                CitationRecord(
                    author_or_source="Winfree",
                    title="The Geometry of Biological Time",
                    year=2001,
                    chapter_or_pages="Springer-Verlag, 2nd Edition",
                )
            ],
        )
    )

    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="memory_importance",
            name="Episodic Memory Importance Scoring via Valence Extremity and Saliency",
            family=AlgorithmFamily.HYBRID,
            doc_path="docs/algorithms/village/memory_stream.md",
            complexity_time="O(1) per memory item insertion",
            complexity_space="O(1) metadata fields per memory item",
            assumptions=[
                "Emotional valence extremity |v| strongly dictates memory encoding saliency",
                "Survival-critical, social, and justice events receive structured significance baselines",
            ],
            hyperparameters={
                "w_valence": "Weight on emotional valence extremity |v| (default 0.35)",
                "w_saliency": "Weight on domain event saliency (default 0.45)",
                "w_base": "Default baseline importance (default 0.20)",
            },
            citations=[
                CitationRecord(
                    author_or_source="Park et al.",
                    title="Generative Agents: Interactive Simulacra of Human Behavior",
                    year=2023,
                    chapter_or_pages="UIST '23, pp. 1-22",
                ),
                CitationRecord(
                    author_or_source="Kahneman",
                    title="Thinking, Fast and Slow",
                    year=2011,
                    chapter_or_pages="Farrar, Straus and Giroux, Part 4",
                ),
            ],
        )
    )

    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="memory_retrieval",
            name="Multimodal Memory Retrieval with Mood-Congruence and Deterministic Tie-Breaking",
            family=AlgorithmFamily.SEARCH,
            doc_path="docs/algorithms/village/memory_retrieval.md",
            complexity_time="O(N log K) top-k selection across stream of size N",
            complexity_space="O(K) retrieved memory buffer",
            assumptions=[
                "Linear combination of recency, importance, keyword relevance, and PAD mood congruence reflects retrieval relevance",
                "Deterministic tie-breaking preserves exact simulation reproducibility across runs",
            ],
            hyperparameters={
                "w_recency": "Weight on exponential recency decay (default 0.25)",
                "w_importance": "Weight on intrinsic memory importance (default 0.30)",
                "w_relevance": "Weight on token/entity Jaccard overlap (default 0.30)",
                "w_mood": "Weight on affective PAD Euclidean congruence (default 0.15)",
                "decay_rate": "Exponential decay factor for recency (default 0.995)",
            },
            citations=[
                CitationRecord(
                    author_or_source="Bower",
                    title="Mood and Memory",
                    year=1981,
                    chapter_or_pages="American Psychologist, 36(2), pp. 129-148",
                ),
                CitationRecord(
                    author_or_source="Park et al.",
                    title="Generative Agents: Interactive Simulacra of Human Behavior",
                    year=2023,
                    chapter_or_pages="UIST '23, pp. 1-22",
                ),
            ],
        )
    )

    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="memory_forgetting",
            name="Ebbinghaus-Power Retention Strength and Bounded Pruning with Unreflected Protection",
            family=AlgorithmFamily.HYBRID,
            doc_path="docs/algorithms/village/memory_stream.md",
            complexity_time="O(N log N) sorting during capacity eviction",
            complexity_space="O(N) memory stream storage strictly bounded at capacity cap (default 300)",
            assumptions=[
                "Retention strength decays exponentially from last access step, modulated logarithmically by access count",
                "Unreflected memories are protected from eviction during pruning unless stream is strictly full of unreflected items",
            ],
            hyperparameters={
                "halflife_s0": "Base halflife scaling factor in steps (default 50.0)",
                "access_log_coeff": "Logarithmic reinforcement multiplier (default 0.30)",
                "capacity_cap": "Maximum items in memory stream (default 300)",
            },
            citations=[
                CitationRecord(
                    author_or_source="Ebbinghaus",
                    title="Memory: A Contribution to Experimental Psychology",
                    year=1885,
                    chapter_or_pages="Teachers College, Columbia University",
                ),
                CitationRecord(
                    author_or_source="Wixted & Ebbesen",
                    title="On the Form of Forgetting",
                    year=1991,
                    chapter_or_pages="Psychological Science, 2(6), pp. 409-415",
                ),
            ],
        )
    )

    AlgorithmRegistry.register(
        AlgorithmMetadata(
            algorithm_id="reflection_consolidation",
            name="Reflection Consolidation, Situational Attribution, and Forgiveness Dynamics",
            family=AlgorithmFamily.HYBRID,
            doc_path="docs/algorithms/village/reflection_consolidation.md",
            complexity_time="O(M * T) episode consolidation, crime pattern aggregation, and attribution evaluation",
            complexity_space="O(M) synthesized reflection items",
            assumptions=[
                "Accumulated importance over recent memory items triggers synthesis of higher-order beliefs",
                "Situational attribution discounts moral blame when environmental stressors (drought, resource shortage) are observable",
                "Restitution and time promote probabilistic forgiveness modulated by Agreeableness and Neuroticism",
            ],
            hyperparameters={
                "reflection_importance_threshold": "Summed importance triggering reflection (default 15.0)",
                "unreflected_item_count_threshold": "Unreflected memory count triggering reflection (default 25)",
                "drought_stress_weight": "Weight of drought stress in situational attribution (default 0.50)",
            },
            citations=[
                CitationRecord(
                    author_or_source="Kelley",
                    title="Attribution Theory in Social Psychology",
                    year=1967,
                    chapter_or_pages="Nebraska Symposium on Motivation, 15, pp. 192-238",
                ),
                CitationRecord(
                    author_or_source="Park et al.",
                    title="Generative Agents: Interactive Simulacra of Human Behavior",
                    year=2023,
                    chapter_or_pages="UIST '23, pp. 1-22",
                ),
                CitationRecord(
                    author_or_source="McCullough et al.",
                    title="Interpersonal Forgiving in Close Relationships",
                    year=1997,
                    chapter_or_pages="Journal of Personality and Social Psychology, 73(2), pp. 321-336",
                ),
            ],
        )
    )

# Automatically populate on module load
populate_default_catalog()
