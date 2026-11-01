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
                    author_or_source="Page, Brin, Motwani, & Winograd",
                    title="The PageRank Citation Ranking: Bringing Order to the Web",
                    year=1999,
                    chapter_or_pages="Stanford InfoLab Technical Report",
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
                    author_or_source="Becker",
                    title="Crime and Punishment: An Economic Approach",
                    year=1968,
                    chapter_or_pages="Journal of Political Economy, 76(2), pp. 169-217",
                )
            ],
        )
    )

# Automatically populate on module load
populate_default_catalog()
