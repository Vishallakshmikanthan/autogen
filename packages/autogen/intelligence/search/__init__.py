"""Search algorithms and problem solving package (SYS-06, Module M06)."""

from autogen.intelligence.search.contracts import (
    HeuristicFn,
    SearchNode,
    SearchProblem,
    SearchResult,
    SearchStepEvent,
    reconstruct_path,
)
from autogen.intelligence.search.duel import AlgorithmDuelRunner
from autogen.intelligence.search.grid_problem import GridSearchProblem
from autogen.intelligence.search.heuristics import (
    chebyshev_distance,
    euclidean_distance,
    manhattan_distance,
    null_heuristic,
    octile_distance,
)
from autogen.intelligence.search.informed import (
    AStarSearch,
    GreedyBestFirstSearch,
    IterativeDeepeningAStar,
    WeightedAStarSearch,
)
from autogen.intelligence.search.local import (
    HillClimbingSearch,
    LocalBeamSearch,
    SimulatedAnnealingSearch,
)
from autogen.intelligence.search.uninformed import (
    BidirectionalBFS,
    BreadthFirstSearch,
    DepthFirstSearch,
    DepthLimitedSearch,
    IterativeDeepeningSearch,
    UniformCostSearch,
)

__all__ = [
    "AStarSearch",
    "AlgorithmDuelRunner",
    "BidirectionalBFS",
    "BreadthFirstSearch",
    "DepthFirstSearch",
    "DepthLimitedSearch",
    "GridSearchProblem",
    "GreedyBestFirstSearch",
    "HeuristicFn",
    "HillClimbingSearch",
    "IterativeDeepeningAStar",
    "IterativeDeepeningSearch",
    "LocalBeamSearch",
    "SearchNode",
    "SearchProblem",
    "SearchResult",
    "SearchStepEvent",
    "SimulatedAnnealingSearch",
    "UniformCostSearch",
    "WeightedAStarSearch",
    "chebyshev_distance",
    "euclidean_distance",
    "manhattan_distance",
    "null_heuristic",
    "octile_distance",
    "reconstruct_path",
]

