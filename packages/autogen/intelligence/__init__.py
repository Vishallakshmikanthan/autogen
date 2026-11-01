"""Intelligence package: algorithms from search to deep RL (SYS-06 through SYS-15)."""

from autogen.intelligence import evolution, hybrid, multiagent, neural, search
from autogen.intelligence.registry import (
    AlgorithmFamily,
    AlgorithmMetadata,
    AlgorithmRegistry,
    CitationRecord,
)

__all__ = [
    "search",
    "neural",
    "multiagent",
    "evolution",
    "hybrid",
    "AlgorithmRegistry",
    "AlgorithmMetadata",
    "AlgorithmFamily",
    "CitationRecord",
]
