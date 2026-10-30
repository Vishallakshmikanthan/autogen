"""Unit tests for KnowledgeBase consistency, deduction, and contradiction detection (Module M08)."""

from __future__ import annotations

import pytest

from autogen.intelligence.logic.knowledge_base import KnowledgeBase
from autogen.intelligence.logic.parser import parse_formula


def test_kb_tell_retract_ask() -> None:
    kb = KnowledgeBase()
    kb.tell("P => Q")
    kb.tell("P")

    # Ask Q -> True with proof tree
    is_proved, proof = kb.ask("Q")
    assert is_proved is True
    assert proof is not None
    assert proof.is_proved is True
    assert len(proof.steps) > 0

    # Retract P -> Q no longer provable
    retracted = kb.retract("P")
    assert retracted is True

    is_proved_after, _ = kb.ask("Q")
    assert is_proved_after is False


def test_kb_contradiction_detection_oracle() -> None:
    """Oracle verification: Contradictory KB is detected, producing a valid refutation proof."""
    kb = KnowledgeBase()
    kb.tell("Safe_1_1")
    kb.tell("Safe_1_1 => ~Hazard")
    kb.tell("Hazard")

    # Consistent check must detect contradiction
    is_consistent, proof = kb.is_consistent()
    assert is_consistent is False
    assert proof is not None
    assert proof.is_proved is True
    # Verify provenance contains refutation contradiction step
    contradiction_steps = [s for s in proof.steps if s.conclusion == "[]"]
    assert len(contradiction_steps) > 0


def test_kb_empty_consistency() -> None:
    kb = KnowledgeBase()
    is_consistent, proof = kb.is_consistent()
    assert is_consistent is True
    assert proof is None

