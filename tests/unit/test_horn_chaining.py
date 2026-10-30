"""Unit tests for Horn clause Forward and Backward Chaining (Module M08)."""

from __future__ import annotations

import pytest

from autogen.intelligence.logic.horn import (
    DefiniteClause,
    pl_bc_entails,
    pl_fc_entails,
)


def _build_horn_kb() -> list[DefiniteClause]:
    # A & B => L
    # A & P => L
    # B & L => M
    # L & M => P
    # P => Q
    # Facts: A, B
    return [
        DefiniteClause(("A", "B"), "L"),
        DefiniteClause(("A", "P"), "L"),
        DefiniteClause(("B", "L"), "M"),
        DefiniteClause(("L", "M"), "P"),
        DefiniteClause(("P",), "Q"),
        DefiniteClause((), "A"),
        DefiniteClause((), "B"),
    ]


def test_horn_forward_and_backward_chaining_agreement() -> None:
    """Oracle verification: Forward and Backward chaining agree on all Horn queries."""
    kb = _build_horn_kb()
    all_symbols = ["A", "B", "L", "M", "P", "Q", "Unreachable_Z", "Fake_X"]

    for sym in all_symbols:
        fc_result, fc_proof = pl_fc_entails(kb, sym)
        bc_result, bc_proof = pl_bc_entails(kb, sym)

        assert fc_result == bc_result, f"Chaining disagreement on symbol '{sym}': FC={fc_result}, BC={bc_result}"

        if fc_result:
            assert fc_proof is not None
            assert fc_proof.is_proved is True
            assert bc_proof is not None
            assert bc_proof.is_proved is True


def test_horn_chaining_cycle_resilience() -> None:
    """Verifies that cyclic rules (A => B, B => A) do not cause infinite recursion in backward chaining."""
    cyclic_kb = [
        DefiniteClause(("A",), "B"),
        DefiniteClause(("B",), "A"),
        DefiniteClause((), "C"),
    ]
    # Query A: not provable, must terminate cleanly
    bc_result, bc_proof = pl_bc_entails(cyclic_kb, "A")
    assert bc_result is False
    assert bc_proof is None
