"""Unit tests for truth-table model checking, validity, and entailment (Module M08)."""

from __future__ import annotations

import pytest

from autogen.intelligence.logic.parser import parse_formula
from autogen.intelligence.logic.truth_table import (
    enumerate_models,
    is_satisfiable,
    is_valid,
    tt_entails,
)


def test_tautology_validity() -> None:
    # Excluded middle: P | ~P
    assert is_valid(parse_formula("P | ~P"))
    # Modus Ponens tautology: ((P => Q) & P) => Q
    assert is_valid(parse_formula("((P => Q) & P) => Q"))
    # De Morgan equivalence: ~(P & Q) <=> (~P | ~Q)
    assert is_valid(parse_formula("~(P & Q) <=> (~P | ~Q)"))
    # Contingency is not valid
    assert not is_valid(parse_formula("P => Q"))


def test_satisfiability() -> None:
    # P & ~P is unsatisfiable
    sat1, model1 = is_satisfiable(parse_formula("P & ~P"))
    assert sat1 is False
    assert model1 is None

    # (P | Q) & ~P is satisfiable by {P: False, Q: True}
    sat2, model2 = is_satisfiable(parse_formula("(P | Q) & ~P"))
    assert sat2 is True
    assert model2 is not None
    assert model2["P"] is False
    assert model2["Q"] is True


def test_semantic_entailment_modus_ponens() -> None:
    kb = parse_formula("(P => Q) & P")
    query = parse_formula("Q")
    assert tt_entails(kb, query) is True

    # False conclusion does not follow
    query_false = parse_formula("~Q")
    assert tt_entails(kb, query_false) is False


def test_semantic_entailment_wumpus_style() -> None:
    # B_1_1 <=> (P_1_2 | P_2_1)
    # ~B_1_1
    # Query: ~P_1_2 and ~P_2_1
    kb = parse_formula("(B_1_1 <=> (P_1_2 | P_2_1)) & ~B_1_1")
    assert tt_entails(kb, parse_formula("~P_1_2")) is True
    assert tt_entails(kb, parse_formula("~P_2_1")) is True
    assert tt_entails(kb, parse_formula("P_1_2")) is False

