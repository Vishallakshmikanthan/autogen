"""Unit tests for Conjunctive Normal Form (CNF) transformation (Module M08)."""

from __future__ import annotations

import pytest

from autogen.intelligence.logic.cnf import (
    distribute_or_over_and,
    eliminate_implications,
    move_not_inwards,
    to_cnf,
)
from autogen.intelligence.logic.parser import parse_formula
from autogen.intelligence.logic.truth_table import enumerate_models


def _assert_cnf_equivalent(orig_str: str) -> None:
    formula = parse_formula(orig_str)
    clauses = to_cnf(formula)

    # Check truth equivalence on all variable assignments
    syms = list(formula.variables())
    for m in enumerate_models(syms):
        orig_val = formula.evaluate(m)
        # Evaluate CNF clauses manually
        cnf_val = True
        for c in clauses:
            clause_val = False
            for lit in c.literals:
                val = m[lit.variable]
                if lit.is_negated:
                    val = not val
                if val:
                    clause_val = True
                    break
            if not clause_val:
                cnf_val = False
                break
        assert orig_val == cnf_val, f"CNF mismatch on model {m} for '{orig_str}'"


def test_cnf_basic_transformations() -> None:
    _assert_cnf_equivalent("P => Q")
    _assert_cnf_equivalent("P <=> Q")
    _assert_cnf_equivalent("~(P & Q)")
    _assert_cnf_equivalent("~(P | Q)")
    _assert_cnf_equivalent("P | (Q & R)")
    _assert_cnf_equivalent("(A & B) | (C & D)")
    _assert_cnf_equivalent("(P => Q) & (Q => R)")


def test_cnf_tautology_removal() -> None:
    # P | ~P is a tautology; should yield empty CNF (trivially satisfied)
    clauses = to_cnf(parse_formula("P | ~P"))
    assert len(clauses) == 0


def test_cnf_contradiction() -> None:
    # P & ~P should yield contradiction (resolvable to empty clause)
    clauses = to_cnf(parse_formula("P & ~P"))
    assert len(clauses) == 2

