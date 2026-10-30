"""Unit tests for First-Order Logic terms, unification, and occurs-check (Module M08)."""

from __future__ import annotations

import pytest

from autogen.intelligence.logic.fol import (
    Constant,
    Function,
    Predicate,
    Variable,
    compose_substitutions,
    occurs_check,
    unify,
)


def test_standard_unification_oracle() -> None:
    """Oracle verification: Knows(x, Food) with Knows(Agent1, Food) gives {x: Agent1}."""
    # Predicates: Knows(x, Food) and Knows(Agent1, Food)
    p1 = Predicate("Knows", [Variable("x"), Constant("Food")])
    p2 = Predicate("Knows", [Constant("Agent1"), Constant("Food")])

    theta = unify(p1, p2)
    assert theta is not None
    assert "x" in theta
    assert theta["x"] == Constant("Agent1")


def test_occurs_check_fails_correctly() -> None:
    """Oracle verification: Unifying x with f(x) must fail (return None) due to occurs check."""
    var_x = Variable("x")
    fx = Function("f", [Variable("x")])

    # Direct occurs check
    assert occurs_check("x", fx, {}) is True

    # Unification must return None (failure)
    theta = unify(var_x, fx)
    assert theta is None

    # Nested occurs check: unify P(x, y) with P(g(y), f(x))
    p1 = Predicate("P", [Variable("x"), Variable("y")])
    p2 = Predicate("P", [Function("g", [Variable("y")]), Function("f", [Variable("x")])])
    theta_nested = unify(p1, p2)
    assert theta_nested is None


def test_compound_unification_and_chains() -> None:
    """Tests variable chaining and multi-argument unification."""
    # P(x, B, x) with P(A, y, z) -> {x: A, y: B, z: A}
    p1 = Predicate("P", [Variable("x"), Constant("B"), Variable("x")])
    p2 = Predicate("P", [Constant("A"), Variable("y"), Variable("z")])

    theta = unify(p1, p2)
    assert theta is not None
    assert theta["x"] == Constant("A")
    assert theta["y"] == Constant("B")
    assert theta["z"] == Constant("A")


def test_substitution_composition() -> None:
    # theta1 = {x: A, y: z}
    # theta2 = {z: B}
    # composed = {x: A, y: B, z: B}
    t1 = {"x": Constant("A"), "y": Variable("z")}
    t2 = {"z": Constant("B")}

    comp = compose_substitutions(t1, t2)
    assert comp["x"] == Constant("A")
    assert comp["y"] == Constant("B")
    assert comp["z"] == Constant("B")
