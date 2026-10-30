"""Unit tests for Propositional Logic AST, Lexer, and Parser (Module M08)."""

from __future__ import annotations

import pytest

from autogen.intelligence.logic.ast import (
    And,
    ConstantBool,
    Equiv,
    Implies,
    Not,
    Or,
    PropVar,
)
from autogen.intelligence.logic.parser import parse_formula, tokenize


def test_tokenize_basic() -> None:
    tokens = tokenize("P & (Q | ~R) => Safe_1")
    types = [t.token_type.value for t in tokens[:-1]]
    assert types == ["VAR", "AND", "LPAREN", "VAR", "OR", "NOT", "VAR", "RPAREN", "IMPLIES", "VAR"]


def test_parser_roundtrip() -> None:
    expr = parse_formula("A & B | ~C => D <=> E")
    assert isinstance(expr, Equiv)
    assert isinstance(expr.left, Implies)
    assert isinstance(expr.left.antecedent, Or)


def test_operator_precedence() -> None:
    # ~ binds tighter than &
    e1 = parse_formula("~A & B")
    assert isinstance(e1, And)
    assert isinstance(e1.operands[0], Not)

    # & binds tighter than |
    e2 = parse_formula("A & B | C & D")
    assert isinstance(e2, Or)
    assert len(e2.operands) == 2
    assert isinstance(e2.operands[0], And)
    assert isinstance(e2.operands[1], And)

    # | binds tighter than =>
    e3 = parse_formula("A | B => C")
    assert isinstance(e3, Implies)
    assert isinstance(e3.antecedent, Or)

    # => is right-associative: A => B => C is A => (B => C)
    e4 = parse_formula("A => B => C")
    assert isinstance(e4, Implies)
    assert isinstance(e4.consequent, Implies)


def test_evaluation_truth_values() -> None:
    f = parse_formula("(P | Q) & ~R")
    assert f.evaluate({"P": True, "Q": False, "R": False}) is True
    assert f.evaluate({"P": False, "Q": False, "R": False}) is False
    assert f.evaluate({"P": True, "Q": True, "R": True}) is False


def test_syntax_errors() -> None:
    with pytest.raises(SyntaxError):
        parse_formula("A & (B | ")

    with pytest.raises(SyntaxError):
        parse_formula("A & & B")
