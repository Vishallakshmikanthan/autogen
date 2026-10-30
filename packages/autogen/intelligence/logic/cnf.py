"""Conjunctive Normal Form (CNF) transformation engine (SYS-07, Module M08)."""

from __future__ import annotations

from autogen.intelligence.logic.ast import (
    And,
    Clause,
    ConstantBool,
    Equiv,
    Formula,
    Implies,
    Literal,
    Not,
    Or,
    PropVar,
)


def eliminate_implications(formula: Formula) -> Formula:
    """Recursively replaces material implications and biconditionals with basic connectives (~, &, |)."""
    if isinstance(formula, PropVar) or isinstance(formula, ConstantBool):
        return formula

    if isinstance(formula, Not):
        return Not(eliminate_implications(formula.operand))

    if isinstance(formula, And):
        return And(*(eliminate_implications(op) for op in formula.operands))

    if isinstance(formula, Or):
        return Or(*(eliminate_implications(op) for op in formula.operands))

    if isinstance(formula, Implies):
        # A => B  becomes  ~A | B
        ant = eliminate_implications(formula.antecedent)
        cons = eliminate_implications(formula.consequent)
        return Or(Not(ant), cons)

    if isinstance(formula, Equiv):
        # A <=> B becomes (~A | B) & (~B | A)
        left = eliminate_implications(formula.left)
        right = eliminate_implications(formula.right)
        return And(Or(Not(left), right), Or(Not(right), left))

    raise TypeError(f"Unknown formula node type: {type(formula)}")


def move_not_inwards(formula: Formula) -> Formula:
    """Applies De Morgan's laws and double negation elimination recursively."""
    if isinstance(formula, (PropVar, ConstantBool)):
        return formula

    if isinstance(formula, Not):
        op = formula.operand
        # Double negation: ~~A -> A
        if isinstance(op, Not):
            return move_not_inwards(op.operand)
        # De Morgan: ~(A & B) -> ~A | ~B
        if isinstance(op, And):
            return Or(*(move_not_inwards(Not(arg)) for arg in op.operands))
        # De Morgan: ~(A | B) -> ~A & ~B
        if isinstance(op, Or):
            return And(*(move_not_inwards(Not(arg)) for arg in op.operands))
        # Negated constant
        if isinstance(op, ConstantBool):
            return ConstantBool(not op.value)
        # Negated variable
        return formula

    if isinstance(formula, And):
        return And(*(move_not_inwards(op) for op in formula.operands))

    if isinstance(formula, Or):
        return Or(*(move_not_inwards(op) for op in formula.operands))

    return formula


def distribute_or_over_and(formula: Formula) -> Formula:
    """Distributes disjunctions over conjunctions: A | (B & C) -> (A | B) & (A | C)."""
    if isinstance(formula, (PropVar, ConstantBool, Not)):
        return formula

    if isinstance(formula, And):
        return And(*(distribute_or_over_and(op) for op in formula.operands))

    if isinstance(formula, Or):
        # Recursively distribute children first
        distributed_ops = [distribute_or_over_and(op) for op in formula.operands]

        # Find first operand that is an And
        for i, op in enumerate(distributed_ops):
            if isinstance(op, And):
                # We have (op_1 | ... | (A1 & A2) | ... | op_n)
                # Combine other operands into rest
                others = distributed_ops[:i] + distributed_ops[i + 1:]
                rest = Or(*others) if len(others) > 1 else (others[0] if others else ConstantBool(False))

                new_operands = [distribute_or_over_and(Or(rest, and_arg)) for and_arg in op.operands]
                return distribute_or_over_and(And(*new_operands))

        return Or(*distributed_ops)

    return formula


def _extract_and_operands(formula: Formula) -> list[Formula]:
    """Recursively unpacks nested And operators into a flat list."""
    if isinstance(formula, And):
        res: list[Formula] = []
        for op in formula.operands:
            res.extend(_extract_and_operands(op))
        return res
    return [formula]


def _extract_or_operands(formula: Formula) -> list[Formula]:
    """Recursively unpacks nested Or operators into a flat list."""
    if isinstance(formula, Or):
        res: list[Formula] = []
        for op in formula.operands:
            res.extend(_extract_or_operands(op))
        return res
    return [formula]


def flatten_to_clauses(formula: Formula) -> frozenset[Clause]:
    """Flattens a distributed CNF formula into a frozenset of Clause objects."""
    if isinstance(formula, ConstantBool):
        if formula.value:
            return frozenset()  # Empty conjunction is True
        else:
            return frozenset({Clause([])})  # Empty clause is False

    # Extract top-level conjunction recursively
    and_operands = _extract_and_operands(formula)
    clauses: set[Clause] = set()

    for item in and_operands:
        or_operands = _extract_or_operands(item)
        literals: set[Literal] = set()
        is_true_clause = False

        for term in or_operands:
            if isinstance(term, PropVar):
                literals.add(Literal(term.name, is_negated=False))
            elif isinstance(term, Not) and isinstance(term.operand, PropVar):
                literals.add(Literal(term.operand.name, is_negated=True))
            elif isinstance(term, ConstantBool):
                if term.value:
                    is_true_clause = True
                    break
                else:
                    continue
            else:
                raise ValueError(f"Term {term} could not be converted to a literal in CNF clause.")

        if is_true_clause:
            continue

        clause = Clause(literals)
        # Filter out tautological clauses (containing P and ~P)
        if not clause.is_tautology:
            clauses.add(clause)

    return frozenset(clauses)


def to_cnf(formula: Formula) -> frozenset[Clause]:
    """Transforms an arbitrary propositional formula into an equisatisfiable Conjunctive Normal Form (CNF)."""
    f1 = eliminate_implications(formula)
    f2 = move_not_inwards(f1)
    f3 = distribute_or_over_and(f2)
    return flatten_to_clauses(f3)
