"""Propositional logic AST nodes, clauses, and literal representations (SYS-07, Module M08)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable


@runtime_checkable
class Formula(Protocol):
    """Protocol for all propositional logic formulas."""

    def variables(self) -> frozenset[str]:
        """Returns the set of unique variable symbols present in the formula."""
        ...

    def evaluate(self, model: Mapping[str, bool]) -> bool:
        """Evaluates truth value under the given variable assignment model."""
        ...

    def __str__(self) -> str:
        """Returns standard symbolic string representation."""
        ...


@dataclass(frozen=True, slots=True)
class PropVar:
    """Atomic propositional variable symbol (e.g. 'P', 'Safe_1_2')."""

    name: str

    def variables(self) -> frozenset[str]:
        return frozenset({self.name})

    def evaluate(self, model: Mapping[str, bool]) -> bool:
        if self.name not in model:
            raise KeyError(f"Variable '{self.name}' unassigned in truth assignment model.")
        return bool(model[self.name])

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class ConstantBool:
    """Boolean constant truth value: True (Top) or False (Bottom)."""

    value: bool

    def variables(self) -> frozenset[str]:
        return frozenset()

    def evaluate(self, model: Mapping[str, bool]) -> bool:
        return self.value

    def __str__(self) -> str:
        return "True" if self.value else "False"


@dataclass(frozen=True, slots=True)
class Not:
    """Negation connective: ~phi."""

    operand: Formula

    def variables(self) -> frozenset[str]:
        return self.operand.variables()

    def evaluate(self, model: Mapping[str, bool]) -> bool:
        return not self.operand.evaluate(model)

    def __str__(self) -> str:
        if isinstance(self.operand, (PropVar, ConstantBool)):
            return f"~{self.operand}"
        return f"~({self.operand})"


@dataclass(frozen=True, slots=True)
class And:
    """Conjunction connective: phi_1 & phi_2 & ... & phi_n."""

    operands: tuple[Formula, ...]

    def __init__(self, *args: Formula) -> None:
        object.__setattr__(self, "operands", args)

    def variables(self) -> frozenset[str]:
        return frozenset().union(*(op.variables() for op in self.operands))

    def evaluate(self, model: Mapping[str, bool]) -> bool:
        return all(op.evaluate(model) for op in self.operands)

    def __str__(self) -> str:
        if not self.operands:
            return "True"
        return " & ".join(f"({op})" if isinstance(op, (Or, Implies, Equiv)) else str(op) for op in self.operands)


@dataclass(frozen=True, slots=True)
class Or:
    """Disjunction connective: phi_1 | phi_2 | ... | phi_n."""

    operands: tuple[Formula, ...]

    def __init__(self, *args: Formula) -> None:
        object.__setattr__(self, "operands", args)

    def variables(self) -> frozenset[str]:
        return frozenset().union(*(op.variables() for op in self.operands))

    def evaluate(self, model: Mapping[str, bool]) -> bool:
        return any(op.evaluate(model) for op in self.operands)

    def __str__(self) -> str:
        if not self.operands:
            return "False"
        return " | ".join(f"({op})" if isinstance(op, (And, Implies, Equiv)) else str(op) for op in self.operands)


@dataclass(frozen=True, slots=True)
class Implies:
    """Material implication: antecedent => consequent."""

    antecedent: Formula
    consequent: Formula

    def variables(self) -> frozenset[str]:
        return self.antecedent.variables() | self.consequent.variables()

    def evaluate(self, model: Mapping[str, bool]) -> bool:
        if not self.antecedent.evaluate(model):
            return True
        return self.consequent.evaluate(model)

    def __str__(self) -> str:
        left = f"({self.antecedent})" if isinstance(self.antecedent, (Implies, Equiv)) else str(self.antecedent)
        right = f"({self.consequent})" if isinstance(self.consequent, (Implies, Equiv)) else str(self.consequent)
        return f"{left} => {right}"


@dataclass(frozen=True, slots=True)
class Equiv:
    """Biconditional / Equivalence: left <=> right."""

    left: Formula
    right: Formula

    def variables(self) -> frozenset[str]:
        return self.left.variables() | self.right.variables()

    def evaluate(self, model: Mapping[str, bool]) -> bool:
        return self.left.evaluate(model) == self.right.evaluate(model)

    def __str__(self) -> str:
        return f"({self.left}) <=> ({self.right})"


@dataclass(frozen=True, slots=True)
class Literal:
    """Atomic literal in clausal logic: variable or negated variable."""

    variable: str
    is_negated: bool = False

    def complement(self) -> Literal:
        """Returns the complementary literal (~L if L, or L if ~L)."""
        return Literal(self.variable, not self.is_negated)

    def __str__(self) -> str:
        return f"~{self.variable}" if self.is_negated else self.variable


@dataclass(frozen=True, slots=True)
class Clause:
    """Disjunction of literals (C = L_1 | ... | L_m)."""

    literals: frozenset[Literal]

    def __init__(self, literals: frozenset[Literal] | set[Literal] | list[Literal]) -> None:
        object.__setattr__(self, "literals", frozenset(literals))

    @property
    def is_empty(self) -> bool:
        """Returns True if this clause is the empty clause (contradiction / Box)."""
        return len(self.literals) == 0

    @property
    def is_unit(self) -> bool:
        """Returns True if this clause contains exactly one literal."""
        return len(self.literals) == 1

    @property
    def is_tautology(self) -> bool:
        """Returns True if the clause contains both L and ~L for some variable."""
        positives = {lit.variable for lit in self.literals if not lit.is_negated}
        negatives = {lit.variable for lit in self.literals if lit.is_negated}
        return bool(positives & negatives)

    def __str__(self) -> str:
        if self.is_empty:
            return "[]"  # Box / Contradiction
        return " | ".join(str(lit) for lit in sorted(self.literals, key=lambda l: (l.variable, l.is_negated)))

