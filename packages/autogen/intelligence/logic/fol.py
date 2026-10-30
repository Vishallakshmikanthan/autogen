"""First-order logic terms, predicates, unification with occurs-check, and substitution (SYS-07, Module M08)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Term(Protocol):
    """Protocol for First-Order Logic terms (variables, constants, function terms)."""

    @property
    def name(self) -> str: ...

    @property
    def is_variable(self) -> bool: ...

    def substitute(self, theta: dict[str, Term]) -> Term: ...

    def __str__(self) -> str: ...


@dataclass(frozen=True, slots=True)
class Variable:
    """Logical variable term (e.g. 'x', 'y')."""

    name: str

    @property
    def is_variable(self) -> bool:
        return True

    def substitute(self, theta: dict[str, Term]) -> Term:
        if self.name in theta:
            return theta[self.name].substitute(theta)
        return self

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class Constant:
    """Ground constant term (e.g. 'Agent1', 'Food', 'Cell_0_0')."""

    name: str

    @property
    def is_variable(self) -> bool:
        return False

    def substitute(self, theta: dict[str, Term]) -> Term:
        return self

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class Function:
    """Compound function term: f(t_1, ..., t_k)."""

    name: str
    args: tuple[Term, ...]

    def __init__(self, name: str, args: Sequence[Term]) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "args", tuple(args))

    @property
    def is_variable(self) -> bool:
        return False

    def substitute(self, theta: dict[str, Term]) -> Term:
        return Function(self.name, [arg.substitute(theta) for arg in self.args])

    def __str__(self) -> str:
        return f"{self.name}({', '.join(str(a) for a in self.args)})"


@dataclass(frozen=True, slots=True)
class Predicate:
    """First-order atomic predicate: P(t_1, ..., t_k)."""

    name: str
    args: tuple[Term, ...]

    def __init__(self, name: str, args: Sequence[Term]) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "args", tuple(args))

    def substitute(self, theta: dict[str, Term]) -> Predicate:
        return Predicate(self.name, [arg.substitute(theta) for arg in self.args])

    def __str__(self) -> str:
        return f"{self.name}({', '.join(str(a) for a in self.args)})"


def occurs_check(var_name: str, term: Term, theta: dict[str, Term]) -> bool:
    """Returns True if variable var_name occurs inside term under substitution theta."""
    if term.is_variable:
        if term.name == var_name:
            return True
        if term.name in theta:
            return occurs_check(var_name, theta[term.name], theta)
        return False

    if isinstance(term, Function):
        return any(occurs_check(var_name, arg, theta) for arg in term.args)

    return False


def unify(
    x: Term | Predicate | Sequence[Term],
    y: Term | Predicate | Sequence[Term],
    theta: dict[str, Term] | None = None,
) -> dict[str, Term] | None:
    """Most General Unifier (MGU) algorithm with strict occurs-check.

    Returns the substitution dict theta unifying x and y, or None if not unifiable.
    """
    if theta is None:
        theta = {}

    if x == y:
        return theta

    # Both are sequences of arguments
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        if len(x) != len(y):
            return None
        curr_theta = dict(theta)
        for arg_x, arg_y in zip(x, y):
            sub_res = unify(arg_x, arg_y, curr_theta)
            if sub_res is None:
                return None
            curr_theta = sub_res
        return curr_theta

    # Predicates
    if isinstance(x, Predicate) and isinstance(y, Predicate):
        if x.name != y.name or len(x.args) != len(y.args):
            return None
        return unify(x.args, y.args, theta)

    # Variables
    if isinstance(x, Term) and x.is_variable:
        return _unify_var(x.name, y, theta)
    if isinstance(y, Term) and y.is_variable:
        return _unify_var(y.name, x, theta)

    # Compound functions
    if isinstance(x, Function) and isinstance(y, Function):
        if x.name != y.name or len(x.args) != len(y.args):
            return None
        return unify(x.args, y.args, theta)

    return None


def _unify_var(var_name: str, x: Any, theta: dict[str, Term]) -> dict[str, Term] | None:
    if var_name in theta:
        return unify(theta[var_name], x, theta)

    if isinstance(x, Term) and x.is_variable and x.name in theta:
        return unify(Variable(var_name), theta[x.name], theta)

    if isinstance(x, Term) and occurs_check(var_name, x, theta):
        # Occurs-check failed: cyclic infinite term prevented
        return None

    new_theta = dict(theta)
    new_theta[var_name] = x
    return new_theta


def compose_substitutions(
    theta1: dict[str, Term],
    theta2: dict[str, Term],
) -> dict[str, Term]:
    """Composes two substitutions theta = theta1 o theta2 such that x(theta1 o theta2) == (x theta1) theta2."""
    composed: dict[str, Term] = {}
    for k, v in theta1.items():
        composed[k] = v.substitute(theta2)
    for k, v in theta2.items():
        if k not in composed:
            composed[k] = v
    return composed
