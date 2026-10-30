"""Truth-table enumeration, model checking, and semantic entailment (SYS-07, Module M08)."""

from __future__ import annotations

import itertools
from typing import Iterator, Sequence

from autogen.intelligence.logic.ast import Formula


def enumerate_models(symbols: Sequence[str]) -> Iterator[dict[str, bool]]:
    """Generates all 2^N complete truth assignment models for the specified variable symbols."""
    sorted_syms = sorted(symbols)
    n = len(sorted_syms)
    for bits in itertools.product([False, True], repeat=n):
        yield dict(zip(sorted_syms, bits))


def is_valid(formula: Formula) -> bool:
    """Returns True if the formula is a tautology (satisfied by every possible assignment model)."""
    syms = list(formula.variables())
    for model in enumerate_models(syms):
        if not formula.evaluate(model):
            return False
    return True


def is_satisfiable(formula: Formula) -> tuple[bool, dict[str, bool] | None]:
    """Returns True and a satisfying model if one exists, otherwise (False, None)."""
    syms = list(formula.variables())
    for model in enumerate_models(syms):
        if formula.evaluate(model):
            return True, model
    return False, None


def tt_entails(kb_formula: Formula, query_formula: Formula) -> bool:
    """Determines semantic entailment KB |= alpha via exhaustive truth-table model checking.

    Invariant: KB |= alpha iff for all models M, M |= KB implies M |= alpha.
    """
    all_syms = list(kb_formula.variables() | query_formula.variables())
    for model in enumerate_models(all_syms):
        if kb_formula.evaluate(model):
            if not query_formula.evaluate(model):
                return False  # Countermodel found: KB is True but alpha is False
    return True


def generate_truth_table_report(formula: Formula) -> list[dict[str, bool]]:
    """Generates a complete tabular record of truth values across all variable models."""
    syms = sorted(formula.variables())
    report: list[dict[str, bool]] = []
    for model in enumerate_models(syms):
        row = dict(model)
        row["__result__"] = formula.evaluate(model)
        report.append(row)
    return report
