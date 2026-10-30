"""Davis-Putnam-Logemann-Loveland (DPLL) SAT solver algorithm (SYS-07, Module M08)."""

from __future__ import annotations

from typing import Mapping

from autogen.intelligence.logic.ast import Clause, Literal


def dpll_satisfiable(
    clauses: frozenset[Clause],
    symbols: frozenset[str] | None = None,
    model: dict[str, bool] | None = None,
) -> tuple[bool, dict[str, bool]]:
    """Determines satisfiability of a set of CNF clauses using the recursive DPLL algorithm.

    Returns:
        (True, satisfying_model) if satisfiable, or (False, {}) if unsatisfiable.
    """
    if model is None:
        model = {}

    if symbols is None:
        all_syms: set[str] = set()
        for c in clauses:
            for lit in c.literals:
                all_syms.add(lit.variable)
        symbols = frozenset(all_syms)

    # 1. Check clause status under current partial model
    all_clauses_true = True
    for c in clauses:
        clause_val = _eval_clause_under_partial_model(c, model)
        if clause_val is False:
            return False, {}  # Conflict: at least one clause is completely violated
        if clause_val is None:
            all_clauses_true = False

    if all_clauses_true:
        # All clauses satisfied! Assign any remaining unassigned symbols arbitrarily (e.g. False)
        complete_model = dict(model)
        for s in symbols:
            if s not in complete_model:
                complete_model[s] = False
        return True, complete_model

    # 2. Pure literal elimination heuristic
    pure_sym, pure_val = _find_pure_symbol(symbols, clauses, model)
    if pure_sym is not None:
        new_model = dict(model)
        new_model[pure_sym] = pure_val
        return dpll_satisfiable(clauses, symbols - {pure_sym}, new_model)

    # 3. Unit clause propagation heuristic
    unit_sym, unit_val = _find_unit_clause(clauses, model)
    if unit_sym is not None:
        new_model = dict(model)
        new_model[unit_sym] = unit_val
        return dpll_satisfiable(clauses, symbols - {unit_sym}, new_model)

    # 4. Branching heuristic: pick first unassigned symbol and branch
    unassigned = [s for s in sorted(symbols) if s not in model]
    if not unassigned:
        return False, {}

    p = unassigned[0]
    rest_syms = symbols - {p}

    # Branch True first
    model_true = dict(model)
    model_true[p] = True
    sat, res_model = dpll_satisfiable(clauses, rest_syms, model_true)
    if sat:
        return True, res_model

    # Branch False
    model_false = dict(model)
    model_false[p] = False
    return dpll_satisfiable(clauses, rest_syms, model_false)


def _eval_clause_under_partial_model(clause: Clause, model: Mapping[str, bool]) -> bool | None:
    """Evaluates a clause under a partial assignment.

    Returns:
        True: if at least one literal is satisfied.
        False: if all literals are falsified.
        None: if undecided (no literal satisfied, but some literals remain unassigned).
    """
    has_unassigned = False
    for lit in clause.literals:
        if lit.variable in model:
            val = model[lit.variable]
            lit_val = not val if lit.is_negated else val
            if lit_val is True:
                return True
        else:
            has_unassigned = True

    return None if has_unassigned else False


def _find_pure_symbol(
    symbols: frozenset[str],
    clauses: frozenset[Clause],
    model: Mapping[str, bool],
) -> tuple[str | None, bool]:
    """Finds a symbol that appears with only one polarity across all currently unsatisfied clauses."""
    # Collect unsatisfied clauses
    active_clauses = [c for c in clauses if _eval_clause_under_partial_model(c, model) is not True]

    for sym in sorted(symbols):
        if sym in model:
            continue
        has_pos = False
        has_neg = False
        for c in active_clauses:
            for lit in c.literals:
                if lit.variable == sym:
                    if lit.is_negated:
                        has_neg = True
                    else:
                        has_pos = True

        if has_pos and not has_neg:
            return sym, True
        if has_neg and not has_pos:
            return sym, False

    return None, False


def _find_unit_clause(
    clauses: frozenset[Clause],
    model: Mapping[str, bool],
) -> tuple[str | None, bool]:
    """Finds an unsatisfied clause with exactly one unassigned literal."""
    for c in clauses:
        if _eval_clause_under_partial_model(c, model) is True:
            continue

        unassigned_lits: list[Literal] = []
        for lit in c.literals:
            if lit.variable not in model:
                unassigned_lits.append(lit)

        if len(unassigned_lits) == 1:
            unit_lit = unassigned_lits[0]
            # Must satisfy this literal
            return unit_lit.variable, not unit_lit.is_negated

    return None, False
