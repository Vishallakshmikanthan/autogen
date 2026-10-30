"""Mathematical verification of DPLL and Resolution Refutation against Truth-Table oracle on 5000 random formulas (Module M08)."""

from __future__ import annotations

import random
import pytest

from autogen.intelligence.logic.ast import (
    And,
    Equiv,
    Formula,
    Implies,
    Not,
    Or,
    PropVar,
)
from autogen.intelligence.logic.cnf import to_cnf
from autogen.intelligence.logic.dpll import dpll_satisfiable
from autogen.intelligence.logic.resolution import pl_resolution
from autogen.intelligence.logic.truth_table import is_satisfiable, tt_entails


def generate_random_formula(
    symbols: list[str],
    max_depth: int = 2,
    rng: random.Random | None = None,
) -> Formula:
    """Recursively generates a random propositional logic formula."""
    if rng is None:
        rng = random.Random()

    if max_depth <= 0 or rng.random() < 0.3:
        return PropVar(rng.choice(symbols))

    r = rng.random()
    if r < 0.2:
        return Not(generate_random_formula(symbols, max_depth - 1, rng))
    elif r < 0.4:
        return And(
            generate_random_formula(symbols, max_depth - 1, rng),
            generate_random_formula(symbols, max_depth - 1, rng),
        )
    elif r < 0.6:
        return Or(
            generate_random_formula(symbols, max_depth - 1, rng),
            generate_random_formula(symbols, max_depth - 1, rng),
        )
    elif r < 0.8:
        return Implies(
            generate_random_formula(symbols, max_depth - 1, rng),
            generate_random_formula(symbols, max_depth - 1, rng),
        )
    else:
        return Equiv(
            generate_random_formula(symbols, max_depth - 1, rng),
            generate_random_formula(symbols, max_depth - 1, rng),
        )


def test_dpll_vs_truth_table_satisfiability_5000_formulas() -> None:
    """Oracle verification: DPLL SAT agrees with Truth-Table satisfiability on 5000 random formulas."""
    symbols = ["A", "B", "C", "D"]
    rng = random.Random(42)

    for i in range(5000):
        formula = generate_random_formula(symbols, max_depth=2, rng=rng)
        clauses = to_cnf(formula)

        dpll_sat, model = dpll_satisfiable(clauses, symbols=frozenset(formula.variables()))
        tt_sat, _ = is_satisfiable(formula)

        assert dpll_sat == tt_sat, (
            f"Trial {i}: DPLL={dpll_sat} != TT={tt_sat} on formula: {formula}"
        )
        if dpll_sat:
            for v in formula.variables():
                if v not in model:
                    model[v] = False
            # Verify that the model produced by DPLL actually satisfies the original formula
            assert formula.evaluate(model) is True, f"Trial {i}: DPLL model does not satisfy original formula"


def test_resolution_refutation_vs_truth_table_entailment() -> None:
    """Oracle verification: Resolution refutation theorem prover matches Truth-Table semantic entailment."""
    symbols = ["P", "Q", "R"]
    rng = random.Random(1337)

    for i in range(500):
        kb = generate_random_formula(symbols, max_depth=2, rng=rng)
        query = generate_random_formula(symbols, max_depth=1, rng=rng)

        tt_res = tt_entails(kb, query)
        res_proved, proof = pl_resolution(kb, query)

        assert res_proved == tt_res, (
            f"Trial {i}: Resolution={res_proved} != TT={tt_res} for KB={kb} |= {query}"
        )
        if res_proved:
            assert proof is not None
            assert proof.is_proved is True
