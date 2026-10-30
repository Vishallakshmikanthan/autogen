"""Propositional resolution refutation theorem prover with proof provenance (SYS-07, Module M08)."""

from __future__ import annotations

from autogen.intelligence.logic.ast import And, Clause, Formula, Not
from autogen.intelligence.logic.cnf import to_cnf
from autogen.intelligence.logic.provenance import ProofRule, ProofStep, ProofTree


def pl_resolve(ci: Clause, cj: Clause) -> list[Clause]:
    """Resolves two clauses on complementary literals, returning all valid non-tautological resolvents."""
    resolvents: list[Clause] = []

    for lit_i in ci.literals:
        comp_i = lit_i.complement()
        if comp_i in cj.literals:
            # Found complementary pair lit_i and comp_i
            new_lits = (ci.literals - {lit_i}) | (cj.literals - {comp_i})
            resolvent = Clause(new_lits)
            if not resolvent.is_tautology:
                resolvents.append(resolvent)

    return resolvents


def pl_resolution(
    kb_formula: Formula,
    query_formula: Formula,
    max_steps: int = 5000,
) -> tuple[bool, ProofTree | None]:
    """Proves KB |= alpha via resolution refutation by deriving the empty clause from KB & ~alpha.

    Returns:
        (True, proof_tree) if entailed, or (False, None) if not entailed.
    """
    # 1. Assert KB & ~query
    refutation_sentence = And(kb_formula, Not(query_formula))
    initial_clauses = to_cnf(refutation_sentence)

    # Provenance tracking: Clause -> step_id
    clause_to_step: dict[Clause, int] = {}
    proof_steps: list[ProofStep] = []
    step_counter = 0

    # Record initial axioms / premises
    clause_set: set[Clause] = set()
    for c in initial_clauses:
        step_counter += 1
        clause_to_step[c] = step_counter
        proof_steps.append(
            ProofStep(
                step_id=step_counter,
                conclusion=str(c),
                rule=ProofRule.AXIOM,
                premises=(),
            )
        )
        clause_set.add(c)
        if c.is_empty:
            # Immediate contradiction in premises
            return True, ProofTree(
                query=str(query_formula),
                is_proved=True,
                steps=tuple(proof_steps),
                root_step_id=step_counter,
            )

    # 2. Main resolution loop
    iterations = 0
    while iterations < max_steps:
        iterations += 1
        new_clauses: set[Clause] = set()
        clause_list = list(clause_set)
        n = len(clause_list)

        found_empty = False
        empty_step_id = None

        for i in range(n):
            for j in range(i + 1, n):
                ci = clause_list[i]
                cj = clause_list[j]
                resolvents = pl_resolve(ci, cj)

                for res in resolvents:
                    if res not in clause_set and res not in new_clauses:
                        step_counter += 1
                        clause_to_step[res] = step_counter
                        rule = ProofRule.REFUTATION_CONTRADICTION if res.is_empty else ProofRule.RESOLUTION
                        proof_steps.append(
                            ProofStep(
                                step_id=step_counter,
                                conclusion=str(res),
                                rule=rule,
                                premises=(clause_to_step[ci], clause_to_step[cj]),
                            )
                        )
                        new_clauses.add(res)

                        if res.is_empty:
                            found_empty = True
                            empty_step_id = step_counter
                            break
                if found_empty:
                    break
            if found_empty:
                break

        if found_empty and empty_step_id is not None:
            # Extract only relevant steps leading to empty clause
            relevant_steps = _filter_relevant_proof_steps(proof_steps, empty_step_id)
            return True, ProofTree(
                query=str(query_formula),
                is_proved=True,
                steps=tuple(relevant_steps),
                root_step_id=empty_step_id,
            )

        if new_clauses.issubset(clause_set):
            # No new resolvents can be derived: KB does not entail query
            return False, None

        clause_set.update(new_clauses)

    return False, None


def _filter_relevant_proof_steps(steps: list[ProofStep], target_step_id: int) -> list[ProofStep]:
    """Filters proof steps to keep only those that contributed to deriving target_step_id."""
    step_dict = {s.step_id: s for s in steps}
    needed_ids: set[int] = set()

    def mark_dependencies(sid: int) -> None:
        if sid in needed_ids or sid not in step_dict:
            return
        needed_ids.add(sid)
        for premise in step_dict[sid].premises:
            mark_dependencies(premise)

    mark_dependencies(target_step_id)
    return [s for s in steps if s.step_id in needed_ids]

