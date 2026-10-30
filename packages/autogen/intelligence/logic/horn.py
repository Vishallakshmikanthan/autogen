"""Horn clause knowledge representation, forward chaining, and backward chaining (SYS-07, Module M08)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

from autogen.intelligence.logic.provenance import ProofRule, ProofStep, ProofTree


@dataclass(frozen=True, slots=True)
class DefiniteClause:
    """Propositional definite clause: (p_1 & p_2 & ... & p_k) => q.

    If premises is empty, represents a ground fact 'q'.
    """

    premises: tuple[str, ...]
    conclusion: str

    @property
    def is_fact(self) -> bool:
        return len(self.premises) == 0

    def __str__(self) -> str:
        if self.is_fact:
            return self.conclusion
        return f"{' & '.join(self.premises)} => {self.conclusion}"


def pl_fc_entails(
    clauses: Sequence[DefiniteClause],
    query: str,
) -> tuple[bool, ProofTree | None]:
    """Forward chaining algorithm determining entailment for propositional Horn KBs in linear time."""
    count: dict[int, int] = {i: len(c.premises) for i, c in enumerate(clauses)}
    inferred: dict[str, bool] = {}
    agenda: deque[str] = deque()

    # Track provenance
    proof_steps: list[ProofStep] = []
    symbol_to_step: dict[str, int] = {}
    step_counter = 0

    # 1. Enqueue all initial facts
    for i, c in enumerate(clauses):
        if c.is_fact:
            if c.conclusion not in symbol_to_step:
                step_counter += 1
                symbol_to_step[c.conclusion] = step_counter
                proof_steps.append(
                    ProofStep(
                        step_id=step_counter,
                        conclusion=c.conclusion,
                        rule=ProofRule.AXIOM,
                        premises=(),
                    )
                )
                agenda.append(c.conclusion)

    # 2. Process agenda
    while agenda:
        p = agenda.popleft()
        if p == query:
            return True, ProofTree(
                query=query,
                is_proved=True,
                steps=tuple(proof_steps),
                root_step_id=symbol_to_step[p],
            )

        if not inferred.get(p, False):
            inferred[p] = True
            for i, c in enumerate(clauses):
                if p in c.premises:
                    count[i] -= 1
                    if count[i] == 0:
                        q = c.conclusion
                        if q not in symbol_to_step:
                            step_counter += 1
                            symbol_to_step[q] = step_counter
                            premise_steps = tuple(symbol_to_step[prem] for prem in c.premises)
                            proof_steps.append(
                                ProofStep(
                                    step_id=step_counter,
                                    conclusion=q,
                                    rule=ProofRule.MODUS_PONENS,
                                    premises=premise_steps,
                                )
                            )
                            agenda.append(q)

    return False, None


def pl_bc_entails(
    clauses: Sequence[DefiniteClause],
    query: str,
) -> tuple[bool, ProofTree | None]:
    """Backward chaining algorithm determining entailment from goal query backwards to facts."""
    proof_steps: list[ProofStep] = []
    step_counter = 0
    visited_goals: set[str] = set()

    def bc_prove(goal: str, path: set[str]) -> int | None:
        nonlocal step_counter

        # Cycle detection
        if goal in path:
            return None

        # Check if already a known ground fact
        for c in clauses:
            if c.is_fact and c.conclusion == goal:
                step_counter += 1
                sid = step_counter
                proof_steps.append(
                    ProofStep(
                        step_id=sid,
                        conclusion=goal,
                        rule=ProofRule.AXIOM,
                        premises=(),
                    )
                )
                return sid

        # Search for rules that conclude this goal
        for c in clauses:
            if c.conclusion == goal and not c.is_fact:
                sub_premises: list[int] = []
                all_proved = True
                new_path = path | {goal}

                for prem in c.premises:
                    prem_step = bc_prove(prem, new_path)
                    if prem_step is None:
                        all_proved = False
                        break
                    sub_premises.append(prem_step)

                if all_proved:
                    step_counter += 1
                    sid = step_counter
                    proof_steps.append(
                        ProofStep(
                            step_id=sid,
                            conclusion=goal,
                            rule=ProofRule.BACKWARD_CHAINING,
                            premises=tuple(sub_premises),
                        )
                    )
                    return sid

        return None

    root_id = bc_prove(query, set())
    if root_id is not None:
        return True, ProofTree(
            query=query,
            is_proved=True,
            steps=tuple(proof_steps),
            root_step_id=root_id,
        )
    return False, None

