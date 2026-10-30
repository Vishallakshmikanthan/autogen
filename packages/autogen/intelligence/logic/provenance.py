"""Proof provenance contracts, step records, and proof tree models (SYS-07, Module M08)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProofRule(str, Enum):
    """Canonical inference rules utilized in deductive derivations."""

    AXIOM = "Axiom"
    MODUS_PONENS = "ModusPonens"
    RESOLUTION = "Resolution"
    UNIT_PROPAGATION = "UnitPropagation"
    AND_ELIMINATION = "AndElimination"
    FORWARD_CHAINING = "ForwardChaining"
    BACKWARD_CHAINING = "BackwardChaining"
    REFUTATION_CONTRADICTION = "RefutationContradiction"


@dataclass(frozen=True, slots=True)
class ProofStep:
    """Atomic deductive inference step with premise references and optional substitution."""

    step_id: int
    conclusion: str
    rule: ProofRule
    premises: tuple[int, ...] = field(default_factory=tuple)
    substitution: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProofTree:
    """Complete provenance tree reconstructing the derivation of a query sentence."""

    query: str
    is_proved: bool
    steps: tuple[ProofStep, ...]
    root_step_id: int | None = None

    def pretty_print(self) -> str:
        """Renders an indented textual representation of the derivation proof."""
        if not self.is_proved:
            return f"Proof for '{self.query}': FAILED (Query not entailed)"

        lines = [f"=== Proof Derivation for: {self.query} ==="]
        step_dict = {s.step_id: s for s in self.steps}

        for s in self.steps:
            premise_str = f" from steps {list(s.premises)}" if s.premises else ""
            subst_str = f" with {s.substitution}" if s.substitution else ""
            lines.append(f" [{s.step_id}] {s.conclusion}  via {s.rule.value}{premise_str}{subst_str}")

        return "\n".join(lines)
