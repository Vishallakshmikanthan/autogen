"""Knowledge Base engine with tell, retract, ask, and contradiction detection (SYS-07, Module M08)."""

from __future__ import annotations

from autogen.intelligence.logic.ast import And, ConstantBool, Formula, Not
from autogen.intelligence.logic.cnf import to_cnf
from autogen.intelligence.logic.dpll import dpll_satisfiable
from autogen.intelligence.logic.parser import parse_formula
from autogen.intelligence.logic.provenance import ProofRule, ProofStep, ProofTree
from autogen.intelligence.logic.resolution import pl_resolution


class KnowledgeBase:
    """Symbolic Knowledge Base storing sentences, checking consistency, and answering queries with provenance."""

    def __init__(self, initial_sentences: list[Formula | str] | None = None) -> None:
        self._sentences: list[Formula] = []
        if initial_sentences:
            for s in initial_sentences:
                self.tell(s)

    def tell(self, sentence: Formula | str) -> None:
        """Adds a fact or rule to the Knowledge Base."""
        f = parse_formula(sentence) if isinstance(sentence, str) else sentence
        self._sentences.append(f)

    def retract(self, sentence: Formula | str) -> bool:
        """Removes a sentence from the Knowledge Base if present. Returns True if removed."""
        target = parse_formula(sentence) if isinstance(sentence, str) else sentence
        target_str = str(target)
        for i, s in enumerate(self._sentences):
            if str(s) == target_str:
                self._sentences.pop(i)
                return True
        return False

    def ask(self, query: Formula | str) -> tuple[bool, ProofTree | None]:
        """Queries whether KB |= alpha. Returns (is_entailed, proof_tree)."""
        q = parse_formula(query) if isinstance(query, str) else query
        if not self._sentences:
            return False, None

        kb_formula = self._conjoined_kb()
        return pl_resolution(kb_formula, q)

    def is_consistent(self) -> tuple[bool, ProofTree | None]:
        """Checks whether the Knowledge Base is consistent (free of internal contradiction).

        Returns:
            (True, None) if consistent, or (False, proof_tree) with contradiction derivation if contradictory.
        """
        if not self._sentences:
            return True, None

        kb_formula = self._conjoined_kb()
        clauses = to_cnf(kb_formula)

        # Check satisfiability via DPLL
        sat, _ = dpll_satisfiable(clauses)
        if sat:
            return True, None

        # Contradiction exists: derive proof of contradiction via resolution against False
        _, proof = pl_resolution(kb_formula, ConstantBool(False))
        return False, proof

    def sentences(self) -> list[Formula]:
        """Returns a copy of all sentences currently in the Knowledge Base."""
        return list(self._sentences)

    def _conjoined_kb(self) -> Formula:
        if not self._sentences:
            return ConstantBool(True)
        if len(self._sentences) == 1:
            return self._sentences[0]
        return And(*self._sentences)

