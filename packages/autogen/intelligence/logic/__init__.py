"""Knowledge Representation & Logical Reasoning engine (SYS-07, Module M08)."""

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
from autogen.intelligence.logic.cnf import to_cnf
from autogen.intelligence.logic.dpll import dpll_satisfiable
from autogen.intelligence.logic.fol import (
    Constant,
    Function,
    Predicate,
    Term,
    Variable,
    compose_substitutions,
    occurs_check,
    unify,
)
from autogen.intelligence.logic.horn import (
    DefiniteClause,
    pl_bc_entails,
    pl_fc_entails,
)
from autogen.intelligence.logic.knowledge_base import KnowledgeBase
from autogen.intelligence.logic.parser import parse_formula, tokenize
from autogen.intelligence.logic.provenance import (
    ProofRule,
    ProofStep,
    ProofTree,
)
from autogen.intelligence.logic.resolution import pl_resolution, pl_resolve
from autogen.intelligence.logic.truth_table import (
    enumerate_models,
    is_satisfiable,
    is_valid,
    tt_entails,
)

__all__ = [
    "And",
    "Clause",
    "Constant",
    "ConstantBool",
    "DefiniteClause",
    "Equiv",
    "Formula",
    "Function",
    "Implies",
    "KnowledgeBase",
    "Literal",
    "Not",
    "Or",
    "Predicate",
    "ProofRule",
    "ProofStep",
    "ProofTree",
    "PropVar",
    "Term",
    "Variable",
    "compose_substitutions",
    "dpll_satisfiable",
    "enumerate_models",
    "is_satisfiable",
    "is_valid",
    "occurs_check",
    "parse_formula",
    "pl_bc_entails",
    "pl_fc_entails",
    "pl_resolution",
    "pl_resolve",
    "to_cnf",
    "tokenize",
    "tt_entails",
    "unify",
]
