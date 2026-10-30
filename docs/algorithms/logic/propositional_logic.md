# Propositional Logic AST, Parser, and Truth-Table Semantics

> **Module:** M08 — Knowledge Representation & Logical Reasoning (SYS-07)  
> **Status:** Production / Verified

---

## 1. Theory & Mathematical Definition
Propositional logic evaluates sentences constructed from atomic proposition symbols $P, Q, R, \dots$ and logical connectives $\{\neg, \land, \lor, \implies, \iff\}$. A model $M: V \to \{0, 1\}$ assigns a truth value to each symbol. A formula $\alpha$ is valid (tautology) if $M \models \alpha$ for all models $M$. A knowledge base $KB$ semantically entails $\alpha$ ($KB \models \alpha$) if every model satisfying $KB$ also satisfies $\alpha$.

## 2. Assumptions
1. **Classical Bivalent Logic:** Every proposition is either strictly True or False (Law of Excluded Middle; Principle of Non-Contradiction).
2. **Compositional Semantics:** Truth value of compound formulas depends purely on the truth values of sub-formulas and truth-functional connectives.

## 3. Equations & Variable Definitions

Truth functions:
$$\text{eval}(M, P) = M(P)$$
$$\text{eval}(M, \neg \phi) = 1 - \text{eval}(M, \phi)$$
$$\text{eval}(M, \phi \land \psi) = \min(\text{eval}(M, \phi), \text{eval}(M, \psi))$$
$$\text{eval}(M, \phi \lor \psi) = \max(\text{eval}(M, \phi), \text{eval}(M, \psi))$$
$$\text{eval}(M, \phi \implies \psi) = \max(1 - \text{eval}(M, \phi), \text{eval}(M, \psi))$$
$$\text{eval}(M, \phi \iff \psi) = \mathbb{I}[\text{eval}(M, \phi) = \text{eval}(M, \psi)]$$

### Semantic Entailment Condition
$$KB \models \alpha \iff \forall M \in \{0, 1\}^{|V|}, \quad (M \models KB \implies M \models \alpha)$$
Equivalently:
$$KB \models \alpha \iff KB \land \neg \alpha \text{ is unsatisfiable}$$

## 4. Derivation & CNF Equivalence
Any formula can be converted into an equisatisfiable Conjunctive Normal Form (CNF) $\bigwedge_{i} \bigvee_{j} L_{i,j}$ via three semantics-preserving rewrite stages:
1. Eliminate $\iff$ and $\implies$:
   $$A \iff B \equiv (\neg A \lor B) \land (\neg B \lor A)$$
   $$A \implies B \equiv \neg A \lor B$$
2. De Morgan's laws (pushing $\neg$ inward):
   $$\neg (A \land B) \equiv \neg A \lor \neg B, \quad \neg (A \lor B) \equiv \neg A \land \neg B, \quad \neg \neg A \equiv A$$
3. Distribute $\lor$ over $\land$:
   $$A \lor (B \land C) \equiv (A \lor B) \land (A \lor C)$$

## 5. Pseudocode (Truth-Table Entailment)

```text
function TT-Entails(KB, alpha):
    symbols = KB.variables() union alpha.variables()
    return TT-Check-All(KB, alpha, symbols, {})

function TT-Check-All(KB, alpha, symbols, model):
    if symbols is empty:
        if KB.evaluate(model):
            return alpha.evaluate(model)
        return True  # Vacuously true when KB is False
    P = FIRST(symbols)
    rest = REST(symbols)
    return (TT-Check-All(KB, alpha, rest, model + {P: True}) and
            TT-Check-All(KB, alpha, rest, model + {P: False}))
```

## 6. Computational Complexity
- **Truth-Table Time Complexity:** $\mathcal{O}(2^n)$, where $n$ is the number of distinct proposition symbols.
- **Truth-Table Space Complexity:** $\mathcal{O}(n)$ recursive call stack depth.
- **CNF Conversion Time:** Worst-case $\mathcal{O}(2^m)$ due to repeated disjunction distribution, but linear for Horn and bounded formulas.

## 7. Implementation Notes
- The parser implements a recursive descent grammar with precedence levels: $\iff$ (lowest), $\implies$, $\lor$, $\land$, $\neg$ (highest).
- Literals and Clauses are immutable `frozenset` collections with tautology elimination.

## 8. Verification Strategy & Test Oracles
1. **Model Checking Invariants:** Verify $(P \implies Q) \equiv (\neg P \lor Q)$ and $P \lor \neg P \equiv \top$.
2. **Entailment Oracle:** $P \land (P \implies Q) \models Q$ (Modus Ponens) verified by exhaustive truth table.
3. **Equivalence of CNF:** For 5,000 random formulas, $\text{eval}(M, \phi) == \text{eval}(M, \text{to\_cnf}(\phi))$ for all models.

## 9. Known Limitations
- Truth-table enumeration scales as $2^n$ and is unsuitable for large symbol sets ($n > 20$); DPLL and Resolution must be used for scalability.

## 10. References
- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed., Chapter 7).
