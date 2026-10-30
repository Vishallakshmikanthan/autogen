# Horn Clause Knowledge Bases, Forward & Backward Chaining

> **Module:** M08 — Knowledge Representation & Logical Reasoning (SYS-07)  
> **Status:** Production / Verified

---

## 1. Theory & Mathematical Definition
A Horn clause is a disjunction of literals containing at most one positive literal. A definite clause has exactly one positive literal:
$$\neg p_1 \lor \neg p_2 \lor \dots \lor \neg p_k \lor q \iff (p_1 \land p_2 \land \dots \land p_k) \implies q$$
When $k=0$, the clause represents an unconditional atomic fact $q$. Horn clauses represent the logical foundation of deductive databases, production rule systems, and logic programming (e.g., Prolog).

## 2. Forward Chaining vs Backward Chaining

- **Forward Chaining (Data-Driven):** Starts from known ground facts, repeatedly fires rules whose premises are satisfied (Modus Ponens), adding their conclusions to the agenda until the goal is proved or no further conclusions follow. Operates in linear time $\mathcal{O}(N)$ where $N$ is KB length.
- **Backward Chaining (Goal-Driven):** Starts from the target query, matches it against rule heads, and recursively attempts to prove all premises. Avoids deriving irrelevant facts.

## 3. Equations & Variable Definitions

Rule representation:
$$\text{Premises}(C) = \{p_1, \dots, p_k\}, \quad \text{Conclusion}(C) = q$$
Modus Ponens inference rule:
$$\frac{p_1, \dots, p_k, \quad (p_1 \land \dots \land p_k) \implies q}{q}$$

## 4. Complexity & Invariants
- **Forward Chaining Time Complexity:** $\mathcal{O}(N)$, where each clause $C$ is processed at most $|\text{Premises}(C)|$ times by maintaining a decrementing counter table.
- **Space Complexity:** $\mathcal{O}(S)$ where $S$ is the number of distinct proposition symbols.

## 5. Pseudocode

```text
function PL-FC-Entails(KB, query):
    count = {c: len(c.premises) for c in KB}
    inferred = {s: False for s in symbols}
    agenda = queue of facts
    while agenda is not empty:
        p = agenda.pop()
        if p == query: return True
        if not inferred[p]:
            inferred[p] = True
            for each clause c in KB where p in c.premises:
                count[c] -= 1
                if count[c] == 0:
                    agenda.push(c.conclusion)
    return False
```

## 6. Implementation Notes
- Both forward and backward chaining record step-by-step rule applications with premise IDs into `ProofTree` for full provenance.
- Backward chaining incorporates cycle detection to prevent infinite recursion on mutual implications ($A \implies B$ and $B \implies A$).

## 7. Verification Strategy & Test Oracles
1. **Chaining Agreement Oracle:** For any Horn KB and query $q$:
   $$\text{PL-FC-Entails}(KB, q)[0] == \text{PL-BC-Entails}(KB, q)[0]$$
2. **Resolution Equivalence:** On Horn KBs, Forward Chaining, Backward Chaining, and Resolution refutation yield identical entailment results.

## 8. Known Limitations
- Restricted to Horn clauses (cannot directly represent general disjunctions $A \lor B$).

## 9. References
- Dowling, W. F., & Gallier, J. H. (1984). Linear-time algorithms for testing the satisfiability of propositional Horn formulae. *The Journal of Logic Programming*, 1(3), 267-284.

