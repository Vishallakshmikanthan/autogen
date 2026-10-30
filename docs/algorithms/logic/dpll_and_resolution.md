# DPLL SAT Solver & Propositional Resolution Refutation

> **Module:** M08 — Knowledge Representation & Logical Reasoning (SYS-07)  
> **Status:** Production / Verified

---

## 1. Theory & Mathematical Definition
Propositional Resolution Refutation is an automated theorem proving technique operating on clauses. To prove that a knowledge base $KB$ entails query $\alpha$, we assert $KB \land \neg \alpha$, convert the resulting sentence to CNF, and repeatedly apply the resolution inference rule:
$$\frac{l_1 \lor \dots \lor l_i \lor \dots \lor l_k, \quad m_1 \lor \dots \lor \neg l_i \lor \dots \lor m_n}{l_1 \lor \dots \lor l_{i-1} \lor l_{i+1} \dots \lor l_k \lor m_1 \lor \dots \lor m_{n}}$$
If the empty clause $\square$ (Box / False) is derived, a contradiction is established, proving $KB \models \alpha$.

DPLL (Davis-Putnam-Logemann-Loveland) is a backtracking search algorithm deciding satisfiability of CNF formulas by leveraging unit propagation and pure literal elimination.

## 2. Assumptions
1. **Refutation Completeness:** Propositional resolution is sound and refutation complete: $KB \models \alpha \iff KB \land \neg \alpha \vdash \square$.
2. **Clausal Form:** Input formulas are converted to CNF clauses before provers are invoked.

## 3. Equations & Variable Definitions

Let $C_1, C_2$ be two clauses with complementary literals $L \in C_1$ and $\neg L \in C_2$. The resolvent $R(C_1, C_2)$ is:
$$R(C_1, C_2) = (C_1 \setminus \{L\}) \cup (C_2 \setminus \{\neg L\})$$
When $C_1 = \{L\}$ and $C_2 = \{\neg L\}$, $R(C_1, C_2) = \emptyset = \square$.

### Unit Clause Rule (DPLL)
If clause $C$ contains exactly one unassigned literal $L$ while all other literals in $C$ are falsified under partial model $M$:
$$M' = M \cup \{L \mapsto 1\}$$

### Pure Literal Rule (DPLL)
If literal $L$ appears in active clauses but $\neg L$ does not appear in any active clause:
$$M' = M \cup \{L \mapsto 1\}$$

## 4. Derivation of Soundness
If $C_1$ and $C_2$ are satisfied under model $M$, then either $L$ is true or $L$ is false. If $L$ is true, then $\neg L$ is false, so some other literal in $C_2$ must be true. If $L$ is false, then some other literal in $C_1$ must be true. In either case, at least one literal in $(C_1 \setminus \{L\}) \cup (C_2 \setminus \{\neg L\})$ must be true under $M$. Thus, $C_1 \land C_2 \models R(C_1, C_2)$.

## 5. Pseudocode

```text
function PL-Resolution(KB, alpha):
    clauses = TO-CNF(KB & ~alpha)
    new = {}
    loop:
        for each pair C_i, C_j in clauses:
            resolvents = PL-Resolve(C_i, C_j)
            if empty clause in resolvents:
                return (True, ReconstructProofTree(C_i, C_j))
            new = new union resolvents
        if new is subset of clauses:
            return (False, None)
        clauses = clauses union new
```

## 6. Computational Complexity
- **Resolution Space & Time:** Worst-case exponential in number of propositions, but polynomial for Horn clauses.
- **DPLL Time:** Significantly faster than naive truth tables on structured industrial formulas through unit propagation.

## 7. Implementation Notes
- Subsumption elimination and tautology elimination ($P \lor \neg P$) are applied to prune redundant clauses.
- Every derived resolvent tracks parent clause indices to construct a complete provenance `ProofTree`.

## 8. Verification Strategy & Test Oracles
1. **5,000 Random Formulas Oracle:**
   $$\forall \phi, \quad \text{DPLL}(\text{CNF}(\phi)) \iff \text{TruthTableSatisfiable}(\phi)$$
   $$KB \models_{\text{TT}} \alpha \iff \text{PL-Resolution}(KB, \alpha)[0] == \text{True}$$
2. **Contradiction Oracle:** Asserting $P$ and $\neg P$ directly produces empty clause $\square$ with `ProofRule.REFUTATION_CONTRADICTION`.

## 9. Known Limitations
- General resolution can generate many intermediate clauses if heuristics like unit resolution are not prioritized.

## 10. References
- Davis, M., Logemann, G., & Loveland, D. (1962). A machine program for theorem-proving. *Communications of the ACM*, 5(7), 394-397.
- Robinson, J. A. (1965). A machine-oriented logic based on the resolution principle. *Journal of the ACM*, 12(1), 23-41.

