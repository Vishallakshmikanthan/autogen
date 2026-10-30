# Constraint Satisfaction Problems & Arc Consistency (AC-3)

## 1. Theory & Mathematical Definition

A **Constraint Satisfaction Problem (CSP)** consists of a triple $\langle X, D, C \rangle$:
- $X = \{X_1, X_2, \dots, X_n\}$: a finite set of variables.
- $D = \{D_1, D_2, \dots, D_n\}$: a set of finite domains, where each $D_i$ denotes the set of allowable values for variable $X_i$.
- $C = \{C_1, C_2, \dots, C_m\}$: a set of constraints. Each constraint $C_k = \langle \text{scope}(C_k), R_k \rangle$ specifies a relation $R_k$ over a subset of variables $\text{scope}(C_k) \subseteq X$.

An **assignment** is a mapping from a subset of variables to values: $\{X_i \mapsto v_i \mid v_i \in D_i\}$. An assignment is **consistent** (or legal) if it violates no constraints in $C$. An assignment is **complete** if all variables in $X$ are assigned. A **solution** is a complete, consistent assignment.

### Arc Consistency (AC-3)
A variable $X_i$ is **arc-consistent** with respect to another variable $X_j$ if for every value $x \in D_i$, there exists at least one value $y \in D_j$ such that the pair $(x, y)$ satisfies all binary constraints on $(X_i, X_j)$.
A CSP is arc-consistent if every directed arc $(X_i, X_j)$ is arc-consistent.

$$
\forall x \in D_i, \quad \exists y \in D_j \quad \text{s.t.} \quad (X_i = x, X_j = y) \models C_{ij}
$$

---

## 2. Assumptions

1. Domains $D_i$ are discrete and finite.
2. Constraints are deterministic predicates over their scope.
3. Arcs are directed pairs $(X_i, X_j)$ where $X_j \in \text{neighbors}(X_i)$.

---

## 3. Equations & Algorithms

### AC-3 (Mackworth 1977)
Let $Q$ be the queue of all directed arcs in the CSP:
$$
Q = \{(X_i, X_j) \mid X_i \in X, X_j \in \text{neighbors}(X_i)\}
$$

While $Q$ is not empty, pop $(X_i, X_j)$ and revise $D_i$:
$$
\text{Revise}(X_i, X_j): \quad D_i \leftarrow \{x \in D_i \mid \exists y \in D_j \text{ with } \text{satisfied}(x, y)\}
$$
If $D_i$ is modified, re-enqueue all incoming arcs $(X_k, X_i)$ for $X_k \in \text{neighbors}(X_i) \setminus \{X_j\}$. If $|D_i| = 0$, the CSP is inconsistent.

### Heuristics
- **Minimum Remaining Values (MRV)**: Choose variable $X \in X_{\text{unassigned}}$ minimizing $|D(X)|$.
- **Degree Heuristic**: Tie-breaker choosing $X$ with the maximum number of constraints on remaining unassigned variables.
- **Least Constraining Value (LCV)**: Order values $v \in D(X)$ by the minimum number of values ruled out in neighboring unassigned variables.

---

## 4. Pseudocode

```text
function BACKTRACKING-SEARCH(csp):
    return BACKTRACK(csp, {})

function BACKTRACK(csp, assignment):
    if |assignment| == |csp.variables|:
        return assignment
    var = SELECT-UNASSIGNED-VARIABLE(csp, assignment)
    for val in ORDER-DOMAIN-VALUES(csp, var, assignment):
        if IS-CONSISTENT(csp, var, val, assignment):
            assignment[var] = val
            pruned = INFERENCE(csp, var, val, assignment)
            if pruned is not None:
                result = BACKTRACK(csp, assignment)
                if result is not None:
                    return result
            del assignment[var]
            RESTORE-DOMAINS(csp, pruned)
    return None
```

---

## 5. Computational Complexity

- **AC-3 Time Complexity**: $O(c \cdot d^3)$, where $c$ is the number of binary constraints and $d = \max |D_i|$.
- **AC-3 Space Complexity**: $O(c)$ for arc queue storage.
- **Backtracking Search Time**: $O(d^n)$ worst-case; with MRV, LCV, and Forward Checking/AC-3, typical average-case search complexity is reduced by orders of magnitude.

---

## 6. Implementation Notes

- Indexed constraint lookup `constraints_by_var` avoids scanning irrelevant constraints.
- Pruning sets track exact values removed at each step to allow $O(1)$ backtrack restorations without deep object cloning.

---

## 7. Verification Strategy & Test Oracles

- **Independent Constraint Oracle**: `verify_csp_solution` iterates over all constraints and variable domains independently of the solver state.
- **AC-3 Support Oracle**: Verified against exhaustive brute-force pairwise support checks.
- **Benchmark Suites**: Map Coloring (Australia and random planar maps), $N$-Queens ($N=8, 16$), and 9x9 Sudoku.

---

## 8. Known Limitations

- High-arity non-binary constraints require decomposition or generalized arc consistency (GAC).
- Local search (`min_conflicts`) can get trapped in local plateaus without random restarts.

---

## 9. References

- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.), Chapter 5.
- Mackworth, A. K. (1977). Consistency in networks of relations. *Artificial Intelligence*, 8(1), 99-118.

