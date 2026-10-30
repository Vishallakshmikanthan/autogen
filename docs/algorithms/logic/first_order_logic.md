# First-Order Logic Unification & Substitution

> **Module:** M08 — Knowledge Representation & Logical Reasoning (SYS-07)  
> **Status:** Production / Verified

---

## 1. Theory & Mathematical Definition
First-Order Logic (FOL) enriches propositional logic with terms (Constants, Variables, and Functions) and Predicates representing relations over objects. The core algorithmic operation for automated deduction in FOL is **Unification**: finding a variable substitution $\theta$ that makes two syntactic expressions identical:
$$\text{UNIFY}(\phi, \psi) = \theta \iff \phi \theta = \psi \theta$$

## 2. Most General Unifier (MGU) & Occurs Check
A substitution $\theta$ is a Most General Unifier if any other unifying substitution $\sigma$ can be obtained by further substituting through $\theta$ ($\sigma = \theta \circ \delta$).

### The Occurs-Check Invariant
Before binding variable $x$ to term $t$, the algorithm must verify that $x$ does not appear in $t$. If $x$ occurs in $t$ (e.g. attempting to unify $x$ with $f(x)$), binding $x \mapsto f(x)$ creates an infinite recursive term $x = f(f(f(\dots)))$, which is unsound in first-order semantics. The occurs check rejects such unifications, returning failure (`None`).

## 3. Equations & Variable Definitions

Let $\theta = \{v_1/t_1, \dots, v_n/t_n\}$ and $\sigma = \{w_1/s_1, \dots, w_m/s_m\}$. The composition $\theta \circ \sigma$ is:
$$\theta \circ \sigma = \{v_i / (t_i \sigma) \mid v_i \neq t_i \sigma\} \cup \{w_j / s_j \mid w_j \notin \{v_1, \dots, v_n\}\}$$

Unification satisfies:
$$\text{UNIFY}(P(t_1, \dots, t_k), P(s_1, \dots, s_k)) = \theta \implies \forall i \in \{1, \dots, k\}, \quad t_i \theta = s_i \theta$$

## 4. Pseudocode

```text
function UNIFY(x, y, theta):
    if theta is failure: return failure
    else if x == y: return theta
    else if IS-VARIABLE(x): return UNIFY-VAR(x, y, theta)
    else if IS-VARIABLE(y): return UNIFY-VAR(y, x, theta)
    else if IS-COMPOUND(x) and IS-COMPOUND(y):
        if op(x) != op(y) or length(args(x)) != length(args(y)): return failure
        return UNIFY(args(x), args(y), UNIFY(op(x), op(y), theta))
    else: return failure

function UNIFY-VAR(var, x, theta):
    if var in theta: return UNIFY(theta[var], x, theta)
    else if x is variable and x in theta: return UNIFY(var, theta[x], theta)
    else if OCCURS-CHECK(var, x, theta): return failure
    else: return theta + {var: x}
```

## 5. Computational Complexity
- **Time Complexity:** Linear to almost-linear $\mathcal{O}(n \cdot \alpha(n))$ with efficient term DAG sharing, or $\mathcal{O}(n^2)$ with occurs check on standard tree structures.
- **Space Complexity:** $\mathcal{O}(d)$ recursion depth.

## 6. Implementation Notes
- Pure functional substitution composition and term recursion.
- All terms (`Variable`, `Constant`, `Function`, `Predicate`) are immutable slots dataclasses.

## 7. Verification Strategy & Test Oracles
1. **Unification Oracle:** $\text{UNIFY}(\text{Knows}(x, \text{Food}), \text{Knows}(\text{Agent1}, \text{Food}))$ yields $\{x \mapsto \text{Agent1}\}$.
2. **Occurs-Check Oracle:** $\text{UNIFY}(x, f(x))$ returns `None` (failure), confirming cyclic infinite terms are rejected.
3. **Multi-Variable Oracle:** $\text{UNIFY}(P(x, g(x)), P(A, y))$ yields $\{x \mapsto A, y \mapsto g(A)\}$.

## 8. References
- Robinson, J. A. (1965). A machine-oriented logic based on the resolution principle. *Journal of the ACM*, 12(1), 23-41.
- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed., Section 9.2).
