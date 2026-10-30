# Domain-Independent Delete-Relaxation Heuristics

## 1. Theory & Mathematical Definition

A domain-independent heuristic automatically estimates the distance from an arbitrary planning state $S$ to goal $G$ without hand-crafted domain knowledge.
The dominant paradigm is **delete relaxation**: constructing a simplified planning problem $\Pi^+ = \langle \mathcal{P}, \mathcal{A}^+, S, G \rangle$ in which delete effects are eliminated:
$$
\text{Del}^+(a) = \emptyset \quad \forall a \in \mathcal{A}
$$

Because facts can only be added and never deleted, any proposition achieved at step $t$ remains true for all future steps $t' \ge t$. Consequently, relaxed reachability costs can be computed in polynomial time via fixed-point iterations.

---

## 2. Heuristic Formulations

### A. $h_{\text{max}}$ (Max Relaxed Cost)
The cost to achieve a fact $p \in \mathcal{P}$ under $h_{\text{max}}$ is defined recursively:
$$
\text{cost}_{\text{max}}(p) = 0 \quad \text{if } p \in S
$$
$$
\text{cost}_{\text{max}}(p) = \min_{a \in \text{Achievers}(p)} \left( \text{Cost}(a) + \max_{q \in \text{Pre}(a)} \text{cost}_{\text{max}}(q) \right) \quad \text{if } p \notin S
$$
The state heuristic is the maximum cost over all goal facts:
$$
h_{\text{max}}(S) = \max_{g \in G} \text{cost}_{\text{max}}(g)
$$
**Admissibility**: Since achieving the set of subgoals $G$ cannot cost less than achieving the single most expensive subgoal in the delete-relaxation, and relaxation cannot increase costs, $h_{\text{max}}(S) \le h^*(S)$. $h_{\text{max}}$ is admissible and consistent.

---

### B. $h_{\text{add}}$ (Additive Relaxed Cost)
Assumes subgoals are independent and sums their costs:
$$
\text{cost}_{\text{add}}(p) = 0 \quad \text{if } p \in S
$$
$$
\text{cost}_{\text{add}}(p) = \min_{a \in \text{Achievers}(p)} \left( \text{Cost}(a) + \sum_{q \in \text{Pre}(a)} \text{cost}_{\text{add}}(q) \right) \quad \text{if } p \notin S
$$
$$
h_{\text{add}}(S) = \sum_{g \in G} \text{cost}_{\text{add}}(g)
$$
**Inadmissibility**: When an action simultaneously achieves or enables multiple subgoals, $h_{\text{add}}$ counts that action multiple times, overestimating true cost. While inadmissible, it provides strong guidance for greedy search.

---

### C. $h_{\text{FF}}$ (Fast-Forward Relaxed Plan Heuristic)
Constructs the layered relaxed planning graph until $G \subseteq P_k$, then performs a backward extraction to identify a specific relaxed plan $\pi^+$. The heuristic value is the cardinality of the relaxed plan:
$$
h_{\text{FF}}(S) = |\pi^+|
$$
$h_{\text{FF}}$ avoids the extreme overcounting of $h_{\text{add}}$ while accounting for shared actions among subgoals.

---

## 3. Mathematical Properties & Invariants

1. **Monotonicity**: $h_{\text{max}}(S) \le h_{\text{add}}(S)$ for all states $S$.
2. **Goal Zero**: $h(G) = 0$ for all heuristics $h \in \{h_{\text{max}}, h_{\text{add}}, h_{\text{FF}}\}$.
3. **Dead-End Detection**: If any $g \in G$ has $\text{cost}(g) = \infty$, the goal is unreachable even in the relaxed problem; hence the original problem is strictly unsolvable from $S$.

---

## 4. References

- Bonet, B., & Geffner, H. (2001). Planning as heuristic search. *Artificial Intelligence*, 129(1-2), 5-33.
- Hoffmann, J., & Nebel, B. (2001). The FF planning system: Fast plan generation through heuristic search. *Journal of Artificial Intelligence Research*, 14, 253-302.
