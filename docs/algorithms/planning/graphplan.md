# The Graphplan Algorithm

## 1. Theory & Mathematical Definition

**Graphplan** (Blum & Furst 1997) decouples planning into two alternating phases:
1. **Planning Graph Expansion**: Polynomial-time construction of a layered directed graph encoding reachable propositions and mutual exclusions (mutexes).
2. **Backward Plan Extraction**: Constraint-satisfaction search backward from the goal level to find a conflict-free plan.

A planning graph consists of alternating levels:
$$
P_0, A_0, P_1, A_1, \dots, P_k
$$
- $P_i$: Proposition layer containing facts that might be true at time step $i$.
- $A_i$: Action layer containing ground problem actions and persistence (no-op) actions $p \to p$ whose preconditions are mutually non-exclusive in $P_i$.

---

## 2. Mutual Exclusion (Mutex) Relations

### A. Action Mutexes
Two actions $a_1, a_2 \in A_i$ are mutually exclusive ($\text{Mutex}(a_1, a_2)$) if any of the following hold:
1. **Inconsistent Effects**: $a_1$ deletes an effect that $a_2$ adds, or vice versa:
   $$
   (\text{Del}(a_1) \cap \text{Add}(a_2) \ne \emptyset) \lor (\text{Del}(a_2) \cap \text{Add}(a_1) \ne \emptyset)
   $$
2. **Interference**: $a_1$ deletes a precondition of $a_2$, or vice versa:
   $$
   (\text{Del}(a_1) \cap \text{Pre}(a_2) \ne \emptyset) \lor (\text{Del}(a_2) \cap \text{Pre}(a_1) \ne \emptyset)
   $$
3. **Competing Needs**: A precondition of $a_1$ is mutex with a precondition of $a_2$ in $P_i$:
   $$
   \exists p_1 \in \text{Pre}(a_1), p_2 \in \text{Pre}(a_2) \quad \text{s.t.} \quad \text{Mutex}(p_1, p_2) \in P_i
   $$

### B. Proposition Mutexes
Two propositions $p, q \in P_{i+1}$ are mutually exclusive if:
- **Inconsistent Support**: Every action in $A_i$ that achieves $p$ is mutex with every action in $A_i$ that achieves $q$:
  $$
  \forall a_p \in \text{Achievers}(p), \forall a_q \in \text{Achievers}(q): \quad a_p \ne a_q \land \text{Mutex}(a_p, a_q) \in A_i
  $$

---

## 3. Algorithm Lifecycle

```text
function GRAPHPLAN(problem):
    pg = InitializePlanningGraph(problem.initial_facts)
    while True:
        if problem.goal_facts <= pg.prop_levels[-1].facts and NoMutexBetweenGoals(pg, problem.goal_facts):
            plan = ExtractPlan(pg, level=len(pg.action_levels), goals=problem.goal_facts)
            if plan is not None:
                return FilterNoOps(plan)
        if pg.has_leveled_off():
            return UNSOLVABLE
        pg.expand_level()
```

---

## 4. Complexity & Guarantees

- **Graph Expansion**: Polynomial time and space $O(k \cdot (|\mathcal{P}|^2 + |\mathcal{A}|^2))$.
- **Completeness**: If a plan exists, Graphplan is guaranteed to find a plan with the minimal number of time steps (parallel steps).
- **Termination on Unsolvable Problems**: When two consecutive levels are identical in propositions and mutexes (leveled off), and goal extraction still fails, the problem is guaranteed to be unsolvable.

---

## 5. References

- Blum, A. L., & Furst, M. L. (1997). Fast planning through planning graph analysis. *Artificial Intelligence*, 90(1-2), 281-300.

