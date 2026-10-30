# Uninformed Search Algorithms (BFS, DFS, DLS, IDS, Bi-BFS, UCS)

> **Module:** M06 — Search Algorithms and Problem Solving (SYS-06, Gate 2)  
> **Source:** Russell & Norvig, AIMA (4th Ed.), Chapter 3.

---

## 1. Theory & Mathematical Definition

Uninformed (blind) search algorithms explore a problem space $\mathcal{S}$ without domain-specific estimates of distance to the goal.

A search problem is formally defined as a 5-tuple:
$$\mathcal{P} = \langle \mathcal{S}, s_0, \mathcal{A}, \mathcal{T}, \mathcal{G}, c \rangle$$
- $\mathcal{S}$: State space.
- $s_0 \in \mathcal{S}$: Initial state.
- $\mathcal{A}$: Set of valid actions.
- $\mathcal{T}: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$: Transition function $\mathcal{T}(s, a) = s'$.
- $\mathcal{G} \subseteq \mathcal{S}$: Goal state set.
- $c(s, a, s') \ge 0$: Cost of transitioning from $s$ to $s'$ via action $a$.

---

## 2. Assumptions

1. The state space $\mathcal{S}$ and transitions are discrete and deterministic.
2. Edge transition costs are non-negative: $c(s, a, s') \ge 0$.
3. For Uniform-Cost Search, all step costs exceed a strictly positive lower bound: $c(e) \ge \epsilon > 0$.

---

## 3. Equations & Complexities

Let $b$ be the effective branching factor, $d$ be the shallowest goal depth, and $m$ be the maximum search tree depth.

| Algorithm | Frontier Data Structure | Time Complexity | Space Complexity | Completeness | Optimality |
|---|---|---|---|---|---|
| **Breadth-First Search (BFS)** | FIFO Queue (`deque`) | $\mathcal{O}(b^d)$ | $\mathcal{O}(b^d)$ | Yes (if $b < \infty$) | Yes (unweighted / unit cost) |
| **Depth-First Search (DFS)** | LIFO Stack | $\mathcal{O}(b^m)$ | $\mathcal{O}(bm)$ | No (infinite spaces) | No |
| **Depth-Limited Search (DLS)** | Stack with limit $L$ | $\mathcal{O}(b^L)$ | $\mathcal{O}(bL)$ | Yes if $d \le L$ | No |
| **Iterative Deepening (IDS)** | Successive DLS ($0 \dots d$) | $\mathcal{O}(b^d)$ | $\mathcal{O}(bd)$ | Yes (if $b < \infty$) | Yes (unweighted / unit cost) |
| **Bidirectional BFS (Bi-BFS)** | Dual FIFO Queues | $\mathcal{O}(b^{d/2})$ | $\mathcal{O}(b^{d/2})$ | Yes (if $b < \infty$) | Yes (unweighted / unit cost) |
| **Uniform-Cost Search (UCS)** | Priority Queue ($g(n)$) | $\mathcal{O}(b^{1 + \lfloor C^* / \epsilon \rfloor})$ | $\mathcal{O}(b^{1 + \lfloor C^* / \epsilon \rfloor})$ | Yes (if $c(e) \ge \epsilon > 0$) | Yes (arbitrary non-negative costs) |

### Uniform-Cost Path Evaluation
Cumulative cost of node $n$ with path $(e_1, e_2, \dots, e_k)$:
$$g(n) = \sum_{i=1}^k c(e_i)$$
Priority queue pops $\arg\min_{n \in \text{Frontier}} g(n)$.

---

## 4. Pseudocode

### Uniform Cost Search (Dijkstra)
```text
function UniformCostSearch(problem):
    node = SearchNode(state=problem.initial_state, g=0)
    frontier = PriorityQueue(key=g)
    frontier.push(node)
    best_g = {node.state: 0.0}
    closed = set()

    while not frontier.is_empty():
        current = frontier.pop()
        if current.state in closed: continue
        if current.g > best_g[current.state]: continue
        closed.add(current.state)

        if problem.is_goal(current.state):
            return reconstruct_path(current)

        for (action, next_state, cost) in problem.successors(current.state):
            new_g = current.g + cost
            if next_state not in best_g or new_g < best_g[next_state]:
                best_g[next_state] = new_g
                child = SearchNode(state=next_state, parent=current, action=action, g=new_g)
                frontier.push(child)

    return failure
```

---

## 5. Verification Strategy & Test Oracles

- **Dijkstra Oracle:** `UniformCostSearch` path costs are rigorously verified against `scipy.sparse.csgraph.dijkstra` on 500 randomized grid topographies across diverse seeds (`tests/integration/test_search_dijkstra_oracle.py`).
- **BFS Edge Minimality:** Validated on unweighted grids against topological shortest path length.

---

## 6. References

- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach (4th ed.)*. Pearson. Chapter 3.
- Dijkstra, E. W. (1959). *A note on two problems in connexion with graphs*. Numerische Mathematik, 1(1), 269–271.
