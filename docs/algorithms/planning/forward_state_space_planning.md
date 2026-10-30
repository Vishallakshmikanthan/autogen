# Forward State-Space Planning (STRIPS Progression)

## 1. Theory & Mathematical Definition

Classical planning addresses deterministic, fully observable, static, discrete environments.
A planning task is formulated in **STRIPS** (Stanford Research Institute Problem Solver):
$$
\Pi = \langle \mathcal{P}, \mathcal{A}, S_0, G \rangle
$$
- $\mathcal{P}$: finite set of ground positive relational propositions (facts).
- $\mathcal{A}$: finite set of ground actions. Each action $a \in \mathcal{A}$ is defined by:
  - $\text{Pre}(a) \subseteq \mathcal{P}$: Preconditions that must hold for $a$ to be applicable.
  - $\text{Add}(a) \subseteq \mathcal{P}$: Facts added by the execution of $a$.
  - $\text{Del}(a) \subseteq \mathcal{P}$: Facts deleted by the execution of $a$.
  - $\text{Cost}(a) \in \mathbb{R}^+$: Non-negative execution cost.
- $S_0 \subseteq \mathcal{P}$: Initial world state under the Closed World Assumption (facts not in $S_0$ are false).
- $G \subseteq \mathcal{P}$: Goal condition (a partial state that must be satisfied).

### State Transition (Progression)
An action $a$ is applicable in state $S$ if and only if:
$$
\text{Pre}(a) \subseteq S
$$
The successor state $\text{Result}(S, a)$ is determined by set-theoretic progression:
$$
\text{Result}(S, a) = (S \setminus \text{Del}(a)) \cup \text{Add}(a)
$$
A state $S$ satisfies the goal condition if:
$$
G \subseteq S
$$

---

## 2. Assumptions

1. Fully observable world state represented as positive ground facts.
2. Actions are deterministic and instantaneous.
3. Closed World Assumption: any fact not explicitly enumerated in state $S$ is false.

---

## 3. Forward Progression Search

The forward search starts at root node $S_0$ and applies all applicable actions $\mathcal{A}(S) = \{a \in \mathcal{A} \mid \text{Pre}(a) \subseteq S\}$.
- **Breadth-First Search (BFS)**: Explores state space level-by-level; guarantees shortest-length plan.
- **$A^*$ Search**: Orders exploration by evaluation function $f(n) = g(n) + h(n)$, where $g(n)$ is cumulative path cost and $h(n)$ is an admissible delete-relaxation heuristic ($h_{\text{max}}$).

---

## 4. Pseudocode

```text
function FORWARD-ASTAR-PLANNER(problem, heuristic):
    frontier = PriorityQueue(key=lambda node: node.g + node.h)
    root = Node(state=problem.initial_facts, g=0, h=heuristic(problem.initial_facts), path=())
    frontier.push(root)
    best_g = {problem.initial_facts: 0}

    while frontier is not empty:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return Plan(actions=node.path, cost=node.g)
        if node.g > best_g[node.state]:
            continue

        for a in problem.actions:
            if a.preconditions <= node.state:
                succ_state = (node.state - a.delete_effects) | a.add_effects
                succ_g = node.g + a.cost
                if succ_g < best_g.get(succ_state, inf):
                    best_g[succ_state] = succ_g
                    h = heuristic(succ_state)
                    if h < inf:
                        frontier.push(Node(succ_state, succ_g, h, node.path + (a,)))
    return UNSOLVABLE
```

---

## 5. Computational Complexity

- **State Space**: Up to $2^{|\mathcal{P}|}$ distinct states.
- **Forward Search Time**: $O(b^d)$, where $b$ is the average action branching factor and $d$ is plan depth.
- Pruning and heuristics reduce expanded states dramatically compared to blind search.

---

## 6. Verification Strategy & Test Oracles

- **BFS vs A* Invariant**: Optimal A* with admissible $h_{\text{max}}$ produces plans of identical length and cost to uniform-cost BFS.
- **Independent Plan Validator**: `PlanValidator.validate_plan` checks each precondition and effect along the generated plan without referencing search internals.
- **Dead-End Detection**: Formally certifies unreachable goal states as unsolvable.

---

## 7. References

- Fikes, R. E., & Nilsson, N. J. (1971). STRIPS: A new approach to the application of theorem proving to problem solving. *Artificial Intelligence*, 2(3-4), 189-208.
- Ghallab, M., Nau, D., & Traverso, P. (2004). *Automated Planning: Theory and Practice*. Elsevier.
