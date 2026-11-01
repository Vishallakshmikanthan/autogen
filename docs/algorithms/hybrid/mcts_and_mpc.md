# Monte Carlo Tree Search and Model Predictive Control (SYS-15, Module M17)

## 1. Theory & Mathematical Definition
Monte Carlo Tree Search (MCTS) and Model Predictive Control (MPC) represent forward-simulation planning paradigms over generative world models. MCTS builds an asymmetric search tree using Upper Confidence Bounds applied to Trees (UCT), while MPC samples or optimizes finite-horizon action trajectories using the receding-horizon principle.

## 2. Assumptions
- **Forward Generative Model:** The agent has access to a forward simulation step function $T(s, a) \to (s', r, \text{done})$.
- **State Representability:** State $s$ can be serialized into a discrete lookup key for tree node hashing.
- **Horizon Sufficiency:** The planning horizon $H$ is sufficiently long to detect key reward landmarks and hazard traps.

## 3. Equations & Variable Definitions

### MCTS Upper Confidence Bounds (UCT)
At parent node $p$ with total visits $N_p$ and child node $i$ with visits $n_i$ and cumulative reward $W_i$:
$$\text{UCT}_i = \frac{W_i}{n_i} + c \sqrt{\frac{\ln N_p}{n_i}}$$
where $c = \sqrt{2} \approx 1.414$ is the theoretical exploration constant balancing exploitation ($\bar{X}_i = W_i / n_i$) and exploration.

### Four Canonical MCTS Phases
1. **Selection:** Descend tree from root using $\arg\max_i \text{UCT}_i$ until reaching a leaf node or non-fully expanded node.
2. **Expansion:** Sample an untried legal action $a \in \mathcal{A}(s)$, transition model to $s' = T(s, a)$, and create a new child node.
3. **Simulation (Rollout):** Perform default random rollout for depth $d \le D_{\text{rollout}}$:
$$R_{\text{rollout}} = \sum_{k=0}^{D-1} \gamma^k r_{t+k+1}$$
4. **Backpropagation:** Increment visit counts $n \leftarrow n + 1$ and add $R_{\text{rollout}}$ to all ancestor nodes up to root.

### Model Predictive Control (Receding Horizon)
Given current state $s_t$, solve:
$$\mathbf{a}^*_{t:t+H-1} = \arg\max_{\mathbf{a} \in \mathcal{A}^H} \sum_{k=0}^{H-1} \gamma^k R(s_{t+k}, a_{t+k}) \quad \text{s.t. } s_{t+k+1} = T(s_{t+k}, a_{t+k})$$
Execute solely first action $a_t^* = \mathbf{a}^*[0]$, observe real transition $s_{t+1}$, and re-plan over $[t+1, t+H]$.

## 4. Computational Complexity
- **MCTS Time Complexity:** $\mathcal{O}(K \cdot (D_{\text{tree}} + D_{\text{rollout}}))$ where $K$ is iterations, $D_{\text{tree}}$ is tree depth, and $D_{\text{rollout}}$ is simulation horizon.
- **MPC Time Complexity:** $\mathcal{O}(N_{\text{samples}} \cdot H)$ per planning tick.
- **Space Complexity:** $\mathcal{O}(K \cdot |\mathcal{A}|)$ for stored MCTS tree nodes.

## 5. Implementation Notes
- Fully implemented from scratch in pure NumPy in `mcts.py` and `mpc.py`.
- Zero external dependencies. Uses robust child selection criterion (most visited child) at root.

## 6. Verification Strategy & Test Oracles
- **Bandit Convergence Oracle:** On a multi-armed bandit tree, MCTS must allocate over 80% of visits to the true optimal arm.
- **Minimax Equivalence:** On a deterministic finite game tree, MCTS with sufficient budget must converge to the minimax value.

## 7. Known Limitations
- Deep trees in large branching factor environments require heuristic rollout policies or neural value guidance.

## 8. References
- Kocsis, L., & Szepesvári, C. (2006). *Bandit Based Monte-Carlo Planning*. ECML.
- Camphuijsen, M., & Garcia, C. E. (1989). *Model Predictive Control: Theory and Practice*.

