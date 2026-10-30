# Minimax & Alpha-Beta Pruning Algorithm Specification

> **Module:** M07 — Adversarial Search & Game Theory (SYS-10)  
> **Status:** Production / Verified

---

## 1. Theory & Mathematical Definition
In a two-player, zero-sum sequential game with perfect information, player MAX seeks to maximize utility while player MIN seeks to minimize it. The game tree is defined by an initial state $s_0$, an active player function $\text{Player}(s) \in \{\text{MAX}, \text{MIN}\}$, a set of legal actions $\text{Actions}(s)$, a deterministic transition function $\text{Result}(s, a)$, and a terminal utility function $U(s)$.

Alpha-Beta pruning computes the exact Minimax value while avoiding the expansion of game tree subtrees that cannot influence the final decision.

## 2. Assumptions
1. **Zero-Sum Payoffs:** $U_{\text{MAX}}(s) + U_{\text{MIN}}(s) = 0 \iff U_{\text{MIN}}(s) = -U_{\text{MAX}}(s)$.
2. **Perfect Information:** Both agents observe the complete subgame state.
3. **Rationality:** Each player assumes the opponent will choose an action minimizing their payoff.
4. **Finite Horizon / Heuristic Evaluation:** When search reaches depth limit $d_{\max}$, a static heuristic evaluation $h(s)$ approximates the true game-theoretic value $V^*(s)$.

## 3. Equations & Variable Definitions

The minimax value function $V(s)$ is defined recursively:
$$V(s) = \begin{cases}
U(s), & \text{if } \text{Terminal}(s) \\
h(s), & \text{if } \text{Depth}(s) \ge d_{\max} \\
\max_{a \in \text{Actions}(s)} V(\text{Result}(s, a)), & \text{if } \text{Player}(s) = \text{MAX} \\
\min_{a \in \text{Actions}(s)} V(\text{Result}(s, a)), & \text{if } \text{Player}(s) = \text{MIN}
\end{cases}$$

### Alpha-Beta Pruning Invariants
Let $\alpha$ be the value of the best choice found so far along the search path for MAX.  
Let $\beta$ be the value of the best choice found so far along the search path for MIN.

- At any node $s$, if $v \ge \beta$ at a MAX node, MIN will never allow play to reach this subtree (beta cutoff).
- If $v \le \alpha$ at a MIN node, MAX will never choose this branch earlier in the tree (alpha cutoff).
- **Pruning condition:** Prune immediately when $\alpha \ge \beta$.

## 4. Derivation
Consider a subtree under a MIN node $m$ with parent MAX node $M$. Suppose $M$ already has a known alternative yielding value $\alpha$. If evaluating the first child of $m$ yields a value $v_1 \le \alpha$, then because $m$ takes the minimum over all its children, the final value of $m$ satisfies $V(m) \le v_1 \le \alpha$. Therefore, the parent $M$ will never select branch $m$, rendering evaluation of all remaining children of $m$ irrelevant.

## 5. Pseudocode

```text
function AlphaBetaSearch(game, state, max_depth, eval_fn, move_order_fn):
    alpha = -infinity
    beta = +infinity
    best_action = None
    best_value = -infinity
    for action in move_order_fn(game.legal_actions(state), state, player=MAX):
        succ = game.successor(state, action)
        val = MinValue(game, succ, max_depth - 1, alpha, beta, eval_fn, move_order_fn)
        if val > best_value:
            best_value = val
            best_action = action
        alpha = max(alpha, best_value)
    return (best_action, best_value)

function MaxValue(game, state, depth, alpha, beta, eval_fn, order_fn):
    if game.is_terminal(state): return game.utility(state, 0)
    if depth <= 0: return eval_fn(state, 0)
    v = -infinity
    for action in order_fn(game.legal_actions(state), state, player=MAX):
        succ = game.successor(state, action)
        v = max(v, MinValue(game, succ, depth - 1, alpha, beta, eval_fn, order_fn))
        if v >= beta: return v  # Beta cutoff
        alpha = max(alpha, v)
    return v

function MinValue(game, state, depth, alpha, beta, eval_fn, order_fn):
    if game.is_terminal(state): return game.utility(state, 0)
    if depth <= 0: return eval_fn(state, 0)
    v = +infinity
    for action in order_fn(game.legal_actions(state), state, player=MIN):
        succ = game.successor(state, action)
        v = min(v, MaxValue(game, succ, depth - 1, alpha, beta, eval_fn, order_fn))
        if v <= alpha: return v  # Alpha cutoff
        beta = min(beta, v)
    return v
```

## 6. Computational Complexity
- **Minimax Time Complexity:** $\mathcal{O}(b^d)$, where $b$ is the effective branching factor and $d$ is search depth.
- **Alpha-Beta Worst-Case Time:** $\mathcal{O}(b^d)$ (when moves are ordered worst-first).
- **Alpha-Beta Best-Case Time:** $\mathcal{O}(b^{d/2})$ (when moves are ordered best-first).
- **Space Complexity:** $\mathcal{O}(b \cdot d)$ linear memory storing recursive stack call frames.

## 7. Implementation Notes
- Move ordering heuristic `order_predator_moves` sorts actions by proximity to target to maximize the likelihood of early cutoffs.
- Optional tree instrumentation records a structured `GameTreeNode` hierarchy for inspection and visualization.
- Pure Python/NumPy implementation with no external C-extensions or solvers.

## 8. Verification Strategy & Test Oracles
1. **Equivalence Oracle on 1000 Random Trees:**
   $$\forall T \in \text{RandomTrees}, \quad V_{\text{AlphaBeta}}(T) = V_{\text{Minimax}}(T)$$
   $$\text{NodesExpanded}_{\text{AlphaBeta}}(T) \le \text{NodesExpanded}_{\text{Minimax}}(T)$$
2. **Tic-Tac-Toe Solved Value Oracle:**
   $$V^*(\text{TicTacToe initial board}) = 0 \quad (\text{draw with optimal play})$$
3. **Move Ordering Pruning Bound:**
   Verifies that best-case ordered trees expand substantially fewer nodes than worst-case ordered trees, asymptotically approaching $\mathcal{O}(b^{d/2})$.

## 9. Known Limitations
- Fixed depth limits can suffer from the *horizon effect* where critical opponent counter-tactics lie just beyond $d_{\max}$.
- Pruning depends heavily on the accuracy of move-ordering heuristics.

## 10. References
- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed., Chapter 5).
- Knuth, D. E., & Moore, R. W. (1975). An analysis of alpha-beta pruning. *Artificial Intelligence*, 6(4), 293-326.

