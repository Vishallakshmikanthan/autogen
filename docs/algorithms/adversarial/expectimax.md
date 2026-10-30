# Expectimax Search Algorithm Specification

> **Module:** M07 — Adversarial Search & Game Theory (SYS-10)  
> **Status:** Production / Verified

---

## 1. Theory & Mathematical Definition
Expectimax extends adversarial search to games with stochastic elements or non-adversarial opponents (e.g., opponents that act randomly, nature chance nodes, or non-optimal adversaries). Instead of computing worst-case minimization over opponent moves, Expectimax computes the mathematical expectation of the utility over candidate transition outcomes.

## 2. Assumptions
1. **Probabilistic Opponent / Chance Model:** Opponent or chance transitions are governed by a known probability distribution $P(s' \mid s, a)$.
2. **Risk Neutrality:** The maximizing agent operates according to Von Neumann-Morgenstern expected utility axioms.
3. **Finite Horizon / Cutoff:** State values at cutoff depth are approximated by static evaluation function $h(s)$.

## 3. Equations & Variable Definitions

The Expectimax value $V_{\text{expectimax}}(s)$ is defined as:
$$V(s) = \begin{cases}
U(s), & \text{if } \text{Terminal}(s) \\
h(s), & \text{if } \text{Depth}(s) \ge d_{\max} \\
\max_{a \in \text{Actions}(s)} V(\text{Result}(s, a)), & \text{if } \text{Player}(s) = \text{MAX} \\
\sum_{s'} P(s' \mid s) \cdot V(s'), & \text{if } \text{Player}(s) \in \{\text{CHANCE}, \text{STOCHASTIC}\}
\end{cases}$$

Where:
- $s$ is the current game state.
- $\text{Actions}(s)$ is the set of legal actions.
- $P(s' \mid s)$ is the transition probability distribution satisfying $\sum_{s'} P(s' \mid s) = 1$ and $P(s' \mid s) \ge 0$.

## 4. Derivation
When an opponent does not minimize utility $U$, standard Minimax search acts over-pessimistically, missing opportunities where average payoff is high despite low-probability worst-case outcomes. By taking expectation $E[V] = \sum_i p_i V_i$, the decision maker chooses action $a^* = \arg\max_a E[V \mid a]$ which maximizes long-term average return.

## 5. Pseudocode

```text
function ExpectimaxSearch(game, state, max_depth, eval_fn):
    best_act = None
    best_val = -infinity
    for act in game.legal_actions(state):
        outcomes = game.chance_outcomes(state, act)
        expected_val = 0.0
        for succ, prob in outcomes:
            val = ExpectimaxValue(game, succ, max_depth - 1, eval_fn)
            expected_val += prob * val
        if expected_val > best_val:
            best_val = expected_val
            best_act = act
    return (best_act, best_val)

function ExpectimaxValue(game, state, depth, eval_fn):
    if game.is_terminal(state): return game.utility(state, 0)
    if depth <= 0: return eval_fn(state, 0)
    p = game.player(state)
    if p == MAX:
        v = -infinity
        for act in game.legal_actions(state):
            outcomes = game.chance_outcomes(state, act)
            act_val = sum(prob * ExpectimaxValue(game, succ, depth - 1, eval_fn) for succ, prob in outcomes)
            v = max(v, act_val)
        return v
    else: # Chance / Stochastic
        legal_acts = game.legal_actions(state)
        uniform_p = 1.0 / len(legal_acts)
        expected_val = 0.0
        for act in legal_acts:
            outcomes = game.chance_outcomes(state, act)
            branch_val = sum(prob * ExpectimaxValue(game, succ, depth - 1, eval_fn) for succ, prob in outcomes)
            expected_val += uniform_p * branch_val
        return expected_val
```

## 6. Computational Complexity
- **Time Complexity:** $\mathcal{O}(b^d)$, where $b$ is the branching factor. Unlike Alpha-Beta, general chance nodes cannot be pruned without upper and lower bounds on evaluation functions.
- **Space Complexity:** $\mathcal{O}(d)$ stack depth.

## 7. Implementation Notes
- Supports both discrete uniform opponent actions and arbitrary explicit chance outcome distributions.
- Yields substantially higher survival rates against `StochasticOpponent` agents than standard Minimax.

## 8. Verification Strategy & Test Oracles
1. **Known Probability Trees:** Evaluated against hand-computed chance trees where $V(s) = \sum p_i V_i$.
2. **Deterministic Degeneration:** If all chance branches have probability 1.0 to a single outcome, Expectimax matches Minimax value when opponent plays that fixed move.

## 9. Known Limitations
- Evaluation values cannot be arbitrarily scaled; positive monotonic transformations preserve Minimax decisions but alter Expectimax decisions (requires true cardinal utilities, not just ordinal ranks).

## 10. References
- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed., Section 5.5).
- Von Neumann, J., & Morgenstern, O. (1944). *Theory of Games and Economic Behavior*.

