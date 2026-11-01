# Independent Q-Learning & Non-Stationarity (SYS-14, Module M15)

## 1. Theory & Mathematical Definition
Independent Q-Learning (IQL) distributes standard single-agent Q-learning across $N$ autonomous agents. Each agent $i$ updates its own action-value function $Q_i(s_i, a_i)$ independently, viewing all other agents as part of the environment. However, because concurrent learning causes other agents to alter their policies simultaneously, the Markov property is violated from the perspective of each individual learner, resulting in environmental non-stationarity:
$$P(s'_{t+1} \mid s_t, a_{i, t}) = \sum_{\mathbf{a}_{-i}} P(s'_{t+1} \mid s_t, a_{i, t}, \mathbf{a}_{-i}) \prod_{j \neq i} \pi_j(a_{j, t} \mid s_{j, t})$$

## 2. Assumptions
1. Decentralized execution with local observation $s_i$ and private reward $r_i$.
2. Agents do not observe the actions or internal parameters of peers.

## 3. Equations & Variable Definitions
- **Individual Bellman Update**:
  $$Q_i(s_i, a_i) \leftarrow Q_i(s_i, a_i) + \alpha \left[r_i + \gamma \max_{a'_i} Q_i(s'_i, a'_i) - Q_i(s_i, a_i)\right]$$
- **Moving Target Problem**:
  $$\lim_{t \to \infty} Q_i^t \text{ may fail to converge if } \pi_{-i}^t \text{ oscillates indefinitely}$$
- **Fictitious Self-Play Elo Update**:
  $$R'_A = R_A + K(S_A - E_A), \quad E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

## 4. Derivation of Non-Stationary Transition Probability
Let global state transition be stationary given joint action: $P(\mathbf{s}' \mid \mathbf{s}, \mathbf{a})$. When marginalized over opponent joint policy $\boldsymbol{\pi}_{-i}(\mathbf{a}_{-i} \mid \mathbf{s})$:
$$P_i^t(s'_i \mid s_i, a_i) = \mathbb{E}_{\mathbf{a}_{-i} \sim \boldsymbol{\pi}_{-i}^t}\left[ P(s'_i \mid \mathbf{s}, a_i, \mathbf{a}_{-i}) \right]$$
Because $\boldsymbol{\pi}_{-i}^t$ changes as opponents learn, the transition operator is inherently time-dependent ($t$).

## 5. Pseudocode
```text
function IQL_STEP(observations, rewards, next_observations, terminated):
    for each agent i in 1 to N:
        s_i = observations[i]
        a_i = actions[i]
        r_i = rewards[i]
        s_next_i = next_observations[i]
        td_target = r_i if terminated else r_i + gamma * max_a(Q_i(s_next_i, a))
        Q_i(s_i, a_i) = Q_i(s_i, a_i) + alpha * (td_target - Q_i(s_i, a_i))
```

## 6. Computational Complexity
- **Time Complexity**: $\mathcal{O}(N \cdot |\mathcal{A}|)$ per step across $N$ agents.
- **Space Complexity**: $\mathcal{O}(N \cdot |\mathcal{S}_i| \cdot |\mathcal{A}|)$ tabular storage.

## 7. Implementation Notes
- Implemented in `IndependentQLearning` and `IQLAgent`.
- Tested against frozen opponent snapshots in `OpponentPool`.

## 8. Verification Strategy & Test Oracles
- Matrix Game Oracle: Strict convergence to Nash equilibrium on 2-player games.
- Reduction Oracle: When $N=1$, IQL behavior and updates match M13 single-agent Q-learning bit-for-bit.

## 9. Known Limitations
- IQL is prone to coordination failure in cooperative environments without centralized value estimation.

## 10. References
- Tan, M. (1993). Multi-agent reinforcement learning: Independent vs. cooperative agents. *ICML*, 330-337.

