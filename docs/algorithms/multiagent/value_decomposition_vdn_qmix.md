# Multi-Agent Value Decomposition: VDN & QMIX (SYS-14, Module M15)

## 1. Theory & Mathematical Definition
Value decomposition addresses multi-agent credit assignment in cooperative MARL. Rather than learning a single unconstrained joint action-value $Q_{\text{tot}}(\mathbf{s}, \mathbf{a})$, value decomposition factorizes $Q_{\text{tot}}$ into individual utilities $Q_i(s_i, a_i)$ such that the joint greedy action selection matches individual greedy actions, satisfying the Individual-Global-Max (IGM) condition.

## 2. Assumptions
1. Fully cooperative multi-agent team with joint reward $R_{\text{team}}$.
2. Monotonicity: Increasing an individual agent's action utility $Q_i$ must never decrease the global team utility $Q_{\text{tot}}$.

## 3. Equations & Variable Definitions
- **Individual-Global-Max (IGM) Condition**:
  $$\operatorname{argmax}_{\mathbf{a}} Q_{\text{tot}}(\mathbf{s}, \mathbf{a}) = \begin{pmatrix} \operatorname{argmax}_{a_1} Q_1(s_1, a_1) \\ \vdots \\ \operatorname{argmax}_{a_N} Q_N(s_N, a_N) \end{pmatrix}$$
- **Value-Decomposition Networks (VDN)**:
  $$Q_{\text{tot}}(\mathbf{s}, \mathbf{a}) = \sum_{i=1}^N Q_i(s_i, a_i)$$
- **QMIX Monotonicity Condition**:
  $$\frac{\partial Q_{\text{tot}}}{\partial Q_i} \ge 0 \quad \forall i \in \{1, \dots, N\}$$
- **QMIX Non-Negative Mixing Network**:
  $$Q_{\text{tot}} = W_2 \cdot \operatorname{ELU}(W_1 \mathbf{Q} + b_1) + b_2, \quad \text{where } W_1 \ge 0, W_2 \ge 0$$

## 4. Proof of IGM Monotonicity
For any monotonic function $f$ satisfying $\frac{\partial f}{\partial Q_i} \ge 0$:
$$\operatorname{argmax}_{a_i} f(Q_1, \dots, Q_i(s_i, a_i), \dots, Q_N) = \operatorname{argmax}_{a_i} Q_i(s_i, a_i)$$
Because $f$ is strictly non-decreasing with respect to each argument, maximizing each component independently is mathematically guaranteed to maximize the joint function.

## 5. Pseudocode
```text
function QMIX_FORWARD(agent_qs, global_state, hypernet_w1, hypernet_w2):
    # Produce strictly non-negative weights via absolute value / exp
    W1 = abs(hypernet_w1(global_state)).reshape(num_agents, embed_dim)
    W2 = abs(hypernet_w2(global_state)).reshape(embed_dim, 1)
    
    hidden = max(0, agent_qs @ W1 + b1)
    Q_tot = hidden @ W2 + b2
    return Q_tot
```

## 6. Computational Complexity
- **Time Complexity**: $\mathcal{O}(N \cdot |\mathcal{A}|)$ for decentralized action selection; $\mathcal{O}(N \cdot d_{\text{embed}})$ for joint mixing.
- **Space Complexity**: $\mathcal{O}(N \cdot d_{\text{embed}})$ parameter storage.

## 7. Implementation Notes
- Implemented in `ValueDecompositionNetworks` and `QMIXLite`.
- Monotonicity verified mathematically across extensive randomized tests.

## 8. Verification Strategy & Test Oracles
- IGM Verification Oracle: 10,000 randomized Q-value configurations asserting joint argmax equals individual argmax tuple.
- Monotonicity test: Verifies that positive perturbations $\Delta Q_i > 0$ strictly produce non-negative changes in $Q_{\text{tot}}$.

## 9. Known Limitations
- Monotonic mixing functions cannot represent payoff matrices requiring non-monotonic coordination (e.g. general matrix games with asymmetric penalties).

## 10. References
- Sunehag, P., Lever, G., Gruslys, A., et al. (2018). Value-decomposition networks for cooperative multi-agent learning based on team reward. *AAMAS*, 2085-2087.
- Rashid, T., Samvelyan, M., Schroeder, C., et al. (2018). QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning. *ICML*, 4295-4304.
