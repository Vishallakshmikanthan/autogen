# Temporal Difference Control: Q-Learning, SARSA, Expected SARSA, and Double Q-Learning

## 1. Theory & Mathematical Definition
Temporal Difference (TD) control methods learn state-action values $Q(s, a)$ directly from environmental experience without requiring a transition model. They combine Monte Carlo sampling with dynamic programming bootstrapping.

---

## 2. Assumptions

1. **Discrete State-Action Spaces:** Tabular arrays maintain explicit entries for all $(s, a) \in \mathcal{S} \times \mathcal{A}$.
2. **Infinite Exploration in the Limit:** Every state-action pair is visited infinitely often to guarantee convergence.
3. **Robbins-Monro Learning Rate Schedule:** $\sum_{t=1}^\infty \alpha_t = \infty$ and $\sum_{t=1}^\infty \alpha_t^2 < \infty$.
4. **Stationary Dynamics:** Transition probabilities $P(s' \mid s, a)$ and reward distributions do not drift over time.

---

## 3. Equations & Mathematical Model

### Q-Learning (Watkins 1989, Off-Policy TD Control)
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma (1 - d_t) \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$
where $d_t = 1$ if $S_{t+1}$ is terminal, and $0$ otherwise.

### SARSA (Rummery & Niranjan 1994, On-Policy TD Control)
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma (1 - d_t) Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$
where $A_{t+1} \sim \pi(\cdot \mid S_{t+1})$ is the action actually executed by the policy.

### Expected SARSA
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma (1 - d_t) \sum_a \pi(a \mid S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$

### Double Q-Learning (van Hasselt 2010)
Eliminates positive maximization bias $\mathbb{E}[\max_a Q(s', a)] \ge \max_a \mathbb{E}[Q(s', a)]$ by decoupling maximization from evaluation using two estimators $Q_1$ and $Q_2$:
$$Q_1(S_t, A_t) \leftarrow Q_1(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma (1 - d_t) Q_2(S_{t+1}, \arg\max_a Q_1(S_{t+1}, a)) - Q_1(S_t, A_t) \right]$$

---

## 4. Strict Terminal Target Invariant
For any terminal transition where $S_{t+1} \in \mathcal{S}^+ \setminus \mathcal{S}$, the Bellman bootstrap target must satisfy:
$$\text{Target} = R_{t+1}$$
exactly, with zero contribution from future states: $(1 - d_t) = 0$.

---

## 5. Computational Complexity

- **Time Complexity:** $\mathcal{O}(1)$ update time per environment step.
- **Space Complexity:** $\mathcal{O}(|\mathcal{S}| \cdot |\mathcal{A}|)$ memory for the tabular $Q$-table.

---

## 6. Implementation Notes

- Epsilon-greedy exploration schedule anneals $\epsilon_t = \max(\epsilon_{\min}, \epsilon_0 \cdot \lambda^t)$.
- Decoupled terminal masks ensure terminal returns do not bootstrap off invalid next states.

---

## 7. Verification Strategy & Test Oracles

- **Convergence to Exact $Q^*$:** Learned $Q$-values on small grid MDPs are verified against closed-form values computed by Value Iteration in Module M11.
- **Maximization Bias Demonstration:** Double Q-learning is verified to avoid overestimating state values compared to vanilla Q-learning on stochastic payoff branches.

---

## 8. Known Limitations

- **Tabular Memory Scaling:** Tabular representations cannot scale to continuous observation spaces without feature approximation or neural representations.
- **Sample Inefficiency:** Requires many environment interactions to propagate rewards backwards through long credit assignment horizons.

---

## 9. References / Citations

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed., Chapter 6). MIT Press.
- Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards*. Ph.D. thesis, Cambridge University.
