# Temporal Difference Control: Q-Learning, SARSA, Expected SARSA, and Double Q-Learning

## 1. Theory & Mathematical Definition
Temporal Difference (TD) control methods learn state-action values $Q(s, a)$ directly from environmental experience without requiring a transition model. They combine Monte Carlo sampling with dynamic programming bootstrapping.

## 2. Update Equations
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

## 3. Strict Terminal Target Invariant
For any terminal transition where $S_{t+1} \in \mathcal{S}^+ \setminus \mathcal{S}$, the Bellman bootstrap target must satisfy:
$$\text{Target} = R_{t+1}$$
exactly, with zero contribution from future states: $(1 - d_t) = 0$.

## 4. Verification Strategy
- Compare learned Q-values on small tabular MDPs against exact $Q^*$ exported from Module M11.
- Invariant unit test explicitly validating that when `terminated=True`, target equals `reward` bit-for-bit.

