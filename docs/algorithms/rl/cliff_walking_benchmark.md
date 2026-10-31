# CliffWalking Benchmark: On-Policy SARSA vs Off-Policy Q-Learning

## 1. Description & Environment Setup
CliffWalking is the classic discrete grid benchmark from Sutton & Barto (Example 6.6):
- Grid dimensions: 12 columns $\times$ 4 rows.
- Start cell: $S = (0, 0)$ (bottom-left).
- Goal cell: $G = (11, 0)$ (bottom-right).
- The Cliff: cells $(1, 0), \dots, (10, 0)$.
- Step reward: $-1.0$.
- Cliff penalty: $-100.0$ and instantaneous reset to start $S$.
- Episode ends when reaching Goal $G$.

## 2. Theoretical Policy Divergence
Under $\epsilon$-greedy exploration ($\epsilon = 0.1$):
- **Q-Learning (Off-Policy):** Learns the optimal policy that walks right along the cliff edge (row 0, moving through row 1). Its target assumes greedy play ($\max Q(s', \cdot)$). However, because the agent still explores with probability $\epsilon$, it occasionally falls into the cliff during training, suffering large negative episodic returns.
- **SARSA (On-Policy):** Accounts for the agent's actual exploratory policy, learning the longer but *safer* path along the upper perimeter (rows 2 and 3). By maintaining distance from the cliff edge, random exploratory deviations do not result in falling off the cliff.

## 3. Expected Experimental Returns
- On evaluation (greedy $\epsilon = 0$): Q-Learning achieves the shortest path with return $-13$.
- During training ($\epsilon = 0.1$): SARSA achieves higher average return ($\approx -17$) than Q-Learning ($\approx -40$ to $-90$).

## 4. Verification Strategy
- Verify that SARSA trajectories stay above row 1 while Q-learning trajectories follow the lower perimeter edge.

