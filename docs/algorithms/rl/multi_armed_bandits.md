# Multi-Armed Bandits

## 1. Theory & Mathematical Definition
A multi-armed bandit is a single-state Markov Decision Process $\langle \mathcal{A}, \mathcal{R} \rangle$ formalizing the fundamental exploration vs. exploitation dilemma. The agent selects an action $A_t \in \{1, \dots, K\}$ at each time step $t$ and receives a stochastic reward $R_t \sim \mathcal{D}(A_t)$ with unknown mean $q^*(a) \doteq \mathbb{E}[R_t \mid A_t = a]$.

## 2. Algorithms
### A. Epsilon-Greedy with Exponential Decay
$$A_t = \begin{cases} \text{Uniform}(\mathcal{A}), & \text{with probability } \epsilon_t \\ \arg\max_a \hat{Q}_t(a), & \text{with probability } 1 - \epsilon_t \end{cases}$$
where $\epsilon_t = \max(\epsilon_{\min}, \epsilon_0 \lambda^t)$ with decay factor $\lambda \in (0, 1]$.
Sample-average incremental update:
$$\hat{Q}_{n+1}(a) = \hat{Q}_n(a) + \frac{1}{n} (R_n - \hat{Q}_n(a))$$

### B. Upper Confidence Bound (UCB1)
Based on Hoeffding's Inequality:
$$A_t = \arg\max_a \left[ \hat{Q}_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]$$
where $N_t(a)$ is the number of times arm $a$ has been selected prior to step $t$.

### C. Thompson Sampling (Posterior Sampling)
Maintains Beta-Bernoulli conjugate priors $\text{Beta}(\alpha_a, \beta_a)$ over each arm's payoff probability:
1. Sample $\theta_a \sim \text{Beta}(\alpha_a, \beta_a)$ for all $a \in \mathcal{A}$.
2. Select $A_t = \arg\max_a \theta_a$.
3. Observe reward $R_t \in \{0, 1\}$ and update:
   $$\alpha_{A_t} \leftarrow \alpha_{A_t} + R_t, \quad \beta_{A_t} \leftarrow \beta_{A_t} + (1 - R_t)$$

## 3. Complexity
- Time per step: $O(K)$.
- Space: $O(K)$.

## 4. Verification Strategy
- Verify that cumulative regret for UCB1 and Thompson Sampling grows sublinearly ($\le O(\ln t)$) across stationary Bernoulli/Gaussian arms.

