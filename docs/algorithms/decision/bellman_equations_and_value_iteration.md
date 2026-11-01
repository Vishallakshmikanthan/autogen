# Bellman Equations & Value Iteration

## 1. Theory & Mathematical Definition

A finite Markov Decision Process is defined as a tuple $\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma \rangle$:
- $\mathcal{S}$: finite set of states, $|\mathcal{S}| < \infty$.
- $\mathcal{A}$: finite set of actions, $|\mathcal{A}| < \infty$.
- $\mathcal{T}(s, a, s') = P(S_{t+1} = s' \mid S_t = s, A_t = a)$: stationary state transition probability.
- $\mathcal{R}(s, a, s')$: immediate real-valued reward received on transition $(s, a, s')$.
- $\gamma \in [0, 1)$: discount factor.

### Bellman Expectation Equation
For any policy $\pi(a \mid s)$, the value function $V^\pi(s)$ is the unique fixed point of the Bellman expectation operator:
$$
V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} \mathcal{T}(s, a, s') \left[ \mathcal{R}(s, a, s') + \gamma V^\pi(s') \right]
$$

In matrix form: $V^\pi = R^\pi + \gamma P^\pi V^\pi \implies V^\pi = (I - \gamma P^\pi)^{-1} R^\pi$.

### Bellman Optimality Equation
The optimal value function $V^*(s) = \max_\pi V^\pi(s)$ satisfies:
$$
V^*(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} \mathcal{T}(s, a, s') \left[ \mathcal{R}(s, a, s') + \gamma V^*(s') \right]
$$
The corresponding state-action value function $Q^*(s, a)$ satisfies:
$$
Q^*(s, a) = \sum_{s' \in \mathcal{S}} \mathcal{T}(s, a, s') \left[ \mathcal{R}(s, a, s') + \gamma V^*(s') \right]
$$
The optimal policy is greedy with respect to $Q^*$:
$$
\pi^*(s) = \arg\max_{a \in \mathcal{A}} Q^*(s, a)
$$

---

## 2. Assumptions

1. **Stationary Dynamics:** Transition matrix $\mathcal{T}(s, a, s')$ and reward function $\mathcal{R}(s, a, s')$ do not drift over time.
2. **Markov Property:** Next state $s'$ depends solely on current state $s$ and chosen action $a$.
3. **Discount Factor:** $\gamma \in [0, 1)$ ensures contraction mapping and bounded cumulative returns.
4. **Finite State-Action Cardinality:** $|\mathcal{S}| < \infty$ and $|\mathcal{A}| < \infty$.

---

## 3. Contraction Mapping & Convergence Proof

The Bellman optimality operator $\mathcal{T}^* : \mathbb{R}^{|\mathcal{S}|} \to \mathbb{R}^{|\mathcal{S}|}$ is defined by:
$$
(\mathcal{T}^* V)(s) = \max_{a \in \mathcal{A}} \sum_{s'} \mathcal{T}(s, a, s') [ \mathcal{R}(s, a, s') + \gamma V(s') ]
$$

**Theorem (Contraction Mapping)**:
For any two value functions $U, V \in \mathbb{R}^{|\mathcal{S}|}$:
$$
\|\mathcal{T}^* U - \mathcal{T}^* V\|_\infty \le \gamma \|U - V\|_\infty
$$
Since $\gamma \in [0, 1)$, $\mathcal{T}^*$ is a $\gamma$-contraction in the supremum norm. By Banach's Fixed Point Theorem, repeated applications $V_{k+1} = \mathcal{T}^* V_k$ converge to a unique fixed point $V^*$ independent of initial $V_0$.

**Error Bound & Stopping Criterion**:
If $\|V_{k+1} - V_k\|_\infty < \theta \frac{1 - \gamma}{2\gamma}$, then $\|V_k - V^*\|_\infty < \theta$, guaranteeing that the greedy policy extracted from $V_k$ is at most $\theta$-suboptimal.

---

## 3. Algorithm Pseudocode

```text
function VALUE-ITERATION(mdp, theta):
    V = zeros(|S|)
    threshold = theta * (1 - gamma) / (2 * gamma) if 0 < gamma < 1 else theta
    loop:
        delta = 0
        V_new = zeros(|S|)
        for s in S:
            if s in terminal: continue
            for a in A:
                Q[s, a] = sum_s' T[s, a, s'] * (R[s, a, s'] + gamma * V[s'])
            V_new[s] = max_a Q[s, a]
            delta = max(delta, |V_new[s] - V[s]|)
        V = V_new
        if delta < threshold:
            break
    policy[s] = argmax_a Q[s, a]
    return V, Q, policy
```

---

## 4. Computational Complexity

- **Time Complexity per Iteration**: $O(|\mathcal{S}|^2 \cdot |\mathcal{A}|)$.
- **Number of Iterations**: $O\left( \frac{1}{1 - \gamma} \log \frac{1}{\theta} \right)$.
- **Space Complexity**: $O(|\mathcal{S}| \cdot |\mathcal{A}|)$ for storing $Q$ and policy.

---

## 5. Verification Strategy & Test Oracles

- **Analytically Solved MDPs**: Tested against hand-derived closed-form 2-state and 3-state chain MDPs.
- **Linear Solve Agreement**: Policy evaluation matches $(I - \gamma P_\pi)^{-1} R_\pi$ to machine precision.
- **VI vs PI Equivalence**: Value Iteration and Policy Iteration produce identical $V^*$ ($< 10^{-9}$).

---

## 7. Known Limitations

- **Curse of Dimensionality:** Requires iterating over all states in $|\mathcal{S}|$, rendering tabular updates intractable for high-dimensional continuous or exponential combinatorial spaces.
- **Model Requirement:** Assumes exact ground-truth transition $\mathcal{T}$ and reward $\mathcal{R}$ matrices are known in advance.

---

## 8. References / Citations

- Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 4.

