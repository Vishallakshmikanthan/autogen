# Policy Iteration (Howard's Algorithm)

## 1. Theory & Mathematical Definition

While Value Iteration updates values through successive approximations, **Policy Iteration** (Howard 1960) manipulates the policy directly, alternating between exact policy evaluation and policy improvement until strict convergence.

Because the policy space is finite ($|\mathcal{A}|^{|\mathcal{S}|}$ possible deterministic policies), and every improvement step yields a strictly superior policy unless already optimal, Policy Iteration terminates in a finite number of iterations.

---

## 2. Algorithmic Phases

### Phase 1: Policy Evaluation
For the current deterministic policy $\pi_k$, compute the exact value function $V^{\pi_k}$ by solving the linear system:
$$
V^{\pi_k}(s) = \sum_{s'} \mathcal{T}(s, \pi_k(s), s') \left[ \mathcal{R}(s, \pi_k(s), s') + \gamma V^{\pi_k}(s') \right]
$$
$$
(I - \gamma P_{\pi_k}) V^{\pi_k} = R_{\pi_k} \implies V^{\pi_k} = (I - \gamma P_{\pi_k})^{-1} R_{\pi_k}
$$

### Phase 2: Policy Improvement
Update the policy greedily with respect to the evaluated $V^{\pi_k}$:
$$
\pi_{k+1}(s) = \arg\max_{a \in \mathcal{A}} \sum_{s'} \mathcal{T}(s, a, s') \left[ \mathcal{R}(s, a, s') + \gamma V^{\pi_k}(s') \right]
$$

### Termination Criterion
If $\pi_{k+1}(s) = \pi_k(s)$ for all $s \in \mathcal{S}$ (policy is stable), the algorithm terminates with the globally optimal policy $\pi^* = \pi_k$ and optimal value function $V^* = V^{\pi_k}$.

---

## 3. Policy Improvement Theorem

**Theorem**: Let $\pi$ and $\pi'$ be any pair of deterministic policies such that, for all $s \in \mathcal{S}$:
$$
q_\pi(s, \pi'(s)) \ge v_\pi(s)
$$
Then the policy $\pi'$ must be globally as good as or better than $\pi$:
$$
v_{\pi'}(s) \ge v_\pi(s) \quad \forall s \in \mathcal{S}
$$
Furthermore, if strict inequality holds for at least one state, then $v_{\pi'}$ is strictly better than $v_\pi$.

---

## 4. Complexity & Comparison with Value Iteration

- **Policy Iteration**: Fewer iterations (often single digits), but each iteration requires solving an $|\mathcal{S}| \times |\mathcal{S}|$ linear system ($O(|\mathcal{S}|^3)$).
- **Value Iteration**: Cheaper iterations ($O(|\mathcal{S}|^2 |\mathcal{A}|)$), but requires many iterations to converge numerically within $\theta$.

---

## 5. References

- Howard, R. A. (1960). *Dynamic Programming and Markov Processes*. MIT Press.
- Puterman, M. L. (1994). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. John Wiley & Sons.
