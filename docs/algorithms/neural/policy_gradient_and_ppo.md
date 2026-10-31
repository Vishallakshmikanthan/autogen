# Policy Gradients & Proximal Policy Optimization (PPO) (SYS-13, Module M14)

## 1. Theory & Mathematical Definition
Policy gradient methods optimize parameterized stochastic policies $\pi_\theta(a|s)$ directly via gradient ascent on the expected cumulative reward $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$. Proximal Policy Optimization (PPO) stabilizes policy updates by constraining the probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ within a clipped surrogate objective interval $[1-\epsilon, 1+\epsilon]$.

## 2. Assumptions
1. Continuous policy parameterization $\pi_\theta(a|s) > 0$ with differentiable log-probabilities.
2. Trajectories collected under stationary environment dynamics during policy rollout.

## 3. Equations & Variable Definitions
- **Policy Gradient Theorem**:
  $$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s, a)\right]$$
- **REINFORCE with Baseline $b$**:
  $$\nabla_\theta J(\theta) \approx \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) (G_t - b)$$
- **PPO Clipped Surrogate Objective**:
  $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right) \right]$$
- **Full PPO Loss Function**:
  $$L^{\text{PPO}}(\theta, \phi) = -L^{\text{CLIP}}(\theta) + c_1 L^{\text{VF}}(\phi) - c_2 S[\pi_\theta]$$
  where $L^{\text{VF}}(\phi) = \frac{1}{2} (V_\phi(s_t) - V_t^{\text{targ}})^2$ is value function MSE, and $S[\pi_\theta] = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$ is policy entropy.

## 4. Derivation of the PPO Clipping Bound
The unclipped objective $r_t(\theta) \hat{A}_t$ can cause destructive policy updates if $\pi_\theta$ changes excessively. The minimum operation:
$$\min\left(r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)$$
ensures that when $\hat{A}_t > 0$ and $r_t > 1+\epsilon$, the objective plateaus, zeroing out further policy push in that direction. Similarly, when $\hat{A}_t < 0$ and $r_t < 1-\epsilon$, the loss is bounded from above.

## 5. Pseudocode
```text
for iteration = 1, 2, ... do:
    Collect set of trajectories under pi_{theta_old}
    Compute advantages A_t = G_t - V(s_t) and normalize
    for epoch = 1 to K do:
        Compute r_t(theta) = pi_theta(a_t|s_t) / pi_old(a_t|s_t)
        Compute L_clip, L_value, S_entropy
        Total Loss = - L_clip + c_1 * L_value - c_2 * S_entropy
        Update actor and critic parameters via Adam
    pi_old <- pi_theta
```

## 6. Computational Complexity
- **Time Complexity**: $\mathcal{O}(K \cdot T \cdot |\text{net}|)$ for $K$ optimization epochs over $T$ rollout steps.
- **Space Complexity**: $\mathcal{O}(T \cdot (|s| + |a|))$ rollout memory.

## 7. Implementation Notes
- Pure NumPy implementation of Actor-Critic network with decoupled policy and value parameter paths.
- Provides glass-box `DecisionTrace` capturing state value $V(s)$, policy entropy, and individual action probabilities.

## 8. Verification Strategy & Test Oracles
- Clipping boundary oracle test: verifies zero gradient updates outside the $[1-\epsilon, 1+\epsilon]$ threshold for dominant advantage directions.
- REINFORCE baseline test: proves monotonic reduction in gradient variance when subtracting running mean returns.

## 9. Known Limitations
- Standard episodic PPO requires episodic resets or sufficiently long truncation horizons to yield low-bias advantage estimates.

## 10. References
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3), 229-256.

