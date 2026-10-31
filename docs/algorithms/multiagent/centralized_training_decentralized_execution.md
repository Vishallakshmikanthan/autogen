# Centralized Training with Decentralized Execution (CTDE) (SYS-14, Module M15)

## 1. Theory & Mathematical Definition
Centralized Training with Decentralized Execution (CTDE) resolves non-stationarity in cooperative multi-agent reinforcement learning. During training, a centralized critic conditions on global state $\mathbf{s}$ and joint actions $\mathbf{a}$, providing stationary value estimates and reducing variance. During execution, the centralized critic is discarded; each agent executes its decentralized actor policy $\pi_{\theta_i}(a_i \mid o_i)$ using only local observations $o_i$.

## 2. Assumptions
1. Global state $\mathbf{s}$ is available during simulated training, but inaccessible during decentralized execution.
2. Cooperative setting with shared team reward $R_{\text{team}} = \sum_{i=1}^N r_i$ or mixed individual/team rewards.

## 3. Equations & Variable Definitions
- **Centralized Value Function**:
  $$V_\phi(\mathbf{s}) \approx \mathbb{E}_{\boldsymbol{\pi}}\left[\sum_{t=0}^\infty \gamma^t R_{\text{team}, t} \mid \mathbf{S}_0 = \mathbf{s}\right]$$
- **Critic TD Target & Error**:
  $$y = R_{\text{team}} + \gamma V_\phi(\mathbf{s}'), \quad \delta = y - V_\phi(\mathbf{s})$$
- **Decentralized Actor Policy Gradient**:
  $$\nabla_{\theta_i} J(\theta_i) = \mathbb{E}\left[\nabla_{\theta_i} \log \pi_{\theta_i}(a_i \mid o_i) \cdot \hat{A}(\mathbf{s})\right]$$
  where centralized advantage $\hat{A}(\mathbf{s}) = y - V_\phi(\mathbf{s})$.

## 4. Derivation of the Advantage Baseline
Subtracting state-value baseline $V_\phi(\mathbf{s})$ from return $y$ leaves policy gradient expectation unchanged because:
$$\mathbb{E}_{a_i \sim \pi_i}\left[\nabla_{\theta_i} \log \pi_{\theta_i}(a_i \mid o_i) V_\phi(\mathbf{s})\right] = V_\phi(\mathbf{s}) \nabla_{\theta_i} \sum_{a_i} \pi_{\theta_i}(a_i \mid o_i) = V_\phi(\mathbf{s}) \nabla_{\theta_i}(1) = 0$$
This reduces gradient variance without introducing bias.

## 5. Pseudocode
```text
function CTDE_TRAIN_STEP(global_s, local_obs_map, actions_map, team_reward, next_global_s, terminated):
    v_next = Critic(next_global_s)
    target = team_reward if terminated else team_reward + gamma * v_next
    v_current = Critic(global_s)
    advantage = target - v_current
    
    # Update Centralized Critic
    loss_critic = (target - v_current)^2
    Backprop(Critic, loss_critic)
    
    # Update Decentralized Actors
    for each agent i in 1 to N:
        probs = Actor_i(local_obs_map[i])
        loss_actor = -log(probs[actions_map[i]]) * advantage
        Backprop(Actor_i, loss_actor)
```

## 6. Computational Complexity
- **Time Complexity**: $\mathcal{O}(|\text{Critic}| + \sum_{i=1}^N |\text{Actor}_i|)$ per gradient update.
- **Space Complexity**: $\mathcal{O}(\dim(\mathbf{s}) + \sum_{i=1}^N \dim(o_i))$ memory.

## 7. Implementation Notes
Implemented in `CentralizedCriticMARL` and `MARLAgent`. Built entirely with NumPy neural layers (`DenseLayer`, `ReLU`, `Softmax`, `Adam`).

## 8. Verification Strategy & Test Oracles
- Convergence oracle: In cooperative non-stationary tasks, CTDE achieves higher and more stable team returns than IQL.
- Local observation privacy test: Verifies that during `decide()`, `MARLAgent` inspects only its local `Observation`.

## 9. Known Limitations
- Does not address the credit assignment problem between individual contributions (addressed by VDN/QMIX).

## 10. References
- Lowe, R., Wu, Y., Tamar, A., et al. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. *NeurIPS*, 30.
- Foerster, J., Farquhar, G., Afouras, T., et al. (2018). Counterfactual multi-agent policy gradients. *AAAI*, 32(1).
