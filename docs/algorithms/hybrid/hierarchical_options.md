# Hierarchical Reinforcement Learning and the Options Framework (SYS-15, Module M17)

## 1. Theory & Mathematical Definition
The Options Framework formalizes temporally abstract actions within Semi-Markov Decision Processes (SMDPs). An option $\omega$ is a high-level closed-loop macro-action consisting of an initiation set $\mathcal{I}$, an internal subpolicy $\pi_\omega$, and a stochastic or deterministic termination condition $\beta_\omega$.

## 2. Assumptions
- Environment dynamics satisfy the Markov property at the primitive level.
- Options execute over discrete variable durations $\tau \in \{1, 2, \dots\}$.
- High-level decision making occurs upon option termination.

## 3. Equations & Variable Definitions

### Option Formal Definition
An option is a 3-tuple $\omega = \langle \mathcal{I}_\omega, \pi_\omega, \beta_\omega \rangle$:
1. **Initiation Set:** $\mathcal{I}_\omega \subseteq \mathcal{S}$ specifying states where option $\omega$ can be initiated.
2. **Subpolicy:** $\pi_\omega: \mathcal{S} \times \mathcal{A} \to [0, 1]$ governing primitive action selection during option execution.
3. **Termination Condition:** $\beta_\omega: \mathcal{S} \to [0, 1]$ defining the probability that option $\omega$ terminates upon arriving in state $s$.

### SMDP Cumulative Discounted Option Return
For an option executing from time $t$ for duration $\tau$ steps:
$$r_s^\omega = \mathbb{E}\left[ \sum_{k=0}^{\tau-1} \gamma^k r_{t+k+1} \,\Big|\, s_t = s, \omega \right]$$

### SMDP Q-Learning Bellman Update
Upon option termination in state $s'$ after $\tau$ steps:
$$Q(s, \omega) \leftarrow Q(s, \omega) + \alpha \left[ r_s^\omega + \gamma^\tau \max_{\omega' \in \Omega(s')} Q(s', \omega') - Q(s, \omega) \right]$$

## 4. Pseudocode
```python
def smdp_agent_step(s, observation):
    if active_option is None or active_option.should_terminate(s, observation):
        if active_option is not None:
            update_smdp_q(start_state, active_option, accumulated_reward, duration, s)
        active_option = select_option_epsilon_greedy(s, observation)
        start_state = s
        duration = 0
        accumulated_reward = 0.0

    action = active_option.select_action(s, observation)
    return action
```

## 5. Computational Complexity
- **Action Selection:** $\mathcal{O}(|\Omega|)$ to evaluate initiation sets and select active option.
- **Space Complexity:** $\mathcal{O}(|\mathcal{S}| \cdot |\Omega|)$ for high-level SMDP Q-table.

## 6. Implementation Notes
- Canonical options implemented: `ForageOption`, `FleeHazardOption`, and `WanderOption`.
- Step-wise reward discounting automatically accounts for multi-tick option durations.

## 7. Verification Strategy & Test Oracles
- **Option Invariance:** Initiation predicate must strictly gate invalid option launches.
- **Return Consistency:** SMDP accumulated return $r_s^\omega$ must match manual sum of discounted step rewards.

## 8. Known Limitations
- Hand-crafted options require domain knowledge; end-to-end option discovery requires gradient-based meta-learning.

## 9. References
- Sutton, R. S., Precup, D., & Singh, S. (1999). *Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning*. Artificial Intelligence, 112(1-2), 181-211.

