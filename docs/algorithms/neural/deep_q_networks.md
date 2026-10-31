# Deep Q-Networks (DQN) (SYS-13, Module M14)

## 1. Theory & Mathematical Definition
Deep Q-Networks approximate the optimal action-value function $Q^*(s, a)$ of a Markov Decision Process using parameterized neural networks $Q(s, a; \theta)$. To overcome instability arising from non-stationary targets and correlated training sequences, DQN introduces Experience Replay and a separate Target Network $\theta^-$.

## 2. Assumptions
1. Environment satisfies the discrete-action Markov Decision Process property.
2. Experience replay buffer size is sufficient to decorrelate consecutive transitions.

## 3. Equations & Variable Definitions
- **Bellman Optimality Equation**:
  $$Q^*(s, a) = \mathbb{E}\left[R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') \mid S_t = s, A_t = a\right]$$
- **Bellman Target ($y_i$)**:
  $$y_i = \begin{cases} r_i & \text{if } s'_i \text{ is terminal} \\ r_i + \gamma \max_{a'} Q(s'_i, a'; \theta^-) & \text{otherwise} \end{cases}$$
- **Huber Loss**:
  $$L(\theta) = \frac{1}{B} \sum_{i=1}^B \mathcal{H}_\delta\left(y_i - Q(s_i, a_i; \theta)\right)$$
- **Target Network Synchronization**:
  Hard update: $\theta^- \leftarrow \theta$ every $C$ steps.  
  Soft (Polyak) update: $\theta^- \leftarrow (1 - \tau)\theta^- + \tau \theta$.

## 4. Derivation of the Semi-Gradient Update
The objective treats target $y_i$ as fixed with respect to $\theta$:
$$\nabla_\theta L(\theta) = -\frac{1}{B} \sum_{i=1}^B \left(y_i - Q(s_i, a_i; \theta)\right) \nabla_\theta Q(s_i, a_i; \theta)$$
This semi-gradient bypasses differentiating through the max operator of the target.

## 5. Pseudocode
```text
Initialize Q_theta, Q_theta_minus with identical random weights
Initialize ReplayBuffer D of capacity N
for each environment step:
    Select action a_t via epsilon-greedy policy on Q_theta(s_t)
    Execute a_t, observe r_t, s_{t+1}, terminated
    Store transition (s_t, a_t, r_t, s_{t+1}, terminated) in D
    if |D| >= batch_size:
        Sample mini-batch B from D
        Compute targets y_i (setting y_i = r_i when terminated)
        Update theta via Adam optimizer on Huber loss
        Update theta_minus periodically or via Polyak smoothing
```

## 6. Computational Complexity
- **Time Complexity**: $\mathcal{O}(B \cdot |\text{net}|)$ per gradient step.
- **Space Complexity**: $\mathcal{O}(N \cdot (|s| + |a| + 1))$ memory for circular buffer storage.

## 7. Implementation Notes
- Strict terminal invariant: Future bootstrap term is identically zero for terminal transitions.
- Fully compatible with `BaseAgent` and generates comprehensive `DecisionTrace` with candidate Q-values.

## 8. Verification Strategy & Test Oracles
- Terminal invariant oracle test: verifies $y = r$ when terminal flag is true, regardless of target network Q-value magnitude.
- Small grid environment: verified convergence to optimal actions matching M11 $Q^*$.
- Replay buffer test: circular overwrite behavior and uniform random sampling statistics.

## 9. Known Limitations
- Standard DQN suffers from maximization bias; double DQN (M15) mitigates this overestimation.

## 10. References
- Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.

