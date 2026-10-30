# GridWorld to Tabular MDP Compilation

## 1. Mathematical Formulation

The `GridWorldMDPBuilder` converts a 2D spatial simulation grid of dimensions $W \times H$ with obstacle cells $\mathcal{W} \subset \mathbb{Z}^2$ and terminal goal/hazard cells $\mathcal{G} \subset \mathbb{Z}^2$ into a formal tabular Markov Decision Process $\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma \rangle$.

### State Space $\mathcal{S}$
$$
\mathcal{S} = \{ (x, y) \in \{0, \dots, W-1\} \times \{0, \dots, H-1\} \mid (x, y) \notin \mathcal{W} \}
$$
$|\mathcal{S}| = W \cdot H - |\mathcal{W}|$.

### Action Space $\mathcal{A}$
$$
\mathcal{A} = \{ \text{UP}, \text{DOWN}, \text{LEFT}, \text{RIGHT} \}
$$

### Transition Probability Tensor $\mathcal{T}(s, a, s')$
Movement exhibits stochastic slip parameter $\epsilon \in [0, 1]$:
- Intended direction: probability $1 - \epsilon$.
- Orthogonal slip 1: probability $\epsilon / 2$.
- Orthogonal slip 2: probability $\epsilon / 2$.

If target cell $(x + \Delta x, y + \Delta y)$ is out of bounds or an obstacle wall in $\mathcal{W}$, the agent bounces back and remains at $(x, y)$.

### Reward Tensor $\mathcal{R}(s, a, s')$
$$
\mathcal{R}(s, a, s') = \begin{cases}
R_{\text{terminal}}(s') & \text{if } s' \in \mathcal{G} \\
R_{\text{step}} & \text{otherwise}
\end{cases}
$$

---

## 2. Canonical Russell & Norvig 4x3 Benchmark

The classic $4 \times 3$ GridWorld (Russell & Norvig, AIMA Chapter 17):
- Grid: $W = 4, H = 3$.
- Obstacle: $(1, 1)$.
- Terminals: $(3, 2)$ with reward $+1.0$, $(3, 1)$ with reward $-1.0$.
- Step cost: $-0.04$.
- Slip probability: $\epsilon = 0.2$ (0.8 forward, 0.1 left, 0.1 right).
- Discount factor: $\gamma = 1.0$.

### Analytical Ground Truth Values:
- $V^*(2, 2) \approx 0.812$
- $V^*(0, 2) \approx 0.645$
- $V^*(0, 0) \approx 0.705$
- $V^*(3, 0) \approx 0.388$

---

## 3. Export Schema for M13 Reinforcement Learning

The exact solution $(V^*, Q^*, \pi^*)$ is serialized to JSON/NPZ with schema:
```json
{
  "v_star": [0.705, ...],
  "q_star": [[...], ...],
  "policy": [0, 3, ...],
  "iterations": 23,
  "max_residual": 8.4e-9,
  "converged": true,
  "runtime_sec": 0.0012,
  "state_labels": ["(0,0)", "(1,0)", ...],
  "action_labels": ["UP", "DOWN", "LEFT", "RIGHT"]
}
```
This artifact serves as the ground truth oracle against which Q-learning, SARSA, and DQN convergence will be tested in Module M13.
