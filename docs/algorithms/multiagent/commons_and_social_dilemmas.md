# Tragedy of the Commons & Social Dilemma Games (SYS-14, Module M15)

## 1. Theory & Mathematical Definition
Social dilemmas capture situations in which individual rationality leads to collective irrationality. The Tragedy of the Commons models shared finite renewable resources where unconstrained private consumption degrades or destroys the common asset. Normal-form 2-player bimatrix games (Prisoner's Dilemma, Stag Hunt, Matching Pennies) provide foundational game-theoretic models for analyzing competition, defection dominance, and payoff coordination.

## 2. Assumptions
1. Discrete-time interactions with shared pool stock $R_t \in [0, K]$.
2. Non-linear logistic renewal subject to critical tipping-point thresholds $R_{\text{collapse}}$.

## 3. Equations & Variable Definitions
- **Renewable Logistic Regrowth**:
  $$R_{t+1} = \begin{cases} 0 & \text{if } R_t \le R_{\text{collapse}} \\ \min\left(K, R_t + r R_t \left(1 - \frac{R_t}{K}\right) - \sum_{i=1}^N h_{i, t}\right) & \text{otherwise} \end{cases}$$
  where $K$ is carrying capacity, $r$ is intrinsic growth rate, and $h_{i, t}$ is agent $i$'s harvest.
- **Prisoner's Dilemma Payoff Structure**:
  $$T > R > P > S \quad \text{and} \quad 2R > T + S$$
  where $T=5$ (Temptation to defect), $R=3$ (Reward for mutual cooperation), $P=1$ (Punishment for mutual defection), $S=0$ (Sucker's payoff).
- **Gini Coefficient of Inequality**:
  $$G = \frac{\sum_{i=1}^N \sum_{j=1}^N |w_i - w_j|}{2 N^2 \bar{w}}$$
  measuring resource distribution inequality among $N$ agents.

## 4. Derivation of Defection Dominance
For player 1 facing unknown mixed strategy $p = P(\text{Player 2 Cooperates})$:
$$\mathbb{E}[U(\text{Cooperate})] = 3p + 0(1-p) = 3p$$
$$\mathbb{E}[U(\text{Defect})] = 5p + 1(1-p) = 4p + 1$$
Because $4p + 1 > 3p$ for all $p \in [0, 1]$, Defection strictly dominates Cooperation regardless of the opponent's strategy in one-shot and uncoordinated settings.

## 5. Pseudocode
```text
function STEP_COMMONS(stock, harvest_requests, K, r, collapse_threshold):
    total_harvest = sum(min(req, stock) for req in harvest_requests)
    stock = stock - total_harvest
    if stock <= collapse_threshold:
        return stock, 0.0, True  # Permanent collapse
    regrowth = r * stock * (1.0 - stock / K)
    stock = min(K, stock + regrowth)
    return stock, regrowth, False
```

## 6. Computational Complexity
- **Time Complexity**: $\mathcal{O}(N)$ per time step to process $N$ agent harvests.
- **Space Complexity**: $\mathcal{O}(1)$ scalar pool state tracking.

## 7. Implementation Notes
Implemented in `CommonsResourcePool` and `MatrixGame`. Integrates with `MultiAgentMetrics` to record depletion time and Gini indices.

## 8. Verification Strategy & Test Oracles
- Matrix Game Oracle: IQL convergence to mutual defection $(D, D)$ on Prisoner's Dilemma.
- Stag Hunt Oracle: Convergence to Pareto-optimal $(S, S)$ when initialized towards cooperation.
- Tipping-point test: Proves that stock below threshold halts regrowth permanently.

## 9. Known Limitations
- The standard model assumes homogeneous resource quality across spatial cells.

## 10. References
- Hardin, G. (1968). The tragedy of the commons. *Science*, 162(3859), 1243-1248.
- Axelrod, R., & Hamilton, W. D. (1981). The evolution of cooperation. *Science*, 211(4489), 1390-1396.
