# Predator-Prey Adversarial Lookahead & Subgame Modeling

> **Module:** M07 — Adversarial Search & Game Theory (SYS-10)  
> **Status:** Production / Verified

---

## 1. Theory & Mathematical Definition
In the AUTOGEN simulation engine (M02), agent actions occur simultaneously in discrete ticks. However, modeling joint actions over extended lookahead horizons incurs an exponential branching factor of $|A|^2$ per step. To provide tractability and structured adversarial anticipation, we formulate a turn-based subgame abstraction `PredatorPreySubGame` where predator and prey execute alternating moves in virtual lookahead time.

## 2. Assumptions & Modeling Validity
1. **Alternating Turn Approximation:** Valid when predator reaction time is sufficiently rapid to adjust to evasive maneuvers.
2. **Local Horizon:** Tactical chase/evasion subgames are evaluated over small depth horizons ($d \in [2, 4]$) centered on current spatial coordinates.
3. **Impassable Obstacles:** Terrain with `walkable == False` constrains legal candidate moves for both pursuer and evader.

## 3. Equations & Variable Definitions

The state $s$ of the subgame is defined as:
$$s = (p_{\text{pred}}, p_{\text{prey}}, E_{\text{pred}}, E_{\text{prey}}, \text{turn}, B)$$
where $p \in \mathbb{Z}^2$ are grid coordinates, $E \ge 0$ are energy levels, $\text{turn} \in \{0, 1\}$ designates active player, and $B \subset \mathbb{Z}^2$ is the set of impassable obstacles.

### Evaluation Function Formulation
From the perspective of MAX (predator):
$$h_{\text{predator}}(s) = \begin{cases}
+C_{\text{bonus}}, & \text{if } p_{\text{pred}} = p_{\text{prey}} \text{ (Captured)} \\
-w_{\text{dist}} \cdot d_{\text{Manhattan}}(p_{\text{pred}}, p_{\text{prey}}) - w_{\text{escape}} \cdot \text{deg}_{\text{free}}(p_{\text{prey}}) + w_{\text{energy}} \cdot (E_{\text{pred}} - E_{\text{prey}}), & \text{otherwise}
\end{cases}$$

Where:
- $d_{\text{Manhattan}}(p_1, p_2) = |x_1 - x_2| + |y_1 - y_2|$.
- $\text{deg}_{\text{free}}(p_{\text{prey}}) = \sum_{a \in \text{Cardinals}} \mathbb{I}[p_{\text{prey}} + \Delta a \notin B \text{ and } p_{\text{prey}} + \Delta a \neq p_{\text{pred}}]$ measures prey escape routes.
- $w_{\text{dist}} = 10.0, w_{\text{escape}} = 5.0, w_{\text{energy}} = 0.5, C_{\text{bonus}} = 1000.0$.

From the perspective of MIN (prey):
$$h_{\text{prey}}(s) = -h_{\text{predator}}(s)$$

## 4. Move Ordering Heuristic
To optimize Alpha-Beta pruning, `order_predator_moves` sorts actions by proximity to target:
- Predator evaluates moves minimizing distance to prey first.
- Prey evaluates moves maximizing distance from predator first.

## 5. Computational Complexity
- Branching factor $b \le 5$ (Stay, Up, Down, Left, Right).
- For depth $d=4$, total search nodes without pruning $\approx 5^4 = 625$.
- With Alpha-Beta and distance move ordering, average nodes expanded $\approx 5^{4/2} = 25$ nodes per tick, executing in $< 2$ milliseconds.

## 6. Integration Hook & Visualization ("Hunt" Preset)
- `PredatorAgent` records root lookahead trees in `last_search_tree`.
- `ViewerDataAdapter` serializes this tree into `ViewerState.lookahead_tree`.
- `PygameViewer` visualizes evaluated branches, candidate scores, and pruned cutoffs in real time when toggling the Hunt overlay (`P` key).

## 7. Known Limitations
- When prey is beyond sensory visibility radius, subgame cannot be initialized; agents fallback to exploratory movement.
- True simultaneous moves can result in head-on collisions or simultaneous swaps not captured by turn-based alternating tree lookahead.

## 8. References
- Bopardikar, S. D., Suri, S., & Hespanha, J. P. (2010). Pursuit-evasion games with limited visibility. *Discrete Applied Mathematics*, 158(11), 1146-1157.

