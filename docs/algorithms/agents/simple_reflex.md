# Algorithm Specification: Simple Reflex Agent

> **Subsystem:** SYS-03 (Agent Framework)  
> **Module:** M03 (Gate 1 Part 3)  
> **Source Ref:** Blueprint §4.2, §4.3; Russell & Norvig (AIMA 4th ed.) §2.4.1

---

## 1. Theory & Mathematical Definition
A simple reflex agent selects actions based exclusively on the current percept (or sensory observation $o_t$), ignoring the history of prior percepts:
$$a_t = \pi(o_t)$$
The policy $\pi$ is implemented as a set of condition-action production rules:
$$\text{IF } \text{condition}(o_t) \text{ THEN } a_t$$

---

## 2. Assumptions
1. Environment is partially observable, but sufficient immediate cues exist in the agent's observation window.
2. Complete rationality is unachievable under infinite loops without random tie-breaking or internal state.
3. Prioritized rule ordering resolves conflicting rule triggers deterministically.

---

## 3. Equations & Production Rules
Given observation $o_t = (p_t, E_t, H_t, V_{cells}, V_{agents})$:

1. **Harvest Rule:**
   $$\text{IF } \text{food}(p_t) > 0 \implies a_t = \text{COLLECT}$$
2. **Food Pursuit Rule:**
   $$\text{IF } \exists c \in V_{cells} \text{ s.t. } \text{food}(c) > 0 \implies a_t = \arg\min_{d \in \text{WalkableDirs}} \|(p_t + d) - c_{nearest}\|_1$$
3. **Obstacle Evasion Rule:**
   $$\text{IF } \neg\text{walkable}(p_t + \vec{h}_t) \implies a_t = \text{turn}(p_t, \vec{h}_t)$$
4. **Default Advance Rule:**
   $$\text{OTHERWISE } a_t = \vec{h}_t$$

---

## 4. Pseudocode
```text
Algorithm: SimpleReflexDecide(observation, heading)
Input: observation o_t, current heading h_t
Output: action a_t, trace

1. If food_at(o_t.position) > 0:
     Return COLLECT, Score(100.0)
2. visible_food <- { c in o_t.visible_cells | c.food > 0 }
3. If visible_food is not empty:
     target <- argmin_{c in visible_food} Manhattan(o_t.position, c.position)
     best_dir <- dir in {UP, DOWN, LEFT, RIGHT} that is walkable and minimizes dist to target
     If best_dir exists:
       Return best_dir, Score(50.0)
4. If not walkable(o_t.position + h_t):
     turn_dir <- first walkable direction in clockwise order from h_t
     If turn_dir exists:
       Return turn_dir, Score(20.0)
5. If walkable(o_t.position + h_t):
     Return h_t, Score(10.0)
6. Return STAY, Score(0.0)
```

---

## 5. Computational Complexity
- **Time Complexity:** $O(|V_{cells}| + |A|)$ where $|V_{cells}| = (2r+1)^2$ for Chebyshev radius $r$. With $r=3$, $|V_{cells}| = 49$, which evaluates in microseconds.
- **Space Complexity:** $O(1)$ auxiliary storage (no memory retention across steps).

---

## 6. Implementation Notes
- Pure functional condition-action matching; no ground truth world state is accessible.
- Candidate actions are scored according to rule activation level and attached to the verifiable `DecisionTrace`.

---

## 7. Verification Strategy & Test Oracles
1. **Harvesting Test:** Place food at agent position; assert agent selects `COLLECT` with candidate score 100.
2. **Navigation Test:** Place food 1 tile away; assert agent takes directional step toward food.
3. **Evasion Test:** Place wall ahead; assert agent turns to clear walkable tile.
4. **Statistical Benchmark:** Over $\ge 30$ seeds, assert reflex agent discounted return $J$ statistically beats `RandomAgent` with paired t-test $p < 0.01$.

---

## 8. Known Limitations
- Susceptible to infinite cycles in symmetric concave obstacles (e.g. U-shaped traps) without internal state or stochastic perturbation.
- Cannot recall previously observed food once it leaves sensory field of view.

---

## 9. References
- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach (4th ed.)*. Pearson. §2.4.1 "Simple reflex agents".

