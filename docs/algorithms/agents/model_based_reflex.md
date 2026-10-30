# Algorithm Specification: Model-Based Reflex Agent

> **Subsystem:** SYS-03 (Agent Framework)  
> **Module:** M03 (Gate 1 Part 3)  
> **Source Ref:** Blueprint §4.2; Russell & Norvig (AIMA 4th ed.) §2.4.2

---

## 1. Theory & Mathematical Definition
A model-based reflex agent maintains internal state $M_t$ representing the agent's current knowledge of the world:
$$M_t = \text{UPDATE}(M_{t-1}, a_{t-1}, o_t)$$
$$a_t = \pi(M_t)$$
This internal state decouples immediate sensory field-of-view from deliberative action choice, enabling long-range food navigation and frontier exploration.

---

## 2. Assumptions
1. World is static or quasi-static with respect to terrain features (walls and mountains do not move).
2. Food items observed at step $t$ remain at their recorded coordinates until consumed or expired.
3. The internal model $M_t(x, y)$ strictly reflects observed cells:
   $$M_t(x, y) \ne \emptyset \iff \exists \tau \le t \text{ s.t. } (x, y) \in o_\tau.V_{cells}$$

---

## 3. Equations & Model Representation
1. **Internal Spatial Map:**
   $$M_t: \mathbb{Z}^2 \to \langle \text{terrain}, \text{walkable}, \text{food}, \tau_{last} \rangle$$
2. **Belief Update Rule:**
   $$\forall c \in o_t.V_{cells}: M_t(c.pos) \leftarrow \langle c.terrain, \text{walkable}(c.terrain), c.food, t \rangle$$
3. **Known Food Set:**
   $$\mathcal{F}_t = \{pos \in \text{dom}(M_t) \mid M_t(pos).food > 0\}$$
4. **Target Selection:**
   - If $p_t \in \mathcal{F}_t$: harvest ($a_t = \text{COLLECT}$).
   - Else if $\mathcal{F}_t \ne \emptyset$: target $f^* = \arg\min_{f \in \mathcal{F}_t} \text{dist}(p_t, f)$ via shortest path over known walkable tiles.
   - Else: target frontier cell $c^* = \arg\min_{c \in \text{Frontier}} \tau_{last}(c)$.

---

## 4. Pseudocode
```text
Algorithm: ModelBasedReflexUpdate(observation, internal_map, known_food)
Input: o_t, M, F
Output: updated M, updated F

1. For each cell in o_t.visible_cells:
     M[cell.pos] <- CellInfo(cell.terrain, cell.walkable, cell.food, o_t.step)
     If cell.food > 0:
       F[cell.pos] <- cell.food
     Else if cell.pos in F:
       Remove cell.pos from F

Algorithm: ModelBasedReflexDecide(observation, M, F)
Input: o_t, M, F
Output: action a_t, trace

1. If food_amount(M[o_t.position]) > 0:
     Return COLLECT, Score(100.0)
2. If F is not empty:
     path <- BFS_Shortest_Path(o_t.position, F, M_walkable)
     If path is not empty:
       next_step <- path[0]
       Return DirectionTo(o_t.position, next_step), Score(80.0)
3. frontiers <- AdjacentWalkableCells(o_t.position, M)
4. best_dir <- argmin_{dir in frontiers} M[o_t.position + dir].last_observed_tick
5. If best_dir exists:
     Return best_dir, Score(30.0)
6. Return STAY, Score(0.0)
```

---

## 5. Computational Complexity
- **Time Complexity:**
  - Update: $O(|V_{cells}|)$
  - Deliberation: $O(|V_{known}| + |E_{known}|)$ via BFS over known mapped subgraph, bounded by $O(W \times H)$.
- **Space Complexity:** $O(|\text{Explored Cells}|) \le O(W \times H)$.

---

## 6. Implementation Notes
- `ModelBasedReflexAgent` explicitly declares `capabilities = AgentCapabilities(memory=True)`.
- Internal map copies scalar attributes; it never stores environment pointers.

---

## 7. Verification Strategy & Test Oracles
1. **Map Invariant Test:** Assert `agent.internal_map` coordinates $\equiv \bigcup_{\tau=0}^t \text{coords}(o_\tau.V_{cells})$.
2. **Ground Truth Comparison:** Assert cell terrain and food in `internal_map` match world state if and only if observed.
3. **Out-of-Sight Navigation:** Place food 10 tiles away, allow agent to observe once, remove agent from direct line of sight; assert agent successfully navigates to food using internal memory.

---

## 8. Known Limitations
- When another agent consumes food out of sight, `ModelBasedReflexAgent` will navigate to the empty cell until it arrives and re-observes the location.
- High memory usage on massive open-world maps ($> 1000 \times 1000$).

---

## 9. References
- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach (4th ed.)*. Pearson. §2.4.2 "Model-based reflex agents".

