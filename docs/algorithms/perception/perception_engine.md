# Subsystem Specification: Perception Engine

> **Subsystem:** SYS-04 (Perception Engine)  
> **Module:** M03 (Gate 1 Part 3)  
> **Source Ref:** Blueprint §2.1 (SYS-04), §4.1; M03 Specification

---

## 1. Theory & Mathematical Definition
The Perception Engine acts as the sensory mediator between the true environment world state $\mathcal{S}$ and an agent $i$'s partial observation $o_i \in \mathcal{O}_i$:
$$\mathcal{P}_{\theta_i}: \mathcal{S} \times \mathcal{A}_i \to \mathcal{O}_i$$
where $\theta_i = \langle \text{mode}, r, \sigma_{\text{noise}}, \text{sensing\_cfg} \rangle$.

### Sensory Modes
1. **Full Observability (`FULL`):**
   $$\mathcal{O}_{cells} = \{s(x, y) \mid 0 \le x < W, 0 \le y < H\}$$
2. **Local Chebyshev Window (`LOCAL_WINDOW`):**
   $$\mathcal{O}_{cells} = \{s(x, y) \mid \|(x, y) - p_i\|_\infty \le r\}$$
3. **Line of Sight with Occlusion (`LINE_OF_SIGHT`):**
   $$\mathcal{O}_{cells} = \left\{s(x, y) \in \mathcal{B}_r(p_i) \;\middle|\; \forall c \in \text{BresenhamRay}(p_i, (x, y)) \setminus \{(x, y)\}, \text{walkable}(s(c))\right\}$$
   Opaque obstacles (e.g. `WALL`, `MOUNTAIN`) terminate the ray: the obstacle cell itself is visible, but cells behind it are occluded.

---

## 2. Sensor Noise Formulation
When noise is enabled:
$$\tilde{F}_{x, y} = \max\left(0, \text{round}\left(F_{x, y} + \epsilon, 2\right)\right), \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$
For other agents, detection is a Bernoulli trial with success probability $p_{\text{detect}}$:
$$D_{\text{agent}} \sim \text{Bernoulli}(p_{\text{detect}})$$

---

## 3. Security & Anti-Leakage Invariant
The Perception Engine strictly guarantees that no ground truth state outside the permitted field of view leaks to the agent:
$$\forall (x, y) \notin \text{PermittedCoords}(\theta_i, p_i), \quad s(x, y) \cap \mathcal{O}_i = \emptyset$$
This property is formally tested using sentinel values injected outside the sensory window.

---

## 4. Computational Complexity
- **Local Window:** $O((2r+1)^2)$ coordinate lookups.
- **Line of Sight:** $O(r \cdot 8r) = O(r^2)$ ray steps via Bresenham integer arithmetic.
- **Space:** $O(r^2)$ tuples allocated per agent per tick.

---

## 5. References
- Bresenham, J. E. (1965). "Algorithm for computer control of a digital plotter". *IBM Systems Journal*, 4(1), 25–30.

