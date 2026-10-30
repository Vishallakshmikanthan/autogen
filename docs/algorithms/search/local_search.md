# Local & Stochastic Search (Hill Climbing, Simulated Annealing, Local Beam)

> **Module:** M06 — Search Algorithms and Problem Solving (SYS-06, Gate 2)  
> **Source:** Kirkpatrick, Gelatt & Vecchi (1983); Russell & Norvig, AIMA (4th Ed.), Chapter 4.

---

## 1. Theory & Mathematical Definition

Local search algorithms operate using a single current state (or a small set of $k$ states) rather than multiple paths, moving only to neighboring states in state space.

Energy formulation:
$$E(s) = h(s) \quad (\text{Objective to minimize})$$

---

## 2. Simulated Annealing & Boltzmann Probability

Simulated Annealing models statistical mechanics cooling processes:
Given current state $s$ with energy $E(s)$, pick a random neighbor $s'$ with energy $E(s')$.
Energy change:
$$\Delta E = E(s') - E(s)$$

### Acceptance Rule:
1. If $\Delta E < 0$ (improvement): Accept transition unconditionally.
2. If $\Delta E \ge 0$ (worse state): Accept with Boltzmann probability:
   $$P(\text{accept}) = \exp\left(-\frac{\Delta E}{T}\right)$$
   Where $T > 0$ is the current temperature.

### Cooling Schedules:
- **Geometric:** $T_{k+1} = \alpha T_k, \quad \alpha \in (0, 1)$ (default: $\alpha = 0.95$)
- **Linear:** $T_{k+1} = \max(T_{\min}, T_0 - \beta k)$
- **Logarithmic:** $T_k = \frac{T_0}{1 + \ln(1 + k)}$

As $T \to 0$, $P(\text{accept}) \to 0$, causing Simulated Annealing to smoothly converge to pure greedy descent.

---

## 3. Local Beam Search

Maintains $k$ parallel candidate states:
1. Generate all successors of all $k$ states.
2. Sort all successors globally by evaluation function $E(s)$.
3. Retain the top $k$ states as the new beam.
Unlike $k$ independent hill-climbing searches, Local Beam shares information across parallel search threads.

---

## 4. References

- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). *Optimization by Simulated Annealing*. Science, 220(4598), 671–680.
- Metropolis, N., et al. (1953). *Equation of State Calculations by Fast Computing Machines*. The Journal of Chemical Physics, 21(6), 1087–1092.
