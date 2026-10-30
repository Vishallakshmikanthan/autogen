# Mathematical Foundations: World Model & Environment Dynamics

> **Module:** M01 (World Model and Environment Engine)  
> **Subsystem:** SYS-02 (Environment Engine)  
> **Source Ref:** Blueprint §3.1–3.6

---

## 1. Spatial Domain & Grid Representation

The environment is modeled as a bounded two-dimensional discrete coordinate lattice:

$$\mathcal{G} = \{ (x, y) \in \mathbb{Z}^2 \mid 0 \le x < W, \, 0 \le y < H \}$$

where $W \in \mathbb{N}_{\ge 4}$ is the grid width and $H \in \mathbb{N}_{\ge 4}$ is the grid height. Dirichlet boundary conditions apply: coordinates outside $\mathcal{G}$ are impassable boundaries with infinite movement cost.

Each cell $C_{xy} \in \mathcal{G}$ is characterized by a 4-tuple:

$$C_{xy} = (\tau_{xy}, \, F_{xy}, \, \delta_{xy}, \, c_{xy})$$

where:
- $\tau_{xy} \in \mathcal{T}_{\text{terrain}}$ is the discrete terrain classification.
- $F_{xy} \in [0, F_{\max}(\tau_{xy})]$ is the current consumable food/resource quantity.
- $\delta_{xy} \in [0, 1]$ is the environmental hazard/danger index.
- $c_{xy} \in \mathbb{R}_{\ge 1}$ is the movement energy impedance coefficient.

---

## 2. Terrain Classification & Physical Properties

The terrain alphabet $\mathcal{T}_{\text{terrain}}$ and physical properties are defined as:

$$\mathcal{T}_{\text{terrain}} = \{ \text{Grassland}, \, \text{Forest}, \, \text{Desert}, \, \text{Water}, \, \text{Mountain}, \, \text{Obstacle} \}$$

| Terrain Type ($\tau$) | Walkable ($\omega$) | Base Move Cost ($c$) | Resource Capacity ($F_{\max}$) | Regen Rate ($g$) | Danger ($\delta$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Grassland** | $\text{True}$ | $1.0$ | $10.0$ | $0.20$ | $0.00$ |
| **Forest** | $\text{True}$ | $1.5$ | $25.0$ | $0.50$ | $0.00$ |
| **Desert** | $\text{True}$ | $2.0$ | $2.0$ | $0.05$ | $0.05$ |
| **Water** | $\text{False}$| $\infty$ | $0.0$ | $0.00$ | $0.00$ |
| **Mountain** | $\text{False}$| $\infty$ | $0.0$ | $0.00$ | $0.00$ |
| **Obstacle** | $\text{False}$| $\infty$ | $0.0$ | $0.00$ | $0.00$ |

---

## 3. World State Formalization

The global world state at discrete time step $t \in \mathbb{N}_0$ is represented by the 4-tuple:

$$S_t = (E_t, \, A_t, \, R_t, \, W_t)$$

where:
- $E_t$: Environmental ambient conditions (e.g. weather parameters, seasonal modifiers $\xi_t$).
- $A_t = \{ s_t^1, s_t^2, \dots, s_t^{N_t} \}$: The set of active agent state tuples indexed by agent ID $i$.
- $R_t \in \mathbb{R}^{H \times W}$: The spatial resource matrix representing $F_{xy}(t)$ across all cells.
- $W_t$: Static configuration parameters of the world (dimensions $W, H$, capacities, seeds).

### Individual Agent State
Each agent $i \in A_t$ has a discrete state tuple:

$$s_t^i = (p_t^i, \, e_t^i, \, h_t^i, \, f_t^i, \, \sigma_t^i, \, m_t^i)$$

where:
- $p_t^i = (x_t^i, y_t^i) \in \mathcal{G}$: Discrete spatial location.
- $e_t^i \in [0, E_{\max}]$: Metabolic energy store ($E_{\max} \in \mathbb{R}_{>0}$).
- $h_t^i \in [0, H_{\max}]$: Physical structural health ($H_{\max} \in \mathbb{R}_{>0}$).
- $f_t^i \in [0, F_{\text{carry\_max}}]$: Carried unconsumed food inventory.
- $\sigma_t^i \in \mathbb{N}_0$: Consecutive starvation ticks counter.
- $m_t^i$: Reference to internal cognitive memory structures.

---

## 4. Action Space & Kinematics

The discrete action space $\mathcal{A}$ available to agents is:

$$\mathcal{A} = \{ \text{UP}, \, \text{DOWN}, \, \text{LEFT}, \, \text{RIGHT}, \, \text{STAY}, \, \text{COLLECT} \}$$

Directional displacements $\vec{d}(a)$ are defined as:
$$\vec{d}(\text{UP}) = (0, -1), \quad \vec{d}(\text{DOWN}) = (0, 1), \quad \vec{d}(\text{LEFT}) = (-1, 0), \quad \vec{d}(\text{RIGHT}) = (1, 0), \quad \vec{d}(\text{STAY}) = (0, 0)$$

### Kinematic Transition & Collision Rule
For an intended movement action $a \in \{ \text{UP}, \text{DOWN}, \text{LEFT}, \text{RIGHT}, \text{STAY} \}$:
$$p_{\text{target}} = p_t^i + \vec{d}(a)$$

The tentative movement is validated according to spatial boundaries and walkability:

$$p_{t+1}^i = \begin{cases}
p_{\text{target}}, & \text{if } p_{\text{target}} \in \mathcal{G} \text{ and } \omega(\tau(p_{\text{target}})) = \text{True} \\
p_t^i, & \text{otherwise (collision / illegal move)}
\end{cases}$$

If $p_{\text{target}} \ne p_{t+1}^i$, the action is deemed invalid. The agent remains at $p_t^i$, but is penalized by an idle energy cost $C_{\text{idle}}$.

---

## 5. Environmental Dynamics & Transition Model

The state transition function $\mathcal{T}$ maps the current state and joint actions to the next state:

$$S_{t+1} = \mathcal{T}(S_t, \, \mathbf{a}_t, \, \xi_t)$$

In deterministic baseline mode, $\xi_t = 0$.

### 5.1 Resource Regeneration & Consumption
At each time step $t$, for every cell $(x, y) \in \mathcal{G}$:

$$F_{t+1}(x, y) = \min\left(F_{\max}(\tau_{xy}), \, \max\left(0.0, \, F_t(x, y) + G_t(x, y) - C_t(x, y)\right)\right)$$

where:
- Natural regeneration:
  $$G_t(x, y) = g(\tau_{xy}) \cdot (1 + \xi_{\text{weather}}(t))$$
- Consumption:
  $$C_t(x, y) = \sum_{i: p_t^i = (x, y) \land a_t^i = \text{COLLECT}} \min\left(F_t(x, y), \, \kappa_{\text{collect}}\right)$$
  where $\kappa_{\text{collect}}$ is the maximum food collected per collection tick.

### 5.2 Agent Energy Dynamics
For each agent $i$, action $a_t^i$ incurs an energy expenditure $C_a$:

$$C_a(a_t^i, \tau) = \begin{cases}
c(\tau) \cdot C_{\text{base\_move}}, & \text{if } a_t^i \in \{\text{UP}, \text{DOWN}, \text{LEFT}, \text{RIGHT}\} \\
C_{\text{stay}}, & \text{if } a_t^i = \text{STAY} \\
C_{\text{collect}}, & \text{if } a_t^i = \text{COLLECT}
\end{cases}$$

Let $R_f$ denote the energy recovered by consuming collected food:
$$\Delta E_t^i = R_f \cdot \Delta f_{\text{eaten}}^i - C_a(a_t^i, \tau(p_t^i))$$

The metabolic energy updates as:

$$E_{t+1}^i = \min\left(E_{\max}, \, \max\left(0.0, \, E_t^i + \Delta E_t^i\right)\right)$$

### 5.3 Starvation vs. Health Separation (Strict Rule)

> **CORE INVARIANT:** Energy and health represent distinct biological systems. Energy depletion induces starvation; starvation subsequently damages health over time. Health damage does not directly deplete energy.

The starvation counter and health transition rules are defined:

$$\sigma_{t+1}^i = \begin{cases}
\sigma_t^i + 1, & \text{if } E_{t+1}^i = 0.0 \\
0, & \text{if } E_{t+1}^i > 0.0
\end{cases}$$

$$h_{t+1}^i = \begin{cases}
\max\left(0.0, \, h_t^i - D_{\text{starve}}\right), & \text{if } E_{t+1}^i = 0.0 \text{ (Starvation damage)} \\
\min\left(H_{\max}, \, h_t^i + H_{\text{regen}}\right), & \text{if } E_{t+1}^i > 0.5 \cdot E_{\max} \text{ (Satiated healing)} \\
h_t^i, & \text{otherwise}
\end{cases}$$

where $D_{\text{starve}} > 0$ is the starvation health damage per tick, and $H_{\text{regen}} \ge 0$ is natural recovery.

### 5.4 Population Mortality Dynamics
The active agent count evolves according to:

$$N_{t+1} = N_t + B_t - D_t$$

where:
- Births: $B_t = 0$ for Module M01 through M15.
- Deaths: $D_t = |\{ i \in A_t \mid h_{t+1}^i = 0.0 \}|$.
- Agents with $h_{t+1}^i = 0.0$ are marked `is_alive = False` and removed from the active set.

---

## 6. Procedural Topology & Guaranteed Connectivity

To guarantee that an agent starting at any walkable coordinate can reach any other walkable coordinate, the procedural generator enforces single-component connectivity:

1. Let $\mathcal{W} \subset \mathcal{G}$ be the set of cells with $\omega(\tau_{xy}) = \text{True}$.
2. The connectivity graph is defined as an undirected graph $G = (\mathcal{W}, \mathcal{E})$, where $(u, v) \in \mathcal{E} \iff \|u - v\|_1 = 1$.
3. BFS partitions $G$ into connected components $\mathcal{C} = \{ C_1, C_2, \dots, C_k \}$.
4. If $k > 1$, corridors between non-maximal components and the largest component $C_{\max}$ are carved by converting intervening cells along shortest Manhattan paths into `Grassland`.
5. **Invariant:** Upon generation completion, $|\mathcal{C}| = 1$ and $|\mathcal{W}| \ge \rho_{\min} \cdot W \cdot H$, where $\rho_{\min} \in (0, 1)$ is the minimum allowable walkable ratio.

---

## 7. Canonical State Hashing

For reproducible verification and time-machine verification, the canonical hash $H(S_t)$ is the SHA-256 digest of the canonical byte representation:

$$H(S_t) = \text{SHA256}\left(\text{JSON}_{\text{canonical}}\left(t, \, W, \, H, \, R_t, \, \text{sorted}(A_t)\right)\right)$$

Guarantees identical values across different OS architectures, Python versions, and memory address spaces.
