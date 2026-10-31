# Stigmergic Coordination & Communication Channels (SYS-14, Module M15)

## 1. Theory & Mathematical Definition
Stigmergy is a mechanism of indirect coordination in which agents communicate through traces left in the shared physical environment rather than direct point-to-point signaling. Pheromone trails provide spatial memory and positive feedback for path discovery. Direct communication channels supplement stigmergy via explicit message passing constrained by realistic physical properties (energy dissipation, range attenuation, bandwidth limitations, and channel noise).

## 2. Assumptions
1. Continuous chemical evaporation obeys exponential decay: $\Phi_{t+1} = (1 - \rho)\Phi_t$.
2. Spatial chemical diffusion follows a 2D discrete Laplacian operator with reflective no-flux boundary conditions.
3. Message transmission consumes non-zero agent energy reserves.

## 3. Equations & Variable Definitions
- **Discrete 2D Diffusion-Reaction Equation**:
  $$\frac{\partial \Phi}{\partial t} = D \nabla^2 \Phi - \rho \Phi + S(x, y, t)$$
  where $D \le 0.2$ is the diffusion constant, $\rho \in (0, 1)$ is the evaporation rate, and $S$ is agent deposition.
- **5-Point Discrete Laplacian Stencil**:
  $$\nabla^2 \Phi(x, y) \approx \Phi(x+1, y) + \Phi(x-1, y) + \Phi(x, y+1) + \Phi(x, y-1) - 4\Phi(x, y)$$
- **Spatial Chemical Gradient**:
  $$\nabla \Phi(x, y) = \left(\frac{\Phi(x+1, y) - \Phi(x-1, y)}{2}, \frac{\Phi(x, y+1) - \Phi(x, y-1)}{2}\right)$$
- **Communication Channel Delivery**:
  Delivered if $\|p_{\text{sender}} - p_{\text{recipient}}\|_2 \le d_{\max}$ and sender energy $\ge c_{\text{comm}}$.

## 4. Derivation of Numerical Stability (CFL Condition)
For the 2D explicit Euler diffusion discretization:
$$\Phi^{t+1} = \Phi^t + \frac{D \Delta t}{\Delta x^2} (\nabla^2 \Phi)$$
von Neumann stability analysis requires that the amplification factor satisfies $|\xi| \le 1$, leading to the Courant-Friedrichs-Lewy condition:
$$\frac{D \Delta t}{\Delta x^2} \le \frac{1}{4}$$
With unit grid spacing $\Delta x = 1$ and $\Delta t = 1$, stability requires $D \le 0.25$. Enforcing $D \le 0.20$ guarantees numerical stability without oscillations.

## 5. Pseudocode
```text
function STIGMERGY_STEP(grid, deposition_points, D, rho):
    # 1. Deposition
    for (x, y, amount) in deposition_points:
        grid[y, x] = min(MAX_CONC, grid[y, x] + amount)
        
    # 2. Laplacian diffusion
    laplacian = convolve2d(pad(grid, 1, 'edge'), LAPLACIAN_KERNEL, 'valid')
    grid = grid + D * laplacian
    
    # 3. Evaporation
    grid = max(0, (1 - rho) * grid)
    return grid
```

## 6. Computational Complexity
- **Time Complexity**: $\mathcal{O}(H \cdot W)$ per tick for $H \times W$ grid diffusion.
- **Space Complexity**: $\mathcal{O}(H \cdot W)$ float64 array storage.

## 7. Implementation Notes
- Implemented in `PheromoneField`, `CommunicationBus`, and `CooperativeAgent`.
- Uphill gradient climbing enables resource foraging without global knowledge.

## 8. Verification Strategy & Test Oracles
- Mass conservation and diffusion spread tests: chemical deposited at a single cell spreads symmetrically to all 4 orthogonal neighbors.
- Evaporation verification: unreplenished pheromones decay geometrically.
- Spatial range oracle: messages strictly delivered within Euclidean threshold $d_{\max}$.

## 9. Known Limitations
- High evaporation rates erase long-distance trails before following agents can detect them.

## 10. References
- Grassé, P. P. (1959). La reconstruction du nid et les coordinations interindividuelles chez Bellicositermes natalensis et Cubitermes sp. *Insectes Sociaux*, 6(1), 41-80.
- Dorigo, M., Maniezzo, V., & Colorni, A. (1996). Ant system: optimization by a colony of cooperating agents. *IEEE Transactions on Systems, Man, and Cybernetics, Part B*, 26(1), 29-41.
