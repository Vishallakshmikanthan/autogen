# Convolutional Perception & Terrain Classification (SYS-13, Module M14)

## 1. Theory & Mathematical Definition
Convolutional neural networks apply discrete spatial cross-correlation kernels to multi-channel 2D sensory grids, capturing translation-invariant local patterns. The `CNNPerceptionModule` extracts local grid windows from `Observation.visible_cells` and categorizes terrain cells into canonical classes (`EMPTY`, `OBSTACLE`, `FOOD`, `WATER`, `HAZARD`).

## 2. Assumptions
- Spatial invariance: Local visual/terrain features have consistent semantics regardless of coordinate offsets within the perceptual field.

## 3. Equations & Variable Definitions
- **2D Spatial Cross-Correlation**:
  $$z_{n, c_{out}, h, w} = \sum_{c_{in}=1}^{C_{in}} \sum_{i=0}^{K-1} \sum_{j=0}^{K-1} x_{n, c_{in}, h+i, w+j} W_{c_{out}, c_{in}, i, j} + b_{c_{out}}$$
  where input $x \in \mathbb{R}^{N \times C_{in} \times H \times W}$, weight $W \in \mathbb{R}^{C_{out} \times C_{in} \times K \times K}$.
- **Backward Kernel Gradient**:
  $$\frac{\partial L}{\partial W_{c_{out}, c_{in}, i, j}} = \sum_{n=1}^N \sum_{h, w} \delta_{n, c_{out}, h, w} x_{n, c_{in}, h+i, w+j}$$
- **Backward Input Gradient**:
  $$\frac{\partial L}{\partial x_{n, c_{in}, h+i, w+j}} = \sum_{c_{out}} \delta_{n, c_{out}, h, w} W_{c_{out}, c_{in}, i, j}$$

## 4. Derivation of Spatial Backpropagation
By treating the discrete convolution as a structured sparse linear transformation, applying the chain rule transposes the kernel spatial dimensions and correlates the upstream gradient map $\delta$ with input patches.

## 5. Pseudocode
```text
function CONV2D_FORWARD(x, W, b, padding):
    x_pad = pad(x, padding)
    for h in 0 to H_out - 1:
        for w in 0 to W_out - 1:
            window = x_pad[:, :, h:h+K, w:w+K]
            z[:, :, h, w] = einsum("nchw,ochw->no", window, W) + b
    return sigma(z)
```

## 6. Computational Complexity
- **Time Complexity**: $\mathcal{O}(N \cdot C_{out} \cdot C_{in} \cdot H_{out} \cdot W_{out} \cdot K^2)$ per convolution layer.
- **Space Complexity**: $\mathcal{O}(N \cdot C_{out} \cdot H_{out} \cdot W_{out})$ for activation caching.

## 7. Implementation Notes
- Fully written in NumPy with vectorized tensor operations.
- Interfaced directly with `Observation` via `CNNPerceptionModule`.

## 8. Verification Strategy & Test Oracles
- Numerical gradient verification of Conv2D weights $W$ and biases $b$ via symmetric finite differences ($< 10^{-6}$ error).
- Multi-class synthetic patch classification test separating obstacles, food, water, hazards, and empty cells.

## 9. Known Limitations
- Pure NumPy convolution execution is optimized for small perceptual fields ($5 \times 5$ to $15 \times 15$ patches) rather than high-resolution camera feeds.

## 10. References
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.

