# Multilayer Perceptrons & Backpropagation (SYS-13, Module M14)

## 1. Theory & Mathematical Definition
A Multilayer Perceptron (MLP) is a feedforward artificial neural network mapping input feature vectors $x \in \mathbb{R}^{d_{in}}$ to output representations $\hat{y} \in \mathbb{R}^{d_{out}}$ across $L$ stacked layers. Learning proceeds via reverse-mode automatic differentiation (backpropagation) which applies the multivariate chain rule to compute exact analytical gradients with respect to all layer parameters.

## 2. Assumptions
1. Continuous differentiability (or sub-differentiability at isolated inflection points like $z=0$ for ReLU) of all component activation functions.
2. Independent and identically distributed (i.i.d.) mini-batches during supervised training.

## 3. Equations & Variable Definitions
For layer $l \in \{1, \dots, L\}$:
- Pre-activation:
  $$z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}$$
  where $a^{(0)} = x$, $W^{(l)} \in \mathbb{R}^{d_{l-1} \times d_l}$, $b^{(l)} \in \mathbb{R}^{1 \times d_l}$.
- Activation:
  $$a^{(l)} = \sigma^{(l)}(z^{(l)})$$
- Backpropagation Error Delta:
  $$\delta^{(l)} = \nabla_{a^{(l)}} L \odot {\sigma^{(l)}}'(z^{(l)})$$
- Parameter Gradients:
  $$\frac{\partial L}{\partial W^{(l)}} = (a^{(l-1)})^T \delta^{(l)}$$
  $$\frac{\partial L}{\partial b^{(l)}} = \sum_{i=1}^N \delta^{(l)}_{i, :}$$
- Input Gradient Cascade:
  $$\nabla_{a^{(l-1)}} L = \delta^{(l)} (W^{(l)})^T$$

## 4. Derivation of Backward Delta
By the multivariate chain rule:
$$\frac{\partial L}{\partial z_j^{(l)}} = \sum_k \frac{\partial L}{\partial a_k^{(l)}} \frac{\partial a_k^{(l)}}{\partial z_j^{(l)}}$$
For element-wise activations $\frac{\partial a_k^{(l)}}{\partial z_j^{(l)}} = \sigma'(z_j^{(l)}) \delta_{jk}$, yielding $\delta^{(l)} = \nabla_{a^{(l)}} L \odot \sigma'(z^{(l)})$. For softmax, the full vector Jacobian $\frac{\partial a_k}{\partial z_j} = a_k(\delta_{kj} - a_j)$ is computed via $a \odot (\nabla_a L - \sum (a \odot \nabla_a L))$.

## 5. Pseudocode
```text
function FORWARD(x, layers):
    a = x
    for layer in layers:
        a = layer.forward(a)
    return a

function BACKWARD(grad_loss, layers):
    grad = grad_loss
    for layer in reversed(layers):
        grad = layer.backward(grad)
    return grad
```

## 6. Computational Complexity
- **Time Complexity**: $\mathcal{O}(N \sum_{l=1}^L d_{l-1} d_l)$ for forward and backward passes across batch size $N$.
- **Space Complexity**: $\mathcal{O}(N \sum_{l=1}^L d_l)$ activation caching memory during forward pass.

## 7. Implementation Notes
Implemented in pure NumPy without external autograd engines. Float64 arithmetic is utilized for precision during numerical gradient checks.

## 8. Verification Strategy & Test Oracles
- Finite-difference symmetric numerical gradient check in `float64`:
  $$\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta_i + \epsilon) - L(\theta_i - \epsilon)}{2\epsilon}$$
  Relative error bounded below $10^{-6}$.
- Canonical benchmark: 2-bit non-linear XOR boundary fitted to 100% accuracy.
- Real dataset oracle: scikit-learn `load_digits` fitted above 90% accuracy (achieved 98.0%).

## 9. Known Limitations
- Pure NumPy CPU tensor loops have higher overhead on large vision batches compared to GPU BLAS kernels.
- Gradient vanishing may occur in deep architectures ($L > 6$) when using Sigmoid/Tanh activations without residual skips.

## 10. References
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.

