# Optimizers & Regularization (SYS-13, Module M14)

## 1. Theory & Mathematical Definition
First-order parameter optimization methods update network weights along negative gradient directions. Adaptive gradient methods (RMSProp, Adam) scale per-parameter updates by running exponential estimates of first and second gradient moments. Regularization techniques (inverted dropout, weight decay, gradient norm clipping) constrain capacity to prevent catastrophic overfitting and gradient explosions.

## 2. Assumptions
- Loss surfaces are locally smooth with Lipschitz-continuous gradients.
- Mini-batch stochastic gradients provide unbiased estimates of the true expected gradient.

## 3. Equations & Variable Definitions
For parameters $\theta$, gradient $g_t = \nabla_\theta L(\theta_t) + \lambda \theta_t$:
- **Adam (Adaptive Moment Estimation)**:
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
  $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
  $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
  $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
- **Inverted Dropout**:
  During training with drop rate $p \in [0, 1)$:
  $$\tilde{a} = \frac{a \odot m}{1 - p}, \quad m_i \sim \text{Bernoulli}(1 - p)$$
  During evaluation: $\tilde{a} = a$.
- **Global Gradient Norm Clipping**:
  $$\tilde{g} = g \cdot \min\left(1, \frac{c_{\max}}{\|g\|_2}\right)$$

## 4. Derivation of Adam Bias Corrections
Taking expectations under the assumption that $g_i$ come from a stationary distribution with mean $\mathbb{E}[g]$:
$$\mathbb{E}[m_t] = \mathbb{E}\left[\sum_{i=1}^t (1 - \beta_1)\beta_1^{t-i} g_i\right] = \mathbb{E}[g] (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} = \mathbb{E}[g] (1 - \beta_1^t)$$
Dividing by $(1 - \beta_1^t)$ corrects the initial zero-bias of the running average.

## 5. Pseudocode
```text
function ADAM_STEP(theta, g, m, v, t, lr, beta1, beta2, eps, weight_decay):
    g = g + weight_decay * theta
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * (g * g)
    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)
    theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
    return theta, m, v
```

## 6. Computational Complexity
- **Time Complexity**: $\mathcal{O}(P)$ element-wise operations for $P$ parameters.
- **Space Complexity**: $\mathcal{O}(P)$ auxiliary state for momentum and squared-moment vectors.

## 7. Implementation Notes
All optimizers support $L_2$ weight decay and parameter gradient clipping. Schedulers (`StepLR`, `ExponentialLR`, `CosineAnnealingLR`) adjust learning rates monotonically or cyclically.

## 8. Verification Strategy & Test Oracles
- Unit tests validating moment accumulation, bias correction factors, and gradient zeroing across all optimizers.
- Inverted dropout mean activation preservation test: $\mathbb{E}[\tilde{a}] = \mathbb{E}[a]$.

## 9. Known Limitations
- Adam learning rates may exhibit transient oscillations in non-stationary non-convex reinforcement learning regimes if $\beta_2$ is set too close to 1.0.

## 10. References
- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. *JMLR*, 15(1), 1929-1958.

