# Linear and Logistic Regression

## 1. Theory & Mathematical Definition
Linear Regression models continuous target variables $y \in \mathbb{R}$ as a linear combination of input features $\mathbf{x} \in \mathbb{R}^D$:
$$\hat{y} = \mathbf{x}^T \mathbf{w} + b$$
Logistic Regression extends this to binary classification $y \in \{0, 1\}$ by mapping the linear logit through the standard logistic (sigmoid) function:
$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{x}^T \mathbf{w} + b) = \frac{1}{1 + e^{-(\mathbf{x}^T \mathbf{w} + b)}}$$

## 2. Assumptions
- **Linearity:** The relationship between features and the target (or log-odds) is linear in parameters.
- **Homoscedasticity:** Residual variance is constant across feature space.
- **Absence of Multicollinearity:** Feature columns in $\mathbf{X}$ have full column rank ($L_2$ regularization ensures positive definiteness even when columns are collinear).

## 3. Equations & Derivation
### Ridge Linear Regression Normal Equation
Objective with $L_2$ penalty:
$$J(\mathbf{w}) = \frac{1}{2N} \|\mathbf{X} \mathbf{w} - \mathbf{y}\|_2^2 + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$
Setting $\nabla_\mathbf{w} J = 0$:
$$\mathbf{X}^T (\mathbf{X} \mathbf{w} - \mathbf{y}) + N \lambda \mathbf{w} = 0 \implies \mathbf{w}^* = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$
where the bias $b$ is unpenalized by setting $\mathbf{I}_{0,0} = 0$.

### Logistic Regression Binary Cross-Entropy
Loss function:
$$J(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln p_i + (1 - y_i) \ln(1 - p_i) \right] + \frac{\alpha}{2} \|\mathbf{w}\|_2^2$$
Gradients:
$$\nabla_\mathbf{w} J = \frac{1}{N} \mathbf{X}^T (\mathbf{p} - \mathbf{y}) + \alpha \mathbf{w}, \quad \nabla_b J = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)$$

## 4. Pseudocode
```text
Algorithm MiniBatchSGD(X, y, alpha, eta, max_iter, batch_size):
    w <- 0, b <- 0
    for epoch in 1..max_iter:
        shuffle(indices)
        for batch in batches(indices, batch_size):
            p <- sigmoid(X_b @ w + b)
            grad_w <- (1/m) * X_b^T (p - y_b) + alpha * w
            grad_b <- (1/m) * sum(p - y_b)
            w <- w - eta * grad_w
            b <- b - eta * grad_b
    return w, b
```

## 5. Computational Complexity
- **Normal Equation:** Time $O(N D^2 + D^3)$, Space $O(D^2)$.
- **Gradient Descent:** Time $O(\text{epochs} \cdot N D)$, Space $O(D)$.

## 6. Implementation Notes
- Sigmoid evaluated with numerical clipping to $[-250, 250]$ to prevent exponential overflow.
- Intercept column bias absorption in the design matrix leaves the bias row in the regularization matrix unpenalized.

## 7. Verification Strategy & Test Oracles
- Compare coefficients and $R^2$ directly against `sklearn.linear_model.Ridge` and `LogisticRegression`.
- Tolerances: $\|w_{\text{ours}} - w_{\text{sk}}\|_\infty < 10^{-4}$.

## 8. Known Limitations
- Sensitive to unscaled features when using gradient descent solvers (mitigated by using `StandardScaler`).
