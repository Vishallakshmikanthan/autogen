# Support Vector Machines (Pegasos Algorithm)

## 1. Theory & Mathematical Definition
Support Vector Machines construct maximum-margin separating hyperplanes in feature space. For linearly non-separable distributions, soft margins penalize margin violations via the hinge loss $L(y, \hat{y}) = \max(0, 1 - y \hat{y})$.

## 2. Objective Formulation
$$\min_\mathbf{w, b} \frac{\lambda}{2} \|\mathbf{w}\|_2^2 + \frac{1}{N} \sum_{i=1}^N \max(0, 1 - y_i (\mathbf{w}^T \mathbf{x}_i + b))$$
where $y_i \in \{-1, +1\}$ and $\lambda > 0$ controls regularization.

## 3. Pegasos Algorithm (Primal Estimated Sub-Gradient Solver)
At iteration $t$:
1. Set learning rate $\eta_t = \frac{1}{\lambda t}$.
2. Sample mini-batch $A_t \subset \{1, \dots, N\}$.
3. Identify sub-batch violating margin: $A_t^+ = \{i \in A_t : y_i (\mathbf{w}_t^T \mathbf{x}_i + b) < 1\}$.
4. Gradient update:
   $$\mathbf{w}_{t+1/2} = (1 - \eta_t \lambda) \mathbf{w}_t + \frac{\eta_t}{|A_t|} \sum_{i \in A_t^+} y_i \mathbf{x}_i$$
   $$b_{t+1} = b_t + \frac{\eta_t}{|A_t|} \sum_{i \in A_t^+} y_i$$
5. Ball projection:
   $$\mathbf{w}_{t+1} = \min\left(1, \frac{1 / \sqrt{\lambda}}{\|\mathbf{w}_{t+1/2}\|}\right) \mathbf{w}_{t+1/2}$$

## 4. Kernel Extensions
For non-linear boundaries, inner products $\mathbf{x}_i^T \mathbf{x}_j$ are replaced by Mercer kernels:
- RBF: $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|_2^2)$
- Polynomial: $K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T \mathbf{x}_j + 1)^d$

## 5. Verification Strategy & Test Oracles
- Compare classification accuracy and decision boundaries with `sklearn.svm.SVC` and `LinearSVC`.
