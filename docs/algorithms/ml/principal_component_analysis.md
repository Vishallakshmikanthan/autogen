# Principal Component Analysis (PCA)

## 1. Theory & Mathematical Definition
Principal Component Analysis finds an orthogonal linear transformation $\mathbf{V} \in \mathbb{R}^{D \times k}$ projecting centered data $\mathbf{X}_c \in \mathbb{R}^{N \times D}$ into a lower-dimensional subspace $\mathbf{Z} = \mathbf{X}_c \mathbf{V}_k$ such that the variance of the projected data is maximized, or equivalently, the reconstruction error is minimized.

## 2. Derivation via SVD
Let $\mathbf{X}_c = \mathbf{X} - \mathbf{\mu}$ be the zero-mean data matrix.
Singular Value Decomposition:
$$\mathbf{X}_c = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$
where:
- $\mathbf{U} \in \mathbb{R}^{N \times D}$ has orthonormal columns ($\mathbf{U}^T \mathbf{U} = \mathbf{I}$).
- $\mathbf{\Sigma} = \text{diag}(s_1, \dots, s_D)$ holds singular values sorted descending ($s_1 \ge s_2 \ge \dots \ge 0$).
- $\mathbf{V} \in \mathbb{R}^{D \times D}$ is the orthogonal matrix of principal axes.

Sample covariance matrix:
$$\mathbf{C} = \frac{1}{N - 1} \mathbf{X}_c^T \mathbf{X}_c = \frac{1}{N - 1} \mathbf{V} \mathbf{\Sigma}^2 \mathbf{V}^T$$
Eigenvalues of $\mathbf{C}$ are:
$$\lambda_j = \frac{s_j^2}{N - 1}$$
Explained variance ratio for component $j$:
$$\text{EVR}_j = \frac{\lambda_j}{\sum_{i=1}^D \lambda_i}$$

## 3. Invariants
1. **Orthonormality:** $\mathbf{V}_k^T \mathbf{V}_k = \mathbf{I}_k$.
2. **Reconstruction Residual:** The residual error
   $$E(k) = \frac{1}{N} \|\mathbf{X}_c - \mathbf{X}_c \mathbf{V}_k \mathbf{V}_k^T\|_F^2 = \sum_{j=k+1}^D \lambda_j$$
   is strictly non-increasing with the number of components $k$.

## 4. Verification Strategy & Test Oracles
- Compare singular values, components, and explained variance ratios against `sklearn.decomposition.PCA`.
- Tolerance: $\|s_{\text{ours}} - s_{\text{sk}}\|_\infty < 10^{-6}$.
