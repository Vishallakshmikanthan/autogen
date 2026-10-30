# Clustering: K-Means and Gaussian Mixture Models (EM)

## 1. Theory & Mathematical Definition
Clustering identifies latent grouping structure in unlabeled observations $\mathbf{X} \in \mathbb{R}^{N \times D}$.
- **K-Means:** Hard assignment partitioning observations into $K$ Voronoi cells minimizing squared Euclidean distance to centroids (inertia).
- **Gaussian Mixture Models (GMM):** Soft probabilistic model representing density as a convex combination of $K$ multivariate Gaussian distributions:
  $$p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} \mid \mathbf{\mu}_k, \mathbf{\Sigma}_k)$$

## 2. K-Means++ Initialization & Lloyd's Invariant
1. Choose $\mathbf{\mu}_1 \sim \text{Uniform}(\mathbf{X})$.
2. For $k = 2, \dots, K$:
   $$\mathbf{\mu}_k \sim \frac{D(\mathbf{x})^2}{\sum_{\mathbf{x}'} D(\mathbf{x}')^2}, \quad D(\mathbf{x}) = \min_{j < k} \|\mathbf{x} - \mathbf{\mu}_j\|_2$$
3. **Inertia Monotonicity:** In Lloyd's algorithm, inertia
   $$J = \sum_{i=1}^N \|\mathbf{x}_i - \mathbf{\mu}_{c_i}\|_2^2$$
   is guaranteed to be non-increasing after each assignment and center update step:
   $$J^{(t+1)} \le J^{(t)}$$

## 3. Expectation-Maximization (EM) for GMMs
- **E-step:** Compute posterior responsibilities with log-sum-exp stabilization:
  $$\gamma_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}_i \mid \mathbf{\mu}_k, \mathbf{\Sigma}_k)}{\sum_j \pi_j \mathcal{N}(\mathbf{x}_i \mid \mathbf{\mu}_j, \mathbf{\Sigma}_j)}$$
- **M-step:**
  $$N_k = \sum_{i=1}^N \gamma_{ik}, \quad \pi_k = \frac{N_k}{N}, \quad \mathbf{\mu}_k = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} \mathbf{x}_i$$
  $$\mathbf{\Sigma}_k = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} (\mathbf{x}_i - \mathbf{\mu}_k)(\mathbf{x}_i - \mathbf{\mu}_k)^T + \delta \mathbf{I}$$
- **Log-Likelihood Monotonicity:** Every EM step guarantees $\ln p(\mathbf{X} \mid \mathbf{\theta}^{(t+1)}) \ge \ln p(\mathbf{X} \mid \mathbf{\theta}^{(t)})$.

## 4. Verification Strategy & Test Oracles
- Check monotonicity of $J^{(t)}$ and $\ln p(\mathbf{X} \mid \mathbf{\theta}^{(t)})$ across iterations.
- Benchmark clustering performance against `sklearn.cluster.KMeans` and `sklearn.mixture.GaussianMixture`.
