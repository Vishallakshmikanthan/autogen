# Discrete Probability Distributions & Bayesian Updating

## 1. Theory & Mathematical Definition

A discrete random variable $X$ maps sample space outcomes to a countable set $\mathcal{X}$.
The probability mass function (PMF) satisfies Kolmogorov's probability axioms:
1. Non-negativity: $P(X = x) \ge 0 \quad \forall x \in \mathcal{X}$
2. Normalization: $\sum_{x \in \mathcal{X}} P(X = x) = 1.0$
3. Countable additivity: for disjoint events $A, B \subseteq \mathcal{X}$, $P(A \cup B) = P(A) + P(B)$.

### Bayes' Theorem
Given a prior hypothesis distribution $P(H)$ and an observation likelihood model $P(E \mid H)$, the posterior probability distribution $P(H \mid E)$ is given by:
$$
P(H = h \mid E = e) = \frac{P(E = e \mid H = h) P(H = h)}{\sum_{h' \in \mathcal{H}} P(E = e \mid H = h') P(H = h')}
$$

In binary hypothesis testing ($H \in \{0, 1\}$), Bayes' theorem can be computed additively in **log-odds** space to prevent numerical floating-point underflow:
$$
\mathcal{O}(H) = \frac{P(H)}{1 - P(H)} \implies \text{logit}(P(H \mid E)) = \text{logit}(P(H)) + \log \frac{P(E \mid H)}{P(E \mid \neg H)}
$$

---

## 2. Core Distributions Implemented from Scratch

- **Bernoulli**: $P(X = 1) = p, \quad P(X = 0) = 1 - p$.
- **Binomial**: $P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k} \quad (k \in \{0, \dots, n\})$.
- **Categorical**: Discrete choice over $K$ arbitrary outcomes: $P(X = c_k) = \frac{w_k}{\sum_j w_j}$.
- **Poisson**: $P(X = k) = \frac{\mu^k e^{-\mu}}{k!}$.
- **Discretized Gaussian**: $P(X = x) = \eta \cdot \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$ normalized over a discrete integer lattice.

---

## 3. Information Theoretic Measures

- **Shannon Entropy**:
  $$H(X) = -\sum_{x \in \mathcal{X}} P(x) \log_2 P(x) \quad (\text{with } 0 \log_2 0 \equiv 0)$$
- **Kullback-Leibler (KL) Divergence**:
  $$D_{\text{KL}}(P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \log_2 \frac{P(x)}{Q(x)}$$
- **Numerically Stable LogSumExp**:
  $$\text{LogSumExp}(x_1, \dots, x_n) = m + \log\left(\sum_{i=1}^n \exp(x_i - m)\right) \quad \text{where } m = \max_i x_i$$

---

## 4. Verification Strategy & Test Oracles

- PMF values for Bernoulli, Binomial, Poisson, and Gaussian are verified against `scipy.stats` oracles across test points with absolute tolerance $\epsilon < 10^{-7}$.
- Normalization ($\sum P(x) = 1.0$) and non-negativity verified via property-based invariant testing.

---

## 5. References

- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley.
- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.), Chapter 12.
