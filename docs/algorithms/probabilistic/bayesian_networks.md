# Bayesian Networks: Representation & Inference

## 1. Theory & Mathematical Definition

A **Bayesian Network** is a Directed Acyclic Graph (DAG) $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ paired with Conditional Probability Tables (CPTs) at each node.
The joint probability distribution factorizes according to the chain rule of Bayesian networks:
$$
P(X_1, X_2, \dots, X_n) = \prod_{i=1}^n P(X_i \mid \text{Parents}(X_i))
$$

### Conditional Independence & d-Separation
Two disjoint sets of variables $X$ and $Y$ are conditionally independent given evidence $Z$ ($X \perp\!\!\!\perp Y \mid Z$) if every undirected path between any node in $X$ and any node in $Y$ is **d-separated** by $Z$.

---

## 2. Exact Inference Algorithms

### A. Enumeration
Computes the query distribution by summing the full joint distribution:
$$
P(X \mid E = e) = \alpha \sum_{Y} P(X, E = e, Y)
$$
where $Y = \mathcal{V} \setminus (\{X\} \cup E)$ is the set of hidden variables.

### B. Variable Elimination
Eliminates hidden variables one by one by interchanging sum and product using factor operations:
1. **Factor Multiplication**:
   $$\psi_{12}(X_1, X_2, X_3) = \psi_1(X_1, X_2) \otimes \psi_2(X_2, X_3)$$
2. **Factor Marginalization**:
   $$\sum_{X_2} \psi(X_1, X_2) = \psi'(X_1)$$

---

## 3. Approximate Inference Algorithms

- **Rejection Sampling**: Draws samples from the network prior; discards samples that do not match evidence $E$.
- **Likelihood Weighting**: Avoids rejection by fixing evidence variables and weighting each sample by the joint likelihood of the evidence:
  $$w = \prod_{Z_i \in E} P(Z_i = e_i \mid \text{Parents}(Z_i))$$
- **Gibbs Sampling (MCMC)**: Generates dependent Markov chain transitions by repeatedly sampling each non-evidence variable conditioned on its Markov blanket $\text{MB}(X) = \text{Parents}(X) \cup \text{Children}(X) \cup \text{Parents}(\text{Children}(X))$.

---

## 4. Verification Strategy & Test Oracles

- **Variable Elimination == Enumeration**: Formally tested on Alarm and Burglary-Earthquake networks.
- **Sampling Convergence**: Verified that Monte Carlo empirical frequencies converge to exact posteriors within $O(1/\sqrt{N})$ tolerance.

---

## 5. References

- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference*. Morgan Kaufmann.
- Koller, D., & Friedman, N. (2009). *Probabilistic Graphical Models: Principles and Techniques*. MIT Press.
