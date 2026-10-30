# Hidden Markov Models: Filtering, Smoothing & Viterbi

## 1. Theory & Mathematical Definition

A **Hidden Markov Model (HMM)** is a temporal probabilistic model $\langle \mathcal{S}, \mathcal{O}, T, E, \pi_0 \rangle$:
- Hidden states: $s_t \in \mathcal{S} = \{1, \dots, N\}$.
- Observations: $o_t \in \mathcal{O} = \{1, \dots, M\}$.
- Transition matrix: $T_{ij} = P(s_t = j \mid s_{t-1} = i)$.
- Emission matrix: $E_{ik} = P(o_t = k \mid s_t = i)$.
- Initial prior: $\pi_0(i) = P(s_0 = i)$.

---

## 2. Dynamic Programming Algorithms

### A. Forward Filtering ($\alpha_t$)
Computes the filtered joint probability $\alpha_t(i) = P(o_{1:t}, s_t = i)$:
- Initialization: $\alpha_0(i) = \pi_0(i) \cdot E_{i, o_0}$.
- Induction: $\alpha_t(j) = \left( \sum_{i=1}^N \alpha_{t-1}(i) \cdot T_{ij} \right) \cdot E_{j, o_t}$.

### B. Backward Message Passing ($\beta_t$)
Computes the conditional evidence probability $\beta_t(i) = P(o_{t+1:T} \mid s_t = i)$:
- Initialization: $\beta_{T-1}(i) = 1.0 \quad \forall i$.
- Induction: $\beta_t(i) = \sum_{j=1}^N T_{ij} \cdot E_{j, o_{t+1}} \cdot \beta_{t+1}(j)$.

### C. Forward-Backward Smoothing ($\gamma_t$)
Combines forward and backward messages to evaluate smoothed state posteriors:
$$
\gamma_t(i) = P(s_t = i \mid o_{1:T}) = \frac{\alpha_t(i) \cdot \beta_t(i)}{\sum_{j=1}^N \alpha_t(j) \cdot \beta_t(j)}
$$

### D. Viterbi Algorithm (Most Likely Sequence)
Finds the global optimal state path $\arg\max_{s_{1:T}} P(s_{1:T}, o_{1:T})$ in log-space:
- $\delta_t(j) = \max_i (\delta_{t-1}(i) + \log T_{ij}) + \log E_{j, o_t}$.
- $\psi_t(j) = \arg\max_i (\delta_{t-1}(i) + \log T_{ij})$.

---

## 3. Verification Strategy & Test Oracles

- **Brute-Force Sequence Oracle**: Forward filtering sum $\sum_i \alpha_T(i)$ is compared against the exhaustive enumeration of all $|S|^T$ state trajectories.
- **Brute-Force Viterbi Oracle**: The Viterbi sequence matches the sequence maximizing joint probability across all possible paths.

---

## 4. References

- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257-286.

