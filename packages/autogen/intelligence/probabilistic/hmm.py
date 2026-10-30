"""Hidden Markov Model algorithms: Forward, Backward, Viterbi, and Smoothing (SYS-09, Module M10)."""

from __future__ import annotations

from collections.abc import Sequence
import math
import numpy as np

from autogen.intelligence.probabilistic.contracts import HMMResult


class HiddenMarkovModel:
    """Discrete Hidden Markov Model specified by transition matrix, emission matrix, and initial prior."""

    def __init__(
        self,
        transition_matrix: np.ndarray,
        emission_matrix: np.ndarray,
        initial_prior: np.ndarray,
    ) -> None:
        self.transition_matrix = np.asarray(transition_matrix, dtype=np.float64)
        self.emission_matrix = np.asarray(emission_matrix, dtype=np.float64)
        self.initial_prior = np.asarray(initial_prior, dtype=np.float64)

        self.num_states = self.transition_matrix.shape[0]
        self.num_emissions = self.emission_matrix.shape[1]

        # Invariant checks
        if self.transition_matrix.shape != (self.num_states, self.num_states):
            raise ValueError(f"Transition matrix must be square ({self.num_states}, {self.num_states})")
        if self.emission_matrix.shape[0] != self.num_states:
            raise ValueError(f"Emission matrix rows must match num_states ({self.num_states})")
        if len(self.initial_prior) != self.num_states:
            raise ValueError(f"Initial prior length must match num_states ({self.num_states})")

    def forward(self, observations: Sequence[int]) -> tuple[np.ndarray, float]:
        """Forward algorithm computing filtering probabilities alpha_t(i) = P(o_{1:t}, s_t = i).

        Returns:
            (alpha_table, log_likelihood)
        """
        T = len(observations)
        if T == 0:
            return np.zeros((0, self.num_states)), 0.0

        alpha = np.zeros((T, self.num_states), dtype=np.float64)
        # Initialization t = 0
        alpha[0] = self.initial_prior * self.emission_matrix[:, observations[0]]

        # Induction t = 1 to T-1
        for t in range(1, T):
            obs = observations[t]
            # alpha[t, j] = sum_i (alpha[t-1, i] * T[i, j]) * E[j, obs]
            alpha[t] = (alpha[t - 1] @ self.transition_matrix) * self.emission_matrix[:, obs]

        total_p = float(np.sum(alpha[-1]))
        log_lik = math.log(total_p) if total_p > 0.0 else -math.inf
        return alpha, log_lik

    def backward(self, observations: Sequence[int]) -> np.ndarray:
        """Backward algorithm computing likelihood recursion beta_t(i) = P(o_{t+1:T} | s_t = i)."""
        T = len(observations)
        if T == 0:
            return np.zeros((0, self.num_states))

        beta = np.zeros((T, self.num_states), dtype=np.float64)
        # Initialization t = T-1
        beta[T - 1] = 1.0

        # Induction t = T-2 down to 0
        for t in range(T - 2, -1, -1):
            next_obs = observations[t + 1]
            # beta[t, i] = sum_j (T[i, j] * E[j, next_obs] * beta[t+1, j])
            beta[t] = self.transition_matrix @ (self.emission_matrix[:, next_obs] * beta[t + 1])

        return beta

    def viterbi(self, observations: Sequence[int]) -> tuple[tuple[int, ...], float]:
        """Viterbi algorithm finding the most likely hidden sequence via dynamic programming in log space."""
        T = len(observations)
        if T == 0:
            return (), -math.inf

        log_T = np.log(np.maximum(self.transition_matrix, 1e-300))
        log_E = np.log(np.maximum(self.emission_matrix, 1e-300))
        log_pi = np.log(np.maximum(self.initial_prior, 1e-300))

        delta = np.zeros((T, self.num_states), dtype=np.float64)
        psi = np.zeros((T, self.num_states), dtype=np.int32)

        # Initialization
        delta[0] = log_pi + log_E[:, observations[0]]

        # Recursion
        for t in range(1, T):
            obs = observations[t]
            for j in range(self.num_states):
                # max_i (delta[t-1, i] + log_T[i, j])
                trans_scores = delta[t - 1] + log_T[:, j]
                best_i = int(np.argmax(trans_scores))
                delta[t, j] = trans_scores[best_i] + log_E[j, obs]
                psi[t, j] = best_i

        # Termination & Backtracking
        best_last_state = int(np.argmax(delta[T - 1]))
        max_log_p = float(delta[T - 1, best_last_state])

        path = [best_last_state]
        for t in range(T - 1, 0, -1):
            prev_state = psi[t, path[-1]]
            path.append(prev_state)

        return tuple(reversed(path)), max_log_p

    def forward_backward_smoothing(self, observations: Sequence[int]) -> HMMResult:
        """Full forward-backward smoothing computing posterior marginals gamma_t(i) = P(s_t = i | o_{1:T})."""
        alpha, log_lik = self.forward(observations)
        beta = self.backward(observations)
        viterbi_path, _ = self.viterbi(observations)

        # gamma_t(i) = (alpha_t(i) * beta_t(i)) / P(o_{1:T})
        numerator = alpha * beta
        sums = np.sum(numerator, axis=1, keepdims=True)
        # Avoid division by zero
        sums[sums == 0.0] = 1.0
        smoothed = numerator / sums

        return HMMResult(
            forward_probs=alpha,
            backward_probs=beta,
            smoothed_posteriors=smoothed,
            most_likely_path=viterbi_path,
            log_likelihood=log_lik,
        )

