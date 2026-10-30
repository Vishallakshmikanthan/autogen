"""Exact MDP Solvers: Linear Solve, Policy Evaluation, Value Iteration, and Policy Iteration (SYS-10, Module M11)."""

from __future__ import annotations

import time
import numpy as np

from autogen.intelligence.decision.contracts import MDPSolution, TabularMDP


def linear_policy_evaluation(
    mdp: TabularMDP,
    policy: np.ndarray,
) -> np.ndarray:
    """Exact analytical policy evaluation solving the linear system:

    V_pi = (I - gamma * P_pi)^(-1) * R_pi
    """
    num_s = mdp.num_states
    P_pi = np.zeros((num_s, num_s), dtype=np.float64)
    R_pi = np.zeros(num_s, dtype=np.float64)

    for s in range(num_s):
        if s in mdp.terminal_states:
            # Terminal states have value 0 and transition to themselves with 0 reward
            P_pi[s, s] = 0.0
            R_pi[s] = 0.0
            continue

        a = int(policy[s])
        P_pi[s, :] = mdp.transition_tensor[s, a, :]
        R_pi[s] = np.sum(mdp.transition_tensor[s, a, :] * mdp.reward_tensor[s, a, :])

    # (I - gamma * P_pi) * V_pi = R_pi
    A = np.eye(num_s, dtype=np.float64) - mdp.gamma * P_pi
    # For terminal states, row is row of identity, A[s, s] = 1, R_pi[s] = 0 -> V[s] = 0
    v_pi = np.linalg.solve(A, R_pi)
    return v_pi


def iterative_policy_evaluation(
    mdp: TabularMDP,
    policy: np.ndarray,
    theta: float = 1e-8,
    max_iterations: int = 10000,
) -> np.ndarray:
    """Iterative Bellman expectation backups until max-norm residual < theta."""
    num_s = mdp.num_states
    v = np.zeros(num_s, dtype=np.float64)

    for _ in range(max_iterations):
        delta = 0.0
        v_new = np.zeros(num_s, dtype=np.float64)

        for s in range(num_s):
            if s in mdp.terminal_states:
                v_new[s] = 0.0
                continue

            a = int(policy[s])
            trans_probs = mdp.transition_tensor[s, a, :]
            rewards = mdp.reward_tensor[s, a, :]
            expected_val = np.sum(trans_probs * (rewards + mdp.gamma * v))
            v_new[s] = expected_val
            delta = max(delta, abs(float(v_new[s] - v[s])))

        v = v_new
        if delta < theta:
            break

    return v


def value_iteration(
    mdp: TabularMDP,
    theta: float = 1e-8,
    max_iterations: int = 10000,
) -> MDPSolution:
    """Computes exact V*, Q*, and optimal policy pi* via Value Iteration with contraction threshold."""
    t0 = time.perf_counter()
    num_s = mdp.num_states
    num_a = mdp.num_actions

    v = np.zeros(num_s, dtype=np.float64)
    # Contraction stopping threshold: delta < theta * (1 - gamma) / (2 * gamma)
    if 0.0 < mdp.gamma < 1.0:
        stopping_threshold = theta * (1.0 - mdp.gamma) / (2.0 * mdp.gamma)
    else:
        stopping_threshold = theta

    iterations = 0
    delta = 0.0

    for it in range(1, max_iterations + 1):
        iterations = it
        delta = 0.0
        v_new = np.zeros(num_s, dtype=np.float64)

        for s in range(num_s):
            if s in mdp.terminal_states:
                v_new[s] = 0.0
                continue

            q_sa = np.zeros(num_a, dtype=np.float64)
            for a in range(num_a):
                trans_probs = mdp.transition_tensor[s, a, :]
                rewards = mdp.reward_tensor[s, a, :]
                q_sa[a] = np.sum(trans_probs * (rewards + mdp.gamma * v))

            best_v = float(np.max(q_sa))
            v_new[s] = best_v
            delta = max(delta, abs(best_v - v[s]))

        v = v_new
        if delta < stopping_threshold:
            break

    # Extract final Q* and policy
    q_star = np.zeros((num_s, num_a), dtype=np.float64)
    policy = np.zeros(num_s, dtype=np.int32)

    for s in range(num_s):
        if s in mdp.terminal_states:
            q_star[s, :] = 0.0
            policy[s] = 0
            continue

        for a in range(num_a):
            trans_probs = mdp.transition_tensor[s, a, :]
            rewards = mdp.reward_tensor[s, a, :]
            q_star[s, a] = np.sum(trans_probs * (rewards + mdp.gamma * v))

        policy[s] = int(np.argmax(q_star[s, :]))

    t1 = time.perf_counter()
    return MDPSolution(
        v_star=v,
        q_star=q_star,
        policy=policy,
        iterations=iterations,
        max_residual=delta,
        converged=delta < stopping_threshold,
        runtime_sec=t1 - t0,
        state_labels=mdp.states,
        action_labels=mdp.actions,
    )


def policy_iteration(
    mdp: TabularMDP,
    max_iterations: int = 1000,
) -> MDPSolution:
    """Computes exact V*, Q*, and optimal policy pi* via Howard's Policy Iteration."""
    t0 = time.perf_counter()
    num_s = mdp.num_states
    num_a = mdp.num_actions

    # Start with initial policy (e.g. action 0 everywhere)
    policy = np.zeros(num_s, dtype=np.int32)
    v = np.zeros(num_s, dtype=np.float64)
    iterations = 0

    for it in range(1, max_iterations + 1):
        iterations = it

        # 1. Policy Evaluation via exact linear solve
        v = linear_policy_evaluation(mdp, policy)

        # 2. Policy Improvement
        policy_stable = True
        for s in range(num_s):
            if s in mdp.terminal_states:
                continue

            old_action = policy[s]
            q_sa = np.zeros(num_a, dtype=np.float64)
            for a in range(num_a):
                trans_probs = mdp.transition_tensor[s, a, :]
                rewards = mdp.reward_tensor[s, a, :]
                q_sa[a] = np.sum(trans_probs * (rewards + mdp.gamma * v))

            best_action = int(np.argmax(q_sa))
            if best_action != old_action:
                policy[s] = best_action
                policy_stable = False

        if policy_stable:
            break

    # Final Q-value computation
    q_star = np.zeros((num_s, num_a), dtype=np.float64)
    for s in range(num_s):
        if s in mdp.terminal_states:
            continue
        for a in range(num_a):
            q_star[s, a] = np.sum(mdp.transition_tensor[s, a, :] * (mdp.reward_tensor[s, a, :] + mdp.gamma * v))

    t1 = time.perf_counter()
    return MDPSolution(
        v_star=v,
        q_star=q_star,
        policy=policy,
        iterations=iterations,
        max_residual=0.0,
        converged=True,
        runtime_sec=t1 - t0,
        state_labels=mdp.states,
        action_labels=mdp.actions,
    )

