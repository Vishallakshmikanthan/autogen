"""Module M10 Probabilistic Engine CLI Usage Example.

Demonstrates:
1. Discrete distributions (Bernoulli, Binomial, Poisson, Gaussian) with entropy and normalization.
2. Bayesian updating in linear probability and log-odds space.
3. Bayesian Network exact inference (Variable Elimination vs Enumeration) on Alarm network.
4. Hidden Markov Model (Forward, Backward, Viterbi, Smoothing) on Weather-Umbrella.
5. GridBeliefFilter tracking a moving hazard on a 2D spatial grid.
6. BayesianHazardAgent operating in SimulationEngine with authentic DecisionTraces.
"""

from __future__ import annotations

import numpy as np

from autogen.agents.probabilistic.bayesian_hazard_agent import BayesianHazardAgent
from autogen.intelligence.probabilistic.bayes import BayesUpdater
from autogen.intelligence.probabilistic.bayesian_network import (
    BayesNet,
    BayesNetNode,
    enumeration_ask,
    variable_elimination,
)
from autogen.intelligence.probabilistic.distributions import (
    Bernoulli,
    Binomial,
    DiscretizedGaussian,
    Poisson,
)
from autogen.intelligence.probabilistic.grid_belief_filter import GridBeliefFilter
from autogen.intelligence.probabilistic.hmm import HiddenMarkovModel
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.perception.contracts import PerceptionConfig
from autogen.simulation.state.types import Position


def main() -> None:
    print("=" * 75)
    print("  AUTOGEN MODULE M10: PROBABILITY, BAYESIAN REASONING & BELIEF STATES")
    print("=" * 75)

    # 1. Discrete Distributions
    print("\n--- 1. Discrete Distributions from Scratch ---")
    b = Binomial(n=10, p=0.3)
    pois = Poisson(mu=3.0, max_k=10)
    gauss = DiscretizedGaussian(mean=5.0, std=1.2, min_x=0, max_x=10)
    print(f"Binomial(n=10, p=0.3) Entropy: {b.entropy():.3f} bits, P(X=3) = {b.prob(3):.4f}")
    print(f"Poisson(mu=3.0) Entropy: {pois.entropy():.3f} bits, P(X=3) = {pois.prob(3):.4f}")
    print(f"DiscretizedGaussian(mean=5.0) Entropy: {gauss.entropy():.3f} bits, P(X=5) = {gauss.prob(5):.4f}")

    # 2. Bayesian Updating
    print("\n--- 2. Bayesian Hypothesis Updating ---")
    prior = 0.02
    p_pos_given_disease = 0.98
    p_pos_given_healthy = 0.04
    post = BayesUpdater.update_binary(prior, p_pos_given_disease, p_pos_given_healthy)
    print(f"Medical Test: Prior={prior:.2%}, Likelihood(+|D)={p_pos_given_disease:.2%}, FalsePositive={p_pos_given_healthy:.2%}")
    print(f"Updated Posterior P(Disease | +): {post:.2%}")

    # 3. Bayesian Network: Alarm Network
    print("\n--- 3. Bayesian Network Inference (Alarm Network) ---")
    bn = BayesNet()
    bn.add_node(BayesNetNode("B", (), np.array([0.999, 0.001])))
    bn.add_node(BayesNetNode("E", (), np.array([0.998, 0.002])))

    cpt_a = np.zeros((2, 2, 2))
    cpt_a[0, 0] = [0.999, 0.001]
    cpt_a[0, 1] = [0.71, 0.29]
    cpt_a[1, 0] = [0.06, 0.94]
    cpt_a[1, 1] = [0.05, 0.95]
    bn.add_node(BayesNetNode("A", ("B", "E"), cpt_a))

    cpt_j = np.zeros((2, 2))
    cpt_j[0] = [0.95, 0.05]
    cpt_j[1] = [0.10, 0.90]
    bn.add_node(BayesNetNode("J", ("A",), cpt_j))

    cpt_m = np.zeros((2, 2))
    cpt_m[0] = [0.99, 0.01]
    cpt_m[1] = [0.30, 0.70]
    bn.add_node(BayesNetNode("M", ("A",), cpt_m))

    evidence = {"J": 1, "M": 1}
    p_ve = variable_elimination("B", evidence, bn)
    p_enum = enumeration_ask("B", evidence, bn)
    print(f"Evidence: JohnCalls=True, MaryCalls=True")
    print(f"Variable Elimination P(Burglary=True | J, M) = {p_ve.prob(1):.4f}")
    print(f"Enumeration Ask      P(Burglary=True | J, M) = {p_enum.prob(1):.4f}")

    # 4. Hidden Markov Model: Weather & Umbrella
    print("\n--- 4. Hidden Markov Model (Weather & Umbrella) ---")
    trans = np.array([[0.7, 0.3], [0.3, 0.7]])
    emiss = np.array([[0.1, 0.9], [0.8, 0.2]])  # 0=NoUmbrella, 1=Umbrella
    hmm = HiddenMarkovModel(transition_matrix=trans, emission_matrix=emiss, initial_prior=np.array([0.5, 0.5]))

    obs_sequence = [1, 1, 0, 1]  # Umbrella, Umbrella, No Umbrella, Umbrella
    viterbi_seq, log_p = hmm.viterbi(obs_sequence)
    smoothing_res = hmm.forward_backward_smoothing(obs_sequence)
    weather_names = ["Rainy", "Sunny"]
    print(f"Observation Sequence: [Umbrella, Umbrella, No Umbrella, Umbrella]")
    print(f"Viterbi Most Likely Hidden States: {[weather_names[s] for s in viterbi_seq]}")
    print(f"Smoothed P(Rainy) at each step: {[f'{p[0]:.2f}' for p in smoothing_res.smoothed_posteriors]}")

    # 5. Grid Belief Filter
    print("\n--- 5. Grid Belief Filter Tracking ---")
    gbf = GridBeliefFilter(width=8, height=8, motion_noise=0.1)
    print(f"Initial Uniform Entropy: {gbf.entropy():.2f} bits")
    # Repeated detections at position (4, 4)
    for t in range(1, 4):
        gbf.update((Position(4, 4), 2, True))
        print(f"  After sensor update {t}: Entropy={gbf.entropy():.2f} bits, Top cell={gbf.most_likely_states(k=1)[0]}")

    # 6. BayesianHazardAgent in SimulationEngine
    print("\n--- 6. BayesianHazardAgent in SimulationEngine ---")
    config = EnvironmentConfig(width=10, height=10, seed=42)
    world = GridWorld(config)
    world.reset()

    agent = BayesianHazardAgent(agent_id="bayes_agent_1", grid_width=10, grid_height=10)
    world.spawn_agent("bayes_agent_1")

    engine = SimulationEngine(environment=world, seed=42)
    engine.register_agent(agent, PerceptionConfig(radius=4))

    print("Stepping simulation with BayesianHazardAgent...")
    for step_num in range(1, 6):
        world_state = engine.step()
        ag_state = world_state.agents.get("bayes_agent_1")
        pos = ag_state.position if ag_state else "Unknown"
        trace = agent.latest_trace
        act_name = trace.rationale.get("selected_action", "None") if trace else "None"
        reason = trace.rationale.get("reason", "None") if trace else "None"
        ent = trace.metrics.get("belief_entropy_bits", 0.0) if trace else 0.0
        print(f"  Step {step_num}: Pos={pos}, Action={act_name}, BeliefEntropy={ent:.2f} bits, Reason='{reason}'")

    print("\n[SUCCESS] Module M10 Demonstration Completed.")


if __name__ == "__main__":
    main()

