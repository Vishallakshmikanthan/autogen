# Architecture: System Overview

> **Subsystems:** SYS-01 through SYS-18  
> **Source Ref:** AUTOGEN AI Master Implementation Blueprint §2, §3, §9

---

## 1. System Mission & Boundary

AUTOGEN is an artificial intelligence research platform organized as a modular monolith. Its core purpose is to execute verifiable multi-agent simulations wherein autonomous agents perceive simulated environments under uncertainty, form probabilistic belief states, reason over symbolic/discrete domains, plan sequences of actions, and adapt through reinforcement learning and evolutionary optimization.

The system enforces a strict boundary between:
1. **The Simulation Engine (Python):** Owns ground-truth state, executes physics/rules, validates proposed agent actions, and computes analytics.
2. **The Intelligence Layer (Python):** Contains pure, from-scratch algorithmic implementations (search, probabilistic models, planners, ML estimators) with zero dependency on simulation rendering.
3. **The Presentation Layer (API / Frontend):** Visualizes simulation history, presents algorithm search trees, and provides interactive steerability.

```
┌─────────────────────────────────────────────────────────────┐
│                    Research Studio (Web UI)                 │
└──────────────────────────────▲──────────────────────────────┘
                               │ HTTP / WebSocket JSON stream
┌──────────────────────────────▼──────────────────────────────┐
│                    FastAPI Gateway (apps/api)               │
└──────────────────────────────▲──────────────────────────────┘
                               │ Typed Contracts
┌──────────────────────────────▼──────────────────────────────┐
│             AUTOGEN Simulation Core (packages/autogen)       │
│  ┌───────────────────────┐       ┌───────────────────────┐  │
│  │  Environment Engine   │       │   Perception Engine   │  │
│  │      (World State)    │       │     (Masked View)     │  │
│  └───────────▲───────────┘       └───────────┬───────────┘  │
│              │ Resolve Actions               │ Observation  │
│  ┌───────────┴───────────┐       ┌───────────▼───────────┐  │
│  │   Agent Arbitration   │       │   Autonomous Agents   │  │
│  │  (Collision/Resource) │       │   (Belief / Policy)   │  │
│  └───────────────────────┘       └───────────┬───────────┘  │
│                                              │ Actions      │
│  ┌───────────────────────────────────────────┴───────────┐  │
│  │               Intelligence & Planning Layer           │  │
│  │    (Search, Logic, Bayes, MDP, RL, Deep Models)       │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Major Subsystems Breakdown

| Subsystem ID | Component | Package Directory | Core Responsibilities |
| :--- | :--- | :--- | :--- |
| **SYS-01** | Simulation Core | `autogen.simulation.core` | Controls simulation clock, step ticks, event dispatch, and state snapshots. |
| **SYS-02** | Environment Engine | `autogen.simulation.environment`| 2D grid world, terrain properties, resource regeneration, obstacle dynamics. |
| **SYS-03** | Agent Framework | `autogen.agents.base` | Agent lifecycle, internal state containers, and action proposal protocols. |
| **SYS-04** | Perception Engine | `autogen.agents.perception` | Generates bounded, noisy, or masked observations from ground truth. |
| **SYS-05** | Memory Engine | `autogen.memory` | Spatial maps, episodic buffers, working memory, and semantic graphs. |
| **SYS-06** | Search Engine | `autogen.intelligence.search` | Uninformed, heuristic, and adversarial graph search algorithms. |
| **SYS-07** | Reasoning Engine | `autogen.intelligence.logic` | Propositional/first-order logic, resolution refutation, forward/backward chaining. |
| **SYS-08** | Planning Engine | `autogen.intelligence.planning` | STRIPS/PDDL planners, regression planning, partial-order planning. |
| **SYS-09** | Probabilistic Engine| `autogen.intelligence.probability`| Exact Bayesian networks, HMMs, Kalman/particle filters for belief tracking. |
| **SYS-10** | Game Theory Engine | `autogen.intelligence.game_theory`| Minimax, alpha-beta, Nash equilibria, payoff matrices, auction mechanisms. |
| **SYS-11** | Machine Learning | `autogen.intelligence.ml` | From-scratch linear models, trees, clustering, dimensionality reduction. |
| **SYS-12** | Reinforcement Learning| `autogen.intelligence.rl` | Tabular Q-learning, SARSA, Actor-Critic, Policy Gradients. |
| **SYS-13** | Neural Networks | `autogen.intelligence.deep_learning`| NumPy-based computational graph, autograd engine, MLP, CNN, DQN. |
| **SYS-14** | Multi-Agent Engine | `autogen.intelligence.multi_agent`| Emergent cooperation, communication channels, swarm coordination. |
| **SYS-15** | Evolution Engine | `autogen.intelligence.evolution` | Genetic algorithms, genetic programming, NEAT neural topology evolution. |
| **SYS-16** | Experiment Engine | `autogen.experiments` | Reproducible experiment matrix execution, parameter sweeps, seed isolation. |
| **SYS-17** | Analytics Engine | `autogen.analytics` | Statistical hypothesis testing, metrics computation, Pareto frontier analysis. |
| **SYS-18** | Replay & Visualization| `autogen.visualization` | Step serialization, timeline interpolation, trace formatting. |

---

## 3. Four Distinct State Representations

A central invariant of AUTOGEN is maintaining four explicit, separate state definitions:

1. **World State ($S_t \in \mathcal{S}$):**
   $$S_t = (E_t, A_t, R_t, W_t)$$
   Containing terrain properties ($E_t$), agent physical coordinates and vitals ($A_t$), dynamic resource levels ($R_t$), and weather/global parameters ($W_t$). Stored exclusively within the Environment Engine.
2. **Agent Observation ($O_t \in \mathcal{O}$):**
   $$O_t = \mathcal{P}(S_t, i, \eta_t)$$
   Constructed by the Perception Engine $\mathcal{P}$ for agent $i$ with sensory noise $\eta_t$. Agents never read $S_t$ directly.
3. **Agent Internal State ($I_t \in \mathcal{I}$):**
   The private cognitive state of the agent, containing current goal hierarchy, active execution plan, memory pointers, and energy reserves.
4. **Agent Belief State ($B_t \in \Delta(\mathcal{S})$):**
   The agent's subjective probability distribution over world states:
   $$B_t(s) = P(S_t = s \mid O_{1:t}, a_{1:t-1})$$
   Updated iteratively via Bayesian estimation or particle filtering.

---

## 4. Execution Step Lifecycle

Every discrete simulation tick $\Delta t$ executes through an invariant sequence:

1. **Snapshot:** Simulation Core records world state $S_t$.
2. **Perception:** For each active agent $i \in A_t$, Perception Engine computes observation $O_{i,t} = \mathcal{P}(S_t, i)$.
3. **Reasoning & Deliberation:** Agents update their belief states $B_{i,t} = \text{Update}(B_{i,t-1}, O_{i,t}, a_{i,t-1})$ and select an intended action:
   $$a_{i,t} = \pi_i(B_{i,t}, I_{i,t})$$
   Agents generate a verified `DecisionTrace` recording reasoning metrics.
4. **Action Submission:** Agents submit proposed actions $\{a_{1,t}, \dots, a_{n,t}\}$ to the Simulation Core.
5. **Arbitration & Physics Transition:** Simulation Core resolves action conflicts (e.g. movement into occupied cells, concurrent resource claims) deterministically using explicit priority rules.
6. **State Update:** New world state $S_{t+1} \sim \mathcal{T}(S_t, a_{1:n,t})$ is instantiated.
7. **Event & Replay Logging:** World events, metric diffs, and decision traces are appended to the replay buffer.

