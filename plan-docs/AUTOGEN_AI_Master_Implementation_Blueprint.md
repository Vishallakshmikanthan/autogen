# AUTOGEN AI --- Complete Master Implementation Blueprint

**An Open-Ended Artificial Intelligence Research & Simulation Platform**

> **Vision:** Build a configurable artificial world in which autonomous
> agents perceive their environment, reason under uncertainty, plan
> actions, learn from experience, cooperate or compete, and adapt to
> changing conditions.
>
> **Development philosophy:** Use AI-assisted coding to accelerate
> engineering, but do not outsource the intelligence. Implement,
> inspect, test, and understand the mathematical models and algorithms.

------------------------------------------------------------------------

## 1. Project Definition

AUTOGEN is a modular AI research laboratory built around a multi-agent
simulation ecosystem.

### 1.1 Core objectives

1.  Understand and implement classical AI.
2.  Build agents capable of search, reasoning, planning, and
    decision-making.
3.  Model uncertainty using probability and Bayesian inference.
4.  Train agents using machine learning and reinforcement learning.
5.  Understand neural networks through mathematical derivations and
    implementation.
6.  Investigate cooperation, competition, and emergent behaviour in
    multi-agent systems.
7.  Conduct reproducible experiments and statistically evaluate results.
8.  Build an interactive research interface for inspecting algorithms
    and agent decisions.

### 1.2 Research questions

  -----------------------------------------------------------------------
  Area                                Research question
  ----------------------------------- -----------------------------------
  Search                              How efficiently can an agent find a
                                      goal?

  Reasoning                           Can an agent derive conclusions
                                      from incomplete knowledge?

  Planning                            Can an agent construct a valid
                                      sequence of actions?

  Uncertainty                         How should an agent act with
                                      incomplete information?

  Learning                            Can an agent improve through
                                      experience?

  Memory                              Does retaining experience improve
                                      decisions?

  Cooperation                         Can agents coordinate without
                                      central control?

  Adaptation                          Can agents adapt to changing
                                      environments?

  Evolution                           Can populations develop better
                                      strategies?

  Hybrid AI                           Do combinations of methods improve
                                      performance?
  -----------------------------------------------------------------------

### 1.3 Product components

  -----------------------------------------------------------------------
  Product                             Responsibility
  ----------------------------------- -----------------------------------
  AUTOGEN Engine                        Simulation engine and AI algorithms

  AUTOGEN Studio                        Interactive world and agent
                                      visualization

  AUTOGEN Research                      Experiments, metrics, benchmarks,
                                      and research notes

  AUTOGEN Knowledge Base                Mathematical foundations,
                                      derivations, and algorithm
                                      documentation
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 2. System Architecture

Build AUTOGEN as a modular monorepo. Start with a modular monolith, not
microservices.

### 2.1 Major subsystems

  -----------------------------------------------------------------------
  ID                      Subsystem               Responsibility
  ----------------------- ----------------------- -----------------------
  SYS-01                  Simulation Core         Controls world state
                                                  and simulation
                                                  execution

  SYS-02                  Environment Engine      Terrain, resources, and
                                                  environmental dynamics

  SYS-03                  Agent Framework         Defines autonomous
                                                  agents

  SYS-04                  Perception Engine       Converts world state
                                                  into permitted
                                                  observations

  SYS-05                  Memory Engine           Stores and retrieves
                                                  experiences

  SYS-06                  Search Engine           Classical and heuristic
                                                  search

  SYS-07                  Reasoning Engine        Logic and knowledge
                                                  representation

  SYS-08                  Planning Engine         Constructs action
                                                  sequences

  SYS-09                  Probabilistic Engine    Bayesian inference and
                                                  uncertainty

  SYS-10                  Game Theory Engine      Strategic interactions

  SYS-11                  Machine Learning Engine Supervised and
                                                  unsupervised learning

  SYS-12                  Reinforcement Learning  Learning through
                          Engine                  interaction

  SYS-13                  Neural Network Engine   Deep learning models

  SYS-14                  Multi-Agent Engine      Cooperation and
                                                  competition

  SYS-15                  Evolution Engine        Population-based
                                                  optimization

  SYS-16                  Experiment Engine       Reproducible research
                                                  runs

  SYS-17                  Analytics Engine        Metrics and statistical
                                                  evaluation

  SYS-18                  Visualization Studio    Interactive simulation
                                                  interface
  -----------------------------------------------------------------------

### 2.2 Technology stack

  -----------------------------------------------------------------------
  Layer                   Technology              Purpose
  ----------------------- ----------------------- -----------------------
  Simulation              Python 3.12+            Core simulation and
                                                  algorithms

  Numerical computing     NumPy, SciPy            Numerical operations

  Data analysis           Pandas                  Experiment analysis

  Classical ML            Scikit-learn            Reference models and
                                                  selected baselines

  Deep learning           PyTorch                 Neural network training

  Initial rendering       Pygame                  Local simulation
                                                  visualization

  Frontend                React + TypeScript +    Research interface
                          Vite                    

  Styling                 Tailwind CSS            UI styling

  Charts                  Plotly                  Metrics and analysis

  Backend                 FastAPI                 API and experiment
                                                  control

  Live communication      WebSockets              Simulation updates

  Database                SQLite initially;       Experiment metadata
                          PostgreSQL later        

  Testing                 Pytest                  Automated verification

  Code quality            Ruff, MyPy              Linting and static
                                                  typing

  Documentation           MkDocs                  Technical knowledge
                                                  base
  -----------------------------------------------------------------------

**Architecture rule:** The Python simulation engine owns simulation
state. The frontend visualizes and controls it; the browser must not
become the source of truth for AI decisions.

Start with a local Python runner. Add FastAPI once the engine has stable
interfaces. Run simulation jobs outside HTTP request handlers.

### 2.3 Architecture principles

-   Algorithm independence: algorithms must not depend on rendering.
-   Modularity: algorithms are swappable through stable interfaces.
-   Reproducibility: every run supports explicit random seeds.
-   Testability: each algorithm is independently testable.
-   Observability: important decisions can be inspected.
-   Performance: separate simulation computation from rendering.
-   No hidden intelligence: do not delegate core algorithms to an LLM or
    opaque external API.

------------------------------------------------------------------------

## 3. Mathematical World Specification

Define the mathematical environment before building intelligent agents.

### 3.1 Grid representation

Start with a discrete two-dimensional grid:

\[ G={0,`\ldots`{=tex},H-1}`\times`{=tex}{0,`\ldots`{=tex},W-1} \]

Each cell can contain:

\[ C\_{ij}=(terrain, resource, danger, movement_cost) \]

Possible terrain types:

-   Grassland
-   Water
-   Forest
-   Desert
-   Mountain
-   Obstacle

### 3.2 World state

A conceptual global state is:

\[ S_t=(E_t,A_t,R_t,W_t) \]

where:

-   (E_t): environmental conditions
-   (A_t): agent states
-   (R_t): resource states
-   (W_t): world configuration

An individual agent state may be:

\[ s_t^i=(p_t^i,e_t^i,h_t^i,f_t^i,m_t^i) \]

where:

-   (p): position
-   (e): energy
-   (h): health
-   (f): carried food
-   (m): internal memory state

Keep actual world state, agent observation, internal state, and belief
state distinct.

### 3.3 Initial action space

\[ `\mathcal `{=tex}A={UP,DOWN,LEFT,RIGHT,STAY,COLLECT} \]

The environment transition can be expressed as:

\[ S\_{t+1}=T(S_t,`\mathbf `{=tex}a_t,`\xi`{=tex}\_t) \]

where (`\mathbf `{=tex}a_t) is the joint action of agents and
(`\xi`{=tex}\_t) represents environmental randomness. In deterministic
mode, omit (`\xi`{=tex}\_t).

### 3.4 Resource dynamics

A simple resource regeneration model:

\[ F\_{t+1}=`\min`{=tex}(F\_{`\max`{=tex}},F_t+G_t-C_t) \]

where:

-   (F_t): available food
-   (G_t): food generated
-   (C_t): food consumed
-   (F\_{`\max`{=tex}}): resource capacity

Agent energy:

\[ E\_{t+1}=`\min`{=tex}(E\_{`\max`{=tex}},`\max`{=tex}(0,E_t-C_a+R_f))
\]

where (C_a) is action energy cost and (R_f) is energy recovered from
food.

Define starvation and health damage as separate rules. Do not conflate
energy and health.

### 3.5 Population dynamics

\[ N\_{t+1}=N_t+B_t-D_t \]

where (B_t) is births and (D_t) is deaths.

### 3.6 Simulation tick order

Use a documented discrete-time execution order:

1.  Generate observations.
2.  Update beliefs and memory.
3.  Request agent decisions.
4.  Validate actions.
5.  Resolve simultaneous actions.
6.  Apply environment transitions.
7.  Calculate rewards and costs.
8.  Update metrics and events.
9.  Publish the new state.

The order matters: changing it can change outcomes.

------------------------------------------------------------------------

## 4. Agent Intelligence Architecture

### 4.1 Common interface

``` python
class BaseAgent:
    def observe(self, observation):
        raise NotImplementedError

    def update_belief(self):
        raise NotImplementedError

    def decide(self):
        raise NotImplementedError

    def learn(self, experience):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
```

The environment validates and executes actions. Agents must never mutate
the world directly. Keep decision selection separate from action
execution.

### 4.2 Agent architectures

  Agent                Decision mechanism
  -------------------- -------------------------------
  Random               Random action selection
  Simple reflex        Condition-action rules
  Model-based reflex   Internal world model
  Goal-based           Goal-directed search
  Utility-based        Expected utility maximization
  Learning             Experience-based improvement
  Hybrid               Combination of methods

### 4.3 Simple reflex agent

\[ a_t=`\pi`{=tex}(o_t) \]

Example:

``` python
if food_visible:
    return COLLECT
if obstacle_ahead:
    return TURN
return MOVE
```

### 4.4 Goal-based agent

\[ a\^\*=`\arg`{=tex}`\min`{=tex}\_{a`\in`{=tex}`\mathcal `{=tex}A}
Cost(a) \]

In practice, this should use a search or planning algorithm.

### 4.5 Utility-based agent

\[ a\^\*=`\arg`{=tex}`\max`{=tex}\_a
`\mathbb `{=tex}E\[U(S\_{t+1})`\mid `{=tex}s_t,a\] \]

Example utility:

\[ U(s)=w_eE+w_hH+w_fF-w_rR \]

Weights are experimental parameters, not universal constants.

### 4.6 Learning agent

A policy can be represented as:

\[ `\pi`{=tex}(a`\mid `{=tex}s) \]

Not every agent needs every architectural component. Declare supported
capabilities explicitly.

------------------------------------------------------------------------

# 5. AI Curriculum and Implementation Phases

Each phase should include theory, mathematical foundations,
implementation, tests, experiments, and acceptance criteria. Progress is
completion-based, not calendar-based.

## PHASE 01 --- Intelligent Agents and Classical AI Foundations

**Objective:** Understand rational agents and environment interaction.

### Concepts

-   AI definitions and approaches
-   Rationality and rational agents
-   PEAS framework
-   Agent architectures
-   Fully and partially observable environments
-   Deterministic and stochastic environments
-   Episodic and sequential environments
-   Static and dynamic environments
-   Discrete and continuous environments
-   Reactive and deliberative systems
-   Performance measures

### Mathematics

Agent function:

\[ f:P\^\*`\rightarrow `{=tex}A \]

A conceptual rational action:

\[
a_t\^\*=`\arg`{=tex}`\max`{=tex}\_{a`\in `{=tex}A}`\mathbb `{=tex}E\[U`\mid `{=tex}a,h_t\]
\]

where (h_t) is percept history.

Discounted performance:

\[ J=`\sum`{=tex}\_{t=0}^{T}`\gamma`{=tex}^t r_t \]

### Implement

-   Base agent framework
-   Random agent
-   Simple reflex agent
-   Model-based reflex agent
-   Observation interface
-   Performance evaluator

### Experiments

-   Random vs reactive movement
-   Fully vs partially observable worlds
-   Different sensor ranges
-   Different resource distributions

### Acceptance criteria

-   Agents operate independently.
-   Agents receive only permitted observations.
-   Decisions are inspectable.
-   Performance metrics are recorded.

## PHASE 02 --- Search Algorithms and Problem Solving

**Objective:** Enable agents to find solutions through systematic
exploration.

### Concepts

-   State-space representation
-   Initial and goal states
-   Successor functions
-   Search trees and graphs
-   Frontier and explored sets
-   Completeness and optimality
-   Time and space complexity
-   Heuristic search
-   Local and stochastic search

### Breadth-First Search

\[ O(V+E) \]

BFS finds a shortest path in number of edges on an unweighted graph.

Implement FIFO queue, visited set, parent tracking, and path
reconstruction.

### Depth-First Search

\[ O(V+E) \]

Implement stack or recursion, backtracking, cycle detection, and depth
limits.

### Uniform Cost Search

\[ g(n)=`\sum`{=tex}\_{e`\in `{=tex}path(n)}c(e) \]

Priority:

\[ f(n)=g(n) \]

Use a priority queue.

### Greedy Best-First Search

\[ f(n)=h(n) \]

### A\* Search

\[ `\boxed{f(n)=g(n)+h(n)}`{=tex} \]

where (g(n)) is cost so far and (h(n)) estimates remaining cost.

For a four-direction grid:

\[ h(n)=\|x_n-x_g\|+\|y_n-y_g\| \]

Admissibility:

\[ h(n)`\le `{=tex}h\^\*(n) \]

For standard optimal graph-search A\*, also account for the conditions
required by the implementation, such as heuristic consistency or
reopening nodes.

### Advanced search

-   Iterative Deepening DFS
-   Bidirectional BFS
-   IDA\*
-   Weighted A\*
-   Beam Search
-   Hill Climbing
-   Simulated Annealing
-   Local Beam Search

Simulated annealing acceptance probability for a worse candidate in a
minimization problem:

\[ P(accept)=`\exp`{=tex}(-`\Delta `{=tex}E/T) \]

### Experiments

Compare path cost, nodes expanded, runtime, memory, and solution quality
across algorithms.

### Acceptance criteria

-   Returned paths are valid.
-   Optimality is tested against known solutions.
-   Heuristic properties are tested where required.
-   Algorithms use identical maps and seeds in comparisons.

## PHASE 03 --- Adversarial Search and Game Theory

**Objective:** Create agents that anticipate other agents' actions.

### Concepts

-   Competitive environments
-   Zero-sum games
-   Game trees
-   Minimax
-   Alpha-Beta pruning
-   Evaluation functions
-   Move ordering
-   Depth-limited search
-   Expectimax
-   Nash equilibrium
-   Mixed strategies
-   Utility theory

### Minimax

\[ V(s)=
```{=tex}
\begin{cases}
U(s), & \text{terminal state}\\
\max_a V(Result(s,a)), & \text{MAX}\\
\min_a V(Result(s,a)), & \text{MIN}
\end{cases}
```
\]

### Alpha-Beta pruning

-   (`\alpha`{=tex}): best value MAX can guarantee
-   (`\beta`{=tex}): best value MIN can guarantee

Prune when:

\[ `\alpha`{=tex}`\ge`{=tex}`\beta`{=tex} \]

With ideal move ordering, complexity can improve from (O(b\^d)) toward
(O(b\^{d/2})).

### Expectimax

\[ V(s)=`\sum`{=tex}\_i P(s_i`\mid `{=tex}s)V(s_i) \]

### Strategic game

\[ G=(N,A,U) \]

A Nash equilibrium satisfies:

\[ U_i(s_i^*,s\_{-i}\^*)`\ge `{=tex}U_i(s_i,s\_{-i}^\*) \]

for every player and alternative unilateral strategy.

### Implement

-   Predator and prey agents
-   Minimax engine
-   Alpha-Beta pruning
-   Stochastic opponent
-   Strategic resource competition
-   Payoff matrices

### Acceptance criteria

Agents select valid strategic actions and demonstrate measurable
differences between decision strategies.

## PHASE 04 --- Knowledge Representation and Logical Reasoning

**Objective:** Build agents that reason about facts.

### Concepts

-   Propositional logic
-   First-order logic
-   Syntax and semantics
-   Predicates and quantifiers
-   Logical equivalence
-   Implication
-   Satisfiability and entailment
-   Knowledge bases
-   Resolution
-   Unification
-   Forward and backward chaining
-   Semantic networks and knowledge graphs

### Propositional logic

  Operator                       Meaning
  ------------------------------ ---------
  (`\neg `{=tex}P)               NOT
  (P`\land `{=tex}Q)             AND
  (P`\lor `{=tex}Q)              OR
  (P`\rightarrow `{=tex}Q)       IMPLIES
  (P`\leftrightarrow `{=tex}Q)   IFF

Example:

\[ P`\rightarrow `{=tex}Q \]

where (P) means tracks detected and (Q) means a predator may be nearby.

### First-order logic

\[ `\forall `{=tex}x; Agent(x)`\rightarrow `{=tex}Alive(x) \]

### Modus ponens

\[
P,`\quad `{=tex}P`\rightarrow `{=tex}Q`\quad`{=tex}`\therefore `{=tex}Q
\]

### Resolution

\[ (A`\lor `{=tex}B),`\quad`{=tex}(`\neg `{=tex}B`\lor `{=tex}C) \]

Resolving on (B):

\[ A`\lor `{=tex}C \]

### Unification

Unifying:

\[ Knows(x,Food) \]

with:

\[ Knows(Agent_1,Food) \]

gives:

\[ `\theta`{=tex}={x/Agent_1} \]

### Implement

-   Knowledge base
-   Predicate representation
-   Logical expression parser
-   Inference engine
-   Forward chaining
-   Backward chaining
-   Resolution
-   Unification

### Acceptance criteria

-   Deductions are valid.
-   Contradictions can be detected.
-   Conclusions have traceable provenance.

## PHASE 05 --- Constraint Satisfaction and Automated Planning

**Objective:** Solve problems involving constraints and multi-step
goals.

### Constraint Satisfaction Problem

\[ CSP=(X,D,C) \]

where (X) is variables, (D) domains, and (C) constraints.

Implement:

-   Backtracking
-   Minimum Remaining Values
-   Degree heuristic
-   Least Constraining Value
-   Forward checking
-   Arc consistency
-   AC-3

Arc consistency removes (x_i) when no supporting value exists in (D_j):

\[ `\nexists `{=tex}x_j`\in `{=tex}D_j: C\_{ij}(x_i,x_j)=True \]

### Classical planning

\[ P=(S,A,Pre,Eff,G) \]

where (S) is states, (A) actions, (Pre) preconditions, (Eff) effects,
and (G) goals.

Implement:

-   Initial and goal states
-   Preconditions
-   Add and delete effects
-   State transitions
-   Plan validation
-   STRIPS planning

Advanced topics:

-   Partial-order planning
-   Planning graphs
-   Graphplan
-   Hierarchical Task Networks
-   Heuristic planning
-   Planning under uncertainty

### Acceptance criteria

Agents produce valid plans, validate preconditions, and identify
impossible planning problems.

## PHASE 06 --- Probability, Bayesian Reasoning, and Uncertainty

**Objective:** Enable reasoning with incomplete information.

### Probability foundations

-   Sample spaces and events
-   Conditional probability
-   Independence
-   Random variables
-   Discrete and continuous distributions
-   Expectation and variance
-   Covariance and entropy

Conditional probability:

\[ P(A`\mid `{=tex}B)=`\frac{P(A\cap B)}{P(B)}`{=tex} \]

Bayes' theorem:

\[ `\boxed{P(H\mid E)=\frac{P(E\mid H)P(H)}{P(E)}}`{=tex} \]

### Distributions

Bernoulli:

\[ P(X=x)=p^x(1-p)^{1-x} \]

Binomial:

\[ P(X=k)=`\binom `{=tex}nkp^k(1-p)^{n-k} \]

Gaussian:

\[ f(x)=`\frac{1}{\sqrt{2\pi\sigma^2}}`{=tex}
e\^{-`\frac{(x-\mu)^2}{2\sigma^2}`{=tex}} \]

Poisson:

\[ P(X=k)=`\frac{\lambda^ke^{-\lambda}}{k!}`{=tex} \]

### Bayesian networks

\[
P(X_1,`\ldots`{=tex},X_n)=`\prod`{=tex}\_iP(X_i`\mid `{=tex}Parents(X_i))
\]

Implement DAG representation, conditional probability tables, exact
inference, variable elimination, and sampling-based approximate
inference.

### Markov property

\[
P(S\_{t+1}`\mid `{=tex}S_t,S\_{t-1},`\ldots`{=tex})=P(S\_{t+1}`\mid `{=tex}S_t)
\]

### Hidden Markov Models

\[ P(X\_{1:T},Z\_{1:T})=
P(Z_1)`\prod`{=tex}*{t=2}\^{T}P(Z_t`\mid `{=tex}Z*{t-1})
`\prod`{=tex}\_{t=1}\^{T}P(X_t`\mid `{=tex}Z_t) \]

Implement forward, backward, Viterbi, and forward-backward algorithms.

### Belief states and POMDP foundations

\[ b_t(s)=P(S_t=s`\mid `{=tex}o\_{1:t},a\_{1:t-1}) \]

Belief update:

\[ b\_{t+1}(s')= `\eta `{=tex}O(o\_{t+1}`\mid `{=tex}s')
`\sum`{=tex}\_sT(s'`\mid `{=tex}s,a_t)b_t(s) \]

### Acceptance criteria

The agent distinguishes actual world state from estimated world state
and represents uncertainty quantitatively.

## PHASE 07 --- Decision Theory and Markov Decision Processes

**Objective:** Formalize long-term decision-making.

### Utility

\[ EU(a)=`\sum`{=tex}\_sP(s`\mid `{=tex}a)U(s) \]

\[ a\^\*=`\arg`{=tex}`\max`{=tex}\_a EU(a) \]

### MDP

\[ MDP=(S,A,T,R,`\gamma`{=tex}) \]

where (T(s'\|s,a)) is transition probability, (R) is reward, and
(`\gamma`{=tex}) is the discount factor.

### Bellman expectation equation

\[ V\^`\pi`{=tex}(s)=
`\sum`{=tex}*a`\pi`{=tex}(a`\mid `{=tex}s)`\sum`{=tex}*{s'}T(s'`\mid `{=tex}s,a)
\[R(s,a,s')+`\gamma `{=tex}V\^`\pi`{=tex}(s')\] \]

### Bellman optimality equation

\[ V\^\*(s)= `\max`{=tex}*a`\sum`{=tex}*{s'}T(s'`\mid `{=tex}s,a)
\[R(s,a,s')+`\gamma `{=tex}V\^\*(s')\] \]

### Value iteration

\[ V\_{k+1}(s)= `\max`{=tex}*a`\sum`{=tex}*{s'}T(s'\|s,a)
\[R(s,a,s')+`\gamma `{=tex}V_k(s')\] \]

### Policy iteration

Alternate policy evaluation and policy improvement:

\[ `\pi`{=tex}\_{k+1}(s)=
`\arg`{=tex}`\max`{=tex}*a`\sum`{=tex}*{s'}T(s'\|s,a)
\[R(s,a,s')+`\gamma `{=tex}V\^{`\pi`{=tex}\_k}(s')\] \]

### Implement

-   MDP environment
-   Transition model
-   Reward model
-   Value iteration
-   Policy iteration
-   Policy visualization
-   Convergence monitoring

### Acceptance criteria

Verify results on small finite MDPs with analytically solvable examples.

## PHASE 08 --- Classical Machine Learning

**Objective:** Learn patterns from data.

### General supervised learning objective

\[ D={(x_i,y_i)}\_{i=1}\^{N} \]

\[
`\theta`{=tex}\^\*=`\arg`{=tex}`\min`{=tex}*`\theta`{=tex}`\frac`{=tex}1N
`\sum`{=tex}*{i=1}\^{N}L(f\_`\theta`{=tex}(x_i),y_i) \]

### Linear regression

\[ `\hat `{=tex}y=Xw+b \]

\[ MSE=`\frac`{=tex}1N`\sum`{=tex}\_i(y_i-`\hat `{=tex}y_i)\^2 \]

\[ w\_{t+1}=w_t-`\alpha`{=tex}`\nabla`{=tex}\_wJ(w_t) \]

Implement batch, stochastic, and mini-batch gradient descent, feature
scaling, polynomial regression, and regularization.

### Logistic regression

\[ `\sigma`{=tex}(z)=`\frac1{1+e^{-z}}`{=tex} \]

\[ P(y=1`\mid `{=tex}x)=`\sigma`{=tex}(w\^Tx+b) \]

Binary cross-entropy:

\[
L=-`\frac`{=tex}1N`\sum`{=tex}\_i\[y_i`\log `{=tex}p_i+(1-y_i)`\log`{=tex}(1-p_i)\]
\]

### Decision trees

Entropy:

\[ H(S)=-`\sum`{=tex}\_i p_i`\log`{=tex}\_2p_i \]

Information gain:

\[ IG(S,A)=H(S)-`\sum`{=tex}\_v`\frac{|S_v|}{|S|}`{=tex}H(S_v) \]

Implement ID3, Gini impurity, pruning, random forests, and feature
importance.

### K-Nearest Neighbours

\[ d(x,y)=`\sqrt{\sum_i(x_i-y_i)^2}`{=tex} \]

### Support Vector Machines

Linear hard-margin formulation:

\[ `\min`{=tex}\_{w,b}`\frac12`{=tex}\|w\|\^2 \]

subject to:

\[ y_i(w\^Tx_i+b)`\ge1`{=tex} \]

Extend to soft margins and kernels.

### K-Means

\[ J=`\sum`{=tex}*{i=1}\^{N}\|x_i-`\mu`{=tex}*{c_i}\|\^2 \]

### PCA

\[ `\Sigma`{=tex}=`\frac1{N-1}`{=tex}X_c\^TX_c \]

Find covariance eigenvectors and project onto selected principal
components.

Also study hierarchical clustering, DBSCAN, Gaussian Mixture Models, and
Expectation-Maximization.

### Evaluation

\[ Precision=`\frac{TP}{TP+FP}`{=tex} \]

\[ Recall=`\frac{TP}{TP+FN}`{=tex} \]

\[ F1=2`\frac{Precision\cdot Recall}{Precision+Recall}`{=tex} \]

Regression metrics: MAE, MSE, RMSE, and (R\^2).

### AUTOGEN applications

-   Predict resource availability
-   Classify environmental danger
-   Cluster agent behaviour
-   Predict energy requirements
-   Learn resource distribution patterns

### Acceptance criteria

Use separate training, validation, and test data. Prevent leakage and
compare against simple baselines.

## PHASE 09 --- Reinforcement Learning

**Objective:** Improve policies through interaction and feedback.

### RL model

\[ RL=(S,A,T,R,`\gamma`{=tex}) \]

Interaction:

\[ S_t`\rightarrow `{=tex}A_t`\rightarrow `{=tex}R\_{t+1},S\_{t+1} \]

### Epsilon-greedy

\[ a=
```{=tex}
\begin{cases}
\text{random action}, & \text{with probability }\epsilon\\
\arg\max_aQ(s,a), & \text{otherwise}
\end{cases}
```
\]

Epsilon decay:

\[
`\epsilon`{=tex}*t=`\max`{=tex}(`\epsilon`{=tex}*{`\min`{=tex}},`\epsilon`{=tex}\_0`\lambda`{=tex}\^t)
\]

### Multi-armed bandits

Implement greedy, epsilon-greedy, UCB, and Thompson Sampling.

UCB:

\[ UCB(a)=`\hat `{=tex}Q(a)+c`\sqrt{\frac{\ln t}{N(a)}}`{=tex} \]

### Q-learning

\[ `\boxed{Q(s,a)\leftarrow Q(s,a)+
\alpha[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]}`{=tex} \]

Implement Q-table, state encoding, reward calculation, exploration,
terminal handling, training, and evaluation.

### SARSA

\[ Q(s,a)`\leftarrow `{=tex}Q(s,a)+
`\alpha[r+\gamma Q(s',a')-Q(s,a)]`{=tex}\]

SARSA uses the next action actually selected by the policy; standard
Q-learning uses the maximum estimated next-action value.

### Monte Carlo

\[ G_t=`\sum`{=tex}*{k=0}^{T-t-1}`\gamma`{=tex}^kR*{t+k+1} \]

\[ V(s)=`\mathbb `{=tex}E\[G_t`\mid `{=tex}S_t=s\] \]

Implement first-visit and every-visit methods.

### Temporal Difference

\[ V(S_t)`\leftarrow `{=tex}V(S_t)+
`\alpha[R_{t+1}+\gamma V(S_{t+1})-V(S_t)]`{=tex}\]

\[ `\delta`{=tex}*t=R*{t+1}+`\gamma `{=tex}V(S\_{t+1})-V(S_t) \]

### Advanced RL

-   Expected SARSA
-   Double Q-learning
-   n-step TD
-   Eligibility traces
-   Dyna-Q
-   Policy gradient
-   REINFORCE
-   Actor-Critic
-   A2C
-   PPO
-   DQN

### Policy gradient

\[
J(`\theta`{=tex})=`\mathbb `{=tex}E\_{`\tau`{=tex}`\sim`{=tex}`\pi`{=tex}\_`\theta`{=tex}}\[R(`\tau`{=tex})\]
\]

\[ `\nabla`{=tex}\_`\theta `{=tex}J(`\theta`{=tex})= `\mathbb `{=tex}E\[
`\nabla`{=tex}*`\theta`{=tex}`\log`{=tex}`\pi`{=tex}*`\theta`{=tex}(a\|s)Q\^`\pi`{=tex}(s,a)\]
\]

### DQN

\[ Q(s,a;`\theta`{=tex}) \]

\[ L(`\theta`{=tex})=`\mathbb `{=tex}E\[(y-Q(s,a;`\theta`{=tex}))\^2\]
\]

\[ y=r+`\gamma`{=tex}`\max`{=tex}\_{a'}Q(s',a';`\theta`{=tex}\^-) \]

For terminal transitions, the bootstrap term is zero.

Implement replay buffer, target network, neural Q-function, gradient
optimization, and training diagnostics.

### Acceptance criteria

Demonstrate learning on controlled benchmark environments. Do not treat
a visually convincing reward curve as proof of correctness.

## PHASE 10 --- Neural Networks and Deep Learning

**Objective:** Understand how neural networks represent and learn
functions.

### Perceptron

\[ z=w\^Tx+b \]

\[ `\hat `{=tex}y=f(z) \]

Perceptron update:

\[ w`\leftarrow `{=tex}w+`\eta`{=tex}(y-`\hat `{=tex}y)x \]

### Multilayer perceptron

\[ a^{(l)}=`\sigma`{=tex}(W^{(l)}a^{(l-1)}+b^{(l)}) \]

### Activations

Sigmoid:

\[ `\sigma`{=tex}(x)=`\frac1{1+e^{-x}}`{=tex} \]

Tanh:

\[ `\tanh`{=tex}(x)=`\frac{e^x-e^{-x}}{e^x+e^{-x}}`{=tex} \]

ReLU:

\[ ReLU(x)=`\max`{=tex}(0,x) \]

Leaky ReLU:

\[ f(x)=`\max`{=tex}(`\alpha `{=tex}x,x) \]

Softmax:

\[ P(y=i)=`\frac{e^{z_i}}{\sum_je^{z_j}}`{=tex} \]

### Loss functions

MSE:

\[ L=`\frac`{=tex}1N`\sum`{=tex}\_i(y_i-`\hat `{=tex}y_i)\^2 \]

Binary cross-entropy:

\[ L=-\[y`\log `{=tex}p+(1-y)`\log`{=tex}(1-p)\] \]

Categorical cross-entropy:

\[ L=-`\sum`{=tex}\_i y_i`\log `{=tex}p_i \]

### Backpropagation

Chain rule:

\[ `\frac{\partial L}{\partial w}`{=tex} =
`\frac{\partial L}{\partial a}`{=tex}
`\frac{\partial a}{\partial z}`{=tex}
`\frac{\partial z}{\partial w}`{=tex} \]

For a layer:

\[ z=Wx+b,`\quad `{=tex}a=`\sigma`{=tex}(z) \]

\[ `\delta`{=tex}\^{(l)}= (W^{(l+1)T}`\delta`{=tex}^{(l+1)})
`\odot`{=tex}`\sigma`{=tex}'(z\^{(l)}) \]

\[ `\frac{\partial L}{\partial W^{(l)}}`{=tex}=
`\delta`{=tex}^{(l)}a^{(l-1)T} \]

Implement a complete forward and backward pass using NumPy.

### Optimizers

Momentum, RMSProp, Adam, learning-rate scheduling, weight decay,
dropout, and early stopping.

Adam:

\[ m_t=`\beta`{=tex}*1m*{t-1}+(1-`\beta`{=tex}\_1)g_t \]

\[ v_t=`\beta`{=tex}*2v*{t-1}+(1-`\beta`{=tex}\_2)g_t\^2 \]

\[ `\hat `{=tex}m_t=`\frac{m_t}{1-\beta_1^t}`{=tex}, `\quad`{=tex}
`\hat `{=tex}v_t=`\frac{v_t}{1-\beta_2^t}`{=tex} \]

\[ `\theta`{=tex}*t=`\theta`{=tex}*{t-1}-
`\frac{\alpha\hat m_t}{\sqrt{\hat v_t}+\epsilon}`{=tex} \]

### Advanced deep learning topics

-   CNNs
-   RNNs
-   LSTMs
-   GRUs
-   Attention
-   Transformers
-   Autoencoders
-   Variational Autoencoders
-   Representation learning
-   Self-supervised learning

### Implement

-   NumPy MLP
-   Neural survival predictor
-   CNN terrain classifier
-   DQN
-   Model checkpointing
-   Training visualization

### Acceptance criteria

Verify gradients numerically, test on known datasets, and compare the
NumPy implementation with a PyTorch equivalent.

## PHASE 11 --- Multi-Agent Systems and Collective Intelligence

**Objective:** Study independent agents interacting and making
collective decisions.

### Concepts

-   Multi-Agent Systems
-   Cooperation and competition
-   Communication protocols
-   Coordination
-   Distributed decision-making
-   Decentralized control
-   Multi-Agent Reinforcement Learning
-   Emergent behaviour
-   Social dilemmas
-   Mechanism design

Joint state:

\[ S_t=(s_t^1,s_t^2,`\ldots`{=tex},s_t\^n) \]

Joint action:

\[ A_t=(a_t^1,a_t^2,`\ldots`{=tex},a_t\^n) \]

Individual reward:

\[ r_i=R_i(S_t,A_t,S\_{t+1}) \]

Shared reward:

\[ R\_{team}=`\sum`{=tex}\_{i=1}\^{n}r_i \]

Message:

\[ m_i^t=Message(s_i^t) \]

### Implement

-   Independent Q-learning
-   Independent policy gradient
-   Centralized training with decentralized execution
-   Value decomposition
-   Cooperative policy learning
-   Competitive self-play

### Experiments

-   Independent vs cooperative agents
-   Shared vs individual rewards
-   Centralized vs decentralized coordination
-   Communication vs no communication
-   Team size vs group survival

### Acceptance criteria

Measure individual and group performance. Account for non-stationarity
as other agents learn.

## PHASE 12 --- Evolutionary Computation and Artificial Life

**Objective:** Study population-based optimization and evolving
strategies.

### Genetic algorithms

Chromosome:

\[ C=(g_1,g_2,`\ldots`{=tex},g_n) \]

Fitness:

\[ Fitness(C)=f(C) \]

Fitness-proportionate selection for nonnegative fitness:

\[ P(C_i)=`\frac{Fitness(C_i)}{\sum_jFitness(C_j)}`{=tex} \]

Mutation:

\[ g_i'=
```{=tex}
\begin{cases}
RandomValue, & \text{with probability }p_m\\
g_i, & \text{otherwise}
\end{cases}
```
\]

Implement selection, crossover, mutation, elitism, and replacement.

### Additional algorithms

-   Genetic Algorithms
-   Evolution Strategies
-   Genetic Programming
-   Neuroevolution
-   NEAT
-   Differential Evolution
-   Particle Swarm Optimization

Particle swarm:

\[ v_i\^{t+1}= `\omega `{=tex}v_i\^t+ c_1r_1(p_i-x_i\^t)+
c_2r_2(g-x_i\^t) \]

\[ x_i^{t+1}=x_i^t+v_i\^t \]

### Experiments

-   Evolution of navigation strategies
-   Population diversity
-   Selection pressure
-   Mutation-rate comparisons
-   Neuroevolution vs gradient-based learning

### Acceptance criteria

Track fitness, diversity, convergence, and performance on unseen
environments. Optimizing a chosen fitness function is not equivalent to
achieving general intelligence.

## PHASE 13 --- Advanced Planning, Meta-Learning, and Hybrid AI

**Objective:** Combine methods into more capable architectures.

### Concepts

-   Hierarchical Reinforcement Learning
-   Options framework
-   Model-based RL
-   Model Predictive Control
-   POMDP planning
-   Monte Carlo Tree Search
-   Meta-learning
-   Transfer learning
-   Continual learning
-   Neuro-symbolic AI
-   Hierarchical planning

### Monte Carlo Tree Search

A common UCT selection rule:

\[ UCT_i= `\frac{w_i}{n_i}`{=tex} +c`\sqrt{\frac{\ln N}{n_i}}`{=tex} \]

where (w_i) is accumulated reward, (n_i) is child visits, and (N) is
parent visits.

### Options framework

\[
`\omega`{=tex}=(I\_`\omega`{=tex},`\pi`{=tex}*`\omega`{=tex},`\beta`{=tex}*`\omega`{=tex})
\]

where (I\_`\omega`{=tex}) is the initiation set,
(`\pi`{=tex}*`\omega`{=tex}) the internal policy, and
(`\beta`{=tex}*`\omega`{=tex}) the termination condition.

### Model Predictive Control

\[ `\min`{=tex}*{a*{t:t+H-1}}
`\sum`{=tex}*{k=0}\^{H-1}`\ell`{=tex}(s*{t+k},a\_{t+k})+ V_f(s\_{t+H})
\]

At each step, predict, optimize, execute the first action, observe, and
re-plan.

### Hybrid agent example

-   A\* for navigation
-   Bayesian inference for hazard estimation
-   Symbolic rules for safety
-   Reinforcement learning for action selection
-   Memory for previous experiences

Compare hybrid agents against simpler baselines. More components do not
automatically mean more intelligence.

------------------------------------------------------------------------

## 6. Experimental Research Framework

This subsystem is mandatory across all phases.

### 6.1 Experiment configuration

Example:

``` yaml
experiment:
  name: navigation_comparison
  seed: 42
  episodes: 100

environment:
  width: 50
  height: 50
  obstacle_density: 0.15
  resource_density: 0.08

agents:
  - type: random
  - type: astar

metrics:
  - success_rate
  - average_path_cost
  - nodes_expanded
  - execution_time
```

Validate configuration schemas. This is an illustrative configuration,
not a final API contract.

### 6.2 Metrics

Survival rate:

\[ SR=`\frac{N_{survived}}{N_{initial}}`{=tex} \]

Success rate:

\[ SuccessRate=`\frac{N_{successful}}{N_{trials}}`{=tex} \]

Average cumulative reward:

\[ `\bar `{=tex}R=`\frac`{=tex}1N`\sum`{=tex}\_{i=1}\^{N}R_i \]

Path efficiency, when an optimal reference exists:

\[ PE=`\frac{L_{optimal}}{L_{actual}}`{=tex} \]

Resource efficiency:

\[
RE=`\frac{\text{Useful resources collected}}{\text{Resources consumed}}`{=tex}
\]

Population growth:

\[ GrowthRate=`\frac{N_{final}-N_{initial}}{N_{initial}}`{=tex} \]

Also measure runtime, memory, expanded nodes, model updates, and
simulation throughput.

### 6.3 Statistical evaluation

Mean:

\[ `\bar `{=tex}x=`\frac`{=tex}1N`\sum`{=tex}\_{i=1}\^Nx_i \]

Sample variance:

\[ s\^2=`\frac1{N-1}`{=tex}`\sum`{=tex}\_i(x_i-`\bar `{=tex}x)\^2 \]

Standard error:

\[ SE=`\frac{s}{\sqrt N}`{=tex} \]

Approximate confidence interval under suitable sampling conditions:

\[ `\bar `{=tex}x`\pm`{=tex}1.96SE \]

Also implement:

-   Median
-   Interquartile range
-   Bootstrap confidence intervals
-   Paired comparisons
-   Effect sizes
-   Hypothesis testing

Use a paired t-test only when its assumptions are reasonable. Otherwise
consider non-parametric or permutation-based methods.

### 6.4 Ablation studies

For a hybrid agent with memory, Bayesian inference, A\*, and Q-learning,
remove one component at a time and compare results.

  Configuration   Memory   Bayesian   A\*   Q-learning
  --------------- -------- ---------- ----- ------------
  Full            Yes      Yes        Yes   Yes
  A               No       Yes        Yes   Yes
  B               Yes      No         Yes   Yes
  C               Yes      Yes        No    Yes
  D               Yes      Yes        Yes   No

### 6.5 Reproducibility

Record:

-   Configuration
-   Random seed
-   Algorithm version
-   Environment version
-   Software version
-   Training checkpoint
-   Evaluation metrics
-   Execution metadata

Separate training seeds from evaluation seeds. Test generalization on
unseen environments.

------------------------------------------------------------------------

## 7. AUTOGEN Research Studio --- Interface Specification

Build a research interface, not a generic admin dashboard.

### 7.1 Navigation

  Section               Function
  --------------------- -----------------------------------------------
  Command Center        Project status
  World Builder         Configure environments
  Live Simulation       Watch agent interactions
  Agent Laboratory      Create and configure agents
  Algorithm Library     Explore AI implementations
  Mathematics Lab       Equations and derivations
  Memory Explorer       Inspect agent memories
  Reasoning Inspector   Inspect logical and probabilistic conclusions
  Learning Studio       Train and evaluate ML/RL agents
  Multi-Agent Arena     Cooperation and competition
  Experiment Manager    Configure and run experiments
  Analytics             Compare algorithms
  Research Notebook     Hypotheses and findings
  Documentation         Concepts and architecture

### 7.2 World Builder

-   Grid-size configuration
-   Terrain painting
-   Resource placement
-   Obstacle placement
-   Agent spawning
-   Weather configuration
-   Visibility settings
-   Environment presets
-   Procedural generation
-   Random seed control

### 7.3 Agent Laboratory

-   Create agents
-   Select architecture
-   Configure perception and memory
-   Select algorithms
-   Define reward functions
-   Set exploration parameters
-   Inspect internal state
-   Save configurations
-   Compare agents

### 7.4 Live Simulation

-   Play, pause, reset
-   Step-by-step execution
-   Simulation speed
-   Agent tracking
-   Agent selection
-   Resource heatmaps
-   Movement trails
-   Path visualization
-   Collision indicators
-   Reward visualization
-   Event timeline

### 7.5 Decision Trace Mode

An actual decision trace might display:

``` text
AGENT #014

Observation:
Food detected at (12, 18)

Belief:
P(food available) = 0.92

Current energy:
32 / 100

Available actions:
1. Move north
2. Move east
3. Stay
4. Collect

Selected action:
Move east

Decision mechanism:
A* path planning

Estimated path cost:
6.0

Reason:
Lowest-cost route to the target.
```

These values are illustrative. The application must generate traces from
actual agent execution.

### 7.6 Algorithm Laboratory

Each algorithm page should display:

1.  Formula
2.  Definition
3.  Assumptions
4.  Input graph or state
5.  Frontier or internal state
6.  Step-by-step execution
7.  Numerical values
8.  Output
9.  Complexity
10. Limitations
11. Implementation source
12. Tests and references

### 7.7 Mathematics Laboratory

Every algorithm should have:

-   Definition
-   Assumptions
-   Mathematical notation
-   Derivation
-   Numerical example
-   Pseudocode
-   Implementation
-   Visual demonstration
-   Limitations
-   Related concepts

Allow users to change meaningful parameters and observe their effects.

------------------------------------------------------------------------

## 8. Backend and Data Architecture

### 8.1 Service flow

``` text
React Frontend
      |
      v
FastAPI Application
      |
      +----------------------+
      |                      |
      v                      v
Simulation Manager     Experiment Manager
      |                      |
      v                      v
Simulation Engine      Algorithm Registry
      |                      |
      v                      v
Agent Framework        Evaluation Engine
      |                      |
      +----------+-----------+
                 |
                 v
             Persistence
                 |
                 v
             PostgreSQL
```

Start as a modular monolith. A simulation worker should run separately
from HTTP request handling.

### 8.2 Database entities

  Table                  Purpose
  ---------------------- ---------------------------------------
  users                  Optional user accounts
  environments           Environment configurations
  environment_versions   Versioned world definitions
  agents                 Agent configurations
  agent_versions         Versioned policies
  algorithms             Algorithm metadata
  simulations            Simulation sessions
  simulation_events      Important events
  agent_states           Selected state snapshots
  experiments            Experiment definitions
  experiment_runs        Individual runs
  metrics                Numerical results
  model_checkpoints      Model metadata and storage references
  research_notes         Hypotheses and findings

Do not store every agent state at every tick in PostgreSQL. Use
configurable snapshots, event logs, compressed trajectories, and
aggregate metrics. Store large model artifacts outside ordinary
relational rows and retain references and metadata.

### 8.3 Example API

``` http
POST   /api/v1/environments
GET    /api/v1/environments
GET    /api/v1/environments/{id}

POST   /api/v1/agents
GET    /api/v1/agents
GET    /api/v1/agents/{id}

POST   /api/v1/simulations
POST   /api/v1/simulations/{id}/start
POST   /api/v1/simulations/{id}/pause
POST   /api/v1/simulations/{id}/step
POST   /api/v1/simulations/{id}/reset

GET    /api/v1/simulations/{id}/state
GET    /api/v1/simulations/{id}/metrics

POST   /api/v1/experiments
POST   /api/v1/experiments/{id}/run
GET    /api/v1/experiments/{id}/results
```

Live stream:

``` text
WebSocket: /api/v1/simulations/{id}/stream
```

Use Pydantic schemas and explicit validation.

------------------------------------------------------------------------

## 9. Repository Structure

``` text
nexus-ai/
│
├── apps/
│   ├── api/
│   │   ├── routes/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── main.py
│   │
│   └── web/
│       └── src/
│           ├── components/
│           ├── pages/
│           ├── features/
│           ├── hooks/
│           └── services/
│
├── packages/
│   ├── simulation/
│   │   ├── engine/
│   │   ├── environment/
│   │   ├── physics/
│   │   ├── events/
│   │   └── state/
│   │
│   ├── agents/
│   │   ├── base/
│   │   ├── reflex/
│   │   ├── goal_based/
│   │   ├── utility/
│   │   └── learning/
│   │
│   ├── intelligence/
│   │   ├── search/
│   │   ├── game_theory/
│   │   ├── logic/
│   │   ├── csp/
│   │   ├── planning/
│   │   ├── probability/
│   │   ├── decision_theory/
│   │   ├── machine_learning/
│   │   ├── reinforcement_learning/
│   │   ├── deep_learning/
│   │   ├── multi_agent/
│   │   └── evolution/
│   │
│   ├── memory/
│   ├── experiments/
│   ├── evaluation/
│   ├── analytics/
│   └── visualization/
│
├── research/
│   ├── hypotheses/
│   ├── experiment_designs/
│   ├── findings/
│   └── papers/
│
├── docs/
│   ├── architecture/
│   ├── mathematical_foundations/
│   ├── algorithms/
│   ├── implementation/
│   ├── api/
│   └── decisions/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── mathematical/
│   ├── property_based/
│   └── regression/
│
├── configs/
├── scripts/
├── benchmarks/
├── docker/
├── pyproject.toml
├── package.json
└── README.md
```

Create directories as needed. Do not generate empty implementations for
every future algorithm at bootstrap.

### 9.1 Engineering patterns

  Pattern                 Purpose
  ----------------------- -------------------------------------
  Strategy                Swap decision algorithms
  Factory                 Construct agents from configuration
  Registry                Discover algorithms
  Observer/event system   Publish simulation events
  Command                 Represent agent actions
  State                   Manage simulation lifecycle
  Dependency injection    Independent testing
  Adapter                 Integrate frameworks
  Repository              Separate persistence

------------------------------------------------------------------------

## 10. Vibe Coding Methodology

The greatest risk of AI-assisted development is plausible-looking but
mathematically incorrect or architecturally inconsistent code.

### 10.1 Development cycle

1.  Read the module specification.
2.  Understand the mathematical objective and assumptions.
3.  Ask the coding agent to propose architecture and pseudocode.
4.  Independently review equations and edge cases.
5.  Implement one small unit at a time.
6.  Generate and execute tests.
7.  Inspect outputs and traces.
8.  Integrate through stable interfaces.
9.  Run controlled experiments.
10. Document and commit changes.

### 10.2 Prompt sequence for each module

**Prompt A --- Architecture:** Inspect the existing project and propose
files, classes, interfaces, data structures, dependencies, and
acceptance criteria.

**Prompt B --- Mathematics:** Explain the mathematical model, derive
update equations, state assumptions, and identify edge cases.

**Prompt C --- Implementation:** Implement one algorithm at a time. Do
not modify unrelated modules.

**Prompt D --- Verification:** Independently generate tests, execute
them, and explain discrepancies.

**Prompt E --- Integration:** Integrate through existing contracts and
run regression tests.

**Prompt F --- Research:** Build a reproducible benchmark against a
baseline.

### 10.3 Permanent coding-agent instructions

``` text
You are the senior AI systems engineer assisting with AUTOGEN AI.

AUTOGEN is a modular artificial intelligence research and simulation platform.

CORE RULES:
- Prioritize mathematical correctness over implementation speed.
- Document every algorithm's definition, assumptions, equations,
  variables, derivation where appropriate, pseudocode, complexity,
  limitations, and references.
- Do not replace algorithms intended for from-scratch implementation
  with external APIs or pretrained models.
- Keep simulation, intelligence, persistence, API, and visualization
  layers independent.
- Use typed interfaces and explicit contracts.
- Use reproducible random seeds.
- Separate training and evaluation.
- Agents must never directly mutate the environment.
- The environment validates and resolves actions.
- Record important events and decision traces.
- Do not hardcode experiment results.
- Never fabricate metrics or findings.
- Do not claim completion without running verification.
- Do not rewrite unrelated working modules.
- Do not introduce unnecessary dependencies or microservices.

TESTING:
Every algorithm requires unit tests, mathematical correctness tests,
edge-case tests, and integration tests where applicable.

For stochastic algorithms, test statistical behaviour across repeated
trials rather than expecting different seeds to produce identical outputs.

Use numerical gradient checking for neural-network backpropagation.

WORKFLOW:
1. Inspect the existing architecture.
2. Read relevant documentation.
3. Identify dependencies.
4. Propose implementation and verification plans.
5. Identify mathematical risks.
6. Define acceptance criteria.
7. Implement the smallest functional unit.
8. Write and execute tests.
9. Review regressions.
10. Update documentation.

DEFINITION OF DONE:
- Mathematical specification documented.
- Implementation complete.
- Tests pass.
- Outputs independently verified.
- Integration successful.
- Limitations documented.
- Usage example works.

Build a verifiable AI research system, not merely impressive-looking code.
```

------------------------------------------------------------------------

## 11. Completion-Based Development Roadmap

No calendar deadlines. Advance only when each phase's acceptance
criteria are met.

  -----------------------------------------------------------------------
  Gate                    Phase                   Completion condition
  ----------------------- ----------------------- -----------------------
  0                       Foundation              Repository,
                                                  architecture,
                                                  contracts, and tests

  1                       World and agents        Reproducible
                                                  multi-agent simulation

  2                       Classical AI            Verified search and
                                                  adversarial algorithms

  3                       Symbolic AI             Logic, knowledge base,
                                                  CSP, and planning

  4                       Probabilistic AI        Bayesian inference,
                                                  HMMs, belief states

  5                       Machine learning        Predictive models and
                                                  generalization tests

  6                       Reinforcement learning  Agents learn policies
                                                  through experience

  7                       Deep learning           Numerically verified
                                                  neural networks

  8                       Collective intelligence Multi-agent cooperation
                                                  and competition

  9                       Hybrid intelligence     Comparative
                                                  hybrid-agent
                                                  experiments

  10                      Research platform       Reproducible benchmarks
                                                  and research report
  -----------------------------------------------------------------------

### Dependency notes

-   A\* can be implemented before Bayesian reasoning.
-   Bayesian inference can be studied independently of Q-learning.
-   Neural networks can be built independently of the ecosystem.
-   DQN requires neural networks and reinforcement learning.
-   MARL requires a working multi-agent environment and RL fundamentals.
-   Neuroevolution requires evolutionary algorithms and a defined policy
    representation.

------------------------------------------------------------------------

## 12. Quality Assurance and Mathematical Verification

### 12.1 Test layers

  Test type         Purpose
  ----------------- -------------------------------------------
  Unit              Individual functions and classes
  Integration       Subsystem communication
  Property-based    General mathematical properties
  Numerical         Equations and calculations
  Statistical       Stochastic behaviour
  Regression        Prevent previously fixed bugs
  Performance       Speed and memory
  Reproducibility   Repeatability under controlled conditions

### 12.2 Verification examples

**A\* path cost**

\[ Cost(path)=`\sum`{=tex}\_{e`\in `{=tex}path}c(e) \]

Compare against UCS on small graphs to verify optimality under
appropriate heuristic conditions.

**Normalized probability**

\[ `\sum`{=tex}\_iP(X_i)=1 \]

**Probability bounds**

\[ 0`\le `{=tex}P(A)`\le1`{=tex} \]

**Terminal Q-learning target**

\[ Target=r \]

There must be no future-value bootstrap for a terminal transition.

**Numerical gradient checking**

\[ `\frac{\partial L}{\partial\theta_i}`{=tex} `\approx`{=tex}
`\frac{L(\theta_i+\epsilon)-L(\theta_i-\epsilon)}`{=tex}
{2`\epsilon`{=tex}} \]

Compare numerical gradients with backpropagation.

**Search correctness**

Verify valid paths and compare costs against known optimal solutions
where optimality is expected.

### 12.3 Continuous integration

Initial commands:

``` bash
ruff check .
mypy .
pytest -q
```

Later add coverage checks, integration workflows, benchmark regression
checks, dependency scanning, and frontend type checking/build checks.

Keep expensive research benchmarks separate from fast commit-level CI
tests.

------------------------------------------------------------------------

## 13. Performance and Scalability

Optimize based on profiling, not speculation.

### Initial techniques

-   Efficient grid lookups
-   Spatial partitioning
-   Vectorized resource updates
-   Batched observations
-   Cached pathfinding
-   Event-driven updates
-   Rendering decoupling
-   Memory-efficient snapshots

### Advanced techniques

-   NumPy vectorization
-   Numba
-   Multiprocessing
-   Ray
-   GPU acceleration
-   Distributed experiment execution

Parallelize independent experiment runs before attempting to parallelize
a tightly coupled simulation.

Suggested initial engineering targets:

  Capability            Initial target
  --------------------- --------------------------------------
  Grid world            50 × 50
  Initial population    20 agents
  Expanded population   100+ agents
  Replay                Deterministic under controlled seeds
  Experiments           Multiple independent seeds
  Algorithms            Independently benchmarkable
  Rendering             Independent of simulation speed
  Storage               Persistent metadata and artifacts

These are targets, not performance guarantees.

------------------------------------------------------------------------

## 14. Advanced Extensions

Reserve these for later phases.

### Procedural world generation

-   Perlin noise
-   Cellular automata
-   Fractal terrain
-   Stochastic resource placement

### Continuous and 3D environments

-   Continuous state spaces
-   Physics simulation
-   Continuous control
-   Spatial dynamics

### Neuro-symbolic AI

Combine neural representations with symbolic reasoning.

### Swarm intelligence

Study decentralized collective behaviour and emergent coordination.

### Explainable AI

Inspect feature importance, decision traces, uncertainty, and policy
behaviour.

### Transfer and continual learning

Test reuse of knowledge in new worlds and resistance to catastrophic
forgetting.

------------------------------------------------------------------------

## 15. Bootstrap Implementation Checklist

Start with the foundation. Do not ask a coding agent to build the entire
platform in one generation.

-   [ ] Create project README and scope.
-   [ ] Document architecture and module boundaries.
-   [ ] Define mathematical world specification.
-   [ ] Configure Python and frontend tooling.
-   [ ] Define environment, agent, observation, action, and state
    contracts.
-   [ ] Implement deterministic grid-world engine.
-   [ ] Implement random and reflex agents.
-   [ ] Verify transitions, movement, resources, and replay.
-   [ ] Build the first visualization.
-   [ ] Document the completed foundation before adding search
    algorithms.

### Bootstrap acceptance criteria

The initial foundation is complete only when:

-   The project structure is coherent.
-   Core interfaces are documented and typed.
-   The world model is mathematically specified.
-   A deterministic simulation can run.
-   Agents cannot mutate the world directly.
-   Basic tests pass.
-   A working visualization can inspect the simulation.
-   Documentation reflects actual implementation.

------------------------------------------------------------------------

## 16. First Coding-Agent Initialization Task

Use this prompt in Antigravity, Kiro, or another coding IDE.

``` text
TASK: Initialize AUTOGEN AI — Research Platform Foundation

You are initializing AUTOGEN AI, a long-term artificial intelligence
research and simulation platform.

Do not attempt to implement the entire platform in this task.
Establish a production-quality, extensible foundation.

PROJECT VISION:
AUTOGEN is a modular AI laboratory in which autonomous agents interact
with simulated environments.

Future capabilities include:
- Classical AI and search
- Adversarial search and game theory
- Symbolic AI and logical inference
- Constraint satisfaction and automated planning
- Bayesian reasoning and probabilistic models
- Decision theory and MDPs
- Classical machine learning
- Reinforcement learning
- Neural networks and deep learning
- Multi-agent systems
- Evolutionary computation
- Hybrid AI architectures
- Research experiments and statistical evaluation

TECHNOLOGY:
Core: Python 3.12+, NumPy, SciPy, Pydantic, Pytest
Backend: FastAPI, WebSockets
Frontend: React, TypeScript, Vite, Tailwind CSS, Plotly
Storage: SQLite initially, PostgreSQL-compatible architecture later

TASK 1 — REPOSITORY
Create a clean monorepo structure for simulation, environment, agents,
intelligence algorithms, memory, experiments, evaluation, analytics,
API, frontend, tests, and documentation.

Do not create empty implementations for every future algorithm.

TASK 2 — DOCUMENTATION
Create:
- README.md
- docs/architecture/system_overview.md
- docs/architecture/module_boundaries.md
- docs/architecture/data_contracts.md
- docs/architecture/decision_records.md
- docs/mathematical_foundations/world_model.md
- docs/implementation/roadmap.md

Document module responsibilities, dependencies, data flow, and decisions.

TASK 3 — MATHEMATICAL WORLD SPECIFICATION
Specify the initial discrete 2D environment:
- World dimensions
- Grid coordinates
- Cell representation
- Terrain
- Obstacles
- Resources
- Agent state
- Action space
- Observation space
- State transitions
- Resource regeneration
- Energy consumption
- Reward and termination conditions

Distinguish:
1. Actual environment state
2. Agent internal state
3. Agent observation
4. Agent belief state

TASK 4 — CORE INTERFACES
Define typed interfaces for:
- BaseEnvironment
- BaseAgent
- Observation
- Action
- WorldState
- AgentState
- SimulationEngine
- SimulationEvent

Agents must never directly mutate the environment.
The environment validates and executes actions.

TASK 5 — SIMULATION DESIGN
Design a deterministic discrete-time engine.

Document:
1. Observation generation
2. Agent belief and memory updates
3. Action selection
4. Action validation
5. Simultaneous action resolution
6. Environment transition
7. Reward calculation
8. Metrics collection
9. Event publication

Support seeded reproducibility.

TASK 6 — DEVELOPMENT QUALITY
Configure Pytest, Ruff, MyPy, type annotations, logging,
configuration validation, Git ignore rules, and test commands.

TASK 7 — IMPLEMENTATION ROADMAP
For every future phase document:
- Learning objectives
- Mathematical prerequisites
- Algorithms
- Modules
- Interfaces
- Tests
- Acceptance criteria
- Dependencies

STRICT RESTRICTIONS:
- Do not implement fake AI.
- Do not call external LLM APIs to make agents intelligent.
- Do not hardcode experiment results.
- Do not implement algorithms prematurely.
- Do not introduce microservices.
- Do not create unnecessary abstractions.
- Do not modify unrelated files.
- Do not claim completion without executing verification.

FINAL RESPONSE:
Show actual repository structure, explain foundational modules,
list created files, list executed tests and results, identify incomplete
work, and explain the next implementation milestone.

Wait for approval before beginning the next implementation phase.
```

------------------------------------------------------------------------

## 17. Knowledge Base and Learning Resources

Maintain a technical record for every algorithm.

  Document         Required contents
  ---------------- -------------------------------
  Theory           Conceptual explanation
  Mathematics      Equations and derivations
  Algorithm        Pseudocode and complexity
  Implementation   Code and architecture
  Verification     Test cases and correctness
  Experiments      Configurations and results
  Limitations      Assumptions and weaknesses
  References       Textbooks and research papers

### Core AI

-   **Artificial Intelligence: A Modern Approach** --- Stuart Russell
    and Peter Norvig.
-   [Berkeley CS188](https://inst.eecs.berkeley.edu/~cs188/)
-   [AIMA resources](https://aima.cs.berkeley.edu/)

### Reinforcement learning

-   **Reinforcement Learning: An Introduction** --- Richard Sutton and
    Andrew Barto.
-   [Free textbook](http://incompleteideas.net/book/the-book-2nd.html)

### Machine learning

-   **Pattern Recognition and Machine Learning** --- Christopher Bishop.
-   [An Introduction to Statistical
    Learning](https://www.statlearning.com/)

### Deep learning

-   **Deep Learning** --- Ian Goodfellow, Yoshua Bengio, and Aaron
    Courville.
-   [Dive into Deep Learning](https://d2l.ai/)

### Multi-agent systems

-   **Multiagent Systems** --- Gerhard Weiss, editor.
-   [Multi-Agent Reinforcement Learning book](https://marl-book.com/)

------------------------------------------------------------------------

## 18. Definition of a Successful AUTOGEN Project

At completion, AUTOGEN should provide:

1.  A working artificial ecosystem with configurable environments.
2.  Multiple autonomous agent architectures.
3.  Independently implemented and tested AI algorithms.
4.  Mathematical explanations and derivations.
5.  Inspectable agent decision traces.
6.  Reproducible experiments.
7.  Statistical comparisons and ablation studies.
8.  Interactive visualization and analytics.
9.  A documented, modular codebase.
10. Research reports containing real findings and limitations.

The final research objective is to build a platform capable of
implementing, explaining, comparing, and experimentally evaluating
different forms of artificial intelligence in a common simulated
environment.

## Final Engineering Principles

-   Do not rush to deep learning.
-   Do not confuse complexity with intelligence.
-   Do not let vibe coding turn you into a passive observer.
-   Verify equations and algorithms independently.
-   Treat failed experiments as useful evidence.
-   Do not claim novelty without a literature review and reproducible
    evidence.
-   Build incrementally and advance only when acceptance criteria are
    satisfied.

**The goal:** Be able to open any AUTOGEN module, explain its mathematics,
inspect its implementation, run an experiment, and defend the results
without depending on an LLM.
