# TD(lambda) and Eligibility Traces

## 1. Theory & Mathematical Definition
Eligibility traces bridge temporal difference methods with Monte Carlo returns along a spectrum parameterized by $\lambda \in [0, 1]$. In the backward mechanistic view, an eligibility trace memory vector $e(s, a)$ tracks recent state-action visits, allowing a single temporal difference error $\delta_t$ to update multiple preceding decisions proportionally.

## 2. Multi-Step Return
The $n$-step truncated return is defined as:
$$G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n V_{t+n-1}(S_{t+n})$$
with $G_{t:t+n} \doteq G_t$ if $t + n \ge T$.

## 3. Watkins' Q(lambda) Algorithm
TD error:
$$\delta_t = R_{t+1} + \gamma (1 - d_t) \max_a Q(S_{t+1}, a) - Q(S_t, A_t)$$
Trace accumulation:
$$e_t(S_t, A_t) \leftarrow e_t(S_t, A_t) + 1$$
Global parameter update:
$$Q_{t+1}(s, a) \leftarrow Q_t(s, a) + \alpha \delta_t e_t(s, a) \quad \forall s, a$$
Trace decay:
$$e_{t+1}(s, a) = \begin{cases} \gamma \lambda e_t(s, a), & \text{if } A_{t+1} = \arg\max_{a'} Q(S_{t+1}, a') \\ 0, & \text{otherwise} \end{cases}$$
If an exploratory non-greedy action is chosen, off-policy credit assignment is broken and traces are cut to zero.

## 4. Verification Strategy
- Compare value convergence speed of $n$-step TD and $\text{TD}(\lambda)$ against $\text{TD}(0)$ on finite Markov chains.

