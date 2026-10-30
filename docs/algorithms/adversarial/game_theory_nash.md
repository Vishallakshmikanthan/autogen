# Strategic Normal-Form Games & Nash Equilibrium Specification

> **Module:** M07 — Adversarial Search & Game Theory (SYS-10)  
> **Status:** Production / Verified

---

## 1. Theory & Mathematical Definition
A normal-form game is a tuple $G = (N, A, U)$ where:
- $N = \{1, \dots, n\}$ is a finite set of players.
- $A = A_1 \times \dots \times A_n$ is the set of joint action profiles, with $A_i$ denoting the finite action set of player $i$.
- $U = (U_1, \dots, U_n)$ is the profile of payoff functions, where $U_i: A \to \mathbb{R}$.

A mixed strategy $\sigma_i \in \Delta(A_i)$ is a probability distribution over $A_i$ satisfying $\sigma_i(a_i) \ge 0$ and $\sum_{a_i \in A_i} \sigma_i(a_i) = 1$.  
A strategy profile $\sigma^* = (\sigma_1^*, \dots, \sigma_n^*)$ is a **Nash Equilibrium** if no player can unilaterally deviate to achieve a strictly higher expected payoff:
$$\forall i \in N, \forall \sigma_i \in \Delta(A_i), \quad U_i(\sigma_i^*, \sigma_{-i}^*) \ge U_i(\sigma_i, \sigma_{-i}^*)$$

## 2. Dominance & Best Responses

- **Strict Dominance:** Action $a_i \in A_i$ strictly dominates $a'_i \in A_i$ if:
  $$\forall a_{-i} \in A_{-i}, \quad U_i(a_i, a_{-i}) > U_i(a'_i, a_{-i})$$
- **Weak Dominance:** Action $a_i \in A_i$ weakly dominates $a'_i \in A_i$ if:
  $$\forall a_{-i} \in A_{-i}, \quad U_i(a_i, a_{-i}) \ge U_i(a'_i, a_{-i}) \quad \text{and} \quad \exists a_{-i}, U_i(a_i, a_{-i}) > U_i(a'_i, a_{-i})$$
- **Best Response:** The set of best responses to opponent profile $\sigma_{-i}$ is:
  $$BR_i(\sigma_{-i}) = \arg\max_{a_i \in A_i} U_i(a_i, \sigma_{-i})$$

## 3. Solvers Implemented From Scratch

### 3.1 Pure Strategy Nash Equilibrium (Enumeration)
Joint actions $a^* \in A$ are evaluated across all players. $a^*$ is a pure Nash equilibrium if:
$$\forall i \in N, \quad a_i^* \in BR_i(a_{-i}^*)$$

### 3.2 Analytical 2x2 Mixed Nash Equilibrium
In a 2-player 2x2 game with payoff matrices $A$ (Player 1) and $B$ (Player 2):
Let $p = P(\text{Action } 0)$ for Player 1, and $q = P(\text{Action } 0)$ for Player 2.
At mixed equilibrium, Player 2 is indifferent between actions 0 and 1:
$$p B_{0,0} + (1-p) B_{1,0} = p B_{0,1} + (1-p) B_{1,1} \implies p^* = \frac{B_{1,1} - B_{1,0}}{(B_{0,0} - B_{1,0}) - (B_{0,1} - B_{1,1})}$$
Player 1 is indifferent between actions 0 and 1:
$$q A_{0,0} + (1-q) A_{0,1} = q A_{1,0} + (1-q) A_{1,1} \implies q^* = \frac{A_{1,1} - A_{0,1}}{(A_{0,0} - A_{0,1}) - (A_{1,0} - A_{1,1})}$$
A strictly mixed equilibrium exists iff $0 < p^* < 1$ and $0 < q^* < 1$.

### 3.3 Support Enumeration Algorithm
For $m \times n$ bimatrix games:
1. Iterate over support sizes $k \in \{1, \dots, \min(m, n)\}$.
2. For every subset $I \subseteq A_1, |I|=k$ and $J \subseteq A_2, |J|=k$:
   - Solve the indifference linear equations for $q \in \Delta(J)$ against Player 1's payoffs.
   - Solve the indifference linear equations for $p \in \Delta(I)$ against Player 2's payoffs.
   - Check strict feasibility: $p_i > 0, q_j > 0$.
   - Verify that actions outside support yield payoffs $\le$ equilibrium payoffs.

### 3.4 Independent Epsilon-Nash Equilibrium Verifier
Given any strategy profile $\sigma$, verifies the condition:
$$\max_{a_i \in A_i} U_i(a_i, \sigma_{-i}) - U_i(\sigma_i, \sigma_{-i}) \le \epsilon \quad (\epsilon = 10^{-5})$$
Returns `False` if any player possesses a profitable deviation exceeding $\epsilon$.

## 4. Canonical Resource Competition Games

1. **Prisoner's Dilemma:**
   - Temptation (5) > Reward (3) > Penalty (1) > Sucker (0).
   - Strictly dominant strategy: Defect. Unique Nash: (Defect, Defect).
2. **Hawk-Dove (Evolutionary Resource Competition):**
   - Pure Nash: (Hawk, Dove) and (Dove, Hawk).
   - Mixed Nash: $P(\text{Hawk}) = \frac{V}{C}$.
3. **Matching Pennies (Strictly Zero-Sum):**
   - No pure Nash equilibrium.
   - Unique strictly mixed Nash: $((0.5, 0.5), (0.5, 0.5))$ with expected payoff 0.
4. **Stag Hunt (Coordination & Social Trust):**
   - Payoff-dominant Nash: (Stag, Stag).
   - Risk-dominant Nash: (Hare, Hare).

## 5. Verification Strategy & Test Oracles
- Matching Pennies mixed Nash matches $(0.5, 0.5)$ exactly.
- Prisoner's Dilemma returns single unique pure Nash (Defect, Defect).
- Verifier rejects non-equilibrium profiles and accepts known theoretical equilibria.

## 6. References
- Nash, J. (1950). Equilibrium points in n-person games. *Proceedings of the National Academy of Sciences*, 36(1), 48-49.
- Nisan, N., Roughgarden, T., Tardos, É., & Vazirani, V. V. (Eds.). (2007). *Algorithmic Game Theory*. Cambridge University Press.

