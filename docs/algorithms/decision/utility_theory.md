# Decision Theory & Multi-Attribute Utility

## 1. Principle of Maximum Expected Utility (MEU)

Decision theory combines probability theory with utility theory.
Under uncertainty, an agent chooses an action $a^*$ that maximizes **Expected Utility**:
$$
EU(a) = \sum_{s' \in \mathcal{S}} P(s' \mid a) U(s')
$$
$$
a^* = \arg\max_{a \in \mathcal{A}} EU(a)
$$

### Von Neumann-Morgenstern (VNM) Axioms
Preferences over lotteries satisfy completeness, transitivity, continuity, and independence iff there exists a real-valued utility function $U$ such that preferences correspond to expected utility.

---

## 2. Multi-Attribute Linear Utility

When states have multiple attributes (energy $E$, health $H$, food $F$, danger/risk $R$), the multi-attribute utility under additive independence is:
$$
U(s) = w_e \cdot E(s) + w_h \cdot H(s) + w_f \cdot F(s) - w_r \cdot R(s)
$$
where weights $\mathbf{w} = (w_e, w_h, w_f, w_r)$ encode the agent's risk tolerance and physiological trade-offs.

---

## 3. Decision Making Under Belief States

When the true world state is partially observed, the expectation is taken over the agent's internal belief distribution $b(s) = P(S_t = s \mid o_{1:t})$ from Module M10:
$$
EU(a) = \sum_{s \in \mathcal{S}} b(s) \sum_{s' \in \mathcal{S}} P(s' \mid s, a) U(s')
$$
This forms the foundational transition from belief filtering (M10) to decision-theoretic action selection (M11) and POMDPs.

