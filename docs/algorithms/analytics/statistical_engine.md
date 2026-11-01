# Statistical Analytics & Experiment Engine Specification (SYS-16, SYS-17)

## 1. Theory & Mathematical Definition

The AUTOGEN Analytics and Experiment Engine provides rigorous, from-scratch statistical evaluation tools for multi-agent autonomous simulations. Empirical artificial intelligence research requires distinguishing genuine performance improvements from stochastic variance induced by environmental layout, sensor noise, and multi-agent interaction randomness.

The engine implements:
- Unbiased descriptive statistics ($N-1$ Bessel-corrected sample variance, standard error, median, interquartile range).
- Confidence intervals via Student's $t$ critical values and empirical percentile bootstrap.
- Effect size estimations: parametric Cohen's $d$ (pooled and paired) and non-parametric Cliff's $\delta$.
- Hypothesis testing: diagnostic normality screening, paired Student's $t$-test with exact regularized incomplete beta $p$-values, Monte Carlo paired and two-sample permutation tests, and Wilcoxon signed-rank test.
- Multiple hypothesis testing control via the step-down Holm-Bonferroni procedure.

---

## 2. Assumptions

1. **Independent Seeds:** Each episode $i \in \{1, \dots, N\}$ is seeded by a unique deterministic integer generated via `np.random.SeedSequence`, ensuring mutual independence across runs.
2. **Paired Comparisons:** In paired evaluations (such as ablation or candidate vs baseline), both architectures are evaluated on the exact same environmental layout and seed schedule.
3. **Parametric Assumptions ($t$-test):** The difference vector $d_i = x_i - y_i$ is assumed approximately normally distributed. If skewness $|g_1| > 1.0$ or excess kurtosis $|g_2| > 1.5$, normality is declared violated, requiring fallback to the permutation test.
4. **Permutation Test:** Under $H_0$, exchangeability holds between paired treatment conditions.

---

## 3. Equations & Variable Definitions

### 3.1 Descriptive Statistics

- **Sample Mean:**
  $$\bar{x} = \frac{1}{N} \sum_{i=1}^N x_i$$
- **Sample Variance ($ddof=1$):**
  $$s^2 = \frac{1}{N - 1} \sum_{i=1}^N (x_i - \bar{x})^2$$
- **Standard Error of the Mean:**
  $$SE = \frac{s}{\sqrt{N}}$$
- **Confidence Interval ($1 - \alpha$):**
  $$CI = \left[\bar{x} - t_{1 - \alpha/2, N-1} \cdot SE,\; \bar{x} + t_{1 - \alpha/2, N-1} \cdot SE\right]$$

### 3.2 Effect Sizes

- **Cohen's $d$ (Independent Samples):**
  $$s_{\text{pooled}} = \sqrt{\frac{(N_1 - 1)s_1^2 + (N_2 - 1)s_2^2}{N_1 + N_2 - 2}}, \quad d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}$$
- **Cohen's $d_z$ (Paired Differences):**
  $$d_z = \frac{\bar{d}}{s_d}$$
- **Cliff's Delta $\delta$:**
  $$\delta = \frac{\sum_{i=1}^{N_1}\sum_{j=1}^{N_2} \text{sign}(x_i - y_j)}{N_1 \cdot N_2} \in [-1, 1]$$

### 3.3 Analytic Student's $t$ Two-Sided $p$-Value

Let $t$ be the test statistic with $\nu = N - 1$ degrees of freedom. The two-sided $p$-value is evaluated via the regularized incomplete beta function $I_x(a, b)$:
$$x = \frac{\nu}{\nu + t^2}, \quad p = I_x\left(\frac{\nu}{2}, \frac{1}{2}\right) = \frac{\int_0^x u^{\nu/2 - 1}(1 - u)^{-1/2} du}{B(\nu/2, 1/2)}$$

### 3.4 Monte Carlo Paired Permutation Test

For differences $d_i = x_i - y_i$, under $H_0$ each $d_i$ has random sign $\sigma_i \in \{-1, +1\}$ with equal probability $1/2$. For $M$ sampled sign vectors:
$$T_m = \frac{1}{N} \sum_{i=1}^N \sigma_{m, i} d_i, \quad p = \frac{1 + \sum_{m=1}^M \mathbb{I}(|T_m| \ge |T_{\text{obs}}|)}{M + 1}$$

### 3.5 Holm-Bonferroni Correction

For $m$ hypothesis tests with ordered $p$-values $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$:
$$p_{(k)}^{\text{adj}} = \min\left(1.0, \max_{j \le k} \left[(m - j + 1) \cdot p_{(j)}\right]\right)$$
Reject $H_0^{(k)}$ if $p_{(k)}^{\text{adj}} \le \alpha$.

---

## 4. Derivation & Pseudocode

### 4.1 Incomplete Beta via Lentz's Continued Fraction Method

The continued fraction representation of $I_x(a, b)$:
$$I_x(a, b) = \frac{x^a (1-x)^b}{a B(a, b)} \cdot \cfrac{1}{1 + \cfrac{d_1}{1 + \cfrac{d_2}{1 + \dots}}}$$
where:
$$d_{2m} = \frac{m(b - m)x}{(a + 2m - 1)(a + 2m)}, \quad d_{2m+1} = -\frac{(a + m)(a + b + m)x}{(a + 2m)(a + 2m + 1)}$$

```text
Algorithm: Lentz Incomplete Beta
Input: a, b, x, max_iter=150, eps=1e-14
1. If x > (a + 1)/(a + b + 2): return 1 - Lentz(b, a, 1 - x)
2. c = 1.0, d = 1.0 - (a + b)*x / (a + 1), h = d
3. For m = 1 to max_iter:
     Evaluate even and odd continued fraction step factors aa
     Update d = 1 / (1 + aa * d)
     Update c = 1 + aa / c
     h = h * d * c
     If |d * c - 1| < eps: break
4. Return h * exp(a*ln(x) + b*ln(1-x) - ln_beta(a, b)) / a
```

---

## 5. Computational Complexity

| Operation | Time Complexity | Space Complexity |
| :--- | :--- | :--- |
| Sample mean / variance / SE | $\mathcal{O}(N)$ | $\mathcal{O}(1)$ |
| Median / IQR (Percentile) | $\mathcal{O}(N \log N)$ | $\mathcal{O}(N)$ |
| Empirical Bootstrap ($B$ resamples) | $\mathcal{O}(B \cdot N)$ | $\mathcal{O}(B \cdot N)$ |
| Cliff's Delta (Vectorized) | $\mathcal{O}(N_1 \cdot N_2)$ | $\mathcal{O}(N_1 \cdot N_2)$ |
| Analytic Paired $t$-test | $\mathcal{O}(N + K_{\text{Lentz}})$ | $\mathcal{O}(N)$ |
| Paired Permutation Test ($M$ samples) | $\mathcal{O}(M \cdot N)$ | $\mathcal{O}(M \cdot N)$ |
| Holm-Bonferroni Correction ($m$ tests) | $\mathcal{O}(m \log m)$ | $\mathcal{O}(m)$ |

---

## 6. Implementation Notes

- All core statistics and tests are implemented in pure Python/NumPy without calling `scipy.stats` at runtime.
- Bessel's correction ($ddof=1$) is strictly used for all sample variances.
- Laplace $+1$ continuity correction is maintained on all Monte Carlo permutation $p$-values to prevent zero $p$-values.
- Monotonicity in Holm-Bonferroni is guaranteed by cumulative maximum tracking across ranks.

---

## 7. Verification Strategy & Test Oracles

- **Descriptive Statistics:** Compared directly against `np.mean`, `np.var(ddof=1)`, `np.std(ddof=1)`, and `scipy.stats.sem`.
- **$t$-test $p$-values:** Verified against `scipy.stats.ttest_rel` on 100 randomly generated synthetic normal vectors.
- **Cliff's Delta:** Verified against brute-force combinatorial counts.
- **Bootstrap Coverage:** Verified that the 95% bootstrap CI covers the true population parameter in $\ge 90\%$ of repeated trials on known distributions.

---

## 8. Known Limitations

- For sample sizes $N < 10$, normal approximations for Wilcoxon signed-rank tests are asymptotic approximations.
- Incomplete beta continued fraction requires symmetry transformation $x > (a+1)/(a+b+2)$ to maintain rapid convergence for values near 1.

---

## 9. References

1. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*. Lawrence Erlbaum Associates.
2. Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer ordinal questions. *Psychological Bulletin*, 114(3), 494–509.
3. Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
4. Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007). *Numerical Recipes: The Art of Scientific Computing* (3rd ed.). Cambridge University Press.

