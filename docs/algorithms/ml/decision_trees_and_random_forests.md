# Decision Trees and Random Forests

## 1. Theory & Mathematical Definition
Decision Trees partition feature space into orthogonal hyper-rectangles using recursive greedy binary splits. Ensembles of randomized trees (Random Forests) mitigate tree variance through bootstrap aggregation (bagging) and random feature subspace sampling.

## 2. Splitting Criteria
### Gini Impurity (CART Classification)
$$I_{\text{Gini}}(S) = 1 - \sum_{k=1}^K p_k^2$$
where $p_k = \frac{1}{|S|} \sum_{i \in S} \mathbb{I}(y_i = k)$.

### Shannon Entropy (ID3 Classification)
$$H(S) = -\sum_{k=1}^K p_k \log_2(p_k)$$

### Mean Squared Error / Variance Reduction (Regression)
$$I_{\text{Var}}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y})^2$$

Information / Impurity Gain for split $(j, \theta)$:
$$\Delta I(S, j, \theta) = I(S) - \frac{|S_L|}{|S|} I(S_L) - \frac{|S_R|}{|S|} I(S_R)$$

## 3. Random Forests Algorithm
1. For $b = 1, \dots, B$:
   - Draw bootstrap sample $S^{(b)}$ of size $N$ with replacement from $S$.
   - Grow a decision tree where at each internal node, candidate split features are restricted to a random subset of size $m$:
     - Classification: $m = \lfloor\sqrt{D}\rfloor$
     - Regression: $m = \lfloor D / 3 \rfloor$
2. Aggregation:
   - Classification: majority vote $\hat{y} = \arg\max_k \sum_{b=1}^B \mathbb{I}(\hat{y}^{(b)} = k)$
   - Regression: average $\hat{y} = \frac{1}{B} \sum_{b=1}^B \hat{y}^{(b)}$

## 4. Complexity
- **Tree Training:** $O(D \cdot N \log N \cdot \text{depth})$.
- **Forest Training:** $O(B \cdot m \cdot N \log N \cdot \text{depth})$.
- **Inference:** $O(B \cdot \text{depth})$.

## 5. Verification Strategy & Test Oracles
- Compare classification impurity reduction and tree predictions with `sklearn.tree.DecisionTreeClassifier` and `sklearn.ensemble.RandomForestClassifier`.
- Verify out-of-bag (OOB) accuracy on synthetic non-linear benchmarks (XOR and multi-modal blobs).

