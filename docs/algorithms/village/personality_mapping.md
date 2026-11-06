# Algorithm: Big Five Personality Parameter Modulation & Mehrabian Baseline

> **Family:** Decision  
> **Source Module:** Module M22-H (`autogen.intelligence.village.personality`)  
> **Standard:** AGENTS.md 10-Section Mathematical Specification

---

## 1. Theory & Mathematical Definition
Human decision-making varies systematically across individuals due to stable psychological traits. The Five-Factor Model (Big Five: Openness $O$, Conscientiousness $C$, Extraversion $E$, Agreeableness $A$, Neuroticism $N \in [0, 1]$) provides an empirically validated basis for parameterizing individual differences in risk tolerance, social orientation, threat sensitivity, and emotional inertia without abandoning normative utility optimization.

## 2. Assumptions
1. Trait values remain invariant across short-to-medium simulation horizons (within a single lifespan).
2. Trait effects on utility weights, affective baselines, and attention gains are monotonic.
3. Linear and bounded affine combinations capture dominant psychological shifts while preventing degenerative zero/infinite weights.

## 3. Equations with Variable Definitions
Given traits $\mathbf{T} = (O, C, E, A, N) \in [0, 1]^5$ and occupational role multiplier $m_{\text{role}}$:

### 3.1 Utility Weight Modulation
$$w_{\text{soc}} = 14.0 \cdot (0.7 + 0.6 E)$$
$$w_{\text{econ}} = 1.0 \cdot (0.8 + 0.4 C) \cdot m_{\text{role}}$$
$$\kappa = 18.0 \cdot (0.8 + 0.5 N)$$
$$\theta_{\text{risk}} = \mathrm{clip}\left(0.25 + 0.35(O - 0.5) - 0.30(C - 0.5) - 0.25(A - 0.5) - 0.20(N - 0.5), \, 0.02, \, 0.90\right)$$
$$w_{\text{phys}} = 2.20$$

### 3.2 Affective Baseline Vector (Mehrabian Mapping)
$$v_0 = \mathrm{clip}\left(0.12 + 0.15(E - 0.5) - 0.15(N - 0.5), \, -1.0, \, 1.0\right)$$
$$a_0 = \mathrm{clip}\left(0.05 + 0.10(O - 0.5) + 0.10(N - 0.5), \, -1.0, \, 1.0\right)$$
$$d_0 = \mathrm{clip}\left(0.10 + 0.15(E - 0.5) - 0.10(N - 0.5), \, -1.0, \, 1.0\right)$$

### 3.3 Dynamic Behavioral Parameters
- Threat Salience Gain: $G_{\text{threat}} = 1.0 + 0.50 N$
- Novelty Salience Gain: $G_{\text{novel}} = 1.0 + 0.50 O$
- Empathy Gain: $G_{\text{empathy}} = 1.0 + 0.50 A$
- Switching Cost Scale: $s_{\text{switch}} = 1.0 + 0.40 C$
- Contagion Coupling Factor: $\gamma = 0.08 \cdot (0.60 + 0.40 E + 0.40 A)$
- Grudge Decay Factor: $\lambda_{\text{grudge}} = \min(0.999, \max(0.950, 0.995 + 0.004 N - 0.003 A))$
- Talkativeness: $p_{\text{talk}} = 0.20 + 0.70 E$

## 4. Derivation Where Appropriate
The baseline emotional offsets follow Mehrabian's regression models connecting Big Five personality dimensions to Pleasure, Arousal, and Dominance (PAD). Extraversion positively drives pleasure and dominance; neuroticism suppresses pleasure and dominance while elevating baseline arousal.

## 5. Pseudocode
```python
def weights_from_traits(t: Traits, role: VillageRole) -> tuple[VillagerUtilityWeights, PersonalityParams]:
    m_role = ROLE_ECON_MULTIPLIERS.get(role, 1.0)
    w_soc = 14.0 * (0.7 + 0.6 * t.extraversion)
    w_econ = (0.8 + 0.4 * t.conscientiousness) * m_role
    kappa = 18.0 * (0.8 + 0.5 * t.neuroticism)
    theta_risk = clip(0.25 + 0.35*(t.O - 0.5) - 0.30*(t.C - 0.5) - 0.25*(t.A - 0.5) - 0.20*(t.N - 0.5), 0.02, 0.90)
    
    pad_base = (
        clip(0.12 + 0.15*(t.E - 0.5) - 0.15*(t.N - 0.5), -1, 1),
        clip(0.05 + 0.10*(t.O - 0.5) + 0.10*(t.N - 0.5), -1, 1),
        clip(0.10 + 0.15*(t.E - 0.5) - 0.10*(t.N - 0.5), -1, 1),
    )
    # package and return weights and params
```

## 6. Computational Complexity
- **Time Complexity:** $\mathcal{O}(1)$ closed-form algebraic evaluation.
- **Space Complexity:** $\mathcal{O}(1)$ immutable parameter dataclass.

## 7. Implementation Notes
Implemented in `autogen.intelligence.village.personality`. Archetypes are provided for canonical villagers, but arbitrary valid trait vectors are fully supported.

## 8. Verification Strategy & Test Oracles
1. **Monotonicity:** $\partial w_{\text{soc}}/\partial E > 0$, $\partial w_{\text{econ}}/\partial C > 0$, $\partial \kappa/\partial N > 0$.
2. **Boundary safety:** For all grid points in $[0, 1]^5$, $\theta_{\text{risk}} \in [0.02, 0.90]$ and baseline PAD $\in [-1, 1]^3$.
3. **Reproducibility:** Deterministic output for identical trait vectors.

## 9. Known Limitations
- Traits are static per life stage; adult personality drift or trauma-induced personality shifts are not modeled in this increment.
- Inter-trait interaction terms (e.g., $E \times N$) are simplified into additive first-order effects.

## 10. References / Citations
- Goldberg, L. R. (1993). *The Structure of Phenotypic Personality Traits*. American Psychologist, 48(1), pp. 26-34.
- Mehrabian, A. (1996). *Pleasure-Arousal-Dominance: A General Framework for Describing and Measuring Personality and Emotion*. Current Psychology, 14(4), pp. 261-292.

