# Circadian Daily Routines and Role-Specific Temporal Priors

## 1. Theory & Mathematical Definition
Human societies exhibit strong circadian temporal synchronization. Rather than deciding actions strictly based on immediate visual proximity, daily human routines are synchronized with planetary diurnal cycles: working in the morning, trading at midday, congregating for leisure and worship at dusk, and retreating to safe shelters for sleep at night.

The **Circadian Routine Model** formalizes this temporal synchronization using circular Gaussian utility priors $\pi_I(\tau)$ defined over the 24-hour diurnal topology $\tau \in [0.0, 24.0)$ hours, biased by occupational roles ($VillageRole$).

---

## 2. Assumptions
1. World simulation time follows a continuous 24-hour diurnal clock with period $T = 24.0$ hours.
2. Temporal distance on the clock face wraps around midnight: distance between 23:00 and 01:00 is 2.0 hours, not 22.0 hours.
3. Circadian priors act as soft utility biases that guide intentions rather than rigid deterministic schedules.

---

## 3. Equations with Variable Definitions

### 3.1 Circular Hour Distance ($\delta$)
$$\delta(\tau_1, \tau_2) = \min(|\tau_1 - \tau_2|,\, 24.0 - |\tau_1 - \tau_2|)$$
where $\tau_1, \tau_2 \in [0.0, 24.0)$ hours.

### 3.2 Gaussian Circadian Prior Score ($\pi_I(\tau)$)
$$\pi_I(\tau) = A_I \cdot \exp\left(-\frac{\delta(\tau, \mu_I)^2}{2\sigma_I^2}\right)$$
- $A_I$: Peak amplitude utility boost (typically $2.0 \dots 4.0$)
- $\mu_I$: Peak clock hour of intention $I$ in $[0.0, 24.0)$
- $\sigma_I$: Temporal dispersion standard deviation in hours

### 3.3 Default Schedule Peak Parameters
| Intention Kind | Peak Hour $\mu_I$ | Std Dev $\sigma_I$ | Amplitude $A_I$ | Description |
|---|---|---|---|---|
| `WORK` | 09:00 | 2.5 h | 2.50 | Morning field harvest, well-drawing, crafts |
| `MARKET` | 13:00 | 1.5 h | 2.20 | Midday market trading and double auctions |
| `SOCIALIZE` | 18:00 | 1.5 h | 2.00 | Dusk congregation at well/commons |
| `REST` | 01:00 | 3.0 h | 3.00 | Night shelter and homeostatic recovery |

---

## 4. Derivation
Using linear distance $|\tau_1 - \tau_2|$ creates artificial boundary discontinuities at midnight ($23.99\text{h} \to 0.00\text{h}$). By mapping time to the unit circle $S^1$ through the circular metric $\delta(\tau_1, \tau_2) = \min(|\tau_1 - \tau_2|, 24.0 - |\tau_1 - \tau_2|)$, the Gaussian kernel $\exp(-\delta^2 / 2\sigma^2)$ is smooth, infinitely differentiable, and continuous across the day/night boundary.

---

## 5. Pseudocode
```python
def circular_distance(t1, t2):
    diff = abs((t1 % 24.0) - (t2 % 24.0))
    return min(diff, 24.0 - diff)

def evaluate_circadian_priors(role, time_of_day):
    priors = ROLE_SCHEDULES.get(role, DEFAULT_PRIORS)
    scores = {}
    for p in priors:
        delta = circular_distance(time_of_day, p.peak_hour)
        val = p.amplitude * math.exp(-(delta ** 2) / (2.0 * p.sigma ** 2))
        scores[p.kind] = max(scores.get(p.kind, 0.0), round(val, 4))
    return scores
```

---

## 6. Computational Complexity
- **Time Complexity:** $O(1)$ constant-time circular distance and exponential evaluations per intention.
- **Space Complexity:** $O(1)$ static configuration tables per role.

---

## 7. Implementation Notes
- Role tables allow natural occupational specialization:
  - Guards have night watch peaks at 22:00 and daytime sleep at 05:00.
  - Merchants have elevated market square amplitude ($A = 3.80$).
  - Farmers have early morning agricultural peaks at 08:30 ($A = 3.20$).

---

## 8. Verification Strategy & Test Oracles
- **Circular Continuity Oracle:** Verify $\pi_I(23.9) \approx \pi_I(0.1)$ for intention with peak at 00:00.
- **Peak Hour Maxima Oracle:** Verify $\arg\max_\tau \pi_I(\tau) = \mu_I$.
- **Statistical Autocorrelation Oracle:** Across 3 simulated days (144 ticks, 48 ticks/day), agent zone occupancy shows statistically significant lag-48 autocorrelation ($r > 0.30$).

---

## 9. Known Limitations
- Does not model daylight savings shifts or seasonal variations in photoperiod.
- Fixed Gaussian variance without weather-dependent morning dampening.

---

## 10. References / Citations
- Winfree, A. T. (2001). *The Geometry of Biological Time*. Springer-Verlag, 2nd Edition.
- Aschoff, J. (1965). *Circadian Clocks*. North-Holland Publishing Company.
