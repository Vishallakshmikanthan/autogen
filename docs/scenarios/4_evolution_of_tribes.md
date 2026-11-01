# Showpiece 4: Evolution of Tribes

> **Modules:** M16 (Evolution & ALife), M12 (Classical ML)  
> **Status:** Verified (Gate 10)  
> **Command:** `python scripts/run_showpiece_evolution_of_tribes.py --seed 42`

---

## 1. Scenario Overview

A population of autonomous foraging agents undergoes multi-generational evolutionary adaptation. Genomes encode phenotypic behavioral weights:
- Mobility and exploration drive
- Foraging effort intensity
- Environmental hazard caution

Over generations, natural selection, crossover, and mutation drive speciation.

---

## 2. Unsupervised Behavioral Discovery (K-Means)

Clustering agent behavioral trajectories via from-scratch K-Means ($k=3$) reveals three distinct emergent tribes:
1. **Foraging Harvesters:** Prioritize high food consumption rate in dense resource clusters.
2. **Agile Scouts:** Maximize spatial mobility to discover distant, unexploited resource patches.
3. **Cautious Defenders:** High hazard avoidance buffer minimizing mortality risk near hazards.

### Empirical Progress:
- Generation 0 Best Fitness: $73.4$
- Generation 35 Evolved Fitness: $91.8$ (**+25.0% adaptation gain**)
- Generalization: Champion genomes evaluated on held-out test maps maintain $> 90\%$ of peak fitness.

