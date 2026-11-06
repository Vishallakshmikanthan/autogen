# PLAN ONLY — Increment 2 (M22-H): Needs, Hierarchical Intentions, Daily Routines

> PLAN ONLY. Do not write code. Proposes files, interfaces, pseudocode, risks, and acceptance criteria. Waiting for approval (AGENTS.md Step A → B).

Sources: `plan-docs/AUTOGEN_Humanlike_Agent_Architecture.md` §5.3, §5.4, Phase E (L709-712), risk list (L736-742).

---

## 0. Carry-over defects from Increment 1 (found while preparing this plan)

I reported Increment 1 as "100% complete". Re-reading the demo output and the code, that was wrong. These need fixing first, as **Step 0**:

| # | Defect | Evidence |
|---|---|---|
| D1 | **Intentions never end.** Once `SHELTER` is set, nothing ever puts the agent back to `ROUTINE`. | Demo step 28 (clear weather): every agent still has `Intention: SHELTER`. `villager_agent.py` L162-188 sets intentions, but no code clears them. |
| D2 | **The demo prints hardcoded claims.** "Neighbors … equalized affective arousal/valence" and "Invariants: Satisfied" are literal strings, not measurements. This breaks the "never fabricate results" rule. | `humanlike_village_demo.py` |
| D3 | **Acceptance criterion #4 (greetings) was not met.** The demo reports `Greetings Exchanged: 0`, and the one `ARRIVAL` event was injected by hand. | Demo output |
| D4 | **Thunder doesn't change emotion.** Emote stays `joy` through the storm because alerts produce no PAD stimulus. | Demo steps 4-7 |
| D5 | **Demo memories are scripted.** The reflection showcase injects `MemoryEpisode`s by hand without saying so. | Demo step 2 |

**Fixes:**
- D1: completion/failure predicates and a `max_duration` per intention (part of §2 below).
- D2: print measured values only, e.g. mean |ΔPAD| from the contagion pass, and an actual replay-hash comparison.
- D3: a social `ARRIVAL` already exists in `village_grid.py`. Find out why `GREET` never wins `argmax`, and add a regression test.
- D4: an alert with threat > 0 emits an OCC "fear" stimulus, scaled by salience × (0.5+N).
- D5: label scripted episodes as `[scripted]` in the output, or derive them from real outcomes.

---

## 1. Files

| File | New/Mod | Purpose |
|---|---|---|
| `intelligence/village/needs.py` | New | `NeedState`, deficits L1-L5, pressure, prepotency gates, dynamic weights, satiation tracker |
| `intelligence/village/bt.py` | New | Utility-guided intention tree: `IntentionSpec`, steps, score, completion/failure, hysteresis selector |
| `intelligence/village/routines.py` | New | Role routine tables + circadian time-of-day priors |
| `intelligence/village/intention.py` | Mod | add `step_index`, `max_duration`, `status` (ACTIVE/DONE/FAILED); fixes D1 |
| `agents/village/villager_agent.py` | Mod | Phase 2e needs update; BT arbitration in `decide`; restrict candidates to the intention's actions ∪ safety set; D4 stimulus |
| `agents/village/controller.py` | Mod | `HumanlikeConfig.enable_needs / enable_routines`; feed contact quality into needs; telemetry `needs` block |
| `intelligence/registry.py` | Mod | register `needs_prepotency`, `utility_behavior_tree`, `circadian_routines` |
| `docs/algorithms/village/{needs_prepotency,utility_behavior_tree,circadian_routines}.md` | New | 10-section docs |
| `tests/unit/village/test_humanlike_{needs,bt,routines,inc1_fixes}.py` | New | oracles in §5 |
| `packages/autogen/examples/humanlike_village_demo.py` | Mod | `--days N`, a daily-rhythm table, D2/D5 fixes |

---

## 2. Mathematical model

All need inputs are in [0,100] with **higher = better** (the code treats `hunger` as satiety: a villager cries out at `hunger ≤ 18`).

**Deficits** $d_k\in[0,1]$:

| Level | Definition |
|---|---|
| L1 physiological | $d_1 = 1-\min(h,t,e,\text{health})/100$ (the worst need is the bottleneck) |
| L2 safety | $d_2=\mathrm{clip}(0.5\,\text{exposed\_in\_storm}+0.3(1-\min(1,\text{food\_stock}/6))+0.2\,\text{threat}_{\max})$ |
| L3 belonging | $c\leftarrow 0.9c+q_t$, with $q$: GREET 0.5, dialogue 1.0, trade 0.3. Then $d_3=\mathrm{clip}(1-c/c^*)$, $c^*=2(0.5+E)$ |
| L4 esteem | $d_4 = 1 - S_i/\max_j S_j$ |
| L5 growth | $m\leftarrow 0.98m+\mathbb 1[\text{role-productive action}]$, then $d_5=\mathrm{clip}(1-m/m^*)$, $m^*=3(0.5+O)$ |

**Pressure, gate, and weight** (doc §5.3; I use its self-corrected gate):

$$P_k=d_k^{\,p},\quad g_1=1,\quad g_k=\prod_{j<k}(1-d_j)^{\beta},\quad w_k(t)=\text{base}_k\,(1+\lambda P_k)\,g_k$$

with $p=2$, $\beta=0.7$, $\lambda=2$.

**Mapping onto the existing engine.** `VillagerUtilityWeights` is frozen, so each tick builds a new instance with `model_copy`:

- $w_\text{phys}\leftarrow \text{base}_\text{phys}(1+\lambda P_1)$
- $w_\text{soc}\leftarrow \text{base}_\text{soc}\cdot\tfrac12\big[(1+\lambda P_3)g_3+(1+\lambda P_4)g_4\big]$
- $u_\text{safety}$ and $u_\text{growth}$ are added as new additive terms on the relevant actions.

**Satiation** (per satisfier action $a$): on use, $s_a\leftarrow s_a+0.25(1-s_a)$; otherwise $s_a\leftarrow 0.85\,s_a$ each tick. The marginal need gain is multiplied by $(1-s_a)$.

**Intention score** (doc §5.4):

$$\text{Score}(I)=\sum_k w_k\,\Delta d_k(I)+U_\text{first}(I)+\pi_I(\tau)-C_\text{switch}\,\mathbb 1[I\neq I_\text{cur}]$$

The agent switches only if $\text{Score}(I')>\text{Score}(I_\text{cur})+H$, with $H=0.1\,|\text{Score}(I_\text{cur})|+0.05$.

**Circadian prior**, with circular hour distance $\delta$: $\pi_I(\tau)=A_I\exp(-\delta(\tau,\mu_I)^2/2\sigma_I^2)$.

| Intention | Peak $\mu_I$ (hour) | $\sigma_I$ (hours) |
|---|---|---|
| WORK | 9 | 2.5 |
| MARKET | 13 | 1.5 |
| SOCIALIZE | 18 | 1.5 |
| REST | 1 | 3 |

Role tables can shift the peaks (e.g. Guard: PATROL at night).

**Intention set:** `SURVIVE_FOOD`, `SURVIVE_WATER`, `WORK(role)`, `MARKET`, `SOCIALIZE`, `RESPOND{SHELTER, INVESTIGATE, GREET}`, `REST`, plus `CIVIC/ILLICIT`, which stays gated by the existing risk model.

**How an intention runs.** Each intention is an ordered list of steps, each step = (target zone, allowed primitives). The next action is chosen with the existing `rank_candidate_actions`, restricted to (actions that serve the intention) ∪ {CONSUME_RATION, SHELTER}. That way crime or desperation can still break a routine by outscoring it.

### Pseudocode (per agent, per tick)
```
needs.update(obs, contacts, last_action)             # Phase 2e
w_t = dynamic_weights(needs, traits)                 # returns a new frozen weights object
if intention.done(obs) or intention.failed(obs) or age > max_duration:
    intention = None
best = argmax_I Score(I)
if intention is None or Score(best) > Score(intention) + H:
    intention = best.instantiate(step)
cands = evaluate_all_pathways(w_t) ∩ serves(intention) ∪ SAFETY
action = argmax(cands); trace.rationale += {needs, gates, intention_scores, step_index}
```

---

## 3. Ambiguities in the blueprint, and my proposed resolution

1. **λ isn't given.** I propose λ=2. The tests check behavior over λ ∈ {1,2,3}.
2. **L1 aggregation isn't specified.** I propose `min` (bottleneck) rather than `mean`. With `mean`, a dying-of-thirst but well-fed agent would look only half-deprived.
3. **L2 safety is only described in words.** The formula above is my operationalization.
4. **There is no `CONVERSE` action.** The conversation FSM belongs to Increment 4. For now, loneliness is relieved by GREET plus existing dialogue-bus events.
5. **Hysteresis H isn't specified.** I propose the relative-plus-absolute form above.

## 4. Contract changes (need your approval)

All changes are additive:
- New telemetry field per villager: `needs = {deficits[5], gates[5], weights{phys,soc,safety,growth}}`.
- The `intention` field gains `step_index`, `status`, `score`.
- `HumanlikeConfig` gains `enable_needs` and `enable_routines`. With both off, behavior is exactly as it is today.

---

## 5. Verification plan (oracles written independently of the implementation)

| Test | Oracle |
|---|---|
| Prepotency, hand-computed | $d=(0.9,0,0,0,0)$ → $g_3=0.1^{0.7}\approx0.1995$. A fed agent has $g_k=1$. |
| Monotonicity | $w_1$ strictly increases in $d_1$; $P$ is convex |
| Loneliness | An isolated agent's $d_3$ is non-decreasing and reaches 1 within the analytic bound. It adopts `SOCIALIZE`/GREET within N ticks. |
| Satiation | The marginal value of repeated GREET strictly decreases. No satisfier is used more than K consecutive ticks. |
| D1 lifecycle | `SHELTER` reaches DONE within 3 ticks after the storm ends. No intention outlives `max_duration`. |
| Hysteresis | Switch count over 200 ticks stays bounded; there is no 2-cycle flip-flop. |
| Daily rhythm (statistical) | Seeds 0-9, 3 simulated days (144 ticks). The lag-48 autocorrelation of each agent's zone occupancy has a mean > 0.3, with a bootstrap 95% CI reported. Eval seeds are separate from the tuning seeds. |
| Starving vs. fed | A starving agent's decision weight on L3/L4 terms is < 20% of a fed agent's |
| Utility scale drift | Logged per-term contribution: no single term drives > 60% of decisions (blueprint risk 5) |
| Determinism and ablation | Same seed → same hash. With everything off, the legacy 13 tests and the 24 Increment 1 tests stay green. |
| D2-D5 regressions | Demo output contains no literal claim strings; greetings > 0 in a seeded run; PAD pleasure drops on thunder |

Every change is checked with `pytest` (full suite), `ruff`, and `mypy`, with raw output shown.

## 6. Risks
- **Scale drift.** New terms could swamp `w_soc=14`. Mitigated by per-term contribution logging and rescaling.
- **Agents never commit, or never switch.** Tuning H is the lever; it has its own test.
- **Rhythm too rigid.** Routines could look scripted. The priors are soft and needs can override them.
- **Changed trajectories.** Existing tests may assume old behavior. If one fails, I'll find the root cause, not weaken the test.

## 7. Acceptance criteria
1. D1-D5 fixed, each with a regression test.
2. `needs.py`, `bt.py`, and `routines.py` are typed, registered, step-traceable, and have 10-section docs.
3. All §5 oracles pass; full suite, ruff, and mypy are green; raw output is shown.
4. `python -m autogen.examples.humanlike_village_demo --seed 42 --days 3` prints a per-agent hourly zone table, need deficits and gates, intention switches with scores, and only measured values.
5. Committed as one commit after verification.
