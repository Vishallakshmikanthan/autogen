# Viewer Architecture & Time Machine (Module M05)

> **Document Status:** Complete  
> **Blueprint Reference:** Phase 01 / SYS-18 (Viewer & Visualization Pipeline v0)  
> **Associated Gate:** Gate 1 (Multi-Agent Simulation Platform)

---

## 1. Architectural Overview & Consumer Sovereignty

In AUTOGEN AI, visualization adheres strictly to the **Decoupled Consumer Pattern**:

```
┌─────────────────────────────────────────────────────────────┐
│                    Simulation Engine                        │
│   (Discrete-time deterministic 9-phase resolution loop)     │
└──────────────────────────────┬──────────────────────────────┘
                               │ State / Events / Telemetry
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    ViewerDataAdapter                        │
│     (Render-free serialization into ViewerState DTO)        │
└──────────────┬───────────────────────────────┬──────────────┘
               │                               │
               ▼ (In-process DTO)              ▼ (JSON over WebSocket)
┌──────────────────────────────┐ ┌────────────────────────────┐
│      PygameViewer (M05)      │ │   M20 React Studio         │
│  - Split-screen Viewports    │ │   - Canvas / WebGL         │
│  - Decision Trace Inspector  │ │   - Real-time Telemetry    │
│  - Time Machine Scrubber     │ │   - Interactive Graph      │
└──────────────────────────────┘ └────────────────────────────┘
```

### Sovereign Principles
1. **Zero Write Authority:** The viewer cannot directly modify world or agent states. All mutations flow exclusively through the engine's validated transition pipeline.
2. **AI-Logic Decoupling:** The viewer never executes deliberation, heuristic evaluations, or perception code. It displays state reported by the engine and agents.
3. **Decoupled Frame Rate:** Simulation tick rate and rendering FPS are independent. The simulation can advance at 10,000 ticks/sec headlessly, or single-step at 1 frame per 5 seconds.
4. **Headless Safety:** Neither `ViewerDataAdapter` nor `TimeMachineController` import GUI libraries. In headless mode or CI/CD pipelines, visual state generation executes with 100% fidelity without a video display.

---

## 2. ViewerState Data Transfer Object (DTO)

All visualization data is packaged in the immutable `ViewerState` Pydantic model (`packages/autogen/visualization/contracts.py`).

| DTO Model | Responsibility |
|---|---|
| `ViewerCellState` | Terrain type, resource levels (food/water), hazards, walkability. |
| `ViewerAgentState` | Agent coordinates $(x, y)$, energy, health, alive status, orientation. |
| `ViewerOverlayData` | Fog-of-war visible coordinates, movement trails ($N$ past steps), heatmaps, collision markers. |
| `ViewerTimeMachineState` | Current tick $t$, max tick reached $T_{\max}$, snapshot keyframe steps, pause/speed status. |
| `ViewerAgentMindCell` | Mental map state of selected agent: estimated terrain, food, and confidence $\gamma \in [0, 1]$. |
| `ViewerState` | Root envelope containing world metrics, state hash, selected trace, and all sub-models. |

Because `ViewerState` is a Pydantic model with frozen primitives and standard containers, it serializes losslessly to JSON via `viewer_state.model_dump_json()`. This guarantees that Phase 2's M20 React/TypeScript studio consumes the identical schema over WebSocket.

---

## 3. Time Machine: Scrubber, Keyframes & Forking

The `TimeMachineController` (`packages/autogen/visualization/time_machine.py`) enables non-linear temporal navigation through a simulation run.

### 3.1 Keyframe Indexing and Scrubbing
To avoid unbounded memory consumption while preserving exact instant rewind:
1. Immutable `SimulationSnapshot` instances are saved periodically at a configured cadence (default: every 5 or 10 ticks).
2. When the user scrubs backwards to step $t_{\text{target}}$:
   - The controller finds the greatest snapshot keyframe $t_{\text{snap}} \le t_{\text{target}}$.
   - It restores the engine to $t_{\text{snap}}$ via `engine.restore(snapshot)`.
   - If $t_{\text{snap}} < t_{\text{target}}$, the engine replays recorded historical actions forward until $t_{\text{target}}$ is reached.

### 3.2 Counterfactual Forking ("Fork Here")
At any historical tick $t_{\text{fork}}$:
- The user can fork the simulation via `time_machine.fork(target_step=t_fork, overridden_actions={...})`.
- The engine creates a cloned instance seeded from the snapshot at $t_{\text{fork}}$.
- If an overridden action is specified, the target agent executes the alternate action at $t_{\text{fork}}$.
- The forked simulation immediately branches into an alternate deterministic timeline, demonstrating counterfactual divergence.

---

## 4. PygameViewer Viewports & Inspector

The `PygameViewer` (`packages/autogen/visualization/pygame_viewer.py`) provides an interactive 4-region GUI layout:

```
+----------------------------------------------------------------------------------------------------+
|  AUTOGEN RESEARCH STUDIO | Step 142 | Hash a7f9d8b1                                               |
+------------------------------------+------------------------------------+--------------------------+
|                                    |                                    | DECISION TRACE INSPECTOR |
|       GROUND TRUTH WORLD           |           AGENT MIND               | Agent: agent_00          |
|                                    |         (Fog of War)               | Mode: EXPLORE            |
|   - True terrain tiles             |   - Unseen cells obscured          | Action: MOVE (SOUTH)     |
|   - True resource levels           |   - Perceived food / hazards       | Utility: 0.84            |
|   - All active agents              |   - Spatial confidence shading     | Memory: Recency=0.92     |
|   - Movement trails                |   - Mental map belief              | Candidates:              |
|                                    |                                    |   MOVE (S): 0.84         |
|                                    |                                    |   COLLECT:  0.00         |
|                                    |                                    |   STAY:     0.10         |
+------------------------------------+------------------------------------+--------------------------+
|  TIME MACHINE SCRUBBER & CONTROLS                                                                  |
|  [<<] [|<] [ > / || ] [>|] [>>]  Speed: 1.0x  |  Step: [============|----------------] 142 / 300    |
|  Keys: [SPACE] Pause | [LEFT/RIGHT] Step | [UP/DOWN] Speed | [TAB] Cycle Agent | [F/T/H/C] Toggles |
+----------------------------------------------------------------------------------------------------+
```

### 4.1 Split-Screen Viewports
- **Ground Truth (Left):** Renders the objective grid world as resolved by the simulation engine. Displays actual terrain colors, food density, and all living agents.
- **Agent Mind (Center):** Renders the subjective perception and memory of the currently selected agent. Unvisited regions outside the agent's field-of-view are rendered as dark Fog of War. Explored tiles are rendered using the agent's internal spatial memory with alpha proportional to confidence $\gamma(x, y)$.

### 4.2 Decision Trace Inspector (Right)
Displays the real-time `DecisionTrace` produced by the selected agent's deliberation logic:
- Internal mental state / active goals.
- Sensory observations (detected food, obstacles, neighbors).
- Evaluated candidate actions with computed heuristic/utility scores.
- Chosen action and formal reason string.
- Memory lookups utilized in the decision.

### 4.3 Interactive Overlays
- **Fog of War (`F`):** Toggles sensory visibility masking.
- **Movement Trails (`T`):** Displays fading historical path segments for all agents.
- **Heatmaps (`H`):** Renders food concentration and danger zones.
- **Collision Markers (`C`):** Highlights locations where attempted illegal actions or conflicts occurred.
