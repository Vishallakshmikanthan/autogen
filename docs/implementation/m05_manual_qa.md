# M05 Manual QA Verification Checklist

> **Module:** M05 — Pygame Viewer, Decision Trace Mode, Replay (Gate 1 Completion)  
> **Target Application:** `apps/viewer/run_viewer.py`

This checklist details manual verification steps to confirm visual fidelity, interactive controls, split-screen synchronization, and time-machine scrubbing.

---

## 1. Environment & Setup Verification

- [ ] **Dependency Check:** Run `uv pip list | grep pygame` to verify Pygame is installed.
- [ ] **Headless Verification:** Run `uv run python apps/viewer/run_viewer.py --headless --ticks 50`.
  - *Expected:* Clean console output with tick metrics, zero GUI window opened, zero crashes, exit code 0.
- [ ] **Interactive Launch:** Run `uv run python apps/viewer/run_viewer.py --width 20 --height 20 --agents 4 --seed 42`.
  - *Expected:* Pygame window opens at 1280x720 resolution with title `AUTOGEN Research Studio — Simulation Viewer (M05)`.

---

## 2. Split-Screen & Inspector Viewports

- [ ] **Ground Truth Viewport (Left):**
  - Verify distinct terrain palette colors: Grassland (green), Forest (dark green), Desert (yellow/tan), Water (blue), Mountain (gray), Obstacle (dark charcoal).
  - Verify agents render with directional glyphs and unique agent ID labels.
  - Verify resource dots render on tiles with food > 0.
- [ ] **Agent Mind Viewport (Center):**
  - Verify unseen tiles outside the agent's field of view are obscured in dark Fog-of-War (`#0B0E14`).
  - Verify visited tiles show estimated terrain with brightness proportional to confidence.
  - Verify perceived food matches the selected agent's memory.
- [ ] **Decision Trace Inspector (Right):**
  - Verify panel displays selected agent ID (`agent_00`).
  - Verify sensory context lists observed cells and agent status.
  - Verify candidate actions list scores (e.g. `MOVE_NORTH: 0.85`, `COLLECT: 0.00`).
  - Verify chosen action, deliberation mechanism, and reason string are displayed.
  - Verify trace updates instantaneously on every simulation step.

---

## 3. Interactive Keyboard & Mouse Controls

- [ ] **Agent Selection:**
  - Press `TAB` repeatedly: verify the selected agent cycles through all alive agents, and the Agent Mind / Decision Trace panels immediately switch to that agent.
  - Click on an agent tile on the Ground Truth grid: verify that agent is selected immediately.
- [ ] **Playback Controls:**
  - Press `SPACE`: verify simulation pauses/resumes immediately.
  - When paused, press `RIGHT ARROW`: verify simulation advances exactly 1 step.
  - When paused, press `LEFT ARROW`: verify simulation rewinds exactly 1 step.
  - Press `UP ARROW`: verify playback speed doubles (e.g. 1.0x -> 2.0x -> 4.0x).
  - Press `DOWN ARROW`: verify playback speed halves (e.g. 2.0x -> 1.0x -> 0.5x).
- [ ] **Overlay Toggles:**
  - Press `F`: verify Fog-of-War overlay toggles on/off.
  - Press `T`: verify agent movement trails toggle on/off.
  - Press `H`: verify food/danger heatmaps toggle on/off.
  - Press `C`: verify collision / illegal action markers toggle on/off.

---

## 4. Time Machine & Counterfactual Forking

- [ ] **Scrubber Rewind:**
  - Run simulation to step 50.
  - Pause via `SPACE`.
  - Repeatedly press `LEFT ARROW` or scrub backwards across a snapshot keyframe (e.g. tick 30 -> tick 20 -> tick 10):
    - Verify agent positions, food levels, and state hash match the exact historical values at those ticks.
- [ ] **Scrubber Forward:**
  - Step forward via `RIGHT ARROW` to tick 40:
    - Verify states re-converge identically to the original timeline.
- [ ] **Forking Verification:**
  - Scrub backwards to tick 25.
  - Trigger fork via CLI or test harness:
    - Verify the forked run shares identical state hashes up to tick 25, and diverges thereafter.

---

## 5. Gate 1 Exit Invariant Check

- [ ] **Invariant 1 (Decoupled Consumer):** The viewer contains zero simulation math or AI logic; killing the viewer does not corrupt simulation state.
- [ ] **Invariant 2 (Replay Determinism):** Replaying with identical actions and seed produces bitwise-identical world hashes across all ticks.
- [ ] **Invariant 3 (Agent Mind Isolation):** The Agent Mind viewport displays strictly what the agent observed/remembered, never ground-truth unobserved tiles.

