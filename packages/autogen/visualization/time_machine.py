"""Time Machine Controller: scrubber, replay navigation, and simulation forking."""

from __future__ import annotations

from autogen.simulation.core.contracts import Action
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.core.snapshot import SimulationSnapshot
from autogen.visualization.contracts import ViewerTimeMachineState


class TimeMachineController:
    """Controls simulation execution, step playback, rewind scrubbing, and forking."""

    def __init__(
        self,
        engine: SimulationEngine,
        snapshot_cadence: int = 10,
    ) -> None:
        self.engine = engine
        self.snapshot_cadence = snapshot_cadence
        self.is_paused = True
        self.playback_speed = 1.0

        # Snapshot history index: step -> snapshot
        self.snapshots: dict[int, SimulationSnapshot] = {}
        self.max_step_reached = 0
        # Initial snapshot at tick 0
        self.record_snapshot()

    def record_snapshot(self) -> None:
        """Saves current engine state as an immutable snapshot."""
        current_step = self.engine.environment.state.step
        self.snapshots[current_step] = self.engine.snapshot()
        if current_step > self.max_step_reached:
            self.max_step_reached = current_step

    def play(self) -> None:
        self.is_paused = False

    def pause(self) -> None:
        self.is_paused = True

    def toggle_play_pause(self) -> None:
        self.is_paused = not self.is_paused

    def set_speed(self, multiplier: float) -> None:
        if multiplier <= 0.0:
            raise ValueError(f"Playback speed must be > 0, got {multiplier}")
        self.playback_speed = multiplier

    def step_forward(self) -> int:
        """Executes one simulation step forward and records snapshots on cadence."""
        state = self.engine.step()
        if state.step % self.snapshot_cadence == 0:
            self.record_snapshot()
        if state.step > self.max_step_reached:
            self.max_step_reached = state.step
        return state.step

    def scrub_to(self, target_step: int) -> int:
        """Restores engine to target_step using snapshot keyframes and historical actions."""
        target = max(0, min(target_step, self.max_step_reached))
        current_step = self.engine.environment.state.step

        if target == current_step:
            return current_step

        # Find closest snapshot keyframe <= target
        available_keys = [s for s in sorted(self.snapshots.keys()) if s <= target]
        if not available_keys:
            return current_step

        best_key = available_keys[-1]
        self.engine.restore(self.snapshots[best_key])

        # Step forward until target is reached
        while self.engine.environment.state.step < target:
            self.engine.step()

        return self.engine.environment.state.step

    def step_backward(self) -> int:
        """Rewinds the simulation by one tick."""
        current_step = self.engine.environment.state.step
        return self.scrub_to(current_step - 1)

    def fork(
        self,
        target_step: int | None = None,
        overridden_actions: dict[str, Action] | None = None,
    ) -> TimeMachineController:
        """Forks the simulation from target_step, creating an independent diverging branch."""
        target = target_step if target_step is not None else self.engine.environment.state.step
        # Ensure we are at target step
        self.scrub_to(target)

        # Create forked engine from current snapshot
        snap = self.engine.snapshot()
        forked_engine = self.engine.fork(
            snapshot=snap,
            overridden_actions=overridden_actions,
        )
        return TimeMachineController(
            engine=forked_engine,
            snapshot_cadence=self.snapshot_cadence,
        )

    def get_state(self) -> ViewerTimeMachineState:
        """Constructs serializable time machine state representation."""
        current_step = self.engine.environment.state.step
        return ViewerTimeMachineState(
            current_step=current_step,
            total_steps=self.max_step_reached,
            available_snapshots=sorted(self.snapshots.keys()),
            is_paused=self.is_paused,
            playback_speed=self.playback_speed,
            can_fork=True,
        )
