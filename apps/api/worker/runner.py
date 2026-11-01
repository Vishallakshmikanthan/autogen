"""Simulation worker executing decoupled simulation steps and emitting stream events (M19, SYS-18)."""

from __future__ import annotations

import threading
import time
from typing import Any

from apps.api.persistence.database import DatabaseManager
from apps.api.persistence.repositories.simulation_repo import SimulationRepository
from apps.api.worker.contracts import SimulationCommand, SimulationCommandType
from apps.api.worker.delta_encoder import DeltaEncoder
from apps.api.worker.queue import IJobQueue
from autogen.agents.baseline.random_agent import RandomAgent
from autogen.agents.baseline.simple_reflex_agent import SimpleReflexAgent
from autogen.agents.registry import AgentConfig, AgentRegistry
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.state.types import TerrainType
from autogen.visualization.adapter import ViewerDataAdapter
from autogen.visualization.contracts import ViewerState


class SimulationWorker:
    """Manages active simulation lifecycle, engine stepping, and event streaming."""

    def __init__(
        self,
        cmd_queue: IJobQueue,
        event_queue: IJobQueue,
        db_path: str = ":memory:",
        snapshot_cadence: int = 25,
    ) -> None:
        self.cmd_queue = cmd_queue
        self.event_queue = event_queue
        self.db_manager = DatabaseManager(db_path=db_path)
        self.snapshot_cadence = snapshot_cadence

        self.sim_id: str | None = None
        self.engine: SimulationEngine | None = None
        self.adapter: ViewerDataAdapter | None = None
        self.status: str = "IDLE"
        self.current_step: int = 0
        self.max_steps: int = 200
        self.speed: float = 1.0
        self.prev_viewer_state: ViewerState | None = None
        self._running_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def initialize_simulation(
        self,
        sim_id: str,
        width: int = 20,
        height: int = 20,
        obstacle_density: float = 0.15,
        resource_density: float = 0.08,
        seed: int = 42,
        max_steps: int = 200,
        agents_spec: list[dict[str, Any]] | None = None,
    ) -> None:
        """Sets up the GridWorld and SimulationEngine for a new session."""
        self.sim_id = sim_id
        self.max_steps = max_steps
        self.current_step = 0
        self.speed = 1.0

        terrain_weights = {
            TerrainType.GRASSLAND: max(0.1, 1.0 - obstacle_density - 0.2),
            TerrainType.FOREST: 0.20,
            TerrainType.DESERT: 0.0,
            TerrainType.WATER: 0.0,
            TerrainType.MOUNTAIN: 0.0,
            TerrainType.OBSTACLE: obstacle_density,
        }

        env_cfg = EnvironmentConfig(
            width=width,
            height=height,
            seed=seed,
            energy_max=100.0,
            terrain_weights=terrain_weights,
        )
        world = GridWorld(config=env_cfg)
        world.reset(seed=seed)

        self.engine = SimulationEngine(environment=world, seed=seed)
        specs = agents_spec or [{"agent_id": "agent_1", "agent_type": "simple_reflex"}]

        for spec in specs:
            aid = spec.get("agent_id", "agent_1")
            atype = spec.get("agent_type", "simple_reflex")
            world.spawn_agent(agent_id=aid, energy=100.0)
            try:
                agent_cfg = AgentConfig(
                    agent_type=atype,
                    agent_id=aid,
                    seed=seed,
                    custom_params=spec.get("policy_params", {}),
                )
                agent = AgentRegistry.create(agent_cfg)
            except Exception:
                if "random" in atype.lower():
                    agent = RandomAgent(agent_id=aid, seed=seed)
                else:
                    agent = SimpleReflexAgent(agent_id=aid)
            self.engine.register_agent(agent)

        self.adapter = ViewerDataAdapter(self.engine)
        self.status = "PAUSED"
        self.prev_viewer_state = self.adapter.generate_viewer_state()

        with self.db_manager.connection() as conn:
            repo = SimulationRepository(conn)
            repo.update_simulation_status(
                sim_id=self.sim_id,
                status=self.status,
                current_step=0,
                state_hash=self.prev_viewer_state.state_hash,
            )
            # Save initial snapshot
            repo.save_snapshot(
                sim_id=self.sim_id,
                step=0,
                state_hash=self.prev_viewer_state.state_hash,
                snapshot_dict_or_json=self.prev_viewer_state.model_dump(),
            )

        # Broadcast initial keyframe
        init_frame = DeltaEncoder.encode(None, self.prev_viewer_state, is_keyframe=True)
        self.event_queue.put({
            "event_type": "FRAME",
            "sim_id": self.sim_id,
            "step": 0,
            "delta": init_frame,
            "state_hash": self.prev_viewer_state.state_hash,
        })

    def step_once(self) -> dict[str, Any] | None:
        """Executes a single simulation tick deterministically."""
        if self.engine is None or self.adapter is None or self.sim_id is None:
            return None
        if self.current_step >= self.max_steps:
            self.status = "FINISHED"
            self._sync_db_status()
            return None

        # Execute 9-step simulation tick
        _world_state = self.engine.step()
        self.current_step += 1

        new_viewer_state = self.adapter.generate_viewer_state()
        is_keyframe = (self.current_step % 20 == 0) or (self.current_step == self.max_steps)
        delta = DeltaEncoder.encode(self.prev_viewer_state, new_viewer_state, is_keyframe=is_keyframe)
        self.prev_viewer_state = new_viewer_state

        # Snapshot persistence at cadence
        if self.current_step % self.snapshot_cadence == 0 or self.current_step == self.max_steps:
            with self.db_manager.connection() as conn:
                repo = SimulationRepository(conn)
                repo.save_snapshot(
                    sim_id=self.sim_id,
                    step=self.current_step,
                    state_hash=new_viewer_state.state_hash,
                    snapshot_dict_or_json=new_viewer_state.model_dump(),
                )

        if self.current_step >= self.max_steps:
            self.status = "FINISHED"

        self._sync_db_status()

        event = {
            "event_type": "FRAME",
            "sim_id": self.sim_id,
            "step": self.current_step,
            "delta": delta,
            "state_hash": new_viewer_state.state_hash,
        }
        self.event_queue.put(event)
        return event

    def _sync_db_status(self) -> None:
        if self.sim_id and self.prev_viewer_state:
            with self.db_manager.connection() as conn:
                repo = SimulationRepository(conn)
                repo.update_simulation_status(
                    sim_id=self.sim_id,
                    status=self.status,
                    current_step=self.current_step,
                    state_hash=self.prev_viewer_state.state_hash,
                )

    def handle_command(self, cmd: SimulationCommand) -> None:
        """Processes an incoming control command."""
        if cmd.command == SimulationCommandType.START:
            self.status = "RUNNING"
            self._sync_db_status()
        elif cmd.command == SimulationCommandType.PAUSE:
            self.status = "PAUSED"
            self._sync_db_status()
        elif cmd.command == SimulationCommandType.STEP:
            self.step_once()
        elif cmd.command == SimulationCommandType.RESET:
            if self.engine and self.sim_id:
                seed = cmd.params.get("seed", self.engine.seed)
                self.engine.environment.reset(seed=seed)
                self.current_step = 0
                self.status = "PAUSED"
                if self.adapter:
                    self.prev_viewer_state = self.adapter.generate_viewer_state()
                    init_frame = DeltaEncoder.encode(None, self.prev_viewer_state, is_keyframe=True)
                    self.event_queue.put({
                        "event_type": "FRAME",
                        "sim_id": self.sim_id,
                        "step": 0,
                        "delta": init_frame,
                        "state_hash": self.prev_viewer_state.state_hash,
                    })
                self._sync_db_status()
        elif cmd.command == SimulationCommandType.SET_SPEED:
            self.speed = max(0.1, float(cmd.params.get("speed", 1.0)))
        elif cmd.command == SimulationCommandType.STOP:
            self.status = "STOPPED"
            self._sync_db_status()

    def process_pending_commands(self) -> None:
        """Drains and executes all queued control commands without blocking."""
        while not self.cmd_queue.empty():
            try:
                cmd = self.cmd_queue.get(block=False)
                if isinstance(cmd, SimulationCommand):
                    self.handle_command(cmd)
            except Exception:
                break

    def run_worker_loop(self) -> None:
        """Continuous execution loop processing commands and ticking running simulations."""
        while not self._stop_event.is_set():
            self.process_pending_commands()

            if self.status == "RUNNING":
                self.step_once()
                # Throttled playback pacing
                delay = max(0.001, (1.0 / 30.0) / self.speed)
                time.sleep(delay)
            else:
                time.sleep(0.02)

    def start_background(self) -> None:
        """Launches worker loop in a background daemon thread."""
        self._stop_event.clear()
        self._running_thread = threading.Thread(target=self.run_worker_loop, daemon=True)
        self._running_thread.start()

    def stop_background(self) -> None:
        """Halts the background worker loop cleanly."""
        self._stop_event.set()
        if self._running_thread and self._running_thread.is_alive():
            self._running_thread.join(timeout=2.0)
