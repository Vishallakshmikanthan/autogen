"""AUTOGEN Simulation Viewer CLI Application (SYS-18 v0, Gate 1).

Runs the multi-agent simulation with the interactive Pygame viewer, decision trace inspector,
split-screen agent mind view, and time machine scrubber.
"""

from __future__ import annotations

import argparse
import sys

from autogen.agents import (
    BaseAgent,
    MemoryAugmentedAgent,
    ModelBasedReflexAgent,
    SimpleReflexAgent,
)
from autogen.memory.contracts import MemoryConfig
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.visualization.adapter import ViewerDataAdapter
from autogen.visualization.pygame_viewer import PygameViewer
from autogen.visualization.time_machine import TimeMachineController


def build_simulation(
    width: int = 20,
    height: int = 20,
    num_agents: int = 4,
    seed: int = 42,
) -> tuple[SimulationEngine, ViewerDataAdapter, TimeMachineController]:
    """Initializes grid environment, agents, engine, data adapter, and time machine."""
    config = EnvironmentConfig(
        width=width,
        height=height,
        seed=seed,
    )
    world = GridWorld(config)
    world.reset(seed=seed)
    engine = SimulationEngine(world)

    # Spawn heterogeneous agents
    for i in range(num_agents):
        agent_id = f"agent_{i:02d}"
        world.spawn_agent(agent_id=agent_id, energy=100.0)

        agent: BaseAgent
        if i == 0:
            # Memory-augmented agent
            base_agent = SimpleReflexAgent(agent_id=agent_id)
            mem_cfg = MemoryConfig(
                spatial_memory_capacity=100,
                episodic_memory_capacity=100,
            )
            agent = MemoryAugmentedAgent(base_agent, config=mem_cfg)
        elif i % 2 == 1:
            agent = ModelBasedReflexAgent(agent_id=agent_id)
        else:
            agent = SimpleReflexAgent(agent_id=agent_id)

        engine.register_agent(agent)

    adapter = ViewerDataAdapter(engine)
    time_machine = TimeMachineController(engine, snapshot_cadence=5)

    # Initial time machine snapshot at tick 0
    time_machine.record_snapshot()
    adapter.update_trails()

    return engine, adapter, time_machine


def run_viewer_cli(args: list[str] | None = None) -> int:
    """CLI entry point for simulation viewer."""
    parser = argparse.ArgumentParser(
        description="AUTOGEN Simulation Viewer and Decision Trace Inspector (M05)"
    )
    parser.add_argument("--width", type=int, default=20, help="Grid width (default: 20)")
    parser.add_argument("--height", type=int, default=20, help="Grid height (default: 20)")
    parser.add_argument("--agents", type=int, default=4, help="Number of agents (default: 4)")
    parser.add_argument("--seed", type=int, default=42, help="Simulation PRNG seed (default: 42)")
    parser.add_argument(
        "--ticks",
        type=int,
        default=100,
        help="Max ticks to run in headless mode (default: 100)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying a graphical window",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Target frames per second for viewer (default: 15)",
    )

    parsed = parser.parse_args(args)

    print(
        f"Initializing AUTOGEN Simulation: {parsed.width}x{parsed.height} grid, "
        f"{parsed.agents} agents, seed={parsed.seed}"
    )
    engine, adapter, time_machine = build_simulation(
        width=parsed.width,
        height=parsed.height,
        num_agents=parsed.agents,
        seed=parsed.seed,
    )

    if parsed.headless:
        print(f"Running headless simulation for {parsed.ticks} ticks...")
        time_machine.play()
        for t in range(parsed.ticks):
            step = time_machine.step_forward()
            adapter.update_trails()
            # Capture latest traces from engine agents
            for _aid, agent in engine.agents.items():
                if hasattr(agent, "latest_trace") and agent.latest_trace:
                    adapter.record_decision_trace(agent.latest_trace)

            v_state = adapter.generate_viewer_state(
                selected_agent_id="agent_00",
                time_machine_state=time_machine.get_state(),
            )
            if (t + 1) % 20 == 0 or t == parsed.ticks - 1:
                print(
                    f"Tick {step:03d} | State Hash: {v_state.state_hash[:12]} | "
                    f"Agents Alive: {len([a for a in v_state.agents if a.is_alive])}"
                )
        print("Headless simulation run complete.")
        return 0

    # Interactive GUI mode
    try:
        import pygame
    except ImportError:
        print(
            "ERROR: Pygame is not installed. Run 'uv pip install pygame' or use --headless mode.",
            file=sys.stderr,
        )
        return 1

    viewer = PygameViewer(
        adapter=adapter,
        time_machine=time_machine,
        headless=False,
    )
    viewer.selected_agent_id = "agent_00"

    clock = pygame.time.Clock()
    print("Launching Pygame Viewer. Controls:")
    print("  SPACE       : Play / Pause")
    print("  RIGHT / LEFT: Single Step Forward / Rewind")
    print("  UP / DOWN   : Increase / Decrease Speed")
    print("  TAB / CLICK : Select Agent to Inspect Mind / Decision Trace")
    print("  F / T / H / C: Toggle Fog-of-War / Trails / Heatmaps / Collisions")

    try:
        while True:
            # Sync traces from engine agents
            for _aid, agent in engine.agents.items():
                if hasattr(agent, "latest_trace") and agent.latest_trace:
                    adapter.record_decision_trace(agent.latest_trace)

            adapter.update_trails()
            viewer.step()
            clock.tick(parsed.fps)
    except KeyboardInterrupt:
        print("\nViewer exited by user.")
    finally:
        pygame.quit()

    return 0


if __name__ == "__main__":
    sys.exit(run_viewer_cli())
