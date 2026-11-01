"""Runnable CLI demonstration of AUTOGEN Village Society & Emergent Economy."""

from __future__ import annotations

from autogen.agents.village.controller import VillageSimulationController
from autogen.research.village_experiments import run_all_village_experiments
from autogen.simulation.environment.village.village_grid import VillageWorldConfig


def main() -> None:
    print("=" * 72)
    print("AUTOGEN AI — SIMULATED VILLAGE SOCIETY & EMERGENT ECONOMY DEMO")
    print("=" * 72)

    controller = VillageSimulationController(
        world_config=VillageWorldConfig(width=40, height=40, seed=42),
        enable_llm_persona=False,
    )

    state = controller.get_full_state_payload()
    print(f"Initialized Village (40x40) | Villagers: {len(state['villagers'])} | Seed: 42")

    for _ in range(8):
        state = controller.step()

    print(f"\n--- Step {state['step']} Snapshot (Time of Day: {state['time_of_day']:.1f}h) ---")
    print(f"Canonical State Hash : {state['state_hash'][:24]}...")
    print(f"Granary Storage      : {state['economy']['granary_storage']}")
    print(f"Market Clearing Price: {state['economy']['clearing_prices']}")
    print("\nTop 5 Villagers by PageRank Societal Status S_i(t):")
    for rank, entry in enumerate(state["social_graph"]["leaderboard"][:5], start=1):
        aid = entry["agent_id"]
        v_card = next(v for v in state["villagers"] if v["agent_id"] == aid)
        print(
            f"  {rank}. {v_card['name']:<8} ({v_card['role']:<13}) | "
            f"S_i={v_card['societal_status']:.4f} | Coins={v_card['coins']:>5.1f} | "
            f"Hunger={v_card['hunger']:>5.1f} | Action={v_card['last_action']}"
        )

    print("\nRecent Spatial Dialogue & Gossip Bus:")
    for ev in state["dialogue_events"][-4:]:
        print(f"  [{ev['event_type']}] {ev['speaker_name']}: \"{ev['utterance']}\"")

    print("\n--- Running Phase 6 Empirical Research Scenarios ---")
    experiments = run_all_village_experiments()
    for exp in experiments:
        print(f"  * {exp['title']}: {exp['conclusion']}")

    print("=" * 72)


if __name__ == "__main__":
    main()

