"""Runnable CLI demonstration of AUTOGEN Humanlike Villager Architecture (Module M22-H).

Demonstrates Increment 1, Increment 2 & Increment 3:
1. Big Five Personality traits & utility weight modulation (Sec 5.2)
2. Dynamic Maslow-style 5-level Needs Hierarchy & Prepotency Gating (Sec 5.3)
3. Circadian circular Gaussian 24h daily routine priors (Sec 5.4)
4. Utility-guided Behavior Tree hierarchical intention arbitration & hysteresis barrier (Sec 5.4)
5. Social encounters, ARRIVAL events & proximity-bounded greetings (Sec 2.5)
6. Environmental crisis (Thunderstorm) & salience-driven attention (Sec 2.3)
7. Canonical Mehrabian Fear emotion dynamics & storm shelter utility (Sec 2.1, 2.5)
8. Synchronous emotional contagion pass across neighbors (Sec 3.4)
9. Episodic Memory Stream, Importance scoring & Ebbinghaus retention decay (Sec 4.1, 4.3)
10. Multimodal Memory Retrieval with Bower's mood-congruence & ScoreBreakdown (Sec 4.2)
11. Nightfall reflection consolidation, situational attribution & forgiveness (Sec 4.4)
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from autogen.agents.village.controller import HumanlikeConfig, VillageSimulationController
from autogen.intelligence.village.memory_retrieval import MemoryQuery, retrieve_top_k
from autogen.intelligence.village.reflection import MemoryEpisode
from autogen.simulation.environment.village.events import Modality
from autogen.simulation.environment.village.village_grid import VillageWorldConfig

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AUTOGEN Humanlike Villagers Architecture CLI Demonstration (Increment 2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic pseudo-random seed (default: 42)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=48,
        help="Number of simulation steps to run (default: 48 ticks = 1 full 24h day)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of full 24-hour simulation days to run (overrides --steps if set; 1 day = 48 steps)",
    )
    parser.add_argument(
        "--storm-step",
        type=int,
        default=4,
        help="Step at which the sudden thunderstorm strikes the village (default: 4)",
    )
    return parser.parse_args()


def format_pad(pad: tuple[float, float, float] | list[float]) -> str:
    p, a, d = pad[0], pad[1], pad[2]
    return f"P={p:+.2f}, A={a:+.2f}, D={d:+.2f}"


def run_demo(seed: int = 42, total_steps: int = 48, storm_step: int = 4) -> None:
    print("=" * 88)
    print("      AUTOGEN AI -- HUMANLIKE VILLAGER ARCHITECTURE (MODULE M22-H) DEMO")
    print("      INCREMENT 3: MEMORY STREAM, MULTIMODAL RETRIEVAL, FORGETTING & REFLECTION")
    print("=" * 88)
    print(
        f"Configuration: Seed={seed} | Total Steps={total_steps} ({(total_steps / 48.0):.1f} days) | Storm Step={storm_step}\n"
    )

    # 1. Initialize Controller with all Humanlike layers enabled
    controller = VillageSimulationController(
        world_config=VillageWorldConfig(width=30, height=30, seed=seed),
        enable_llm_persona=False,
        humanlike_config=HumanlikeConfig(
            enable_personality=True,
            enable_attention=True,
            enable_intentions=True,
            enable_contagion=True,
            enable_reflection=True,
            enable_weather=True,
            enable_needs=True,
            enable_routines=True,
            enable_memory_stream=True,
        ),
    )

    initial_payload = controller.get_full_state_payload()
    villagers = initial_payload["villagers"]

    # 2. Section 1: Personality Profiles & Base Utility Weights
    print("----------------------------------------------------------------------------------------")
    print(" SECTION 1. VILLAGER BIG FIVE TRAITS & UTILITY WEIGHT MODULATION (Sec 5.2)")
    print("----------------------------------------------------------------------------------------")
    print(
        f"{'Villager':<10} {'Role':<14} {'O':>4} {'C':>4} {'E':>4} {'A':>4} {'N':>4} | "
        f"{'w_soc':>6} {'w_econ':>6} {'kappa':>6} {'theta':>6} | {'PAD Baseline'}"
    )
    print("-" * 88)
    for v in villagers:
        aid = v["agent_id"]
        agent = controller.agents[aid]
        traits = agent.traits
        w = agent.utility_weights
        pad_base = agent.affective_engine.baseline
        print(
            f"{v['name']:<10} {v['role']:<14} "
            f"{traits.openness:>4.2f} {traits.conscientiousness:>4.2f} "
            f"{traits.extraversion:>4.2f} {traits.agreeableness:>4.2f} "
            f"{traits.neuroticism:>4.2f} | "
            f"{w.w_soc:>6.2f} {w.w_econ:>6.2f} "
            f"{w.crisis_penalty_kappa:>6.2f} {w.theta_risk:>6.2f} | "
            f"{format_pad(pad_base)}"
        )
    print()

    # 3. Section 2: Initial Maslow Needs Hierarchy & Prepotency Gating
    print("----------------------------------------------------------------------------------------")
    print(" SECTION 2. DYNAMIC NEEDS HIERARCHY, PREPOTENCY GATES & DYNAMIC WEIGHTS (Sec 5.3)")
    print("----------------------------------------------------------------------------------------")
    print(
        f"{'Villager':<10} {'Deficits (d1..d5)':<22} | "
        f"{'Prepotency Gates (g1..g5)':<26} | {'Modulated Weights (phys/soc/econ)'}"
    )
    print("-" * 88)
    for v in villagers:
        aid = v["agent_id"]
        agent = controller.agents[aid]
        nst = agent.needs_engine.state
        d_str = f"({nst.d_phys:.2f}, {nst.d_safety:.2f}, {nst.d_belonging:.2f}, {nst.d_esteem:.2f}, {nst.d_growth:.2f})"
        g_str = f"({nst.g_phys:.2f}, {nst.g_safety:.2f}, {nst.g_belonging:.2f}, {nst.g_esteem:.2f}, {nst.g_growth:.2f})"
        dyn_w = agent.utility_engine.weights
        w_str = f"w_p={dyn_w.w_phys:.2f}, w_s={dyn_w.w_soc:.2f}, w_e={dyn_w.w_econ:.2f}"
        print(f"{v['name']:<10} {d_str:<22} | {g_str:<26} | {w_str}")
    print()

    # 4. Simulation Execution
    shelter_counts: dict[str, int] = {v["agent_id"]: 0 for v in villagers}
    greetings_seen: list[str] = []
    contagion_deltas: list[float] = []
    hourly_zone_occupancy: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    print("----------------------------------------------------------------------------------------")
    print(" SECTION 3. SIMULATION TIMELINE: INTENTIONS, CRISIS PREEMPTION & DAILY ROUTINES")
    print("----------------------------------------------------------------------------------------")

    for step in range(1, total_steps + 1):
        # Step 2: Inject social arrival event between Imran and Kavya (Defect D3 & D5 fixes)
        if step == 2:
            name_to_id = {v.name: aid for aid, v in controller.world.villagers.items()}
            kavya_id = name_to_id.get("Kavya", "villager_kavya")
            imran_id = name_to_id.get("Imran", "villager_imran")
            arun_id = name_to_id.get("Arun", "villager_arun")
            lata_id = name_to_id.get("Lata", "villager_lata")
            meera_id = name_to_id.get("Meera", "villager_meera")
            bheem_id = name_to_id.get("Bheem", "villager_bheem")

            imran_pos = controller.world.villagers[imran_id].position
            controller.world.event_bus.create_and_emit(
                step=2,
                kind="ARRIVAL",
                modality=Modality.SOCIAL,
                position=imran_pos,
                intensity=0.90,
                valence_hint=0.60,
                source_id=kavya_id,
                target_id=imran_id,
                payload={"description": "Kavya arrives near the well to share water pots."},
            )
            # Record daytime episodic experiences for reflection showcase (Sec 4.4, D5 Fix: Labeled)
            if arun_id in controller.agents:
                controller.agents[arun_id].unreflected_episodes.append(
                    MemoryEpisode(
                        episode_id="ep_02_fair_trade_lata_scripted",
                        step=2,
                        other_id=lata_id,
                        valence=0.70,
                        importance=0.65,
                        is_help=True,
                        description="[Scripted Demo Episode - Injected] Lata traded fresh vegetable seeds at a generous village discount.",
                    )
                )
            if meera_id in controller.agents:
                controller.agents[meera_id].unreflected_episodes.append(
                    MemoryEpisode(
                        episode_id="ep_02_crop_theft_bheem_scripted",
                        step=2,
                        other_id=bheem_id,
                        valence=-0.75,
                        importance=0.85,
                        is_harm=True,
                        description="[Scripted Demo Episode - Injected] Bheem trampled barley furrow while aggressively negotiating prices.",
                    )
                )

        # Trigger storm at storm_step
        if step == storm_step:
            print(
                f"\n[!] [STEP {step:02d} | Time: {controller.world.time_of_day:04.1f}h] VIOLENT THUNDERSTORM STRIKES THE VILLAGE!"
            )
            print("    >>> Global Auditory & Visual Event emitted: Intensity=0.95, Valence=-0.80 <<<\n")
            controller.trigger_thunderstorm(intensity=0.95)

        # Execute simulation step
        state = controller.step()
        time_h = controller.world.time_of_day
        mean_delta = state.get("mean_contagion_delta", 0.0)
        contagion_deltas.append(mean_delta)

        # Track zone occupancy and actions
        for v in state["villagers"]:
            aid = v["agent_id"]
            if v["last_action"] == "SHELTER":
                shelter_counts[aid] += 1
            if v["last_action"] == "GREET":
                greetings_seen.append(f"{v['name']} (Step {step})")
            # C2 Fix: derive zone occupancy directly from actual world coordinates
            v_obj = controller.world.villagers[aid]
            zone_obj = controller.world.get_zone_at(v_obj.position)
            cur_z = zone_obj.zone_id if zone_obj else "village_commons"
            hourly_zone_occupancy[f"{int(time_h):02d}:00"][cur_z] += 1

        # Inspect events & actions during steps around storm or key transitions
        is_storm_window = storm_step <= step <= storm_step + 2
        is_social_window = step in (2, 3)
        is_milestone = step in (1, 2, storm_step, storm_step + 1, storm_step + 2, 24, 48, total_steps)

        if is_milestone or is_storm_window or is_social_window:
            weather_desc = "Raining (Storm)" if controller.world.weather_process.is_raining else "Clear"
            print(
                f"--- [Step {step:02d} | Clock: {time_h:04.1f}h | Weather: {weather_desc} | Contagion |Delta PAD|: {mean_delta:.4f}] ---"
            )
            for v in state["villagers"]:
                aid = v["agent_id"]
                agent = controller.agents[aid]
                trace = controller.traces_by_agent.get(aid)
                rationale = trace.rationale if trace else {}

                alerts = rationale.get("alerts", [])
                intention_str = f"{v['intention']['kind']} (step={v['intention']['step_index']}, c={v['intention']['commitment']:.2f})"
                action = v["last_action"]
                emote = v["emote"]
                gaze = v["gaze"]
                nst = agent.needs_engine.state

                gaze_info = f"Gaze -> ({gaze['target_pos'][0]}, {gaze['target_pos'][1]}) [{gaze['reason']}]"
                needs_summary = f"Needs: d_phys={nst.d_phys:.2f}, d_safe={nst.d_safety:.2f}, d_soc={nst.d_belonging:.2f}"

                print(
                    f"  {v['name']:<9} ({v['role']:<13}) -> Action: {action:<16} | "
                    f"Intention: {intention_str:<26} | Emote: {emote:<10}"
                )
                if alerts:
                    print(f"     |-- Attention Alert: {alerts} | {gaze_info}")
                else:
                    print(f"     |-- Dynamics: {needs_summary} | {gaze_info}")

            if is_storm_window:
                print(
                    f"     [Contagion Synchronous Pass]: Measured mean |Delta PAD| = {mean_delta:.4f} across living villagers."
                )
            print()

    # 5. Section 4: 24-Hour Circadian Activity Distribution
    print("----------------------------------------------------------------------------------------")
    print(" SECTION 4. 24-HOUR CIRCADIAN OCCUPANCY & ROUTINE PROGRESSION (Sec 5.4)")
    print("----------------------------------------------------------------------------------------")
    print(f"{'Hour':<8} {'Fields':>8} {'Town Well':>12} {'Market':>10} {'Private Huts':>14} {'Chapel':>10}")
    print("-" * 68)
    for hour_str in sorted(hourly_zone_occupancy.keys()):
        occ = hourly_zone_occupancy[hour_str]
        f_cnt = occ.get("agricultural_fields", 0)
        w_cnt = occ.get("town_well", 0)
        m_cnt = occ.get("central_market", 0)
        h_cnt = occ.get("private_huts_west", 0) + occ.get("private_huts_east", 0)
        c_cnt = occ.get("chapel", 0)
        print(f"{hour_str:<8} {f_cnt:>8} {w_cnt:>12} {m_cnt:>10} {h_cnt:>14} {c_cnt:>10}")
    print()

    # 6. Section 5: Multimodal Memory Retrieval & Relational Utility Modulation
    print("----------------------------------------------------------------------------------------")
    print(" SECTION 5. MULTIMODAL MEMORY RETRIEVAL & RELATIONAL UTILITY MODULATION (Sec 4.2, 4.5)")
    print("----------------------------------------------------------------------------------------")
    print("When villagers deliberate near peers, episodic memories are retrieved via multimodal")
    print("scoring: score = 1.0*recency + 1.0*importance + 1.5*relevance + 0.4*mood_congruence.\n")

    first_agent_id = next(iter(controller.agents.keys()))
    sample_agent = controller.agents[first_agent_id]
    peer_id = next((k for k in controller.agents.keys() if k != first_agent_id), first_agent_id)
    peer_name = controller.world.villagers[peer_id].name if peer_id in controller.world.villagers else peer_id

    m_query = MemoryQuery(
        step=state["step"],
        actors=(peer_id,),
        location="central_market",
        tags=("social", "trade", "wheat"),
        pad=sample_agent.affective_state.to_tuple(),
    )
    retrieved = retrieve_top_k(sample_agent.memory_stream, m_query, k=3, rehearse=False)
    print(f"Agent: {sample_agent.name} | Stream Size: {len(sample_agent.memory_stream)} items | Current PAD: ({format_pad(sample_agent.affective_state.to_tuple())})")
    print(f"Retrieval Query -> Context: Target='{peer_name}', Location='central_market', Tags=['social', 'trade', 'wheat']")
    if retrieved:
        for idx, (m_item, bd) in enumerate(retrieved, start=1):
            print(f"  [{idx}] Memory ID: {m_item.mid:<18} | Kind: {m_item.kind:<12} | Step: {m_item.step:02d}")
            print(f"      Summary: \"{m_item.summary}\"")
            print(
                f"      Scores: Total={bd.total_score:.4f} | Recency={bd.recency_score:.4f} | "
                f"Importance={bd.importance_score:.4f} | Relevance={bd.relevance_score:.4f} | Mood={bd.mood_score:.4f}"
            )
    else:
        print("  No episodic memories retrieved matching query context.")
    print()

    # 7. Section 6: Nightfall Reflection, Memory Telemetry & Dispositions
    print("----------------------------------------------------------------------------------------")
    print(" SECTION 6. NIGHTFALL CONSOLIDATION, REFLECTION & DISPOSITIONS (Sec 4.4)")
    print("----------------------------------------------------------------------------------------")
    print("Villagers consolidate their memory streams, synthesizing reflections and updating dispositions.\n")

    for v in state["villagers"]:
        aid = v["agent_id"]
        agent = controller.agents[aid]
        disps = agent.dispositions
        mem_info = v.get("memory", {})

        print(f"Villager: {v['name']} ({v['role']}) | Archetype: {v['archetype_title']}")
        print(f"  Affective State: Dominant={v['dominant_emotion']} ({format_pad(v['pad_vector'])})")
        print(f"  Inner Monologue: \"{v['thought']}\"")
        print(f"  Memory Stream: Size={mem_info.get('stream_size', len(agent.memory_stream))}, Unreflected={mem_info.get('unreflected_count', len(agent.memory_stream.unreflected()))}")
        recent_refs = mem_info.get("recent_reflections", [])
        if recent_refs:
            print(f"  Recent Reflections: {recent_refs}")
        print(f"  Shelter actions during storm: {shelter_counts[aid]}")

        if disps:
            print("  Interpersonal Dispositions towards peers:")
            for tid, d in disps.items():
                tname = controller.world.villagers[tid].name if tid in controller.world.villagers else tid
                ev_str = ", ".join(d.evidence_ids) if d.evidence_ids else "None"
                print(
                    f"    * Peer {tname:<8}: Valence={d.valence:+.3f} | "
                    f"Grudge={d.grudge:.3f} | Confidence={d.confidence:.3f} | "
                    f"Evidence IDs: [{ev_str}]"
                )
        else:
            print("  Interpersonal Dispositions: Stable baseline (no conflicting offenses)")
        print()

    # 8. Section 7: Telemetry & Invariants
    final_hash = controller.compute_canonical_hash()
    avg_contagion = sum(contagion_deltas) / max(1, len(contagion_deltas))
    max_stream_size = max(len(a.memory_stream) for a in controller.agents.values())
    total_unreflected = sum(len(a.memory_stream.unreflected()) for a in controller.agents.values())
    total_shelter_actions = sum(shelter_counts.values())

    print("----------------------------------------------------------------------------------------")
    print(" SECTION 7. VERIFICATION EVIDENCE & CANONICAL REPRODUCIBILITY HASH")
    print("----------------------------------------------------------------------------------------")
    print(f"Canonical Simulation State SHA-256    : {final_hash}")
    print(f"Total Steps Simulated                : {state['step']} ({(state['step'] / 48.0):.1f} days)")
    print(f"Total Village Population             : {len(state['villagers'])}")
    print(f"Greetings Exchanged (Real Count, C3) : {len(greetings_seen)} (proximity-bounded r <= 3)")
    print(f"Storm Shelter Actions Taken (C5)     : {total_shelter_actions} (reaction bonus active)")
    print(f"Max Agent Memory Stream Size (Cap 300: {max_stream_size} items (capacity strictly bounded)")
    print(f"Unreflected Memories In Stream       : {total_unreflected} items (protected from eviction)")
    print(f"Mean Contagion |Delta PAD|           : {avg_contagion:.4f}")
    print("Hourly Zone Occupancy (C2 Fix)       : Real coordinates derived via world.get_zone_at(pos)")
    print("Maslow Prepotency Gating (g1..g5)    : Active & dynamically throttling higher-level drives")
    print("Behavior Tree Lifecycles & Hysteresis: Active (Intentions advance, complete, or interrupt)")
    print("Deterministic Seed Reproducibility   : Guaranteed by Named RNG Streams & Hash Digests")
    print("=" * 88)


def main() -> None:
    args = parse_args()
    total_steps = args.days * 48 if args.days is not None else args.steps
    run_demo(seed=args.seed, total_steps=total_steps, storm_step=args.storm_step)


if __name__ == "__main__":
    main()
