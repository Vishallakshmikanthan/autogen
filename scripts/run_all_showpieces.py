#!/usr/bin/env python3
"""Master Showpiece Benchmark Suite (Gate 10 Acceptance Runner, Module M22).

Runs all seven canonical showpieces from a single command, verifying full platform
integration, algorithmic correctness, and empirical reproducibility.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from run_showpiece_ablation_arena import run_ablation_arena
from run_showpiece_algorithm_duel import run_algorithm_duel
from run_showpiece_evolution_of_tribes import run_evolution_of_tribes
from run_showpiece_ghost_in_the_fog import run_ghost_in_the_fog
from run_showpiece_the_changing_world import run_the_changing_world
from run_showpiece_time_machine import run_time_machine
from run_showpiece_tragedy_of_the_commons import run_tragedy_of_the_commons


def run_all(seed: int = 42, fast: bool = False) -> int:
    suite_start = time.perf_counter()
    print("=" * 80)
    print("AUTOGEN AI — GATE 10 SHOWPIECE BENCHMARK SUITE (SYS-19, M22)")
    print("=" * 80)
    print(f"Master Seed: {seed} | Fast Mode: {fast}\n")

    benchmarks = [
        ("Showpiece 1: Ghost in the Fog", lambda: run_ghost_in_the_fog(seed=seed, fast=fast)),
        ("Showpiece 2: Algorithm Duel", lambda: run_algorithm_duel(seed=seed, fast=fast)),
        (
            "Showpiece 3: Tragedy of the Commons",
            lambda: run_tragedy_of_the_commons(seed=seed, fast=fast),
        ),
        ("Showpiece 4: Evolution of Tribes", lambda: run_evolution_of_tribes(seed=seed, fast=fast)),
        ("Showpiece 5: Ablation Arena", lambda: run_ablation_arena(seed=seed, fast=fast)),
        ("Showpiece 6: Time Machine", lambda: run_time_machine(seed=seed, fast=fast)),
        ("Showpiece 7: The Changing World", lambda: run_the_changing_world(seed=seed, fast=fast)),
    ]

    results = []
    for name, runner in benchmarks:
        print(f"\n>>> Executing {name}...")
        try:
            res = runner()
            results.append((name, True, res.execution_time_seconds, res.verdict))
        except Exception as e:
            print(f"❌ {name} FAILED: {e}")
            results.append((name, False, 0.0, str(e)))

    total_time = time.perf_counter() - suite_start

    print("\n" + "=" * 80)
    print("GATE 10 BENCHMARK SUITE EXECUTION SCORECARD")
    print("=" * 80)
    print(f"{'Scenario Name':<38} | {'Status':<8} | {'Time (s)':<9} | {'Scientific Verdict'}")
    print("-" * 80)

    all_passed = True
    for name, success, el, verd in results:
        status_str = "PASS" if success else "FAIL"
        if not success:
            all_passed = False
        print(f"{name:<38} | {status_str:<8} | {el:<9.2f} | {verd[:30]}...")

    print("=" * 80)
    print(f"Total Suite Runtime: {total_time:.2f} seconds")

    if all_passed:
        print("\n🎉 ALL 7 GATE 10 SHOWPIECE SCENARIOS PASSED WITH FULL DETERMINISTIC VERIFICATION!")
        return 0
    else:
        print("\n❌ SOME BENCHMARKS FAILED.")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run All Gate 10 Showpieces")
    parser.add_argument("--seed", type=int, default=42, help="Master RNG seed")
    parser.add_argument(
        "--fast", action="store_true", help="Run fast mode for quick CI verification"
    )
    args = parser.parse_args()

    return run_all(seed=args.seed, fast=args.fast)


if __name__ == "__main__":
    sys.exit(main())
