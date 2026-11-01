#!/usr/bin/env python3
"""Cross-platform check script for AUTOGEN AI repository.

Usage:
    python scripts/check.py [--check | --test | --lint | --format]
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def run_command(cmd: list[str], description: str) -> int:
    print(f"\n>>> Running: {description} ({' '.join(cmd)})")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"FAILED: {description} exited with code {res.returncode}")
    else:
        print(f"PASSED: {description}")
    return res.returncode


def run_lint() -> int:
    exit_code = 0
    exit_code |= run_command(
        [sys.executable, "scripts/docs_lint.py"],
        "Documentation & Algorithm Registry Linter",
    )
    exit_code |= run_command(
        [sys.executable, "-m", "ruff", "check", "packages", "apps", "tests"],
        "Ruff linter",
    )
    exit_code |= run_command(
        [sys.executable, "-m", "ruff", "format", "--check", "packages", "apps", "tests"],
        "Ruff format check",
    )
    exit_code |= run_command(
        [sys.executable, "-m", "mypy", "packages", "apps", "tests"],
        "Mypy static type checker",
    )
    return exit_code


def run_test() -> int:
    return run_command(
        [sys.executable, "-m", "pytest", "-q"],
        "Pytest test suite",
    )


def run_format() -> int:
    exit_code = 0
    exit_code |= run_command(
        [sys.executable, "-m", "ruff", "format", "packages", "apps", "tests"],
        "Ruff code formatter",
    )
    exit_code |= run_command(
        [sys.executable, "-m", "ruff", "check", "--fix", "packages", "apps", "tests"],
        "Ruff autofix",
    )
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="AUTOGEN repository verification tool")
    parser.add_argument("--check", action="store_true", help="Run full check (lint + test)")
    parser.add_argument("--lint", action="store_true", help="Run lint checks only (ruff + mypy)")
    parser.add_argument("--test", action="store_true", help="Run tests only (pytest)")
    parser.add_argument("--format", action="store_true", help="Autoformat files")
    args = parser.parse_args()

    # Default action is check
    if not (args.check or args.lint or args.test or args.format):
        args.check = True

    if args.format:
        return run_format()

    if args.lint:
        return run_lint()

    if args.test:
        return run_test()

    if args.check:
        lint_code = run_lint()
        if lint_code != 0:
            print("\n❌ LINT CHECKS FAILED. Halting before tests.")
            return lint_code
        test_code = run_test()
        if test_code != 0:
            print("\n❌ TESTS FAILED.")
            return test_code
        print("\n✅ ALL CHECKS PASSED SUCCESSFULLY.")
        return 0

    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
