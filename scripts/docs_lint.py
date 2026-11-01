#!/usr/bin/env python3
"""Static Documentation & Algorithm Registry Linter (SYS-19, Module M21).

Verifies that:
1. Every algorithm registered in AlgorithmRegistry has an existing markdown specification.
2. Each documentation file contains all required mathematical and verification sections.
3. Citation safety is upheld (sources must be approved textbooks or explicitly tagged 'TO VERIFY').
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages"))

from autogen.intelligence.registry import AlgorithmRegistry, populate_default_catalog  # noqa: E402

REQUIRED_SECTION_PATTERNS = [
    (r"theory|mathematical definition", "Theory & Mathematical Definition"),
    (r"assumptions", "Assumptions"),
    (r"equations|variable definitions|mathematical model", "Equations & Mathematical Model"),
    (r"complexity", "Computational Complexity"),
    (r"verification|oracle", "Verification Strategy & Test Oracles"),
    (r"limitations", "Known Limitations"),
    (r"references|citations", "References / Citations"),
]

ALLOWED_CANONICAL_SOURCES = [
    "russell",
    "norvig",
    "sutton",
    "barto",
    "bishop",
    "goodfellow",
    "weiss",
    "hart",
    "nilsson",
    "raphael",
    "acklam",
    "knuth",
    "moore",
    "dijkstra",
    "bellman",
]


def lint_documentation() -> tuple[int, list[str]]:
    """Runs static analysis against all registered algorithms in the registry."""
    populate_default_catalog()
    algorithms = AlgorithmRegistry.get_all()

    errors: list[str] = []
    print(f">>> Inspecting documentation for {len(algorithms)} registered algorithms...")

    for alg in algorithms:
        doc_file = REPO_ROOT / alg.doc_path
        if not doc_file.exists():
            errors.append(f"[{alg.algorithm_id}] Missing document at expected path: {alg.doc_path}")
            continue

        content = doc_file.read_text(encoding="utf-8")
        content_lower = content.lower()

        # 1. Section Presence Check
        for pattern, section_name in REQUIRED_SECTION_PATTERNS:
            if not re.search(pattern, content_lower):
                errors.append(
                    f"[{alg.algorithm_id}] Document '{alg.doc_path}' is missing required section: '{section_name}'"
                )

        # 2. Citation Safety Check
        for cite in alg.citations:
            source_lower = cite.author_or_source.lower()
            if not cite.verified:
                if "to verify" not in content_lower:
                    errors.append(
                        f"[{alg.algorithm_id}] Unverified citation '{cite.author_or_source}' must be explicitly tagged 'TO VERIFY' in {alg.doc_path}."
                    )
            else:
                is_recognized = any(
                    allowed in source_lower for allowed in ALLOWED_CANONICAL_SOURCES
                )
                if not is_recognized:
                    errors.append(
                        f"[{alg.algorithm_id}] Citation '{cite.author_or_source}' is not in approved textbook canon. "
                        "Mark verified=False and tag 'TO VERIFY' if unconfirmed."
                    )

    return len(errors), errors


def main() -> int:
    error_count, errors = lint_documentation()
    if error_count > 0:
        print(f"\n❌ DOCS LINT FAILED: Found {error_count} documentation issues:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("\n✅ DOCS LINT PASSED: All registered algorithms have complete, verified documentation!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
