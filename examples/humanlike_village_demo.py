"""Top-level runnable CLI script for AUTOGEN Humanlike Villagers Demonstration."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure packages/ is on sys.path if run directly
root_dir = Path(__file__).resolve().parent.parent
packages_dir = root_dir / "packages"
if str(packages_dir) not in sys.path:
    sys.path.insert(0, str(packages_dir))

from autogen.examples.humanlike_village_demo import main  # noqa: E402

if __name__ == "__main__":
    main()
