"""Experiment entry point.

Usage:
    # Run with the baseline config only:
    python scripts/main.py

    # Run with a prod/ variant merged on top:
    python scripts/main.py --variant release_summer
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow running as `python scripts/main.py` from the experiment root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from setup import build_params  # noqa: E402
from run import run              # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an OceanTracker experiment."
    )
    parser.add_argument(
        "--variant",
        default=None,
        help=(
            "Name of a YAML file (without .yaml) inside params/prod/ to merge "
            "on top of the baseline config.  Omit to use the baseline only."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    params = build_params(variant=args.variant)
    run(params)
