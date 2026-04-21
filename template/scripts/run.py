"""Run the OceanTracker simulation using a pre-built parameter dict."""

from __future__ import annotations

from oceantracker.main import OceanTracker


def run(params: dict) -> None:
    """Execute an OceanTracker simulation.

    Args:
        params: Fully-built parameter dict from setup.build_params().
    """
    ot = OceanTracker()
    ot.run(params)
