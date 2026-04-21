"""Experiment setup: load the OceanTracker baseline from params/base/, merge a
prod/ variant on top, and return a final OceanTracker-ready parameter dict.

High-level experiment behaviour (reader class, paths, metadata) is controlled
by params/config.yaml and can be read separately via load_experiment_config().

Usage (called by main.py):
    from scripts.setup import build_params, load_experiment_config
    exp = load_experiment_config()
    params = build_params(variant="sensitivity_A")
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

# Paths are relative to the experiment root (one level above scripts/)
_EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
_EXPERIMENT_CONFIG = _EXPERIMENT_ROOT / "params" / "config.yaml"
_BASELINE_CONFIG = _EXPERIMENT_ROOT / "params" / "base" / "config.yaml"
_PROD_DIR = _EXPERIMENT_ROOT / "params" / "prod"


def load_experiment_config() -> dict:
    """Load the experiment-level config (params/config.yaml).

    This controls high-level experiment behaviour: reader class, paths,
    metadata.  It is NOT the OceanTracker run config.
    """
    with _EXPERIMENT_CONFIG.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins on conflicts)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def build_params(variant: str | None = None) -> dict:
    """Build the final OceanTracker parameter dict.

    Args:
        variant: Name of a YAML file (without extension) inside params/prod/.
                 If None, the baseline config alone is used.

    Returns:
        Merged parameter dict ready to pass to OceanTracker.
    """
    with _BASELINE_CONFIG.open("r", encoding="utf-8") as fh:
        params = yaml.safe_load(fh)

    if variant is not None:
        prod_file = _PROD_DIR / f"{variant}.yaml"
        if not prod_file.exists():
            raise FileNotFoundError(
                f"Variant config not found: {prod_file}\n"
                f"Available variants: {[p.stem for p in _PROD_DIR.glob('*.yaml')]}"
            )
        with prod_file.open("r", encoding="utf-8") as fh:
            override = yaml.safe_load(fh) or {}
        params = _deep_merge(params, override)

    # ------------------------------------------------------------------
    # Programmatic extensions — add/modify params here using Python.
    # These run AFTER the YAML merge, so they always take final effect.
    # ------------------------------------------------------------------
    # Example: add a custom particle property
    # params["particle_properties"].append({
    #     "class_name": "oceantracker.particle_properties.age_decay.AgeDecay",
    #     "decay_time_scale": 3600,
    # })

    return params
