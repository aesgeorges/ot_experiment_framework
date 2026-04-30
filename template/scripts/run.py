"""Run a single OceanTracker PTM simulation.

Usage:
    python run.py --params_path ../params/prod/scenario_name.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from oceantracker import main
from oceantracker.util import yaml_util

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_simulation(param_path: str) -> None:
    """
    Run an OceanTracker simulation using a pre-built parameter file.

    Args:
        param_path:  Path to the OceanTracker parameter YAML file.
    """
    parameters = yaml_util.read_YAML(param_path)
    case_info_file = main.run(parameters)
    print(f'Simulation complete. Case info: {case_info_file}')


@click.command()
@click.option('--params_path', required=True,
              help='Path to the OceanTracker parameter YAML file.')
def cli(params_path: str) -> None:
    """Run an OceanTracker PTM simulation."""
    run_simulation(params_path)


if __name__ == '__main__':
    cli()
