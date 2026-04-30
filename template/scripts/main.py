"""Experiment entry point.

Commands:
    setup          Build OceanTracker parameter files for all scenarios.
    run            Submit all scenarios to Slurm.
    run_single     Submit a single scenario to Slurm.
    animate        Pre-cache map tiles, then submit animation jobs to Slurm.

Usage:
    python main.py setup
    python main.py run
    python main.py run_single hist_fall
    python main.py animate
    python main.py animate --scenario no_op_fall
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from setup import setup_ptm, write_params_to_files  # noqa: E402

try:
    from simple_slurm import Slurm
    _SLURM_AVAILABLE = True
except ImportError:
    _SLURM_AVAILABLE = False


def _require_slurm():
    """Raise a clear error if simple_slurm is not installed."""
    if not _SLURM_AVAILABLE:
        raise RuntimeError(
            'simple_slurm is not installed. '
            'Install it (pip install simple-slurm) or run jobs manually.'
        )


# ─── Slurm Helpers ─────────────────────────────────────────────────────────────


def run_ptm_slurm(param_file: str) -> None:
    """
    Submit a Slurm job to run a single PTM simulation.

    Args:
        param_file:  Path to the OceanTracker parameter YAML file.
    """
    _require_slurm()
    job_name = f"ptm_{Path(param_file).stem}"
    slurm = Slurm(
        job_name=job_name,
        partition='work',
        mail_type='END,FAIL',
        mail_user=config['user']['email'],
        output=f'../logs/{job_name}_%j.log',
        ntasks=1,
        nodes=1,
        cpus_per_task=24,
        time='12:00:00',
    )
    slurm.add_cmd(f"source {config['user']['conda_path']}")
    slurm.add_cmd(f"conda activate {config['user']['env']}")
    slurm.add_cmd('export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK')
    slurm.add_cmd('export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK')
    slurm.sbatch(f'python ../scripts/run.py --params_path {param_file}')


def run_all_scenarios() -> None:
    """Submit Slurm jobs for all PTM parameter files in params/prod/."""
    param_dir = '../params/prod'
    for fname in os.listdir(param_dir):
        if fname.endswith('.yaml'):
            run_ptm_slurm(os.path.join(param_dir, fname))


def run_animations_slurm(scenario: str | None = None, cpus: int = 4) -> None:
    """
    Submit one Slurm animation job per scenario.

    Args:
        scenario:  Single scenario name (e.g. 'no_op_fall'). If None,
                   derives scenario names from config flow_configs x simulation_periods.
        cpus:      CPUs per animation Slurm job.
    """
    _require_slurm()
    if scenario:
        scenarios = [scenario]
    else:
        scenarios = [
            f"{gate['suffix']}_{period['name']}"
            for gate in config['flow_configs']
            for period in config['simulation_periods']
        ]

    for scenario_name in scenarios:
        job_name = f'anim_{scenario_name}'
        slurm = Slurm(
            job_name=job_name,
            partition='work',
            mail_type='END,FAIL',
            mail_user=config['user']['email'],
            output=f'../logs/{job_name}_%j.log',
            ntasks=1,
            nodes=1,
            cpus_per_task=cpus,
            time='01:00:00',
        )
        slurm.add_cmd(f"source {config['user']['conda_path']}")
        slurm.add_cmd(f"conda activate {config['user']['viz_env']}")
        slurm.sbatch(
            f'python ../scripts/make_animations.py'
            f' --scenario {scenario_name}'
            f' --n_workers {cpus}'
        )


# ─── CLI Commands ───────────────────────────────────────────────────────────────


@click.group()
def cli():
    """Experiment CLI — set up and execute PTM simulations.

    \b
    Examples:
        python main.py setup
        python main.py run
        python main.py run_single hist_fall
        python main.py animate
        python main.py animate --scenario no_op_fall
    """
    pass


@cli.command('setup')
@click.option('--config', 'config_path', default='../params/config.yaml',
              help='Path to experiment config.yaml.')
@click.option('--output_dir', default='../params/prod',
              help='Directory to write parameter files.')
def setup_cmd(config_path, output_dir):
    """Build OceanTracker parameter files for all scenarios."""
    params, names = setup_ptm(config_path=config_path)
    write_params_to_files(params, names, output_dir=output_dir)
    print(f'Generated {len(params)} parameter files in {output_dir}')


@cli.command('run')
def run_cmd():
    """Submit all PTM scenarios to Slurm."""
    run_all_scenarios()
    print('Submitted all PTM scenarios to Slurm.')


@cli.command('run_single')
@click.argument('scenario_name')
def run_single_cmd(scenario_name):
    """Submit a single PTM scenario to Slurm.

    SCENARIO_NAME is the parameter file stem, e.g. hist_fall or no_op_summer.
    """
    param_file = os.path.join('../params/prod', f'{scenario_name}.yaml')
    run_ptm_slurm(param_file)
    print(f'Submitted {scenario_name} to Slurm.')


@cli.command('animate')
@click.option('--scenario', default=None,
              help='Single scenario name (e.g. no_op_fall). Omit for all scenarios.')
@click.option('--cpus', default=4, show_default=True,
              help='CPUs per animation Slurm job.')
def animate_cmd(scenario, cpus):
    """Pre-cache map tiles, then submit animation jobs to Slurm.

    \b
    Submits one Slurm job per scenario. Each job uses parallel frame
    rendering across its allocated CPUs.

    \b
    Examples:
        python main.py animate                        # all scenarios
        python main.py animate --scenario no_op_fall  # single scenario
    """
    from codebase.plot import precache_tiles
    print('Pre-caching map tiles (requires internet)...')
    precache_tiles(config_path='../params/config.yaml')
    run_animations_slurm(scenario=scenario, cpus=cpus)
    if scenario:
        print(f'Submitted animation job for {scenario}.')
    else:
        print('Submitted animation jobs (one per scenario).')


# ─── Entry Point ───────────────────────────────────────────────────────────────


if __name__ == '__main__':
    with open('../params/config.yaml', 'r') as _f:
        config = yaml.safe_load(_f)
    cli()
