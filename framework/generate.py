"""CLI entry point for experiment generation.

Usage:
    ot-generate experiment_configs/my_experiment.yaml
    python -m framework.generate experiment_configs/my_experiment.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import yaml

from framework.renderer import render_template_dir

# Paths relative to this file's parent (repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TEMPLATE_DIR = _REPO_ROOT / "template"
_EXPERIMENTS_DIR = _REPO_ROOT / "experiments"


def _load_config(config_path: Path) -> dict:
    """Load and validate an experiment YAML config."""
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if not isinstance(config, dict):
        raise click.BadParameter(
            f"Config file must be a YAML mapping, got {type(config).__name__}.",
            param_hint="CONFIG",
        )

    missing = [key for key in ("name",) if key not in config]
    if missing:
        raise click.UsageError(
            f"Config is missing required key(s): {', '.join(missing)}"
        )

    return config


def _build_context(config: dict, config_path: Path) -> dict:
    """Build the Jinja2 rendering context from the experiment config."""
    return {
        # Top-level scalar keys are available directly in templates
        **config,
        # Always inject the config filename for traceability
        "_config_file": config_path.name,
    }


@click.command()
@click.argument(
    "config",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--experiments-dir",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=None,
    help="Override the output experiments/ directory.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print what would be created without writing any files.",
)
def cli(config: Path, experiments_dir: Path | None, dry_run: bool) -> None:
    """Generate an experiment directory from a YAML CONFIG file.

    The experiment is created at experiments/<name>/ (relative to the repo
    root), or under the directory specified by --experiments-dir.

    Example:

        ot-generate experiment_configs/my_experiment.yaml
    """
    experiment_config = _load_config(config)
    experiment_name: str = experiment_config["name"]
    dest_root = experiments_dir or _EXPERIMENTS_DIR
    dest_dir = dest_root / experiment_name
    context = _build_context(experiment_config, config)

    if dry_run:
        click.echo(f"[dry-run] Would create: {dest_dir}")
        click.echo(f"[dry-run] Template:     {_TEMPLATE_DIR}")
        click.echo(f"[dry-run] Context keys: {list(context.keys())}")
        return

    click.echo(f"Generating experiment '{experiment_name}' ...")
    click.echo(f"  Config  : {config}")
    click.echo(f"  Template: {_TEMPLATE_DIR}")
    click.echo(f"  Output  : {dest_dir}")

    try:
        render_template_dir(_TEMPLATE_DIR, dest_dir, context)
    except FileExistsError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Done. Run your experiment with:")
    click.echo(f"  python {dest_dir}/scripts/main.py")


if __name__ == "__main__":
    cli()
