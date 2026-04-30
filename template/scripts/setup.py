"""Experiment setup: build OceanTracker parameter files for all scenarios.

Loads polygon geometries, configures polygon releases and statistics, then
generates one parameter YAML file per (flow_config x simulation_period) pair.

Usage:
    python setup.py
    python setup.py --config ../params/config.yaml --output_dir ../params/prod
"""

from __future__ import annotations

import copy
import os
import sys
from datetime import datetime, timedelta
from glob import glob as _glob
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codebase.util import load_polygons  # noqa: E402
from oceantracker.util import yaml_util  # noqa: E402


# ─── PTM Parameter Builders ────────────────────────────────────────────────────


def add_polygon_releases(df, start_date, n_releases=4,
                         release_spacing_days=7,
                         release_duration=1,
                         max_age=60):
    """
    Build OceanTracker PolygonRelease configurations.

    Args:
        df:                    GeoDataFrame with polygon geometries and
                               release parameters (pulse_size, release_interval).
        start_date:            Simulation start date (datetime.date or 'YYYY-MM-DD' str).
        n_releases:            Number of release pulses per polygon.
        release_spacing_days:  Days between consecutive release starts.
        release_duration:      Duration of each release in days.
        max_age:               Maximum particle age in days.

    Returns:
        List of release group configuration dicts.
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()

    entries = []
    for _, row in df.iterrows():
        for i in range(n_releases):
            release_start = (
                datetime.combine(start_date, datetime.min.time())
                + timedelta(days=i * release_spacing_days)
                + timedelta(minutes=30)  # offset: some hindcasts start at 00:30
            )
            entries.append({
                'class_name': 'PolygonRelease',
                'name': f"{row['name']}_r{i + 1}",
                'pulse_size': int(row['pulse_size']),
                'start': release_start.strftime('%Y-%m-%d %H:%M:%S'),
                'duration': int(24 * 3600 * release_duration),
                'release_interval': int(row['release_interval']),
                'max_age': int(24 * 3600 * max_age),
                'points': [list(pt) for pt in row['geometry'].exterior.coords],
            })
    return entries


def add_polygon_statistics(df, update_interval, max_age, stats_type='age',
                           age_bin_size=3600):
    """
    Build OceanTracker polygon statistics configurations.

    Args:
        df:               GeoDataFrame with polygon geometries.
        update_interval:  Time between statistics updates in seconds.
        max_age:          Maximum particle age in days (age-based stats only).
        stats_type:       'age'  → PolygonStats2D_ageBased
                          'time' → PolygonStats2D_timeBased
        age_bin_size:     Size of age bins in seconds (age-based stats only).

    Returns:
        List of particle_statistics configuration dicts.
    """
    if stats_type == 'age':
        class_name = 'PolygonStats2D_ageBased'
    elif stats_type == 'time':
        class_name = 'PolygonStats2D_timeBased'
    else:
        raise ValueError(f"stats_type must be 'age' or 'time', got {stats_type!r}")

    entries = []
    for _, row in df.iterrows():
        entry = {
            'class_name': class_name,
            'name': f"{row['name']}_{stats_type}_stat",
            'update_interval': int(update_interval),
            'polygon_list': [
                {'points': [list(pt) for pt in row['geometry'].exterior.coords]}
            ],
        }
        if stats_type == 'age':
            entry['max_age_to_bin'] = int(24 * 3600 * max_age)
            entry['age_bin_size'] = int(age_bin_size)
        entries.append(entry)
    return entries


def add_gridded_statistics(df, update_interval,
                           rows=1600, cols=1600,
                           span_x=160000, span_y=160000,
                           release_group_centered_grids=True,
                           status_list=None):
    """
    Build an OceanTracker GriddedStats2D_timeBased configuration.

    Args:
        df:                           GeoDataFrame (unused when
                                      release_group_centered_grids=True;
                                      retained for interface consistency).
        update_interval:              Time between statistics updates in seconds.
        rows:                         Number of grid rows.
        cols:                         Number of grid columns.
        span_x:                       East-west grid extent in metres.
        span_y:                       North-south grid extent in metres.
        release_group_centered_grids: Centre grid on each release point.
        status_list:                  Particle statuses to count; defaults to
                                      ['moving', 'on_bottom', 'stranded_by_tide'].

    Returns:
        List containing a single particle_statistics configuration dict.
    """
    if status_list is None:
        status_list = ['moving', 'on_bottom', 'stranded_by_tide']

    return [{
        'class_name': 'GriddedStats2D_timeBased',
        'name': 'connectivity_grid',
        'rows': rows,
        'cols': cols,
        'span_x': span_x,
        'span_y': span_y,
        'release_group_centered_grids': release_group_centered_grids,
        'update_interval': int(update_interval),
        'status_list': status_list,
    }]


def create_period_input_dir(source_dir, start_id, end_id, period_dir):
    """
    Create a directory of symlinks to SCHISM NetCDF files for a time period.

    Includes file IDs in [start_id, end_id + 1]; the extra file ensures
    OceanTracker has a complete time window at the period boundary.

    Args:
        source_dir:  Directory containing the full set of SCHISM .nc files.
        start_id:    First file ID to include.
        end_id:      Last simulation ID; files up to end_id + 1 are included.
        period_dir:  Destination directory for symlinks (created if absent).
    """
    os.makedirs(period_dir, exist_ok=True)
    abs_source = os.path.abspath(source_dir)
    for fid in range(start_id, end_id + 2):
        for filepath in _glob(os.path.join(abs_source, f'*_{fid}.nc')):
            link = os.path.join(period_dir, os.path.basename(filepath))
            if not os.path.lexists(link):
                os.symlink(filepath, link)


# ─── Main Setup Logic ──────────────────────────────────────────────────────────


def setup_ptm(config_path='../params/config.yaml'):
    """
    Build OceanTracker parameter dicts for all (flow_config x period) scenarios.

    For each combination of flow configuration and simulation period:
      1. Loads the per-gate baseline OceanTracker YAML from params/base/.
      2. Creates a period-specific SCHISM input directory via symlinks.
      3. Adds polygon releases and both age- and time-based polygon statistics.

    Args:
        config_path:  Path to the experiment-level config.yaml.

    Returns:
        (params_list, param_names) — lists of parameter dicts and their names.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    polygons_dir        = config['paths']['polygons_dir']
    base_params_dir     = config['paths']['oceantracker_base_dir']
    release_polygon_cfg = config['polygons']['release_regions']
    dest_polygon_cfg    = config['polygons']['destination_regions']

    release_gdf  = load_polygons(polygons_dir, release_polygon_cfg)
    dest_gdf     = load_polygons(polygons_dir, dest_polygon_cfg)
    all_polygons = pd.concat([release_gdf, dest_gdf], ignore_index=True)

    all_params, param_names = [], []
    for gate in config['flow_configs']:
        base_param_path = os.path.join(
            base_params_dir, f"gate_{gate['suffix']}.yaml"
        )
        base_param    = yaml_util.read_YAML(base_param_path)
        base_input_dir = os.path.join(
            config['paths']['hindcasts_symlink_dir'], gate['suffix'], ''
        )

        for period in config['simulation_periods']:
            run_param = copy.deepcopy(base_param)

            if config['paths'].get('use_symlink_inputs', True):
                period_input_dir = os.path.join(
                    config['paths']['hindcasts_symlink_dir'],
                    f"{gate['suffix']}_{period['name']}",
                )
                create_period_input_dir(
                    base_input_dir,
                    period['start_id'],
                    period['end_id'],
                    period_input_dir,
                )
                run_param['reader']['input_dir'] = period_input_dir
            # else: leave input_dir from base YAML unchanged

            run_param['root_output_dir'] = (
                f"{config['user']['output_dir']}"
                f"{gate['suffix']}_{period['name']}"
            )

            max_age_days = config['run']['duration'] / 86400

            run_param['release_groups'] = add_polygon_releases(
                release_gdf,
                start_date=period['start_date'],
                n_releases=period['n_releases'],
                release_spacing_days=period['release_spacing_days'],
                max_age=max_age_days,
            )
            run_param['particle_statistics'] = (
                add_polygon_statistics(
                    all_polygons, update_interval=3600,
                    max_age=max_age_days, stats_type='age'
                )
                + add_polygon_statistics(
                    all_polygons, update_interval=3600,
                    max_age=max_age_days, stats_type='time'
                )
                + add_gridded_statistics(
                    all_polygons, update_interval=3600
                )
            )

            all_params.append(run_param)
            param_names.append(f"{gate['suffix']}_{period['name']}")

    return all_params, param_names


def write_params_to_files(params_list, param_names,
                          output_dir='../params/prod'):
    """
    Write OceanTracker parameter dicts to YAML files.

    Args:
        params_list:  List of parameter dicts.
        param_names:  Corresponding list of file name stems.
        output_dir:   Directory to write files into (created if absent).
    """
    os.makedirs(output_dir, exist_ok=True)
    for name, param in zip(param_names, params_list):
        path = os.path.join(output_dir, f'{name}.yaml')
        yaml_util.write_YAML(path, param)
        print(f'  Written: {path}')


# ─── Entry Point ───────────────────────────────────────────────────────────────


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate PTM parameter files.')
    parser.add_argument('--config', default='../params/config.yaml',
                        help='Path to experiment config.yaml.')
    parser.add_argument('--output_dir', default='../params/prod',
                        help='Directory to write parameter files.')
    args = parser.parse_args()

    params, names = setup_ptm(config_path=args.config)
    write_params_to_files(params, names, output_dir=args.output_dir)
    print(f'\nGenerated {len(params)} parameter files in {args.output_dir}')
