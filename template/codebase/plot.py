"""Plotting and animation utilities for OceanTracker PTM experiments.

Public API
----------
Data loading:
    load_tracks(case_info_path)
    load_polygon_stats(case_info_path)

Scenario helpers:
    pull_caseInfo_stats(scenario, outputs_dir)

Static figures (HoloViews / Bokeh):
    arrival_age_histograms(...)
    arrival_count_timeseries(...)
    proportion_timeseries(...)
    comparison_count_timeseries(...)
    comparison_age_histograms(...)
    comparison_with_stats(...)
    comparison_proportion_timeseries(...)

Interactive animation (Panel):
    build_background(config_path, ...)
    build_particle_animation(tracks_ds, background, ...)

Matplotlib GIF export:
    precache_tiles(config_path, axis_lims, zoom)
    export_animation(tracks_ds, output_path, ...)
"""

from __future__ import annotations

import json
import os

import holoviews as hv
import geoviews as gv
import numpy as np
import pandas as pd
import panel as pn
import xarray as xr
import yaml
from bokeh.models import GlyphRenderer, LinearAxis, Range1d, Span
from bokeh.palettes import Category10_10
from cartopy import crs as ccrs
from pyproj import Transformer

from oceantracker.read_output.python import load_output_files

# ─── Global Font Sizes ──────────────────────────────────────────────────────────

FONT_TITLE  = 16
FONT_LABELS = 16
FONT_TICKS  = 16
FONT_LEGEND = 15
HV_FONTSIZE = {
    'title':  FONT_TITLE,
    'labels': FONT_LABELS,
    'xticks': FONT_TICKS,
    'yticks': FONT_TICKS,
    'legend': FONT_LEGEND,
}
MPL_FONT_TITLE  = 11
MPL_FONT_LEGEND = 10


# ─── Private Helpers ────────────────────────────────────────────────────────────


def _load_config(config_path):
    """Read config.yaml and return (config_dict, config_dir)."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config, os.path.dirname(os.path.abspath(config_path))


def _load_region_gdfs(config_path, target_crs='epsg:3857'):
    """Load release and destination polygon GeoDataFrames from config."""
    from codebase.util import load_polygons

    config, config_dir = _load_config(config_path)
    polygons_dir = os.path.normpath(
        os.path.join(config_dir, config['paths']['polygons_dir'])
    )
    release_gdf = load_polygons(
        polygons_dir, config['polygons']['release_regions']
    ).to_crs(target_crs)
    dest_gdf = load_polygons(
        polygons_dir, config['polygons']['destination_regions']
    ).to_crs(target_crs)
    return release_gdf, dest_gdf


def _get_tile_source(config_path):
    """Build a CartoDB tile source with local file cache."""
    import cartopy.io.img_tiles as cimgt

    config, config_dir = _load_config(config_path)
    tile_cache_dir = os.path.normpath(
        os.path.join(config_dir, config['paths']['tile_cache_dir'])
    )
    os.makedirs(tile_cache_dir, exist_ok=True)
    tiles = cimgt.GoogleTiles(
        style='street',
        url='https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
        cache=tile_cache_dir,
    )
    return tiles, tile_cache_dir


def _load_release_date_map(output_dir):
    """
    Return a dict mapping release cohort key (r1, r2, ...) to start-date label.

    Reads the *_raw_user_params.json* file produced by OceanTracker.

    Args:
        output_dir:  OceanTracker output directory for a single scenario run.

    Returns:
        Dict of {cohort_key: 'Mon DD'} strings, or empty dict if unavailable.
    """
    import glob
    params_files = glob.glob(os.path.join(output_dir, '*_raw_user_params.json'))
    if not params_files:
        return {}
    with open(params_files[0]) as f:
        raw = json.load(f)
    date_map = {}
    for rg in raw.get('release_groups', []):
        name  = rg.get('name', '')
        start = rg.get('start', '')
        if '_' in name and start:
            cohort = name.rsplit('_', 1)[1]
            if cohort not in date_map:
                date_map[cohort] = pd.Timestamp(start).strftime('%b %d')
    return date_map


# ─── Data Loading ───────────────────────────────────────────────────────────────


def load_tracks(case_info_path, fraction_to_read=None):
    """
    Load OceanTracker particle tracks into an xarray Dataset.

    Args:
        case_info_path:    Path to *_caseInfo.json file.
        fraction_to_read:  Optional float 0-1 to subsample particles.

    Returns:
        xr.Dataset with coordinates (time, particle) and data variables:
            x, y:             (time, particle) UTM positions
            status:           (time, particle) 1=alive, 0=not released, <0=dead
            age:              (time, particle) seconds since release
            IDrelease_group:  (particle,) release group index
        Attrs: grid, release_group_names, axis_lim, particle_status_flags.
    """
    tracks = load_output_files.load_track_data(
        case_info_path, fraction_to_read=fraction_to_read
    )

    rg = tracks.get('particle_release_groups', {})
    release_group_names = {
        i: key
        for i, (key, info) in enumerate(rg.items()) if isinstance(info, dict)
    }

    id_rg = tracks['IDrelease_group']
    if id_rg.ndim == 2:
        id_rg = id_rg[0]

    dims = ('time', 'particle')
    ds = xr.Dataset(
        {
            'x':               (dims, tracks['x'][:, :, 0]),
            'y':               (dims, tracks['x'][:, :, 1]),
            'status':          (dims, tracks['status']),
            'age':             (dims, tracks['age']),
            'IDrelease_group': ('particle', id_rg),
        },
        coords={
            'time':     tracks['time'],
            'particle': tracks['ID'],
        },
    )
    ds.attrs['grid']                 = tracks['grid']
    ds.attrs['release_group_names']  = release_group_names
    ds.attrs['axis_lim']             = tracks.get('axis_lim', [])
    ds.attrs['particle_status_flags'] = tracks.get('particle_status_flags', {})
    return ds


def load_polygon_stats(case_info_path):
    """
    Load all polygon statistics files for a simulation run.

    Args:
        case_info_path:  Path to *_caseInfo.json file.

    Returns:
        Dict mapping region name (str) to xr.Dataset with data variables:
            count, count_all_alive, connectivity_matrix
        and coords (age_bin or time, release_group).
        Each dataset has attrs: stats_type, release_group_names.
    """
    with open(case_info_path, 'r') as f:
        case_info = json.load(f)

    stat_names = case_info.get('output_files', {}).get('particle_statistics', {})
    all_stats = {}

    for name in stat_names:
        raw = load_output_files.load_stats_data(case_info_path, name=name)

        rg = raw.get('particle_release_groups', {})
        rg_names = {
            i: info.get('name', f'group_{i}')
            for i, info in enumerate(rg.values()) if isinstance(info, dict)
        }

        time_var = raw.get('time_var', 'age_bins')
        if time_var == 'age_bins':
            dim_name = 'age_bin'
            dim_vals = raw['age_bins']
        else:
            dim_name = 'time'
            dim_vals = raw.get('date', raw['time'])

        count = raw['count']
        while count.ndim > 2:
            count = count.sum(axis=-1)

        count_alive = raw.get('count_all_alive_particles', np.zeros_like(count))
        conn = raw.get('connectivity_matrix', np.zeros_like(count))
        while conn.ndim > 2:
            conn = conn.sum(axis=-1)

        dims = (dim_name, 'release_group')
        ds = xr.Dataset(
            {
                'count':               (dims, count),
                'count_all_alive':     (dims, count_alive),
                'connectivity_matrix': (dims, conn),
            },
            coords={
                dim_name:        dim_vals,
                'release_group': np.arange(count.shape[1]),
            },
        )
        ds.attrs['stats_type']          = raw.get('stats_type', 'unknown')
        ds.attrs['release_group_names'] = rg_names
        all_stats[name] = ds

    return all_stats


# ─── Release Group Utilities ────────────────────────────────────────────────────


def parse_release_groups(names):
    """
    Group release group names by release cohort (r1, r2, ...).

    Expects names in the format '<source>_r<n>' (e.g. 'Sacramento_r1').

    Args:
        names:  Iterable of release group name strings.

    Returns:
        (releases, all_sources, color_map) where:
            releases:   dict mapping cohort key → list of (index, source) tuples
            all_sources: sorted list of unique source names
            color_map:  dict mapping source → Bokeh Category10 color
    """
    all_sources = sorted({str(n).rsplit('_', 1)[0] for n in names})
    color_map = {
        s: Category10_10[i % len(Category10_10)]
        for i, s in enumerate(all_sources)
    }
    releases = {}
    for i, name in enumerate(names):
        source, cohort = str(name).rsplit('_', 1)
        releases.setdefault(cohort, []).append((i, source))
    return releases, all_sources, color_map


def hide_source_labels(plot, element):
    """Bokeh hook: hide per-source sub-labels on the x-axis."""
    plot.handles['xaxis'].major_label_text_font_size = '0pt'


# ─── Scenario Helpers ───────────────────────────────────────────────────────────


def pull_caseInfo_stats(scenario, outputs_dir='../outputs'):
    """
    Build paths and load caseInfo + stat file mappings for a named scenario.

    The scenario directory is expected at::

        <outputs_dir>/<scenario>/<scenario>/

    with caseInfo at::

        <outputs_dir>/<scenario>/<scenario>/<scenario>_caseInfo.json

    Args:
        scenario:     Scenario name string (e.g. 'hist_summer' or 'no_op_fall').
        outputs_dir:  Base outputs directory.

    Returns:
        (case_info, stat_paths, output_dir) where stat_paths maps cleaned
        keys like 'Montezuma_Slough_time' to stat file basenames.
    """
    from oceantracker.util import json_util

    output_dir  = os.path.join(outputs_dir, scenario, scenario)
    output_path = os.path.join(output_dir, f'{scenario}_caseInfo.json')

    case_info  = json_util.read_JSON(output_path)
    stat_paths = {
        k.replace(' ', '_').replace('_stat', ''): v.replace(' ', '_')
        for k, v in case_info['output_files']['particle_statistics'].items()
    }
    return case_info, stat_paths, output_dir


# ─── Static Figures ─────────────────────────────────────────────────────────────


def arrival_age_histograms(output_dir, stat_paths,
                           destination='Montezuma_Slough_age'):
    """
    Histograms of the age distribution of particles present in a destination,
    grouped by release source, with one panel per release cohort.

    Args:
        output_dir:   OceanTracker output directory for the scenario.
        stat_paths:   Dict mapping destination keys to stat file basenames
                      (from pull_caseInfo_stats).
        destination:  Key for the age-based stat to plot (e.g.
                      'Montezuma_Slough_age').

    Returns:
        hv.NdLayout of bar charts, one panel per release cohort.
    """
    ds = xr.open_mfdataset(f'{output_dir}/{stat_paths[destination]}')
    age_days = ds['age_bins'].values / (3600 * 24)
    releases, all_sources, color_map = parse_release_groups(
        ds['release_group_names'].values
    )
    dest_label = destination.replace('_age', '').replace('_', ' ')
    date_map   = _load_release_date_map(output_dir)

    plots = []
    for release, groups in sorted(releases.items()):
        panel_label = date_map.get(release, release)
        rows = []
        for idx, source in groups:
            src_label = source.replace('_', ' ')
            for ai, age in enumerate(age_days):
                rows.append({
                    'Age (days)': str(int(age)),
                    'Source':     src_label,
                    'Count':      float(ds['count'].values[ai, idx, 0]),
                })
        df   = pd.DataFrame(rows)
        bars = hv.Bars(df, kdims=['Age (days)', 'Source'], vdims='Count').opts(
            width=450, height=300, tools=['hover'],
            cmap={s.replace('_', ' '): color_map[s] for s in all_sources},
            color='Source', show_legend=True,
            hooks=[hide_source_labels],
        )
        plots.append((panel_label, bars))

    layout = hv.NdLayout(dict(plots), kdims='Particles Released on', sort=False).cols(2)
    return layout.opts(
        hv.opts.Bars(fontsize=HV_FONTSIZE),
        hv.opts.NdLayout(title=f'Particle presence in {dest_label} by age'),
    )


def arrival_count_timeseries(output_dir, stat_paths,
                              destination='Montezuma_Slough_time',
                              xlim_end=None):
    """
    Timeseries of particles present in a destination over time,
    grouped by release source, with one panel per release cohort.

    Args:
        output_dir:   OceanTracker output directory for the scenario.
        stat_paths:   Dict mapping destination keys to stat file basenames.
        destination:  Key for the time-based stat to plot.
        xlim_end:     Optional end date for x-axis, e.g. '2025-08-23'.

    Returns:
        hv.NdLayout of timeseries curves.
    """
    ds = xr.open_mfdataset(f'{output_dir}/{stat_paths[destination]}')
    releases, all_sources, color_map = parse_release_groups(
        ds['release_group_names'].values
    )
    dest_label = destination.replace('_time', '').replace('_', ' ')
    date_map   = _load_release_date_map(output_dir)

    plots = []
    for release, groups in sorted(releases.items()):
        panel_label = date_map.get(release, release)
        overlay     = hv.NdOverlay({
            source.replace('_', ' '): hv.Curve(
                (ds['time'].values, ds['count'][:, idx, 0].values),
                kdims='Time', vdims='Count',
            ).opts(color=color_map[source])
            for idx, source in groups
        })
        plots.append((panel_label, overlay))

    layout = hv.NdLayout(dict(plots), kdims='Particles Released on', sort=False).cols(1)
    curve_opts = dict(width=900, height=600, tools=['hover'], fontsize=HV_FONTSIZE)
    if xlim_end is not None:
        curve_opts['xlim'] = (
            pd.Timestamp(ds['time'].values[0]),
            pd.Timestamp(xlim_end),
        )

    def _legend_top_left(plot, element):
        plot.state.legend.location = 'top_left'

    curve_opts['hooks'] = [_legend_top_left]
    return layout.opts(
        hv.opts.Curve(**curve_opts),
        hv.opts.NdLayout(title=f'Particle presence in {dest_label} over time'),
    )


def proportion_timeseries(output_dir, stat_paths, source_filter,
                           destinations=None, xlim_end=None):
    """
    Proportion of released particles present in each destination over time,
    filtered to a single source, one panel per release cohort.

    Proportion = count[:, idx, 0] / num_released[:, idx].

    Args:
        output_dir:    OceanTracker output directory for the scenario.
        stat_paths:    stat_paths dict (from pull_caseInfo_stats).
        source_filter: Source name to plot (e.g. 'Sacramento_Full').
        destinations:  Optional list of stat keys ending in '_time'.
                       Defaults to every '_time' key in stat_paths.
        xlim_end:      Optional x-axis end date string.

    Returns:
        hv.NdLayout with one panel per release cohort.
    """
    if destinations is None:
        destinations = [k for k in stat_paths if k.endswith('_time')]

    dest_colors = {
        d: Category10_10[i % len(Category10_10)]
        for i, d in enumerate(destinations)
    }
    dest_data  = {d: xr.open_mfdataset(f'{output_dir}/{stat_paths[d]}')
                  for d in destinations}
    first_ds   = next(iter(dest_data.values()))
    releases, _, _ = parse_release_groups(first_ds['release_group_names'].values)
    date_map   = _load_release_date_map(output_dir)
    src_label  = source_filter.replace('_', ' ')

    plots = []
    for release, groups in sorted(releases.items()):
        panel_label = date_map.get(release, release)
        idx = next((i for i, src in groups if src == source_filter), None)
        if idx is None:
            continue
        curves = {}
        for dest in destinations:
            ds         = dest_data[dest]
            count      = ds['count'][:, idx, 0].values
            num_rel    = ds['num_released'][:, idx].values
            proportion = np.where(num_rel > 0, count / num_rel, 0.0)
            dest_label = dest.replace('_time', '').replace('_', ' ')
            curves[dest_label] = hv.Curve(
                (ds['time'].values, proportion),
                kdims='Time', vdims='Proportion',
            ).opts(color=dest_colors[dest])
        plots.append((panel_label, hv.NdOverlay(curves)))

    layout     = hv.NdLayout(dict(plots), kdims='Particles Released on',
                              sort=False).cols(2)
    curve_opts = dict(width=900, height=600, tools=['hover'],
                      fontsize=HV_FONTSIZE, ylim=(0, 0.5))
    if xlim_end is not None:
        curve_opts['xlim'] = (
            pd.Timestamp(first_ds['time'].values[0]),
            pd.Timestamp(xlim_end),
        )

    def _legend_top_left(plot, element):
        plot.state.legend.location = 'top_left'

    curve_opts['hooks'] = [_legend_top_left]
    return layout.opts(
        hv.opts.Curve(**curve_opts),
        hv.opts.NdLayout(title=f'Proportion of {src_label} particles by destination'),
    )


# ─── Comparison Figures ─────────────────────────────────────────────────────────


def comparison_count_timeseries(output_dir_a, stat_paths_a,
                                 output_dir_b, stat_paths_b,
                                 source_filter,
                                 destination='Montezuma_Slough_time',
                                 label_a='Scenario A', label_b='Scenario B'):
    """
    Overlay particle count timeseries for two scenarios (e.g. hist vs no-op).

    One panel per release cohort, filtered to a single source.

    Args:
        output_dir_a:   Output directory for scenario A.
        stat_paths_a:   stat_paths dict for scenario A.
        output_dir_b:   Output directory for scenario B.
        stat_paths_b:   stat_paths dict for scenario B.
        source_filter:  Release source name to plot (e.g. 'Sacramento_Full').
        destination:    Stat key for time-based stats.
        label_a:        Legend label for scenario A (default 'Scenario A').
        label_b:        Legend label for scenario B (default 'Scenario B').

    Returns:
        hv.NdLayout of overlaid curves.
    """
    ds_a = xr.open_mfdataset(f'{output_dir_a}/{stat_paths_a[destination]}')
    ds_b = xr.open_mfdataset(f'{output_dir_b}/{stat_paths_b[destination]}')

    releases_a, _, _ = parse_release_groups(ds_a['release_group_names'].values)
    releases_b, _, _ = parse_release_groups(ds_b['release_group_names'].values)

    src_label  = source_filter.replace('_', ' ')
    dest_label = destination.replace('_time', '').replace('_', ' ')
    date_map   = _load_release_date_map(output_dir_a)

    plots = []
    for release in sorted(set(releases_a) & set(releases_b)):
        panel_label = date_map.get(release, release)
        curves = {}
        for idx, source in releases_a.get(release, []):
            if source == source_filter:
                curves[label_a] = hv.Curve(
                    (ds_a['time'].values, ds_a['count'][:, idx, 0].values),
                    kdims='Time', vdims='Count',
                ).opts(color='blue')
        for idx, source in releases_b.get(release, []):
            if source == source_filter:
                curves[label_b] = hv.Curve(
                    (ds_b['time'].values, ds_b['count'][:, idx, 0].values),
                    kdims='Time', vdims='Count',
                ).opts(color='red')
        if curves:
            plots.append((panel_label, hv.NdOverlay(curves)))

    layout = hv.NdLayout(dict(plots), kdims='Release', sort=False).cols(2)
    return layout.opts(
        hv.opts.Curve(width=450, height=300, tools=['hover'], fontsize=HV_FONTSIZE),
        hv.opts.NdLayout(title=f'{src_label} \u2192 {dest_label}: {label_a} vs {label_b}'),
    )


def comparison_age_histograms(output_dir_a, stat_paths_a,
                               output_dir_b, stat_paths_b,
                               source_filter,
                               destination='Montezuma_Slough_age',
                               label_a='Scenario A', label_b='Scenario B'):
    """
    Side-by-side age histograms for two scenarios, filtered to a single source.

    One panel per release cohort with grouped bars colored by scenario.

    Args:
        output_dir_a:   Output directory for scenario A.
        stat_paths_a:   stat_paths dict for scenario A.
        output_dir_b:   Output directory for scenario B.
        stat_paths_b:   stat_paths dict for scenario B.
        source_filter:  Release source name to plot.
        destination:    Stat key for age-based stats.
        label_a:        Legend label for scenario A.
        label_b:        Legend label for scenario B.

    Returns:
        hv.NdLayout of grouped bar charts.
    """
    ds_a = xr.open_mfdataset(f'{output_dir_a}/{stat_paths_a[destination]}')
    ds_b = xr.open_mfdataset(f'{output_dir_b}/{stat_paths_b[destination]}')

    age_days_a = ds_a['age_bins'].values / (3600 * 24)
    age_days_b = ds_b['age_bins'].values / (3600 * 24)

    releases_a, _, _ = parse_release_groups(ds_a['release_group_names'].values)
    releases_b, _, _ = parse_release_groups(ds_b['release_group_names'].values)

    src_label  = source_filter.replace('_', ' ')
    dest_label = destination.replace('_age', '').replace('_', ' ')
    date_map   = _load_release_date_map(output_dir_a)

    plots = []
    for release in sorted(set(releases_a) & set(releases_b)):
        panel_label = date_map.get(release, release)
        rows = []
        for idx, source in releases_a.get(release, []):
            if source == source_filter:
                for ai, age in enumerate(age_days_a):
                    rows.append({'Age (days)': str(int(age)), 'Scenario': label_a,
                                 'Count': float(ds_a['count'].values[ai, idx, 0])})
        for idx, source in releases_b.get(release, []):
            if source == source_filter:
                for ai, age in enumerate(age_days_b):
                    rows.append({'Age (days)': str(int(age)), 'Scenario': label_b,
                                 'Count': float(ds_b['count'].values[ai, idx, 0])})
        if rows:
            df   = pd.DataFrame(rows)
            bars = hv.Bars(df, kdims=['Age (days)', 'Scenario'], vdims='Count').opts(
                width=450, height=300, tools=['hover'],
                cmap={label_a: 'blue', label_b: 'red'},
                color='Scenario', show_legend=True,
                hooks=[hide_source_labels],
            )
            plots.append((panel_label, bars))

    layout = hv.NdLayout(dict(plots), kdims='Release', sort=False).cols(2)
    return layout.opts(
        hv.opts.Bars(fontsize=HV_FONTSIZE),
        hv.opts.NdLayout(
            title=f'{src_label} \u2192 {dest_label}: {label_a} vs {label_b} by age'),
    )


def comparison_with_stats(source, destination, scenario_a, scenario_b,
                           outputs_dir='../outputs', show_pct_diff=True,
                           label_a='Scenario A', label_b='Scenario B'):
    """
    Overlay particle count timeseries for two scenarios with optional
    percent-difference curve (dual y-axis), and return per-cohort statistics.

    Args:
        source:       Release source name (e.g. 'Sacramento_Full').
        destination:  Stat key with '_time' suffix.
        scenario_a:   Name of the first scenario (e.g. 'hist_summer').
        scenario_b:   Name of the second scenario (e.g. 'no_op_summer').
        outputs_dir:  Base outputs directory (default '../outputs').
        show_pct_diff: If True, include a percent-difference curve with dual axis.
        label_a:      Legend label for scenario A.
        label_b:      Legend label for scenario B.

    Returns:
        (plot, stats_df) where stats_df has one row per release cohort with
        columns: release, mean_%, median_%, std_%, min_%, max_%,
                 q25_%, q75_%, iqr_%, pct_a_gt, pct_b_gt.
    """
    _, sp_a, od_a = pull_caseInfo_stats(scenario_a, outputs_dir=outputs_dir)
    _, sp_b, od_b = pull_caseInfo_stats(scenario_b, outputs_dir=outputs_dir)

    ds_a = xr.open_mfdataset(f'{od_a}/{sp_a[destination]}')
    ds_b = xr.open_mfdataset(f'{od_b}/{sp_b[destination]}')

    releases_a, _, _ = parse_release_groups(ds_a['release_group_names'].values)
    releases_b, _, _ = parse_release_groups(ds_b['release_group_names'].values)
    time       = ds_b['time'].values
    dest_label = destination.replace('_time', '').replace('_', ' ')

    def _make_hook(pd_arr):
        """Bokeh hook: move count curves to right y-axis, keep % diff on left."""
        def _hook(plot, element):
            p       = plot.state
            pd_min  = float(np.nanmin(pd_arr))
            pd_max  = float(np.nanmax(pd_arr))
            pad     = max((pd_max - pd_min) * 0.1, 1.0)
            count_range = Range1d(
                start=float(np.nanmin([ds_a['count'].values, ds_b['count'].values])),
                end=float(np.nanmax([ds_a['count'].values, ds_b['count'].values])),
            )
            p.extra_y_ranges['right'] = count_range
            p.add_layout(
                LinearAxis(y_range_name='right', axis_label='Count'), 'right')
            p.y_range.start = pd_min - pad
            p.y_range.end   = pd_max + pad
            glyph_renderers = [r for r in p.renderers if isinstance(r, GlyphRenderer)]
            for r in glyph_renderers[1:3]:
                r.y_range_name = 'right'
            p.add_layout(Span(
                location=0, dimension='width', y_range_name='right',
                line_color='red', line_dash='dashed', line_width=1,
            ))
        return _hook

    def _build_overlay(a, b, title, w, h):
        safe_denom = np.where(b == 0, 1.0, b)
        perc_diff  = np.nan_to_num(
            np.where(b == 0, np.nan, (a - b) / safe_denom * 100), nan=0.0
        )
        q25, q75   = float(np.percentile(perc_diff, 25)), float(np.percentile(perc_diff, 75))
        stats_row  = {
            'mean_%':    float(np.mean(perc_diff)),
            'median_%':  float(np.median(perc_diff)),
            'std_%':     float(np.std(perc_diff)),
            'min_%':     float(np.min(perc_diff)),
            'max_%':     float(np.max(perc_diff)),
            'q25_%':     q25,
            'q75_%':     q75,
            'iqr_%':     q75 - q25,
            'pct_a_gt':  float(np.mean(perc_diff > 0) * 100),
            'pct_b_gt':  float(np.mean(perc_diff < 0) * 100),
        }
        if show_pct_diff:
            overlay = (
                hv.Curve((time, perc_diff), kdims='Time',
                         vdims='Percent Difference (%)', label='% Diff'
                         ).opts(color='black', tools=['hover'])
                * hv.Curve((time, a), kdims='Time',
                           vdims=f'{label_a} Count', label=label_a
                           ).opts(color='blue', tools=['hover'])
                * hv.Curve((time, b), kdims='Time',
                           vdims=f'{label_b} Count', label=label_b
                           ).opts(color='red', tools=['hover'])
                * hv.HLine(0).opts(color='grey', line_dash='dashed', line_width=1)
            ).opts(hv.opts.Overlay(
                width=w, height=h, title=title,
                ylabel='Percent Difference (%)',
                hooks=[_make_hook(perc_diff)],
            ), hv.opts.Curve(fontsize=HV_FONTSIZE))
        else:
            overlay = (
                hv.Curve((time, a), kdims='Time', vdims='Count', label=label_a
                         ).opts(color='blue', tools=['hover'])
                * hv.Curve((time, b), kdims='Time', vdims='Count', label=label_b
                         ).opts(color='red', tools=['hover'])
            ).opts(hv.opts.Overlay(width=w, height=h, title=title, ylabel='Count'),
                   hv.opts.Curve(fontsize=HV_FONTSIZE))
        return overlay, stats_row

    stat_rows = []
    date_map  = _load_release_date_map(od_a)
    plots     = []
    for release in sorted(set(releases_a) & set(releases_b)):
        idx_a = next((i for i, src in releases_a.get(release, []) if src == source), None)
        idx_b = next((i for i, src in releases_b.get(release, []) if src == source), None)
        if idx_a is None or idx_b is None:
            continue
        a           = ds_a['count'][:, idx_a, 0].values
        b           = ds_b['count'][:, idx_b, 0].values
        panel_label = date_map.get(release, release)
        panel_plot, row = _build_overlay(a, b, title=panel_label, w=450, h=300)
        row['release'] = panel_label
        stat_rows.append(row)
        plots.append((panel_label, panel_plot))

    plot = hv.NdLayout(dict(plots), kdims='Release', sort=False).cols(2).opts(
        hv.opts.NdLayout(
            title=f'{source} \u2192 {dest_label}: {label_a} vs {label_b}')
    )
    cols     = ['release', 'mean_%', 'median_%', 'std_%', 'min_%', 'max_%',
                'q25_%', 'q75_%', 'iqr_%', 'pct_a_gt', 'pct_b_gt']
    stats_df = pd.DataFrame(stat_rows)[cols]
    return plot, stats_df


def comparison_proportion_timeseries(source, destination,
                                      scenario_a, scenario_b,
                                      outputs_dir='../outputs', xlim_end=None,
                                      label_a='Scenario A', label_b='Scenario B'):
    """
    Overlay proportion timeseries for two scenarios with per-cohort statistics.

    Proportion = count[:, idx, 0] / num_released[:, idx].

    Args:
        source:       Release source name (e.g. 'Sacramento_Full').
        destination:  Stat key with '_time' suffix.
        scenario_a:   Name of the first scenario.
        scenario_b:   Name of the second scenario.
        outputs_dir:  Base outputs directory.
        xlim_end:     Optional x-axis end date string.
        label_a:      Legend label for scenario A.
        label_b:      Legend label for scenario B.

    Returns:
        (plot, stats_df) \u2014 hv layout and summary DataFrame.
    """
    _, sp_a, od_a = pull_caseInfo_stats(scenario_a, outputs_dir=outputs_dir)
    _, sp_b, od_b = pull_caseInfo_stats(scenario_b, outputs_dir=outputs_dir)

    ds_a = xr.open_mfdataset(f'{od_a}/{sp_a[destination]}')
    ds_b = xr.open_mfdataset(f'{od_b}/{sp_b[destination]}')

    releases_a, _, _ = parse_release_groups(ds_a['release_group_names'].values)
    releases_b, _, _ = parse_release_groups(ds_b['release_group_names'].values)
    time       = ds_a['time'].values
    dest_label = destination.replace('_time', '').replace('_', ' ')
    src_label  = source.replace('_', ' ')
    date_map   = _load_release_date_map(od_a)

    def _build_prop_overlay(prop_a, prop_b, title, w, h):
        diff = np.where((prop_a + prop_b) > 0, prop_a - prop_b, 0.0)
        row  = {
            'mean_a':  float(np.nanmean(prop_a)),
            'mean_b':  float(np.nanmean(prop_b)),
            'mean_diff': float(np.nanmean(diff)),
            'max_a':   float(np.nanmax(prop_a)),
            'max_b':   float(np.nanmax(prop_b)),
            'max_diff': float(np.nanmax(diff)),
            'min_diff': float(np.nanmin(diff)),
        }
        overlay = (
            hv.Curve((time, prop_a), kdims='Time',
                     vdims='Proportion', label=label_a
                     ).opts(color='blue', tools=['hover'])
            * hv.Curve((time, prop_b), kdims='Time',
                       vdims='Proportion', label=label_b
                       ).opts(color='red', tools=['hover'])
        ).opts(hv.opts.Overlay(width=w, height=h, title=title, ylabel='Proportion'),
               hv.opts.Curve(fontsize=HV_FONTSIZE))
        return overlay, row

    stat_rows = []
    plots     = []
    for release in sorted(set(releases_a) & set(releases_b)):
        idx_a = next((i for i, src in releases_a.get(release, []) if src == source), None)
        idx_b = next((i for i, src in releases_b.get(release, []) if src == source), None)
        if idx_a is None or idx_b is None:
            continue
        prop_a = np.where(
            ds_a['num_released'][:, idx_a].values > 0,
            ds_a['count'][:, idx_a, 0].values / ds_a['num_released'][:, idx_a].values,
            0.0)
        prop_b = np.where(
            ds_b['num_released'][:, idx_b].values > 0,
            ds_b['count'][:, idx_b, 0].values / ds_b['num_released'][:, idx_b].values,
            0.0)
        panel_label = date_map.get(release, release)
        panel_plot, row = _build_prop_overlay(prop_a, prop_b, panel_label, w=450, h=300)
        row['release']  = panel_label
        stat_rows.append(row)
        plots.append((panel_label, panel_plot))

    plot = hv.NdLayout(dict(plots), kdims='Release', sort=False).cols(2).opts(
        hv.opts.NdLayout(
            title=f'{src_label} \u2192 {dest_label}: Proportion — {label_a} vs {label_b}')
    )
    cols     = ['release', 'mean_a', 'mean_b', 'mean_diff',
                'max_a', 'max_b', 'max_diff', 'min_diff']
    stats_df = pd.DataFrame(stat_rows)[cols]
    return plot, stats_df


# ─── Animation: Background Layers ──────────────────────────────────────────────


def build_polygon_layer(config_path):
    """
    Build colored polygon overlays for release and destination regions.

    Args:
        config_path:  Path to params/config.yaml.

    Returns:
        hv.Overlay of release (blue) and destination (red) polygon layers.
    """
    release_gdf, dest_gdf = _load_region_gdfs(config_path, target_crs='epsg:3857')

    def _gdf_to_paths(gdf, region_type):
        paths = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            polys = [geom] if geom.geom_type == 'Polygon' else geom.geoms
            for poly in polys:
                xs, ys = poly.exterior.coords.xy
                paths.append({
                    'x': np.array(xs), 'y': np.array(ys),
                    'name': row.get('name', ''),
                    'region_type': region_type,
                })
        return paths

    release_layer = hv.Polygons(
        _gdf_to_paths(release_gdf, 'release'),
        kdims=['x', 'y'], vdims=['name', 'region_type'],
    ).opts(fill_color='steelblue', fill_alpha=0.1,
           line_color='steelblue', line_width=2, tools=['hover'])

    dest_layer = hv.Polygons(
        _gdf_to_paths(dest_gdf, 'destination'),
        kdims=['x', 'y'], vdims=['name', 'region_type'],
    ).opts(fill_color='firebrick', fill_alpha=0.1,
           line_color='firebrick', line_width=2, tools=['hover'])

    return release_layer * dest_layer


def build_background(config_path, tiles='CartoLight', axis_lims=None):
    """
    Compose static background layers: basemap tiles and region polygons.

    Args:
        config_path:  Path to params/config.yaml.
        tiles:        Tile source name — 'CartoLight', 'EsriImagery', etc.
                      Set to None to skip tiles.
        axis_lims:    [lon_min, lon_max, lat_min, lat_max] in WGS84 decimal degrees.

    Returns:
        HoloViews Overlay of tiles * polygons in Web Mercator coordinates.
    """
    bg = getattr(gv.tile_sources, tiles)() if tiles is not None else hv.Overlay()
    bg = bg * build_polygon_layer(config_path)

    plot_opts = dict(width=800, height=600)
    if axis_lims is not None:
        to_mercator = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
        x_min, y_min = to_mercator.transform(axis_lims[0], axis_lims[2])
        x_max, y_max = to_mercator.transform(axis_lims[1], axis_lims[3])
        plot_opts['xlim'] = (x_min, x_max)
        plot_opts['ylim'] = (y_min, y_max)

    return bg.opts(hv.opts.Overlay(**plot_opts))


def build_gridded_stats_layer(gridded_nc_path, release_group=0,
                              time_start=None, time_end=None,
                              normalise=True, cmap='viridis', alpha=0.7,
                              clim=None, utm_crs='EPSG:32610'):
    """
    Build a HoloViews QuadMesh of gridded particle count statistics,
    reprojected to Web Mercator for overlay on a tile basemap.

    Supports both snapshot and time-averaged modes:
      - Snapshot:  set time_start == time_end (or pass a single datetime).
      - Averaging: set time_start < time_end to average over [time_start, time_end].
      - Default (both None): uses the final timestep as a snapshot.

    Args:
        gridded_nc_path:  Path to the stats_gridded_time_2D_*.nc file.
        release_group:    Integer index or string name of the release group.
        time_start:       Start of averaging window. Accepts numpy datetime64,
                          pandas Timestamp, or 'YYYY-MM-DD HH:MM' string.
                          If None, defaults to the final timestep.
        time_end:         End of averaging window (inclusive). Same types as
                          time_start. If equal to time_start, returns a snapshot.
        normalise:        If True, divide counts by total released (proportion).
        cmap:             Colormap name.
        alpha:            Layer opacity (0-1).
        clim:             (min, max) colour limits. Auto-scaled if None.
        utm_crs:          CRS of the grid coordinates in the NetCDF file.

    Returns:
        hv.QuadMesh in Web Mercator (EPSG:3857) coordinates.
    """
    from bokeh.models import HoverTool

    ds = xr.open_dataset(gridded_nc_path)

    rg_names = ds['release_group_names'].values.tolist()
    if isinstance(release_group, str):
        rg_idx = rg_names.index(release_group)
    else:
        rg_idx = int(release_group)

    times = ds['time'].values  # numpy datetime64 array

    if time_start is None and time_end is None:
        count = ds['count'].isel(time_dim=-1, release_group_dim=rg_idx).values
    else:
        ta = np.datetime64(time_start) if time_start is not None else times[-1]
        tb = np.datetime64(time_end)   if time_end   is not None else ta

        mask = (times >= ta) & (times <= tb)
        if not mask.any():
            raise ValueError(
                f'No timesteps found in window [{ta}, {tb}]. '
                f'Available range: {times[0]} – {times[-1]}'
            )

        count_window = ds['count'].isel(release_group_dim=rg_idx).values[mask]
        count = count_window[0] if count_window.shape[0] == 1 else count_window.mean(axis=0)

    if normalise:
        n_released = float(ds['number_released_each_release_group'].values[rg_idx])
        if n_released > 0:
            count = count / n_released

    x_utm = ds['x'].values[rg_idx]
    y_utm = ds['y'].values[rg_idx]
    ds.close()

    to_mercator = Transformer.from_crs(utm_crs, 'EPSG:3857', always_xy=True)
    xx, yy = np.meshgrid(x_utm, y_utm)
    xm, ym = to_mercator.transform(xx, yy)

    connectivity = np.where(count > 0, count, np.nan)

    hover = HoverTool(tooltips=[('connectivity', '@connectivity{0.0000}')])

    mesh = hv.QuadMesh((xm, ym, connectivity), kdims=['x', 'y'], vdims=['connectivity'])

    opts = dict(
        cmap=cmap, alpha=alpha,
        colorbar=True,
        colorbar_opts={'title': 'Connectivity' if normalise else 'Count'},
        tools=[hover],
        width=800, height=600,
        apply_ranges=False,
    )
    if clim is not None:
        opts['clim'] = clim

    return mesh.opts(**opts)


# ─── Animation: Particle Layers ────────────────────────────────────────────────


def build_particle_layer(tracks_ds, time_idx=0, size=4, alpha=0.6,
                          color_by_group=True, sample_fraction=None,
                          utm_crs='EPSG:32610'):
    """
    Build a particle scatter layer for one timestep in Web Mercator coordinates.

    Args:
        tracks_ds:        xr.Dataset from load_tracks().
        time_idx:         Integer index into the time dimension.
        size:             Marker size in screen pixels.
        alpha:            Marker opacity (0\u20131).
        color_by_group:   If True, color particles by release group.
        sample_fraction:  Optional float 0\u20131 to subsample alive particles.
        utm_crs:          CRS of the track coordinates (default EPSG:32610,
                          WGS 84 / UTM zone 10N).  Override for other regions.

    Returns:
        hv.NdOverlay of hv.Points elements in Web Mercator (EPSG:3857).
    """
    from bokeh.models import HoverTool

    rg_names       = tracks_ds.attrs.get('release_group_names', {})
    all_group_names = sorted(rg_names.values())
    color_map      = {name: Category10_10[i % len(Category10_10)]
                      for i, name in enumerate(all_group_names)}

    status = tracks_ds['status'].isel(time=time_idx).values
    alive  = status > 0

    hover    = HoverTool(tooltips=[('Release Group', '@release_group')])
    base_opts = dict(size=size, alpha=alpha, tools=[hover],
                     marker='circle', fill_alpha=0, line_width=1,
                     width=800, height=600, apply_ranges=False)

    if not alive.any():
        return hv.NdOverlay({
            name: hv.Points(
                pd.DataFrame({'x': [], 'y': [], 'release_group': []}),
                kdims=['x', 'y'], vdims=['release_group'],
            ).opts(color=color_map[name], **base_opts)
            for name in all_group_names
        })

    alive_idx = np.where(alive)[0]
    if sample_fraction is not None and 0 < sample_fraction < 1:
        n_sample  = max(1, int(len(alive_idx) * sample_fraction))
        rng       = np.random.default_rng(seed=42)
        alive_idx = rng.choice(alive_idx, size=n_sample, replace=False)

    x_utm = tracks_ds['x'].isel(time=time_idx).values[alive_idx]
    y_utm = tracks_ds['y'].isel(time=time_idx).values[alive_idx]

    to_mercator = Transformer.from_crs(utm_crs, 'EPSG:3857', always_xy=True)
    x_merc, y_merc = to_mercator.transform(x_utm, y_utm)

    rg     = tracks_ds['IDrelease_group'].values[alive_idx]
    labels = np.array([rg_names.get(int(g), f'group_{int(g)}') for g in rg])

    layers = {}
    for name in all_group_names:
        mask = labels == name
        df   = pd.DataFrame({
            'x': x_merc[mask] if mask.any() else [],
            'y': y_merc[mask] if mask.any() else [],
            'release_group': name if mask.any() else [],
        })
        layers[name] = hv.Points(
            df, kdims=['x', 'y'], vdims=['release_group'],
        ).opts(color=color_map[name], **base_opts)

    return hv.NdOverlay(layers)


def build_particle_animation(tracks_ds, background, size=2, alpha=0.3,
                              color_by_group=True, sample_fraction=None,
                              player_interval=200, utm_crs='EPSG:32610'):
    """
    Build an interactive Panel particle animation with play/pause controls.

    Args:
        tracks_ds:        xr.Dataset from load_tracks().
        background:       HoloViews Overlay from build_background().
        size:             Marker size for particles.
        alpha:            Marker opacity (0\u20131).
        color_by_group:   If True, color particles by release group.
        sample_fraction:  Optional float 0\u20131 to subsample particles per frame.
        player_interval:  Milliseconds between frames during playback.
        utm_crs:          CRS of track coordinates (default EPSG:32610).

    Returns:
        pn.Column with timestamp label, map overlay, and Player widget.
    """
    n_times = tracks_ds.dims['time']
    times   = tracks_ds['time'].values

    player = pn.widgets.Player(
        name='Timestep', start=0, end=n_times - 1, value=0,
        interval=player_interval, loop_policy='loop', width=800,
    )
    timestamp_pane = pn.pane.Markdown('', styles={'font-size': f'{FONT_LABELS}px'})

    def _update(time_idx):
        t       = pd.to_datetime(times[time_idx], unit='s')
        frac_str = f'sample={sample_fraction:.0%}' if sample_fraction else 'all particles'
        title   = (f'{t.strftime("%Y-%m-%d %H:%M")}  '
                   f'(step {time_idx}/{n_times - 1}, {frac_str})')
        timestamp_pane.object = f'**{title}**'
        particles = build_particle_layer(
            tracks_ds, time_idx=time_idx,
            size=size, alpha=alpha,
            color_by_group=color_by_group,
            sample_fraction=sample_fraction,
            utm_crs=utm_crs,
        )
        return (background * particles).opts(
            hv.opts.Overlay(framewise=False, title=title)
        )

    dmap = hv.DynamicMap(pn.bind(_update, time_idx=player))
    return pn.Column(timestamp_pane, dmap, player)


# ─── Animation: Frame Rendering (module-level for multiprocessing) ──────────────

# Module-level state populated by export_animation before spawning workers.
# Using fork on Linux, child processes inherit these without serialization.
_frame_ctx: dict = {}


def _render_frame(args):
    """
    Render a single animation frame to a PNG file.

    Designed to run in a multiprocessing.Pool worker.  All state is read
    from the module-level ``_frame_ctx`` dict, shared copy-on-write after fork.

    When ctx['basemap'] is True, the pre-cached CartoDB basemap image is
    composited and tick labels are converted to lat/lon.  Otherwise plain
    UTM axes with Easting/Northing labels are used.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    i, frame_path = args
    ctx   = _frame_ctx
    t     = pd.to_datetime(ctx['times'][i], unit='s')
    sf    = ctx['sample_fraction']
    frac_str = f'sample={sf:.0%}' if sf else 'all particles'
    title = (f'{t.strftime("%Y-%m-%d %H:%M")}  '
             f'(step {i}/{ctx["n_times"] - 1}, {frac_str})')

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_title(title, fontsize=MPL_FONT_TITLE)
    ax.set_aspect('equal')

    ctx['release_gdf'].plot(ax=ax, facecolor='lightgrey', edgecolor='grey',
                             alpha=0.2, linewidth=1.5)
    ctx['dest_gdf'].plot(ax=ax, facecolor='lightgrey', edgecolor='grey',
                          alpha=0.2, linewidth=1.5)

    status = ctx['status'][i]
    alive  = status > 0
    if alive.any():
        alive_idx = np.where(alive)[0]
        if ctx['sample_idx'] is not None:
            alive_idx = np.intersect1d(alive_idx, ctx['sample_idx'])
        x_utm  = ctx['x'][i][alive_idx]
        y_utm  = ctx['y'][i][alive_idx]
        rg     = ctx['id_rg'][alive_idx]
        labels = np.array([
            ctx['rg_names'].get(int(g), f'group_{int(g)}') for g in rg
        ])
        for name in ctx['all_group_names']:
            mask = labels == name
            if mask.any():
                ax.scatter(x_utm[mask], y_utm[mask],
                           s=ctx['size'], alpha=ctx['alpha'], label=name,
                           color=ctx['color_map'][name], edgecolors='none',
                           zorder=5)

    if i == 0:
        ax.legend(loc='upper left', fontsize=MPL_FONT_LEGEND, framealpha=0.8)

    if ctx['xlim'] is not None:
        ax.set_xlim(ctx['xlim'])
        ax.set_ylim(ctx['ylim'])

    if ctx['basemap']:
        cache = ctx['basemap_cache']
        ax.imshow(cache['image'], extent=cache['extent'], aspect='equal', zorder=0)

        to_lonlat = Transformer.from_crs(ctx['utm_crs'], 'EPSG:4326', always_xy=True)
        xlabels = [f'{to_lonlat.transform(xt, ctx["ylim"][0])[0]:.2f}\u00b0'
                   for xt in ax.get_xticks()]
        ylabels = [f'{to_lonlat.transform(ctx["xlim"][0], yt)[1]:.2f}\u00b0'
                   for yt in ax.get_yticks()]
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    else:
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')

    fig.savefig(frame_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return i


# ─── Tile Pre-caching ──────────────────────────────────────────────────────────


def precache_tiles(config_path='../params/config.yaml', axis_lims=None,
                   zoom=12, utm_crs='EPSG:32610'):
    """
    Download CartoDB basemap tiles and save a pre-rendered NPZ cache for
    offline use in animations.

    Run on a machine with internet access before submitting animation jobs
    to compute nodes that may not have internet.

    Args:
        config_path:  Path to config.yaml (reads paths.tile_cache_dir).
        axis_lims:    [lon_min, lon_max, lat_min, lat_max] in WGS84.
                      Required — raises ValueError if None.
        zoom:         Tile zoom level (default 12).
        utm_crs:      CRS for the UTM projection used in animations
                      (default EPSG:32610, WGS 84 / UTM zone 10N).
    """
    import contextily as cx
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if axis_lims is None:
        raise ValueError('axis_lims is required for precache_tiles.')

    config, config_dir = _load_config(config_path)
    tile_cache_dir = os.path.normpath(
        os.path.join(config_dir, config['paths']['tile_cache_dir'])
    )
    os.makedirs(tile_cache_dir, exist_ok=True)

    tile_url = 'https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png'

    to_utm = Transformer.from_crs('EPSG:4326', utm_crs, always_xy=True)
    x_min, y_min = to_utm.transform(axis_lims[0], axis_lims[2])
    x_max, y_max = to_utm.transform(axis_lims[1], axis_lims[3])

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    cx.add_basemap(ax, crs=utm_crs, source=tile_url, zoom=zoom)
    fig.canvas.draw()

    outpath = os.path.join(tile_cache_dir, 'basemap_cache.npz')
    im = ax.images[0]
    np.savez_compressed(outpath, image=im.get_array(), extent=im.get_extent())
    plt.close(fig)
    print(f'Basemap cached at {outpath}')
    print(f'  UTM extent: [{x_min:.0f}, {x_max:.0f}, {y_min:.0f}, {y_max:.0f}]')


# ─── GIF Export ────────────────────────────────────────────────────────────────


def export_animation(tracks_ds, output_path, fps=18,
                     size=4, alpha=0.3, color_by_group=True,
                     sample_fraction=0.05, config_path='../params/config.yaml',
                     axis_lims=None, n_workers=1,
                     basemap=True, zoom=12, utm_crs='EPSG:32610'):
    """
    Export particle animation as a GIF using matplotlib.  No browser required.

    Args:
        tracks_ds:        xr.Dataset from load_tracks().
        output_path:      Destination file path (e.g. 'animation.gif').
        fps:              Frames per second in the output GIF.
        size:             Marker size for particles.
        alpha:            Marker opacity (0\u20131).
        color_by_group:   If True, color particles by release group.
        sample_fraction:  Optional float 0\u20131 to subsample particles.
        config_path:      Path to config.yaml (for polygon layers).
        axis_lims:        [lon_min, lon_max, lat_min, lat_max] in WGS84.
        n_workers:        Number of parallel workers for frame rendering.
        basemap:          If True, composite pre-cached CartoDB tiles (requires
                          precache_tiles() to have been run first).
        zoom:             Tile zoom level (kept for API compatibility with
                          precache_tiles; not used at render time).
        utm_crs:          CRS of track coordinates (default EPSG:32610).
                          Must match the CRS used when running precache_tiles.
    """
    import tempfile
    import imageio.v3 as iio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from codebase.util import load_polygons

    config, config_dir = _load_config(config_path)
    polygons_dir = os.path.normpath(
        os.path.join(config_dir, config['paths']['polygons_dir'])
    )
    release_gdf = load_polygons(polygons_dir, config['polygons']['release_regions'])
    dest_gdf    = load_polygons(polygons_dir, config['polygons']['destination_regions'])

    rg_names        = tracks_ds.attrs.get('release_group_names', {})
    all_group_names = sorted(rg_names.values())
    tab10           = plt.cm.tab10
    color_map       = {name: tab10(i % 10) for i, name in enumerate(all_group_names)}

    if sample_fraction is not None and 0 < sample_fraction < 1:
        n_sample   = max(1, int(tracks_ds.sizes['particle'] * sample_fraction))
        rng        = np.random.default_rng(seed=42)
        sample_idx = rng.choice(tracks_ds.sizes['particle'],
                                size=n_sample, replace=False)
    else:
        sample_idx = None

    n_times = tracks_ds.sizes.get('time', tracks_ds.dims['time'])
    times   = tracks_ds['time'].values

    if axis_lims is not None:
        to_utm = Transformer.from_crs('EPSG:4326', utm_crs, always_xy=True)
        x_min, y_min = to_utm.transform(axis_lims[0], axis_lims[2])
        x_max, y_max = to_utm.transform(axis_lims[1], axis_lims[3])
        xlim, ylim   = (x_min, x_max), (y_min, y_max)
    else:
        xlim = ylim = None

    basemap_cache = None
    if basemap:
        tile_cache_dir = os.path.normpath(
            os.path.join(config_dir, config['paths']['tile_cache_dir'])
        )
        cache_path = os.path.join(tile_cache_dir, 'basemap_cache.npz')
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f'Basemap cache not found: {cache_path}. '
                f'Run precache_tiles() on a node with internet access first.'
            )
        data          = np.load(cache_path)
        basemap_cache = {'image': data['image'], 'extent': data['extent']}

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_paths = [os.path.join(tmp_dir, f'frame_{i:05d}.png')
                       for i in range(n_times)]

        global _frame_ctx
        _frame_ctx = {
            'x':               tracks_ds['x'].values,
            'y':               tracks_ds['y'].values,
            'status':          tracks_ds['status'].values,
            'id_rg':           tracks_ds['IDrelease_group'].values,
            'times':           times,
            'n_times':         n_times,
            'rg_names':        rg_names,
            'all_group_names': all_group_names,
            'color_map':       color_map,
            'sample_idx':      sample_idx,
            'sample_fraction': sample_fraction,
            'release_gdf':     release_gdf,
            'dest_gdf':        dest_gdf,
            'xlim':            xlim,
            'ylim':            ylim,
            'basemap':         basemap,
            'basemap_cache':   basemap_cache,
            'utm_crs':         utm_crs,
            'size':            size,
            'alpha':           alpha,
        }
        work_items = list(zip(range(n_times), frame_paths))

        if n_workers > 1:
            import multiprocessing as mp
            print(f'Rendering {n_times} frames with {n_workers} workers...')
            with mp.Pool(n_workers) as pool:
                for done, _ in enumerate(
                        pool.imap_unordered(_render_frame, work_items), 1):
                    print(f'\rRendered {done}/{n_times}', end='', flush=True)
        else:
            for item in work_items:
                _render_frame(item)
                print(f'\rRendered frame {item[0] + 1}/{n_times}',
                      end='', flush=True)

        print('\nStitching frames...')
        frames   = [iio.imread(p) for p in frame_paths]
        duration = 1000 // fps
        iio.imwrite(output_path, frames, duration=duration, loop=0)

    print(f'Saved to {output_path}')
