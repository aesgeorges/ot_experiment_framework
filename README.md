# OceanTracker Experiment Framework

A lightweight framework for generating and running reproducible [OceanTracker](https://oceantracker.github.io) particle-tracking experiments.

Each experiment is driven by a single YAML file, version-controlled under `experiment_configs/`. The framework copies the shared `template/` scaffold into a generated `experiments/<name>/` directory (excluded from git), rendering Jinja2 placeholders with values from your config.

---

## Repo layout

```
ot_experiment_framework/
├── experiment_configs/     # ← git-tracked YAML configs (one per experiment)
├── experiments/            # ← generated experiment dirs (git-ignored)
├── framework/              # ← core framework code
│   ├── __init__.py
│   ├── generate.py         #   CLI entry point
│   └── renderer.py         #   Jinja2 rendering logic
├── template/               # ← scaffold copied for every new experiment
│   ├── codebase/
│   │   ├── plot.py         #   plotting helpers
│   │   └── util.py         #   general utilities
│   ├── data/
│   │   ├── hydro/          #   raw hydrodynamic input (symlink or copy here)
│   │   ├── polygons/       #   geospatial polygons
│   │   └── processed/      #   preprocessed / derived data
│   ├── logs/               #   model and run logs
│   ├── notebooks/
│   │   └── figures.ipynb   #   example analysis notebook
│   ├── outputs/
│   │   ├── animations/
│   │   ├── figures/
│   │   └── PTM/            #   OceanTracker track output
│   ├── params/
│   │   ├── config.yaml.j2  #   high-level experiment config (Jinja2 template, read by scripts/setup.py)
│   │   ├── base/
│   │   │   └── config.yaml.j2  #   per-scenario OceanTracker run config (Jinja2 template)
│   │   └── prod/           #   generated per-(flow_config × period) OceanTracker YAMLs
│   └── scripts/
│       ├── main.py         #   experiment entry point (setup / run / animate)
│       ├── run.py          #   single OceanTracker simulation runner
│       └── setup.py        #   parameter file generator (builds prod/ YAMLs at runtime)
├── tests/
├── env.yml                 # Conda environment specification
└── pyproject.toml          # Project metadata and dependencies
```

---

## Quickstart

### 1 — Set up the environment

```bash
conda env create -f env.yml
conda activate ot_framework
pip install -e .          # installs the `ot-generate` CLI
```

### 2 — Create an experiment config

Copy the provided example and edit it:

```bash
cp experiment_configs/example.yaml experiment_configs/my_release.yaml
```

Edit `experiment_configs/my_release.yaml`:

```yaml
name: my_release          # becomes experiments/my_release/
description: "Harbour entrance larvae release, summer 2024"

# Cluster / user settings
user:
  email: you@institution.edu
  conda_path: /home/<user>/miniconda3/etc/profile.d/conda.sh
  env: oceantracker
  viz_env: oceantracker_viz
  output_dir: "../outputs/PTM/"

# Data paths (relative to experiment root after generation)
paths:
  hindcasts_source_dir: /scratch/projects/my_release/hindcasts/
  hindcasts_symlink_dir: ../data/hydro/
  oceantracker_base_dir: ../params/base
  tile_cache_dir: ../outputs/animations/tiles_cache
  use_symlink_inputs: false

# Hydrodynamic reader
reader:
  class_name: SCHISMreaderV5
  input_dir: ../data/hydro/
  file_mask: "*.nc"

# OceanTracker run settings (flat top-level fields)
root_output_dir: "../outputs/PTM/"
output_file_base: my_release
max_run_duration: 604800   # 7 days in seconds
time_step: 120             # seconds
write_tracks: true

# One entry per gate/scenario file in params/base/
flow_configs:
  - name: Baseline
    filename: config.yaml

# One entry per time window; setup.py creates one PTM run per (flow_config × period)
simulation_periods:
  - name: summer
    start_date: 2024-06-01
    n_releases: 4
    release_spacing_days: 7
    start_id: 1
    end_id: 30
```

### 3 — Generate the experiment directory

```bash
ot-generate experiment_configs/my_release.yaml
```

This creates `experiments/my_release/` with all template files rendered.
The directory is automatically excluded from git.

### 4 — Add hydrodynamic data

Symlink or copy your NetCDF files into the experiment's `data/hydro/` directory (or set `paths.hindcasts_symlink_dir` / `paths.hindcasts_source_dir` in your config and let `setup.py` build the symlinks automatically):

```bash
ln -s /path/to/ocean/model/output/* experiments/my_release/data/hydro/
```

### 5 — Build and run the experiment

`setup.py` generates one complete OceanTracker YAML per (flow_config × simulation_period) combination and writes them to `params/prod/`.  Then `run` submits them to Slurm:

```bash
# Build all parameter files (writes params/prod/*.yaml):
python experiments/my_release/scripts/main.py setup

# Submit all generated scenarios to Slurm:
python experiments/my_release/scripts/main.py run

# Or submit a single scenario by name:
python experiments/my_release/scripts/main.py run_single Baseline_summer
```

OceanTracker track files are written to `experiments/my_release/outputs/PTM/`.

---

## Experiment YAML schema

| Key | Required | Type | Description |
|---|---|---|---|
| `name` | **yes** | string | Experiment name; used as the directory name under `experiments/` |
| `description` | no | string | Free-text description injected into the generated config header |
| **`user` block** | | | |
| `user.email` | no | string | Email for Slurm job notifications (default: `user@institution.edu`) |
| `user.conda_path` | no | string | Path to `conda.sh` used by Slurm job scripts |
| `user.env` | no | string | Conda environment for OceanTracker runs (default: `oceantracker`) |
| `user.viz_env` | no | string | Conda environment for animation jobs (default: `oceantracker_viz`) |
| `user.output_dir` | no | string | OceanTracker root output path inside each Slurm job (default: `../outputs/PTM/`) |
| **`paths` block** | | | |
| `paths.hindcasts_source_dir` | no | string | Source directory of SCHISM hindcast NetCDF files |
| `paths.hindcasts_symlink_dir` | no | string | Experiment-relative path where hindcast symlinks are placed (default: `../data/hydro/`) |
| `paths.oceantracker_base_dir` | no | string | Location of `params/base/` (default: `../params/base`) |
| `paths.tile_cache_dir` | no | string | CartoDB tile cache for animation (default: `../outputs/animations/tiles_cache`) |
| `paths.use_symlink_inputs` | no | bool | Auto-build per-period symlink dirs from `hindcasts_source_dir` (default: `false`) |
| **`reader` block** | | | |
| `reader.class_name` | no | string | OceanTracker reader class (default: `SCHISMreaderV5`) |
| `reader.input_dir` | no | string | Path to hydrodynamic NetCDF input directory (default: `../data/hydro/`) |
| `reader.file_mask` | no | string | Glob pattern for hydrodynamic NetCDF files (default: `*.nc`) |
| **Run settings (flat top-level)** | | | |
| `max_run_duration` | no | int | Maximum simulation duration in seconds (default: `604800`) |
| `time_step` | no | int | Model time step in seconds (default: `120`) |
| `write_tracks` | no | bool | Write particle tracks to file (default: `true`) |
| `root_output_dir` | no | string | OceanTracker output root (default: `../outputs/PTM/`) |
| `output_file_base` | no | string | Base name for output files (default: `name`) |
| **`flow_configs` list** | | | |
| `flow_configs[].name` | **yes** | string | Human-readable scenario name |
| `flow_configs[].filename` | **yes** | string | Filename of the base OceanTracker config in `params/base/` |
| **`simulation_periods` list** | | | |
| `simulation_periods[].name` | **yes** | string | Period label (used in output file names) |
| `simulation_periods[].start_date` | **yes** | date | Simulation start date (`YYYY-MM-DD`) |
| `simulation_periods[].n_releases` | no | int | Number of sequential release pulses (default: `4`) |
| `simulation_periods[].release_spacing_days` | no | int | Days between consecutive release starts (default: `7`) |
| `simulation_periods[].start_id` | **yes** | int | First SCHISM file index for this period |
| `simulation_periods[].end_id` | **yes** | int | Last SCHISM file index for this period |
| **`polygons` block** | | | |
| `polygons.release_regions[].name` | no | string | Region label |
| `polygons.release_regions[].file` | no | string | Shapefile name in `data/polygons/` |
| `polygons.release_regions[].pulse_size` | no | int | Particles per release interval |
| `polygons.release_regions[].release_interval` | no | int | Seconds between pulses (default: `3600`) |
| `polygons.destination_regions[].name` | no | string | Region label |
| `polygons.destination_regions[].file` | no | string | Shapefile name in `data/polygons/` |

Every key defined in the YAML also becomes available as a Jinja2 variable inside template files (`params/config.yaml.j2`, `params/base/config.yaml.j2`).

---

## How setup.py generates run parameters

`scripts/setup.py` is the runtime parameter builder.  It runs **after** `ot-generate` and is invoked via `main.py setup`.

1. Reads `params/config.yaml` (the rendered, experiment-level config).
2. Loads polygon geometries from `data/polygons/` (as defined in `polygons.release_regions` and `polygons.destination_regions`).
3. For every **(flow_config × simulation_period)** combination, it:
   - Reads the base OceanTracker config from `params/base/<flow_config.filename>`.
   - Builds polygon release groups (`PolygonRelease`) with configurable pulse size, spacing, and max age.
   - Builds polygon statistics (age-based and time-based) and gridded statistics (alive, stranded, outside grids).
   - Optionally creates per-period symlink directories for SCHISM hindcast files (when `paths.use_symlink_inputs: true`).
   - Writes one complete, runnable OceanTracker YAML to `params/prod/<flow_name>_<period_name>.yaml`.

The `params/prod/` directory therefore contains **fully resolved** parameter files that are submitted directly to OceanTracker without further merging.

Manual override YAMLs can still be placed in `params/prod/` as secondary entries if you need to run a one-off variation outside the (flow_config × period) matrix.

To extend the parameter generation with custom logic (e.g. dynamic release points, additional particle properties), edit `scripts/setup.py` directly. Changes you make there take effect the next time you run `main.py setup`.

---

## How templating works

`ot-generate` performs **two-tier rendering** using the experiment YAML as context:

**Tier 1 — Experiment-level config**

`template/params/config.yaml.j2` → `experiments/<name>/params/config.yaml`

This file controls high-level experiment behaviour (user/cluster settings, paths, flow configs, simulation periods, polygon definitions).  It is read by `scripts/setup.py` at runtime to drive parameter generation.

**Tier 2 — Per-scenario OceanTracker base config**

`template/params/base/config.yaml.j2` → `experiments/<name>/params/base/config.yaml`

This is the starting-point OceanTracker run configuration (reader settings, run duration, timestep, etc.).  `setup.py` reads this file (or a gate-specific variant named alongside it), mutates it programmatically, and writes the fully resolved YAML to `params/prod/`.

**How it runs:**

1. `ot-generate` reads your `experiment_configs/*.yaml` into a Python dict.
2. It copies `template/` to `experiments/<name>/`, walking every file.
3. Files with a `.j2` extension are rendered through Jinja2 with the config dict as context; the `.j2` suffix is stripped from the output filename.
4. All other text files (`.py`, `.ipynb`, etc.) are copied as-is; binary files and `.gitkeep` placeholders are copied without modification.
5. `{{ name }}`, `{{ user.email }}`, `{{ flow_configs }}`, and any other key from your YAML become resolved values in the output files.

---

## Extending setup.py

`experiments/<name>/scripts/setup.py` is the place for programmatic extensions to the parameter generation such as adding custom OceanTracker particle property classes, computing release points dynamically, or adjusting run settings per scenario:

```python
# inside setup_ptm(), after building the base params dict:
params["particle_properties"].append({
    "class_name": "oceantracker.particle_properties.age_decay.AgeDecay",
    "decay_time_scale": 3600,
})
```

These additions are applied before each YAML is written to `params/prod/`, so they take effect for every generated scenario.

---

## CLI reference

### `ot-generate` — scaffold a new experiment

```
ot-generate [OPTIONS] CONFIG

  Generate an experiment directory from a YAML CONFIG file.

Arguments:
  CONFIG          Path to an experiment YAML file under experiment_configs/

Options:
  --experiments-dir PATH  Override the output experiments/ directory
  --dry-run               Print what would be created without writing files
  --help                  Show this message and exit
```

### `scripts/main.py` — run and manage an experiment

After generating the experiment directory, all operational commands are issued through `scripts/main.py`:

```
python experiments/<name>/scripts/main.py COMMAND [OPTIONS]

Commands:
  setup                     Build OceanTracker parameter files for all
                            (flow_config × simulation_period) combinations.
                            Writes YAMLs to params/prod/.

  run                       Submit all generated param files in params/prod/
                            to Slurm as individual jobs.

  run_single SCENARIO_NAME  Submit a single scenario to Slurm.
                            SCENARIO_NAME is the stem of the YAML in params/prod/
                            (e.g. Baseline_summer).

  animate [--scenario NAME] Pre-cache CartoDB map tiles, then submit animation
                            rendering jobs to Slurm. If --scenario is omitted,
                            all (flow_config × period) combinations are animated.
```
