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
│   ├── notebooks/          #   analysis notebooks
│   ├── outputs/
│   │   ├── animations/
│   │   ├── figures/
│   │   └── PTM/            #   OceanTracker track output
│   ├── params/
│   │   ├── config.yaml     #   baseline OceanTracker config (Jinja2 template)
│   │   └── prod/           #   run-variant override YAMLs (add as needed)
│   └── scripts/
│       ├── main.py         #   experiment entry point
│       ├── run.py          #   OceanTracker runner
│       └── setup.py        #   config builder (YAML merge + Python extensions)
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

reader:
  file_mask: "schism_*.nc"

run:
  duration: 604800        # 7 days (seconds)
  time_step: 120

release_groups:
  - name: source_1
    points:
      - [174.76, -36.85]
    release_interval: 3600
    pulse_size: 10
```

### 3 — Generate the experiment directory

```bash
ot-generate experiment_configs/my_release.yaml
```

This creates `experiments/my_release/` with all template files rendered.
The directory is automatically excluded from git.

### 4 — Add hydrodynamic data

Symlink or copy your NetCDF files into the experiment's `data/hydro/` directory:

```bash
ln -s /path/to/ocean/model/output/* experiments/my_release/data/hydro/
```

### 5 — Run the experiment

```bash
# Baseline config only:
python experiments/my_release/scripts/main.py

# With a prod/ variant merged on top:
python experiments/my_release/scripts/main.py --variant sensitivity_A
```

OceanTracker track files are written to `experiments/my_release/outputs/PTM/`.

---

## Experiment YAML schema

| Key | Required | Type | Description |
|---|---|---|---|
| `name` | **yes** | string | Experiment name; used as the directory name under `experiments/` |
| `description` | no | string | Free-text description injected into the generated config header |
| `reader.file_mask` | no | string | Glob pattern for hydrodynamic NetCDF files (default: `*.nc`) |
| `run.duration` | no | int | Simulation duration in seconds (default: `86400`) |
| `run.time_step` | no | int | Model time step in seconds (default: `120`) |
| `run.write_tracks` | no | bool | Write particle tracks to file (default: `true`) |
| `release_groups` | no | list | List of release group definitions (see example) |

Every key defined in the YAML becomes available as a Jinja2 variable inside template files.

---

## Run variants (prod/ directory)

Within a single experiment you can define multiple run variants without creating new experiment directories.  Add YAML files to `experiments/<name>/params/prod/`:

```
params/
├── config.yaml         ← shared baseline
└── prod/
    ├── sensitivity_A.yaml   ← override: longer duration
    └── sensitivity_B.yaml   ← override: different release points
```

Each `prod/` file contains only the keys you want to override (deep-merged on top of `config.yaml`):

```yaml
# params/prod/sensitivity_A.yaml
run:
  duration: 1209600    # 14 days
```

Run with:

```bash
python experiments/my_release/scripts/main.py --variant sensitivity_A
```

---

## How templating works

1. `ot-generate` reads your `experiment_configs/*.yaml` into a Python dict.
2. It copies `template/` to `experiments/<name>/`, walking every file.
3. Text files (`.py`, `.yaml`, `.yml`, etc.) are rendered through Jinja2 with the config dict as context.
4. `{{ name }}`, `{{ run.duration }}`, and any other key from your YAML become resolved values in the output files.

Binary files and `.gitkeep` placeholders are copied without rendering.

---

## Extending setup.py

`experiments/<name>/scripts/setup.py` is the place for programmatic config extensions that are easier to express in Python than YAML — for example, building release point lists dynamically or adding OceanTracker particle property classes:

```python
# inside build_params():
params["particle_properties"].append({
    "class_name": "oceantracker.particle_properties.age_decay.AgeDecay",
    "decay_time_scale": 3600,
})
```

These additions run *after* the YAML merge, so they always take final effect regardless of which variant is selected.

---

## CLI reference

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
