# Experiment Config Recipe

Notes on what works and what doesn't when writing experiment YAML configs for `ot-generate`.

## OceanTracker run fields are TOP-LEVEL — not nested under `run:`

**Wrong** (I generated this):
```yaml
run:
  duration: 604800
  time_step: 120
  write_tracks: true
```

**Correct** (actual OceanTracker fields, top-level in the base config):
```yaml
max_run_duration: 604800   # seconds
time_step: 120             # seconds
write_tracks: True
root_output_dir: "../outputs/PTM/"
output_file_base: "diversion_sink"
```

## Reader block needs `class_name` and `input_dir` explicitly

**Wrong**:
```yaml
reader:
  file_mask: "out2d_*.nc"
```

**Correct**:
```yaml
reader:
  class_name: SCHISMreaderV5
  input_dir: "../data/hydro/"
  file_mask: "*.nc"
```

- File mask is `"*.nc"` not `"out2d_*.nc"` — OceanTracker filters by what's in `input_dir`
- `input_dir` must be explicit; `paths.hindcasts_symlink_dir` is for `setup.py` logic only

## `release_groups` and other lists are populated by `setup.py`, not hardcoded

Leave these as empty lists in the base config — `setup.py` populates them programmatically:

```yaml
release_groups: []
particle_statistics: []
particle_properties: []
event_loggers: []
```

Do **not** hardcode point releases in the experiment YAML; the base OceanTracker config is
the right place for static release groups if needed, but setup.py generally owns this.

## `paths.hindcasts_symlink_dir` uses `../data/hydro/`

Not `../data/SCHISM/` — the symlink destination directory is `hydro/` by convention.

## `flow_configs` filename is the base YAML filename stem

The `filename` key is used directly: `setup.py` looks for `params/base/{filename}.yaml`.
Use `hist` / `no_op` for the real historical / no-operations scenarios.
